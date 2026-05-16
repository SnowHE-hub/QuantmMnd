#!/usr/bin/env python3
"""补充复权因子并计算 adj_close.

日线 OHLCV 已完整（2019-2026，1374只），只缺 adj_factor / adj_close。
使用官方 2000 积分 API（tushare.pro_api，不走代理）逐股下载。

输出：直接更新 data/raw/alpha_prices_panel.parquet（添加 adj_factor / adj_close 列）
预计时间：1374只 × ~0.5s = ~12 分钟

运行（须在已安装 tushare 的环境中，例如 conda 环境 quantmind，不要用 base）：
  conda activate quantmind
  python scripts/data_pipeline/patch_adj_close.py
  python scripts/data_pipeline/patch_adj_close.py --resume   # 断点续跑

  # 或直接指定解释器：
  # /path/to/miniforge3/envs/quantmind/bin/python scripts/data_pipeline/patch_adj_close.py
"""
from __future__ import annotations

import os
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH   = ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
CACHE_PATH   = ROOT / "data" / "raw" / "adj_cache.pkl"          # 断点缓存
UNIVERSE_TXT = ROOT / "data" / "alpha_universe" / "alpha_universe.txt"


def _init_pro():
    import tushare as ts
    token = os.environ.get(
        "TUSHARE_TOKEN",
        "64a18c359c1d28fab92fed6bebd1f1662cc6e34872ad9ee643b55f56",
    ).strip()
    ts.set_token(token)
    print("[API] 官方 2000积分（adj_factor专用）")
    return ts.pro_api(timeout=60)


def _retry(fn, label: str, attempts: int = 4, base_sleep: float = 5.0):
    for i in range(attempts):
        try:
            r = fn()
            if r is not None and not r.empty:
                return r
            return pd.DataFrame()
        except Exception as e:
            msg = str(e)[:100]
            print(f"    [{label}] {i+1}/{attempts} 失败: {msg}", flush=True)
            if "token" in msg.lower() or "权限" in msg or "积分" in msg:
                break
            wait = base_sleep * (i + 1)
            time.sleep(wait)
    return pd.DataFrame()


def compute_adj_close(price_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
    """合并复权因子并计算 adj_close."""
    df = price_df.copy()
    if adj_df.empty:
        df["adj_factor"] = np.nan
        df["adj_close"]  = pd.to_numeric(df["close"], errors="coerce")
        return df

    a = adj_df[["trade_date", "adj_factor"]].copy()
    a["trade_date"] = pd.to_datetime(a["trade_date"])
    a["adj_factor"] = pd.to_numeric(a["adj_factor"], errors="coerce")

    df = df.merge(a, on="trade_date", how="left")
    df["adj_factor"] = df["adj_factor"].bfill().ffill()

    latest = df["adj_factor"].dropna().iloc[-1] if df["adj_factor"].notna().any() else np.nan
    if pd.notna(latest) and latest > 0:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce") * df["adj_factor"] / float(latest)
    else:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")

    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="跳过已缓存的股票")
    ap.add_argument("--sleep",  type=float, default=0.35, help="每只间隔秒数")
    args = ap.parse_args()

    # 加载价格面板
    print(f"加载价格面板: {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    print(f"  {len(panel):,} 行 | {panel['ts_code'].nunique()} 只 | {panel['trade_date'].nunique()} 天")

    # 如果已经有 adj_close 且非全 NaN，问是否重跑
    if "adj_close" in panel.columns and panel["adj_close"].notna().sum() > 10000:
        print(f"  adj_close 已存在（{panel['adj_close'].notna().sum():,} 有效值），将覆盖更新")

    tickers = sorted(panel["ts_code"].unique().tolist())

    # 断点缓存
    adj_cache: dict[str, pd.DataFrame] = {}
    if args.resume and CACHE_PATH.exists():
        adj_cache = pickle.loads(CACHE_PATH.read_bytes())
        print(f"  断点续跑：缓存中已有 {len(adj_cache)} 只")

    pro = _init_pro()
    skipped = 0
    failed  = 0

    for i, ticker in enumerate(tickers):
        if ticker in adj_cache:
            skipped += 1
            continue

        adj = _retry(
            lambda t=ticker: pro.adj_factor(
                ts_code=t, start_date="20190101", end_date="20261231"
            ),
            label=f"adj {ticker}",
        )
        adj_cache[ticker] = adj
        if adj.empty:
            failed += 1

        if (i + 1) % 100 == 0:
            # 每100只保存一次缓存
            CACHE_PATH.write_bytes(pickle.dumps(adj_cache))
            coverage = len(adj_cache) - failed
            print(f"  进度 {i+1}/{len(tickers)} | 有效 {coverage} | 失败 {failed}", flush=True)

        time.sleep(args.sleep)

    # 保存最终缓存
    CACHE_PATH.write_bytes(pickle.dumps(adj_cache))
    print(f"\n复权因子下载完成：{len(adj_cache) - failed} 只有效，{failed} 只失败")

    # 合并 adj_close
    print("\n合并 adj_close...")
    parts: list[pd.DataFrame] = []
    for ticker, sub in panel.groupby("ts_code", sort=False):
        adj_df = adj_cache.get(ticker, pd.DataFrame())
        patched = compute_adj_close(sub.copy(), adj_df)
        parts.append(patched)

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["ts_code", "trade_date"])

    # 统计
    valid_adj = result["adj_close"].notna().sum()
    total     = len(result)
    print(f"  adj_close 覆盖率: {valid_adj:,}/{total:,} ({valid_adj/total:.1%})")

    # 保存
    result.to_parquet(PANEL_PATH, index=False, compression="snappy")
    print(f"  已保存: {PANEL_PATH}")

    # 清理缓存
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print("  缓存文件已清理")

    print("\n[完成] alpha_prices_panel.parquet 现在包含 adj_close 列")

    # 通知
    try:
        from scripts.notify import notify
        notify(
            "复权因子补全完成",
            f"adj_close 覆盖率 {valid_adj/total:.1%}，可以开始 Phase 1 了"
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
