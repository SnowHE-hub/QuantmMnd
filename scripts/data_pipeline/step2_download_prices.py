#!/usr/bin/env python3
"""Step 2: 下载 Alpha Universe 日线价格 + 复权因子（2019-2026）.

支持两种模式（--mode 参数）：

  by_stock（默认）：
    每只股票分年度请求，适合官方 API（全量7年 <32分钟）
    代理按年请求约 39 小时，可断点续传

  by_date（推荐用于代理）：
    每个交易日请求一次，返回全市场数据，再筛 Alpha Universe
    共 ~1700 次请求（约 7-10 小时），比 by_stock 快 4 倍

运行示例：
  # 官方API by_stock（推荐，32分钟）
  export TUSHARE_TOKEN="xxxx"
  python step2_download_prices.py --mode by_stock

  # 代理API by_date（推荐代理，7-10小时）
  export TUSHARE_TOKEN_HI="xxxx"
  export TUSHARE_HI_URL="http://tsy.xiaodefa.cn"
  python step2_download_prices.py --mode by_date

  # 代理API by_stock（年请求，39小时，支持断点续传）
  export TUSHARE_TOKEN_HI="xxxx"
  python step2_download_prices.py --mode by_stock --use-hi-api
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_TXT = ROOT / "data" / "alpha_universe" / "alpha_universe.txt"
OUT_DIR = ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUX_DIR = ROOT / "data" / "alpha_universe"
DONE_FILE = AUX_DIR / "price_done.txt"
FAILED_FILE = AUX_DIR / "price_failed.txt"

YEARS = list(range(2019, 2027))  # 2019-2026


# ── API 初始化 ──────────────────────────────────────────────────────────────

def _init_pro(hi_mode: bool = False):
    import tushare as ts

    if hi_mode:
        token = os.environ.get(
            "TUSHARE_TOKEN_HI",
            "5caf9b3022e13d4e915df0af19a076130287cb7837c0b020290691c8",
        ).strip()
        ts.set_token(token)
        # timeout=120s：早期日期（2019）返回约 32s，需要充足余量
        pro = ts.pro_api(timeout=120)
        url = os.environ.get("TUSHARE_HI_URL", "http://tsy.xiaodefa.cn")
        pro._DataApi__http_url = url
        print(f"[API] 高频代理 → {url}（timeout=120s）")
    else:
        token = os.environ.get(
            "TUSHARE_TOKEN",
            "64a18c359c1d28fab92fed6bebd1f1662cc6e34872ad9ee643b55f56",
        ).strip()
        ts.set_token(token)
        pro = ts.pro_api(timeout=120)
        print("[API] 官方 API（2000积分）")
    return pro


def _retry(fn, label: str, attempts: int = 5, base_sleep: float = 10.0):
    for i in range(attempts):
        try:
            r = fn()
            return r if r is not None else pd.DataFrame()
        except Exception as e:
            msg = str(e)[:120]
            print(f"    [{label}] {i+1}/{attempts} 失败: {msg}", flush=True)
            if "token" in msg.lower() or "权限" in msg or "积分" in msg:
                break
            wait = base_sleep * (i + 1)  # 10s / 20s / 30s / 40s
            print(f"    等待 {wait:.0f}s 后重试...", flush=True)
            time.sleep(wait)
    return pd.DataFrame()


# ── 复权合并 ───────────────────────────────────────────────────────────────

def _compute_adj_close(daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    if adj.empty:
        df["adj_factor"] = np.nan
    else:
        a = adj.copy()
        a["trade_date"] = pd.to_datetime(a["trade_date"], format="%Y%m%d", errors="coerce")
        df = df.merge(a[["trade_date", "adj_factor"]], on="trade_date", how="left")
    df["adj_factor"] = pd.to_numeric(df.get("adj_factor", np.nan), errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)
    latest = df["adj_factor"].iloc[-1] if len(df) > 0 else np.nan
    if pd.notna(latest) and latest > 0:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce") * df["adj_factor"] / float(latest)
    else:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


# ── 模式1：by_stock（分年度，支持断点续传）────────────────────────────────

def _download_one_stock(pro, ticker: str, sleep: float) -> pd.DataFrame:
    parts = []
    for year in YEARS:
        sd, ed = f"{year}0101", f"{year}1231"
        df = _retry(
            lambda t=ticker, s=sd, e=ed: pro.daily(ts_code=t, start_date=s, end_date=e),
            label=f"daily {ticker} {year}",
            attempts=3,
            base_sleep=5.0,
        )
        time.sleep(sleep)
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame()
    daily_all = pd.concat(parts, ignore_index=True).drop_duplicates("trade_date")

    # adj_factor：先试全量，超时则分年
    adj = _retry(
        lambda t=ticker: pro.adj_factor(ts_code=t, start_date="20190101", end_date="20261231"),
        label=f"adj {ticker} full",
        attempts=2,
        base_sleep=3.0,
    )
    time.sleep(sleep)

    if adj.empty:
        adj_parts = []
        for year in YEARS:
            sd, ed = f"{year}0101", f"{year}1231"
            a = _retry(
                lambda t=ticker, s=sd, e=ed: pro.adj_factor(ts_code=t, start_date=s, end_date=e),
                label=f"adj {ticker} {year}",
                attempts=2,
                base_sleep=5.0,
            )
            time.sleep(sleep)
            if not a.empty:
                adj_parts.append(a)
        adj = pd.concat(adj_parts, ignore_index=True) if adj_parts else pd.DataFrame()

    return _compute_adj_close(daily_all, adj)


def run_by_stock(pro, tickers: list[str], sleep: float):
    done: set[str] = set(DONE_FILE.read_text().splitlines()) if DONE_FILE.exists() else set()
    todo = [t for t in tickers if t not in done]
    print(f"[by_stock] 共 {len(tickers)} 只，已完成 {len(done)}，待下载 {len(todo)}")
    hrs_est = len(todo) * len(YEARS) * 2 * (6.0 + sleep) / 3600
    print(f"  代理估计时长：~{hrs_est:.0f} 小时（断点续传，可多次中断重跑）", flush=True)

    parts: list[pd.DataFrame] = []
    failed: list[str] = []

    for i, ticker in enumerate(todo):
        try:
            merged = _download_one_stock(pro, ticker, sleep)
            if merged.empty:
                failed.append(ticker)
                continue
            merged["ts_code"] = ticker
            keep = ["trade_date", "ts_code", "open", "high", "low", "close",
                    "vol", "amount", "adj_factor", "adj_close"]
            merged = merged[[c for c in keep if c in merged.columns]]
            parts.append(merged)
            done.add(ticker)
        except Exception as e:
            print(f"  [ERR] {ticker}: {e}", flush=True)
            failed.append(ticker)

        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            elapsed_h = (i + 1) * len(YEARS) * 2 * (6 + sleep) / 3600
            print(f"  进度 {i+1}/{len(todo)} | 成功 {len(parts)} | 失败 {len(failed)} | 约{elapsed_h:.1f}h已过", flush=True)
            DONE_FILE.write_text("\n".join(sorted(done)))
            _flush_parts(parts, tickers)

    _flush_parts(parts, tickers, final=True)
    if failed:
        FAILED_FILE.write_text("\n".join(sorted(set(failed))))
    print(f"[by_stock 完成] 成功 {len(done)} | 失败 {len(failed)}", flush=True)


# ── 模式2：by_date（每日全市场，代理推荐）────────────────────────────────

def _get_trading_dates(pro, start: str = "20190101", end: str = "20261231") -> list[str]:
    df = _retry(
        lambda: pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1",
                              fields="cal_date"),
        label="trade_cal",
    )
    if df.empty:
        return []
    today = pd.Timestamp.today().strftime("%Y%m%d")
    dates = sorted(df["cal_date"].tolist())
    return [d for d in dates if d <= today]


def run_by_date(pro, universe: set[str], sleep: float):
    DONE_DATE = AUX_DIR / "price_date_done.txt"
    done_dates: set[str] = set(DONE_DATE.read_text().splitlines()) if DONE_DATE.exists() else set()

    print("[by_date] 获取交易日历...", flush=True)
    all_dates = _get_trading_dates(pro)
    todo_dates = [d for d in all_dates if d not in done_dates]
    hrs_est = len(todo_dates) * (8 + sleep) / 3600
    print(f"  共 {len(all_dates)} 个交易日，待下载 {len(todo_dates)}", flush=True)
    print(f"  代理估计时长：~{hrs_est:.0f} 小时（断点续传）", flush=True)

    date_parts: list[pd.DataFrame] = []
    adj_cache: dict[str, pd.DataFrame] = {}

    # 倒序：从最新日期开始，优先保证近期数据完整
    todo_dates_ordered = list(reversed(todo_dates))
    success_cnt = 0
    skip_cnt = 0

    for i, dt in enumerate(todo_dates_ordered):
        try:
            df = _retry(
                lambda d=dt: pro.daily(trade_date=d),
                label=f"daily {dt}",
                attempts=2,       # 最多2次，失败就跳过，不卡死
                base_sleep=5.0,
            )
            time.sleep(sleep)
            if df.empty:
                done_dates.add(dt)  # 可能非交易日，标记跳过
                skip_cnt += 1
                continue
            df_filtered = df[df["ts_code"].isin(universe)].copy()
            df_filtered["trade_date"] = dt
            date_parts.append(df_filtered)
            done_dates.add(dt)
            success_cnt += 1
        except Exception as e:
            print(f"  [跳过] {dt}: {str(e)[:80]}", flush=True)
            skip_cnt += 1

        if (i + 1) % 100 == 0 or i + 1 == len(todo_dates_ordered):
            elapsed_h = (i + 1) * (8 + sleep) / 3600
            print(f"  进度 {i+1}/{len(todo_dates_ordered)} | 成功 {success_cnt} | 跳过 {skip_cnt} | 约{elapsed_h:.1f}h", flush=True)
            DONE_DATE.write_text("\n".join(sorted(done_dates)))
            # 每100天写一次中间结果
            if date_parts:
                _flush_date_parts(date_parts, universe)

    if date_parts:
        _flush_date_parts(date_parts, universe, final=True)

    # adj_factor 必须按股票拉（无法按日期）
    print("\n[by_date] 下载复权因子（按股票，官方API更快）...", flush=True)
    adj_parts = []
    for j, ticker in enumerate(sorted(universe)):
        adj = _retry(
            lambda t=ticker: pro.adj_factor(ts_code=t, start_date="20190101", end_date="20261231"),
            label=f"adj {ticker}",
            attempts=3,
            base_sleep=3.0,
        )
        time.sleep(sleep)
        if not adj.empty:
            adj["ts_code"] = ticker
            adj_parts.append(adj)
        if (j + 1) % 100 == 0:
            print(f"  adj 进度 {j+1}/{len(universe)}", flush=True)

    # 合并 daily + adj → adj_close
    if date_parts and adj_parts:
        daily_all = pd.concat(date_parts, ignore_index=True)
        daily_all["trade_date"] = pd.to_datetime(
            daily_all.get("trade_date", daily_all.get("trade_date_dt")),
            format="%Y%m%d", errors="coerce"
        )
        adj_all = pd.concat(adj_parts, ignore_index=True)
        adj_all["trade_date"] = pd.to_datetime(adj_all["trade_date"], format="%Y%m%d", errors="coerce")
        merged = daily_all.merge(adj_all[["ts_code", "trade_date", "adj_factor"]],
                                 on=["ts_code", "trade_date"], how="left")
        for ticker, grp in merged.groupby("ts_code"):
            latest = grp["adj_factor"].iloc[-1] if len(grp) > 0 else np.nan
            if pd.notna(latest) and latest > 0:
                merged.loc[grp.index, "adj_close"] = (
                    pd.to_numeric(grp["close"], errors="coerce") * grp["adj_factor"] / float(latest)
                )
            else:
                merged.loc[grp.index, "adj_close"] = pd.to_numeric(grp["close"], errors="coerce")
        _save_panel(merged)
    print("[by_date 完成]", flush=True)


# ── 共用：写盘 ─────────────────────────────────────────────────────────────

_INCREMENTAL_PARTS: list[pd.DataFrame] = []

def _flush_date_parts(new_parts: list[pd.DataFrame], universe: set, final: bool = False):
    """by_date 模式：每100天增量写盘，避免内存堆积。"""
    if not new_parts:
        return
    chunk = pd.concat(new_parts, ignore_index=True)
    chunk["trade_date"] = pd.to_datetime(chunk["trade_date"], format="%Y%m%d", errors="coerce")
    existing_file = OUT_DIR / "alpha_prices_panel.parquet"
    if existing_file.exists():
        try:
            old = pd.read_parquet(existing_file)
            chunk = pd.concat([old, chunk], ignore_index=True).drop_duplicates(
                subset=["ts_code", "trade_date"]
            )
        except Exception:
            pass
    chunk = chunk[chunk["ts_code"].isin(universe)].sort_values(["ts_code", "trade_date"])
    chunk.to_parquet(existing_file, index=False, compression="snappy")
    coverage = chunk["trade_date"].nunique()
    tickers_covered = chunk["ts_code"].nunique()
    print(f"  [写盘] {existing_file.name}: {len(chunk):,}行 | {tickers_covered}只 | {coverage}天", flush=True)
    new_parts.clear()


def _flush_parts(new_parts: list[pd.DataFrame], all_tickers: list[str], final: bool = False):
    if not new_parts:
        return
    existing_file = OUT_DIR / "alpha_prices_panel.parquet"
    all_new = pd.concat(new_parts, ignore_index=True)
    if existing_file.exists() and not final:
        try:
            old = pd.read_parquet(existing_file)
            all_new = pd.concat([old, all_new], ignore_index=True).drop_duplicates(
                ["ts_code", "trade_date"]
            )
        except Exception:
            pass
    all_new.to_parquet(existing_file, index=False, compression="snappy")
    if final:
        _write_wide(all_new)
        print(f"  长表: {existing_file}  ({len(all_new):,} 行, {all_new['ts_code'].nunique()} 只)")


def _save_panel(panel: pd.DataFrame):
    out = OUT_DIR / "alpha_prices_panel.parquet"
    panel = panel.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"])
    panel.to_parquet(out, index=False, compression="snappy")
    _write_wide(panel)
    print(f"  长表: {out}  ({len(panel):,} 行, {panel['ts_code'].nunique()} 只)", flush=True)


def _write_wide(panel: pd.DataFrame):
    if "adj_close" not in panel.columns:
        return
    wide = panel.pivot_table(
        index="trade_date", columns="ts_code", values="adj_close", aggfunc="last"
    ).sort_index()
    wide.to_parquet(OUT_DIR / "alpha_adj_close_wide.parquet", compression="snappy")
    print(f"  宽表: alpha_adj_close_wide.parquet  ({wide.shape[0]} 日 × {wide.shape[1]} 只)", flush=True)


# ── 主入口 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["by_stock", "by_date"], default="by_stock",
                        help="by_stock=分年度/官方API；by_date=按交易日/代理推荐")
    parser.add_argument("--use-hi-api", action="store_true",
                        help="强制使用高频代理API（by_stock模式下默认用官方API）")
    parser.add_argument("--sleep", type=float, default=None,
                        help="每次请求后等待秒数（by_stock默认0.35；by_date默认0.4）")
    args = parser.parse_args()

    if not UNIVERSE_TXT.exists():
        raise SystemExit(f"先运行 step1，未找到 {UNIVERSE_TXT}")

    tickers = [t.strip() for t in UNIVERSE_TXT.read_text().splitlines() if t.strip()]
    universe = set(tickers)

    hi_mode = args.use_hi_api or args.mode == "by_date"
    pro = _init_pro(hi_mode=hi_mode)

    if args.mode == "by_date":
        sleep = args.sleep if args.sleep is not None else 0.4
        run_by_date(pro, universe, sleep)
    else:
        sleep = args.sleep if args.sleep is not None else 0.35
        run_by_stock(pro, tickers, sleep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
