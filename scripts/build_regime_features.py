"""scripts/build_regime_features.py — 为每个 as_of 季末日计算市场 Regime 指标.

输出: data/features/regime_features.parquet
  索引: as_of (date)
  列:
    csi500_csi300_63d   - CSI500 vs CSI300 63日收益差（小盘溢价）
    chiext_csi300_63d   - 创业板 vs CSI300 63日收益差
    csi300_63d_return   - CSI300 本身 63日收益（市场整体方向）
    csi300_20d_vol      - CSI300 20日已实现波动率（年化）
    small_large_63d_spread - 本期末 alpha 宇宙内小市值20% vs 大市值20% 63日收益差
    breadth_20d         - 近20日正收益股票比例（市场宽度）
    regime_label        - 1=小盘占优 / 0=大盘占优（由 small_large_63d_spread 决定）
    regime_small_prob   - soft signal（small_large_63d_spread 归一化到 [0,1]）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SNAP_DIR   = PROJECT_ROOT / "data" / "snapshots"
PRICE_PATH = PROJECT_ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
OUT_PATH   = PROJECT_ROOT / "data" / "features" / "regime_features.parquet"

# 20 个 Alpha 宇宙快照目录（按时间排好）
TARGET_SNAPS = [
    "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
    "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
    "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-30",
    "2023-03-31", "2023-06-30", "2023-09-28", "2023-12-29",
    "2024-03-29", "2024-06-28", "2024-09-30", "2024-12-31",
]

LOOKBACK_LONG  = 63   # 约3个月：小盘/大盘溢价窗口
LOOKBACK_SHORT = 20   # 约1个月：波动率、宽度窗口


def _index_return(df_idx: pd.DataFrame, code: str, n: int, as_of: pd.Timestamp) -> float:
    """计算指数 code 在 as_of 之前 n 个交易日的区间收益."""
    sub = df_idx[df_idx["ts_code"] == code].copy()
    sub = sub[sub["trade_date"] <= as_of].sort_values("trade_date")
    if len(sub) < n + 1:
        return float("nan")
    ret = sub["close"].iloc[-1] / sub["close"].iloc[-(n + 1)] - 1
    return float(ret)


def _index_vol(df_idx: pd.DataFrame, code: str, n: int, as_of: pd.Timestamp) -> float:
    """计算指数 code 在 as_of 之前 n 个交易日的年化波动率."""
    sub = df_idx[df_idx["ts_code"] == code].copy()
    sub = sub[sub["trade_date"] <= as_of].sort_values("trade_date")
    if len(sub) < n + 2:
        return float("nan")
    rets = sub["close"].pct_change(fill_method=None).dropna().iloc[-n:]
    return float(rets.std() * np.sqrt(252))


def _small_large_spread(
    prices: pd.DataFrame,
    circ_mv: pd.Series,
    n: int,
    as_of: pd.Timestamp,
) -> float:
    """计算 alpha 宇宙内小市值20% vs 大市值20% 的区间收益差."""
    tickers = circ_mv.index.tolist()
    sub = prices[prices["ts_code"].isin(tickers) & (prices["trade_date"] <= as_of)].copy()

    # 取窗口内所有股票的起止 adj_close
    cutoff_start = sub["trade_date"].max() - pd.Timedelta(days=n * 2)
    window = sub[sub["trade_date"] >= cutoff_start]
    pivot = window.pivot(index="trade_date", columns="ts_code", values="adj_close")
    if len(pivot) < n + 1:
        return float("nan")
    # 用前 n+1 行计算区间收益
    pivot = pivot.iloc[-(n + 1):]
    stock_ret = pivot.iloc[-1] / pivot.iloc[0] - 1
    stock_ret = stock_ret.dropna()

    # 与 circ_mv 对齐
    common = stock_ret.index.intersection(circ_mv.index)
    if len(common) < 10:
        return float("nan")
    ret_aligned = stock_ret.loc[common]
    mv_aligned  = circ_mv.loc[common]

    q20 = mv_aligned.quantile(0.2)
    q80 = mv_aligned.quantile(0.8)
    small_ret = ret_aligned[mv_aligned <= q20].mean()
    large_ret = ret_aligned[mv_aligned >= q80].mean()
    return float(small_ret - large_ret)


def _breadth(prices: pd.DataFrame, n: int, as_of: pd.Timestamp) -> float:
    """近 n 日有正收益的股票比例."""
    cutoff = as_of - pd.Timedelta(days=n * 2)
    window = prices[(prices["trade_date"] >= cutoff) & (prices["trade_date"] <= as_of)]
    pivot = window.pivot(index="trade_date", columns="ts_code", values="adj_close")
    if len(pivot) < n + 1:
        return float("nan")
    pivot = pivot.iloc[-(n + 1):]
    ret = pivot.iloc[-1] / pivot.iloc[0] - 1
    return float((ret > 0).mean())


def main() -> None:
    print("=== 计算 Regime 特征 ===")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"加载价格面板: {PRICE_PATH}")
    prices = pd.read_parquet(PRICE_PATH)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])

    records = []
    for snap_str in TARGET_SNAPS:
        as_of = pd.Timestamp(snap_str)
        snap_dir = SNAP_DIR / snap_str
        print(f"\n── {snap_str} ──")

        # ── 指数数据 ──────────────────────────────────────────────────────
        idx_path = snap_dir / "index_daily.parquet"
        if not idx_path.exists():
            print("  WARNING: index_daily.parquet 缺失，跳过")
            continue
        df_idx = pd.read_parquet(idx_path)
        df_idx["trade_date"] = pd.to_datetime(df_idx["trade_date"])

        csi300_63d  = _index_return(df_idx, "000300.SH", LOOKBACK_LONG,  as_of)
        csi500_63d  = _index_return(df_idx, "000905.SH", LOOKBACK_LONG,  as_of)
        chiext_63d  = _index_return(df_idx, "399006.SZ", LOOKBACK_LONG,  as_of)
        csi300_vol  = _index_vol   (df_idx, "000300.SH", LOOKBACK_SHORT, as_of)
        csi500_spread = float("nan") if np.isnan(csi500_63d) or np.isnan(csi300_63d) else csi500_63d - csi300_63d
        chiext_spread = float("nan") if np.isnan(chiext_63d) or np.isnan(csi300_63d) else chiext_63d - csi300_63d

        # ── daily_basic → circ_mv ──────────────────────────────────────────
        db_path = snap_dir / "daily_basic.parquet"
        circ_mv = pd.Series(dtype=float)
        if db_path.exists():
            db = pd.read_parquet(db_path, columns=["ticker", "circ_mv"])
            circ_mv = db.set_index("ticker")["circ_mv"].dropna()

        # ── 小盘/大盘收益差 ───────────────────────────────────────────────
        sl_spread = _small_large_spread(prices, circ_mv, LOOKBACK_LONG, as_of)

        # ── 市场宽度 ─────────────────────────────────────────────────────
        brd = _breadth(prices, LOOKBACK_SHORT, as_of)

        rec = {
            "as_of": as_of.date(),
            "csi300_63d_return":    csi300_63d,
            "csi500_csi300_63d":    csi500_spread,
            "chiext_csi300_63d":    chiext_spread,
            "csi300_20d_vol":       csi300_vol,
            "small_large_63d_spread": sl_spread,
            "breadth_20d":          brd,
        }
        print(f"  CSI300_63d={csi300_63d:+.3f}  CSI500-CSI300={csi500_spread:+.3f}  "
              f"small-large={sl_spread:+.3f}  breadth={brd:.2f}  vol={csi300_vol:.3f}")
        records.append(rec)

    df = pd.DataFrame(records).set_index("as_of")

    # ── Regime label（硬阈值：small_large_63d_spread > 0 → 小盘 regime）──
    df["regime_label"] = (df["small_large_63d_spread"] > 0).astype(int)

    # ── Soft signal：把 spread 归一化到 [0,1] ─────────────────────────────
    s = df["small_large_63d_spread"]
    df["regime_small_prob"] = (s - s.min()) / (s.max() - s.min() + 1e-8)

    df.to_parquet(OUT_PATH)
    print(f"\n✅ 保存 → {OUT_PATH}")
    print(df.to_string())


if __name__ == "__main__":
    main()
