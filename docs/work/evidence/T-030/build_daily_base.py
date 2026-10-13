"""T-030 P0：日频网格上的 35 个量价/市场因子 + delist-aware 标签（12/20/21/40/63d）.

与 v6_1 周频面板（T-017 的 `p2_build_v6.py --panel`）用**同一套函数与同一份输入**，只把 as_of 网格从
`build_asof_grid`（每 5 个交易日）换成完整交易日历的每一天：
  价格 data/raw/alpha_prices_panel_v6_20261009.parquet；日历 data/lake/trade_cal_sse.parquet；
  stock_basic data/lake/stock_basic_full_merged_20261009.parquet；PIT 入池 = list+120td ≤ as_of < delist；
  因子 = FeaturePipeline(technical/sentiment/expansion) 的 WHITELIST_35 + 两道 NaN 守卫；
  标签 = compute_forward_returns_delist_aware（adj_close，信号日收盘 → 第 h 个交易日收盘）。
输出 <data_root>/data/gpu_v1/panel/base/<YYYY-MM>.parquet（index=(as_of, ticker)）；已存在的月份跳过（断点续跑），从不覆盖。
用法：PYTHONPATH=. python build_daily_base.py <data_root> [--workers N] [--months 2019-01,2019-02]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.survivorship.p2_build_v6 as p2

HORIZONS = (12, 20, 21, 40, 63)
LABELS = [f"forward_return_{h}d" for h in HORIZONS]
G: dict = {}


def setup(root: Path) -> None:
    from quantmind.research.gpu.fast_factors import install as _install_fast
    _install_fast()
    p2.configure(data_root=root, prices_file=Path("data/raw/alpha_prices_panel_v6_20261009.parquet"),
                 calendar_file=Path("data/lake/trade_cal_sse.parquet"),
                 stock_basic=Path("data/lake/stock_basic_full_merged_20261009.parquet"),
                 panel_out=root / "data/gpu_v1/_never_written.parquet")
    from quantmind.features.pipeline import FeaturePipeline
    from quantmind.features.weekly_panel import _PRICE_RENAME
    cal = p2.trading_calendar()
    prices = pd.read_parquet(p2.V6_PRICES).rename(columns=_PRICE_RENAME)
    prices["ticker"] = prices["ticker"].astype(str)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices.sort_values("trade_date").set_index("trade_date", drop=False)
    sb_true, sb_seasoned = p2._seasoned_stock_basic(cal)
    sub = prices[["trade_date", "ticker", "adj_close"]].dropna(subset=["adj_close"])
    adj = (sub.drop_duplicates(["trade_date", "ticker"], keep="last")
           .pivot(index="trade_date", columns="ticker", values="adj_close").sort_index())
    has_delist = pd.Series(False, index=adj.columns)
    dmap = sb_true.set_index("ticker")
    common = has_delist.index.intersection(dmap.index)
    has_delist.loc[common] = ((dmap.loc[common, "list_status"] == "D") & dmap.loc[common, "delist_date"].notna()).values
    G.update(cal=cal, prices=prices, priced=set(prices["ticker"].unique()), sb_true=sb_true, sb_seasoned=sb_seasoned,
             adj=adj, ffill=adj.ffill(), last_trade=adj.apply(lambda s: s.last_valid_index()), has_delist=has_delist,
             pipe=FeaturePipeline(groups=["technical", "sentiment", "expansion"], do_standardize=False))


def one_day(as_of: pd.Timestamp) -> pd.DataFrame | None:
    """cmd_panel 循环体的逐字复刻（只换了标签 horizon 集合）."""
    from quantmind.features.weekly_panel import (
        _MIN_TRADING_DAYS,
        _PER_TICKER_MIN_BARS,
        WHITELIST_35,
        assemble_snapshot,
        pit_universe,
    )
    uni = pit_universe(as_of, G["sb_seasoned"], G["priced"])
    if uni.empty:
        return None
    snap = assemble_snapshot(as_of, G["prices"], G["sb_true"], uni, G["cal"])
    raw = G["pipe"].compute_raw_from_snapshot(snap, as_of.date())
    missing = [c for c in WHITELIST_35 if c not in raw.columns]
    if missing:
        raise RuntimeError(f"factor cols missing: {missing}")
    out = raw[WHITELIST_35].copy()
    px = snap["prices"]
    px_days = pd.to_datetime(px["trade_date"]).dt.normalize()
    avail = int(px_days.nunique())
    for col, req in _MIN_TRADING_DAYS.items():
        if avail < req and col in out.columns:
            out[col] = np.nan
    bars = (px.assign(_d=px_days).dropna(subset=["close"])
            .drop_duplicates(["ticker", "_d"]).groupby("ticker")["_d"].size())
    bars = bars.reindex(out.index).fillna(0).astype(int)
    for col, req in _PER_TICKER_MIN_BARS.items():
        if col in out.columns:
            out.loc[bars < req, col] = np.nan
    fr = p2.compute_forward_returns_delist_aware(G["adj"], G["ffill"], as_of, HORIZONS, G["last_trade"],
                                                 G["has_delist"])
    out = out.join(fr.reindex(out.index), how="left")
    out = out.reset_index().rename(columns={"index": "ticker"})
    out["ticker"] = out["ticker"].astype(str)
    out["as_of"] = as_of
    return out


def do_month(args: tuple[str, str]) -> str:
    ym, out_dir = args
    from quantmind.features.weekly_panel import WHITELIST_35
    out = Path(out_dir) / f"{ym}.parquet"
    if out.exists():
        return f"{ym} exists"
    t0 = time.time()
    days = [d for d in G["cal"] if d.strftime("%Y-%m") == ym]
    rows = [r for r in (one_day(pd.Timestamp(d)) for d in days) if r is not None]
    df = pd.concat(rows, ignore_index=True).set_index(["as_of", "ticker"]).sort_index()[WHITELIST_35 + LABELS]
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)
    return f"{ym} {len(days)} days {df.shape} {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--months", default=None)
    a = ap.parse_args()
    t0 = time.time()
    setup(a.root)
    out_dir = a.root / "data/gpu_v1/panel/base"
    out_dir.mkdir(parents=True, exist_ok=True)
    months = a.months.split(",") if a.months else sorted({d.strftime("%Y-%m") for d in G["cal"]})
    todo = [m for m in months if not (out_dir / f"{m}.parquet").exists()]
    print(f"[base] setup {time.time() - t0:.0f}s; {len(todo)}/{len(months)} months to build", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:      # fork：子进程共享已加载的价格表
        for msg in ex.map(do_month, [(m, str(out_dir)) for m in todo]):
            print(f"  {msg} (total {time.time() - t0:.0f}s)", flush=True)
    print(f"[base] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
