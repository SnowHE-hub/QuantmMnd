#!/usr/bin/env python3
"""Validate Phase B1 daily price parquet outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADJ_WIDE = ROOT / "data/prices/csi300_daily_adj_close.parquet"
INDEX_DAILY = ROOT / "data/prices/index_daily.parquet"


def main() -> int:
    rc = 0
    if not ADJ_WIDE.is_file():
        print(f"MISSING: {ADJ_WIDE}")
        return 1

    wide = pd.read_parquet(ADJ_WIDE)
    print("=== csi300_daily_adj_close.parquet ===")
    print("shape:", wide.shape)
    dr = (wide.index.min(), wide.index.max())
    print("date_range:", dr[0], "→", dr[1])

    nan_frac_cols = wide.isna().mean(axis=0)
    mean_nan = float(nan_frac_cols.mean())
    print(f"mean NaN fraction per column: {mean_nan:.4f}")
    if mean_nan >= 0.05:
        print("WARN: mean NaN >= 5%")
        rc = 1

    cols = [c for c in wide.columns if not wide[c].isna().all()][:3]
    if not cols:
        cols = list(wide.columns[:3])
    print("sample tickers (up to 3):", cols)
    tail_n = min(3, len(wide))
    if tail_n:
        sample = wide[cols].tail(tail_n)
        print("last rows adj_close:")
        print(sample.to_string())

    print("\n=== index_daily.parquet ===")
    if not INDEX_DAILY.is_file():
        print(f"MISSING: {INDEX_DAILY}")
        return 1

    ix = pd.read_parquet(INDEX_DAILY)
    if isinstance(ix.index, pd.MultiIndex):
        ix = ix.reset_index()
    print("rows:", len(ix))
    if "trade_date" in ix.columns:
        print("trade_date range:", ix["trade_date"].min(), "→", ix["trade_date"].max())
    codes = ix["ts_code"].unique().tolist() if "ts_code" in ix.columns else []
    print("ts_codes:", codes)
    for need in ("000300.SH", "000905.SH"):
        if need not in codes:
            print(f"WARN: missing index {need}")
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())
