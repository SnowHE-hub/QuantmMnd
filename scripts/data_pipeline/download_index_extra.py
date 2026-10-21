"""scripts/data_pipeline/download_index_extra.py — M 层要用、湖里还没有的指数日线（T-035 P1；03_final_plan_v2 第 5 节 M）.

中证 1000（000852.SH）与中证 2000（932000.CSI）：风格监测里"小盘对大盘的相对强弱和拥挤度"需要它们。
写到新文件 data/lake/index_daily_extra.parquet（不改现有的 index_daily）。已有文件时只追加缺的日期，旧行一行不动。
token 走 backfill_tushare 同一套（api_key.txt，官方端点），不回显、不落盘。
用法：python scripts/data_pipeline/download_index_extra.py --data-root <root> --end 2026-10-20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CODES = ("000852.SH", "932000.CSI")
START = "20190102"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=REPO)
    ap.add_argument("--end", required=True)
    a = ap.parse_args(argv)
    import scripts.backfill_tushare as bf
    bf.REPO_ROOT = str(a.data_root)
    bf._load_token_into_env()
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger as _lg
        _lg.remove()
    except Exception:  # noqa: BLE001
        pass
    out = a.data_root / "data/lake/index_daily_extra.parquet"
    old = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    end = pd.Timestamp(a.end).strftime("%Y%m%d")
    parts = []
    for c in CODES:
        have = old[old["ts_code"] == c]["trade_date"].astype(str) if len(old) else pd.Series(dtype=str)
        lo = (pd.Timestamp(have.max()) + pd.Timedelta(days=1)).strftime("%Y%m%d") if len(have) else START
        if lo > end:
            print(f"[{c}] up to date ({have.max()})")
            continue
        df = tp._call("index_daily", ts_code=c, start_date=lo, end_date=end)
        n = 0 if df is None else len(df)
        print(f"[{c}] {lo}..{end}: {n} rows")
        if n:
            df["trade_date"] = df["trade_date"].astype(str)
            parts.append(df)
    if not parts:
        return 0
    new = pd.concat([old, *parts], ignore_index=True) if len(old) else pd.concat(parts, ignore_index=True)
    new = new.drop_duplicates(["ts_code", "trade_date"], keep="first").sort_values(["ts_code", "trade_date"])
    if len(old):
        chk = new.merge(old, on=["ts_code", "trade_date"], suffixes=("", "_o"))
        assert len(chk) == len(old), "old rows must be kept unchanged"
    tmp = out.with_suffix(".tmp")
    new.to_parquet(tmp, index=False)
    tmp.replace(out)
    print(f"[write] {out} rows={len(new)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
