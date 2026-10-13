"""T-030 P0：基本面因子在日频网格上的取值（= `scripts/survivorship/p63_2_build_factors.py` 的逐 as_of 循环体，只换网格）.

每个交易日：daily_basic@(≤ as_of 的最后交易日) + 财报 ann_date ≤ as_of（PIT）→ compute_all_fundamental_factors + fcf_yield_cf
→ 截面 1%/99% 缩尾（在当日全部 daily_basic 股票上，与原脚本相同）。不 reindex 到面板（训练时与 base 对齐，值不变）。
输出 <data_root>/data/gpu_v1/panel/fnd/<YYYY-MM>.parquet；已存在的月份跳过（断点续跑），从不覆盖。
用法：PYTHONPATH=. python build_daily_fnd.py <data_root> [--workers N]
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.survivorship.p63_2_build_factors as p63
from quantmind.features.fundamental import compute_all_fundamental_factors

G: dict = {}


def setup(root: Path) -> None:
    lake = root / "data/lake"
    db = pd.read_parquet(lake / "daily_basic.parquet")
    db["trade_date"] = pd.to_datetime(db["trade_date"])
    if "ticker" not in db.columns:
        db["ticker"] = db["ts_code"].astype(str)
    fi = pd.read_parquet(lake / "fina_indicator_full.parquet")
    fi["ticker"] = fi["ts_code"].astype(str)
    fi["report_date"] = fi["end_date"]

    def stmt(name):
        d = pd.read_parquet(lake / f"{name}.parquet")
        d["ticker"] = d["ts_code"].astype(str)
        d["report_date"] = d["end_date"]
        d["f_ann_date"] = d["ann_date"]
        return d

    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(lake / "trade_cal_sse.parquet")["cal_date"])).sort_values()
    px = pd.to_datetime(pd.read_parquet(root / "data/raw/alpha_prices_panel_v6_20261009.parquet",
                                        columns=["trade_date"])["trade_date"]).drop_duplicates()
    cal = cal[(cal >= px.min()) & (cal <= px.max())]
    G.update(db=db, db_dates=np.array(sorted(db["trade_date"].unique())), fi=fi, inc=stmt("income"),
             bs=stmt("balancesheet"), cf=stmt("cashflow"), cal=cal)


def one_day(as_of: pd.Timestamp) -> pd.DataFrame | None:
    db, fi, inc, bs, cf = G["db"], G["fi"], G["inc"], G["bs"], G["cf"]
    asof_str = as_of.strftime("%Y%m%d")
    ltd = G["db_dates"][G["db_dates"] <= np.datetime64(as_of)]
    if len(ltd) == 0:
        return None
    db_a = db[db.trade_date == ltd[-1]]
    snap = {"daily_basic": db_a, "financial_indicators": fi[fi["ann_date"] <= asof_str],
            "financials_income": inc[inc["ann_date"] <= asof_str],
            "financials_balance_sheet": bs[bs["ann_date"] <= asof_str],
            "financials_cashflow": cf[cf["ann_date"] <= asof_str]}
    fac = compute_all_fundamental_factors(snap, as_of.date())
    cfa = cf[cf["ann_date"] <= asof_str].sort_values(["ticker", "report_date"], ascending=[True, False])
    cfa = cfa.drop_duplicates("ticker", keep="first").set_index("ticker")
    mv = db_a.set_index("ticker")["total_mv"]
    num = cfa.get("n_cashflow_act", pd.Series(dtype=float)) - cfa.get("c_paid_invest", 0.0)
    common = num.index.intersection(mv.index)
    fac["fcf_yield_cf"] = (num.loc[common] / mv.loc[common].replace(0, np.nan))
    fac = fac.apply(p63._winsor, axis=0)
    fac["as_of"] = as_of
    return fac.reset_index().rename(columns={"index": "ticker"})


def do_month(args: tuple[str, str]) -> str:
    ym, out_dir = args
    out = Path(out_dir) / f"{ym}.parquet"
    if out.exists():
        return f"{ym} exists"
    t0 = time.time()
    rows = [r for r in (one_day(pd.Timestamp(d)) for d in G["cal"] if d.strftime("%Y-%m") == ym) if r is not None]
    df = pd.concat(rows, ignore_index=True)
    df["ticker"] = df["ticker"].astype(str)
    df = df.set_index(["as_of", "ticker"]).sort_index()
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)
    return f"{ym} {df.shape} {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    setup(a.root)
    out_dir = a.root / "data/gpu_v1/panel/fnd"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [m for m in sorted({d.strftime("%Y-%m") for d in G["cal"]}) if not (out_dir / f"{m}.parquet").exists()]
    print(f"[fnd] setup {time.time() - t0:.0f}s; {len(todo)} months", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(do_month, [(m, str(out_dir)) for m in todo]):
            print(f"  {msg} (total {time.time() - t0:.0f}s)", flush=True)
    print(f"[fnd] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
