"""P63-2a：v6 全市场基本面因子面板（PIT by ann_date）.

snapshot/as_of：daily_basic@(last_td<=as_of) + fina_indicator(ann<=as_of, latest/ticker)。
复用 quantmind.features.fundamental.compute_all_fundamental_factors（内部 latest_report_per_ticker
按 ann_date PIT）+ 自定义 fcf_yield_cf=(n_cashflow_act-c_paid_invest)/total_mv（cashflow 表，若已拉）。
逐 as_of 截面 winsorize(1/99%)。输出 data/panel/fundamental_factors_v6.parquet（reindex 到 v6 panel）。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from quantmind.features.fundamental import compute_all_fundamental_factors, FUNDAMENTAL_FACTORS

LAKE = REPO / "data/lake"
V6_PANEL = REPO / "data/panel/alpha_panel_weekly_v6.parquet"
OUT = REPO / "data/panel/fundamental_factors_v6.parquet"


def _winsor(s, lo=0.01, hi=0.99):
    if s.notna().sum() < 20:
        return s
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def main():
    t0 = time.time()
    panel_idx = pd.read_parquet(V6_PANEL, columns=["forward_return_63d"]).index
    as_of_list = sorted(panel_idx.get_level_values("as_of").unique())

    db = pd.read_parquet(LAKE / "daily_basic.parquet")
    db["trade_date"] = pd.to_datetime(db["trade_date"])
    if "ticker" not in db.columns:
        db["ticker"] = db["ts_code"].astype(str)
    db_dates = np.array(sorted(db["trade_date"].unique()))

    # fina_indicator：优先全字段版（解锁 equity_multiplier/op_yoy/q_*/accel/fcff）
    fina_path = LAKE / "fina_indicator_full.parquet"
    if not fina_path.exists():
        fina_path = LAKE / "fina_indicator.parquet"
    fi = pd.read_parquet(fina_path)
    fi["ticker"] = fi["ts_code"].astype(str); fi["report_date"] = fi["end_date"]
    print(f"[p63-2] fina={fina_path.name} ({len(fi):,}行, {len(fi.columns)}列)", flush=True)

    def _load_stmt(name):
        p = LAKE / f"{name}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        d["ticker"] = d["ts_code"].astype(str)
        d["report_date"] = d["end_date"]
        d["f_ann_date"] = d["ann_date"]          # accruals/ytd_to_ttm 用 f_ann_date
        return d

    inc_all = _load_stmt("income")
    bs_all = _load_stmt("balancesheet")
    cf_all = _load_stmt("cashflow")
    print(f"[p63-2] income={None if inc_all is None else len(inc_all)} "
          f"balancesheet={None if bs_all is None else len(bs_all)} "
          f"cashflow={None if cf_all is None else len(cf_all)}", flush=True)
    cf = cf_all  # 自定义 fcf_yield_cf 仍用 cashflow
    if cf is not None:
        print(f"[p63-2] cashflow 可用 → 启用自定义 fcf_yield_cf", flush=True)

    names = [n for n, _ in FUNDAMENTAL_FACTORS]
    rows = []
    for k, as_of in enumerate(as_of_list):
        as_of = pd.Timestamp(as_of); asof_str = as_of.strftime("%Y%m%d")
        ltd = db_dates[db_dates <= np.datetime64(as_of)]
        if len(ltd) == 0:
            continue
        db_a = db[db.trade_date == ltd[-1]]
        fi_a = fi[fi["ann_date"] <= asof_str]
        snap = {"daily_basic": db_a, "financial_indicators": fi_a}
        if inc_all is not None:
            snap["financials_income"] = inc_all[inc_all["ann_date"] <= asof_str]
        if bs_all is not None:
            snap["financials_balance_sheet"] = bs_all[bs_all["ann_date"] <= asof_str]
        if cf_all is not None:
            snap["financials_cashflow"] = cf_all[cf_all["ann_date"] <= asof_str]
        fac = compute_all_fundamental_factors(snap, as_of.date())  # index=ticker
        # 自定义 fcf_yield from cashflow（CFO - capex）/ total_mv
        if cf is not None:
            cfa = cf[cf["ann_date"] <= asof_str].sort_values(["ticker", "report_date"], ascending=[True, False])
            cfa = cfa.drop_duplicates("ticker", keep="first").set_index("ticker")
            mv = db_a.set_index("ticker")["total_mv"]
            num = cfa.get("n_cashflow_act", pd.Series(dtype=float)) - cfa.get("c_paid_invest", 0.0)
            common = num.index.intersection(mv.index)
            fac["fcf_yield_cf"] = (num.loc[common] / mv.loc[common].replace(0, np.nan))
        fac = fac.apply(_winsor, axis=0)
        fac["as_of"] = as_of
        rows.append(fac.reset_index().rename(columns={"index": "ticker"}))
        if k % 25 == 0:
            print(f"  as_of {k+1}/{len(as_of_list)} {as_of.date()} ({time.time()-t0:.0f}s)", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out["ticker"] = out["ticker"].astype(str)
    out = out.set_index(["as_of", "ticker"]).reindex(panel_idx)
    keep = [c for c in out.columns if out[c].notna().sum() > 1000]   # 丢全空因子
    out = out[keep]
    out.to_parquet(OUT)
    cov = {c: int(out[c].notna().sum()) for c in out.columns}
    print(f"[p63-2] saved {OUT} shape={out.shape} 可用因子={len(keep)} ({time.time()-t0:.0f}s)")
    print("  覆盖(非空行):", {k: v for k, v in sorted(cov.items(), key=lambda x: -x[1])})


if __name__ == "__main__":
    main()
