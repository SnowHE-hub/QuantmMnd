"""T-039：lake 财报表的时点（PIT）审计 —— 版本结构、lake 取的是哪个版本、差多大、对冻结基本面特征的影响估计.

输入（只读）：
  <data_root>/data/research/pit_audit/raw/<api>/<ts>.parquet   pit_fetch.py 拉的带版本字段原始行
  <data_root>/data/lake/{income,balancesheet,cashflow,fina_indicator_full}.parquet
  <data_root>/data/lake/daily_basic.parquet（只读样本股票的 total_mv）
  <data_root>/data/gpu_v1/panel/fnd/<YYYY-MM>.parquet          冻结的基本面面板（只读）
  <repo>/docs/work/evidence/T-030/p6/frozen_eval_rows_20d.parquet  参考臂的可评估行
输出：<data_root>/data/research/pit_audit/results/（新路径）

口径：
  - 「版本」= 同一 (ts_code, end_date) 下数值不同的行（三大表另按 f_ann_date 区分）；数值逐位相同的 flag 0/1 重复行算一个版本。
  - 「原始版」= f_ann_date == ann_date 的版本；「后来版」= f_ann_date > ann_date 的版本。
  - PIT 规则（本卡提出的候选）：每个版本从它的 f_ann_date 起可用；某日只用 f_ann_date ≤ 该日的版本，同一期取 f_ann_date 最大的。
    fina_indicator 没有 f_ann_date：把每个版本和同期三大表版本对账（debt_to_assets ↔ 负债/资产，netprofit_margin ↔ 净利润/营收），
    借用对上的那个报表版本的 f_ann_date；对不上的按 ann_date。
  - 特征影响：在参考臂 20d 可评估行（样本股票）上，用冻结构建器同一套函数分别以 lake 输入和 PIT 输入重算，
    再放进冻结面板当日截面看分位变化。不重建面板，不跑回测。

用法：PYTHONPATH=. python docs/work/evidence/T-039/pit_analyze.py --data-root /home/lenovo/projects/quantmind
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantmind.features import fundamental as F  # noqa: E402

VALS = {
    "income": ["total_revenue", "revenue", "n_income", "n_income_attr_p", "operate_profit"],
    "balancesheet": ["total_assets", "total_liab", "total_hldr_eqy_exc_min_int", "total_cur_assets", "total_cur_liab"],
    "cashflow": ["n_cashflow_act", "c_paid_invest", "c_pay_acq_const_fiolta", "free_cashflow"],
    "fina_indicator": ["roe", "roa", "roe_yearly", "roa_yearly", "grossprofit_margin", "netprofit_margin",
                       "assets_turn", "current_ratio", "quick_ratio", "debt_to_assets", "assets_to_eqt", "ebit_ps",
                       "bps", "eps", "ocfps", "fcff", "or_yoy", "tr_yoy", "netprofit_yoy", "dt_netprofit_yoy",
                       "roe_yoy", "op_yoy", "q_sales_yoy", "q_op_qoq", "q_gr_yoy", "q_profit_yoy"],
}
LAKE = {"income": "income", "balancesheet": "balancesheet", "cashflow": "cashflow",
        "fina_indicator": "fina_indicator_full"}
STMT = ("income", "balancesheet", "cashflow")
# 冻结基本面特征实际读的字段
USED = {"income": ["total_revenue", "n_income_attr_p"], "balancesheet": ["total_assets"],
        "cashflow": ["n_cashflow_act", "c_paid_invest"],
        "fina_indicator": ["roe_yearly", "roa_yearly", "grossprofit_margin", "netprofit_margin", "debt_to_assets",
                           "current_ratio", "assets_turn", "assets_to_eqt", "tr_yoy", "op_yoy", "netprofit_yoy",
                           "q_sales_yoy", "fcff", "q_op_qoq"]}
# 判定「版本」用的字段：Tushare 自算的派生字段（free_cashflow、fcff）在 flag 0/1 两行之间经常不同、又没有日期，
# 不算版本，单独统计（derived_field_diff）；fina 只看特征实际读的字段
DERIVED = {"cashflow": "free_cashflow", "fina_indicator": "fcff"}
KEYC = {"income": VALS["income"], "balancesheet": VALS["balancesheet"],
        "cashflow": [c for c in VALS["cashflow"] if c != "free_cashflow"],
        "fina_indicator": [c for c in USED["fina_indicator"] if c != "fcff"]}
FINA_FEATS = ["roe_ttm", "roa_ttm", "gross_margin", "net_margin", "debt_to_assets", "current_ratio", "asset_turnover",
              "equity_multiplier", "revenue_yoy", "operating_profit_yoy", "net_profit_yoy", "quarterly_revenue_yoy",
              "fcf_yield", "earnings_accel_q", "revenue_accel_q"]
STMT_FEATS = ["accruals", "ocf_to_revenue_ttm", "fcf_yield_cf"]
FROZEN_END = "20261009"   # 冻结数据截止日


# ---------------------------------------------------------------- 读入
def load_raw(base: Path, api: str, tickers: list[str]) -> pd.DataFrame:
    parts = [pd.read_parquet(base / "raw" / api / f"{t}.parquet") for t in tickers]
    d = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for c in ("ts_code", "ann_date", "f_ann_date", "end_date", "update_flag"):
        if c in d.columns:
            d[c] = d[c].astype(object).where(d[c].notna(), None)
    for c in VALS[api]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
    return d


def vkey(df: pd.DataFrame, api: str) -> pd.Series:
    """数值元组的键，NaN 视为相等。三大表（元）：10 位有效数字，容忍浮点表示噪声，400 万 / 59 亿这种差仍能分开；
    fina_indicator（百分比与比率）：保留两位小数，flag 0/1 两行常见的末位差（如 2.3640 / 2.3641）不算版本。"""
    fmt = (lambda x: f"{x:.2f}") if api == "fina_indicator" else (lambda x: f"{x:.10g}")
    return df[KEYC[api]].astype("float64").apply(lambda r: tuple("nan" if np.isnan(x) else fmt(x) for x in r), axis=1)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(a - b) / np.maximum(np.abs(b), 1e-12)


def q(s: pd.Series) -> dict:
    s = pd.Series(s, dtype=float).dropna()
    if not len(s):
        return {"n": 0}
    return {"n": int(len(s)), "median": float(s.median()), "p75": float(s.quantile(.75)),
            "p90": float(s.quantile(.9)), "gt_1pct": float((s > .01).mean()), "gt_10pct": float((s > .10).mean())}


# ---------------------------------------------------------------- A. 版本结构
def versions(raw: pd.DataFrame, api: str) -> pd.DataFrame:
    """每个 (ts_code, end_date, 版本) 一行：版本 = 不同数值（三大表再按 f_ann_date 分）。"""
    d = raw.copy()
    d["_v"] = vkey(d, api)
    keys = ["ts_code", "end_date", "_v"] + (["f_ann_date"] if api in STMT else [])
    agg = {"ann_date": "min", "_row": "min", "update_flag": lambda s: ",".join(sorted(set(map(str, s))))}
    v = d.groupby(keys, dropna=False).agg(agg).reset_index()
    first = d.sort_values("_row").drop_duplicates(keys)[keys + VALS[api]]
    return v.merge(first, on=keys, how="left")


def structure(v: pd.DataFrame, api: str) -> tuple[pd.DataFrame, dict]:
    g = v.groupby(["ts_code", "end_date"])
    s = pd.DataFrame({"n_ver": g["_v"].nunique(), "ann0": g["ann_date"].min(), "n_ann": g["ann_date"].nunique()})
    if api in STMT:
        s["fann_min"], s["fann_max"] = g["f_ann_date"].min(), g["f_ann_date"].max()
        s["orig_present"] = s["fann_min"] == s["ann0"]
        s["late_present"] = s["fann_max"] > s["ann0"]
    s = s.reset_index()
    s["annual"] = s["end_date"].str[4:] == "1231"
    out = {"groups": int(len(s)), "multi_value_versions": float((s.n_ver > 1).mean()),
           "multi_ann_date": float((s.n_ann > 1).mean())}
    if api in STMT:
        out.update({
            "late_version_present": float(s.late_present.mean()),
            "late_and_values_differ": float((s.late_present & (s.n_ver > 1)).mean()),
            "orig_missing_only_late": float((~s.orig_present).mean()),
            "annual": {"groups": int(s.annual.sum()), "multi_value_versions": float((s[s.annual].n_ver > 1).mean()),
                       "orig_missing_only_late": float((~s[s.annual].orig_present).mean())},
        })
        lag = (pd.to_datetime(s.loc[s.late_present, "fann_max"]) - pd.to_datetime(s.loc[s.late_present, "ann0"])).dt.days
        out["late_lag_days"] = {k: float(x) for k, x in lag.describe(percentiles=[.25, .5, .75, .9]).items()}
    return s, out


# ---------------------------------------------------------------- B/C. lake 取的是哪个版本、差多大
def lake_match(lake: pd.DataFrame, v: pd.DataFrame, api: str) -> pd.DataFrame:
    lk = lake.copy()
    lk["vk"] = vkey(lk, api)   # itertuples 会改掉下划线开头的列名
    rows = []
    vg = {k: g for k, g in v.groupby(["ts_code", "end_date"])}
    for r in lk.itertuples(index=False):
        g = vg.get((r.ts_code, r.end_date))
        rec = {"ts_code": r.ts_code, "end_date": r.end_date, "ann_date": r.ann_date}
        if g is None:
            rows.append({**rec, "cls": "no_api"})
            continue
        n_ver = g["_v"].nunique()
        m = g[g["_v"] == r.vk]
        rec["n_ver"] = n_ver
        if api in STMT:
            same_ann = g[g.ann_date == r.ann_date]
            orig = same_ann[same_ann.f_ann_date == r.ann_date]
            rec["orig_present"] = len(orig) > 0
            if len(m) == 0:
                rec["cls"] = "match_none"
            elif (m.f_ann_date <= r.ann_date).any():
                rec["cls"] = "match_orig"
            else:
                rec["cls"] = "match_late_only"
                rec["matched_fann"] = m.f_ann_date.min()
            # 与原始版的差（原始版有多个数值时取接口最先返回的）
            if len(orig) and rec["cls"] != "match_orig":
                o = orig.sort_values("_row").iloc[0]
                for c in USED[api]:
                    rec[f"rel_{c}"] = float(rel(getattr(r, c), o[c]))
            fann = sorted(g.f_ann_date.unique())
            if n_ver > 1 and len(m):
                rec["eq_earliest"] = bool((m.f_ann_date == fann[0]).any())
                rec["eq_latest"] = bool((m.f_ann_date == fann[-1]).any())
        else:
            if len(m) == 0:
                rec["cls"] = "match_none"
            else:
                rec["cls"] = "match"
                rec["flag"] = ",".join(sorted(set(",".join(m.update_flag).split(","))))
                rec["first_returned"] = bool(m._row.min() == g._row.min())
        rows.append(rec)
    return pd.DataFrame(rows)


def stmt_diff(v: pd.DataFrame, api: str) -> pd.DataFrame:
    """原始版 vs 最新版（f_ann_date 最大）在特征字段上的相对差；只看两者都在、且数值不同的期。"""
    out = []
    for (ts, end), g in v.groupby(["ts_code", "end_date"]):
        if g["_v"].nunique() < 2:
            continue
        a0 = g.ann_date.min()
        orig = g[g.f_ann_date == a0].sort_values("_row")
        late = g[g.f_ann_date > a0].sort_values(["f_ann_date", "_row"])
        if not len(orig) or not len(late):
            continue
        o, l_ = orig.iloc[0], late.iloc[-1]
        rec = {"ts_code": ts, "end_date": end, "ann_date": a0, "late_fann": l_.f_ann_date,
               "lag_days": (pd.Timestamp(l_.f_ann_date) - pd.Timestamp(a0)).days}
        for c in VALS[api]:
            rec[f"rel_{c}"] = float(rel(l_[c], o[c])) if not (np.isnan(l_[c]) and np.isnan(o[c])) else 0.0
        out.append(rec)
    return pd.DataFrame(out)


def derived_diff(raw: pd.DataFrame, lake: pd.DataFrame, api: str) -> dict:
    """派生字段（free_cashflow / fcff）：同一 (ts_code, end_date, ann_date) 多行之间是否不同、lake 取的是哪一行、差多大。"""
    c = DERIVED[api]
    d = raw.dropna(subset=["ann_date"])
    g = d.groupby(["ts_code", "end_date", "ann_date"])
    k = pd.DataFrame({"n": g.size(), "nuniq": g[c].apply(lambda s: s.round(4).nunique(dropna=False)),
                      "first": g.apply(lambda x: x.sort_values("_row")[c].iloc[0], include_groups=False),
                      "lo": g[c].min(), "hi": g[c].max()}).reset_index()
    multi = k[k.n > 1]
    dif = multi[multi.nuniq > 1]
    m = dif.merge(lake[["ts_code", "end_date", "ann_date", c]].rename(columns={c: "lake"}),
                  on=["ts_code", "end_date", "ann_date"], how="inner")
    lake_is_first = np.isclose(m.lake, m["first"], rtol=1e-9, equal_nan=True)
    return {"field": c, "groups_multi_row": int(len(multi)), "differs": float(len(dif) / max(len(multi), 1)),
            "differs_share_of_all_groups": float(len(dif) / max(len(k), 1)),
            "lake_equals_first_returned": float(lake_is_first.mean()) if len(m) else None,
            "rel_range": q(rel(dif.hi, dif.lo)) if len(dif) else {"n": 0}}


def fina_diff(v: pd.DataFrame) -> pd.DataFrame:
    """fina_indicator 同一期两个数值版本：特征字段的绝对差（百分比口径的字段即百分点）。"""
    out = []
    for (ts, end), g in v.groupby(["ts_code", "end_date"]):
        if g["_v"].nunique() < 2:
            continue
        g = g.sort_values("_row")
        a, b = g.iloc[0], g.iloc[1]
        rec = {"ts_code": ts, "end_date": end, "ann_date": g.ann_date.min(), "flags": f"{a.update_flag}|{b.update_flag}"}
        for c in USED["fina_indicator"]:
            rec[f"abs_{c}"] = abs(a[c] - b[c]) if not (np.isnan(a[c]) and np.isnan(b[c])) else 0.0
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- fina 版本借 f_ann_date
def fina_avail(vfi: pd.DataFrame, vinc: pd.DataFrame, vbs: pd.DataFrame) -> pd.DataFrame:
    """给 fina 每个版本一个可用日：与同期报表版本对账，借对上的报表版本的 f_ann_date；对不上按 ann_date。"""
    inc = vinc.assign(npm=100 * vinc.n_income / vinc.revenue)[["ts_code", "end_date", "f_ann_date", "npm"]]
    bs = vbs.assign(dta=100 * vbs.total_liab / vbs.total_assets)[["ts_code", "end_date", "f_ann_date", "dta"]]
    ig = {k: g for k, g in inc.groupby(["ts_code", "end_date"])}
    bg = {k: g for k, g in bs.groupby(["ts_code", "end_date"])}
    av, how = [], []
    for r in vfi.itertuples(index=False):
        cand = []
        gi, gb = ig.get((r.ts_code, r.end_date)), bg.get((r.ts_code, r.end_date))
        if gb is not None and not np.isnan(r.debt_to_assets):
            cand += gb.loc[(gb.dta - r.debt_to_assets).abs() < 0.01, "f_ann_date"].tolist()
        if gi is not None and not np.isnan(r.netprofit_margin):
            cand += gi.loc[(gi.npm - r.netprofit_margin).abs() < 0.01, "f_ann_date"].tolist()
        if cand:
            av.append(min(cand))
            how.append("borrowed" if min(cand) > r.ann_date else "matched_orig")
        else:
            av.append(r.ann_date)
            how.append("unmatched" if (gi is not None or gb is not None) else "no_stmt")
    out = vfi.assign(avail=av, avail_how=how)
    # 同一期几个数值版本拿到同一个可用日 → 分不出先后（pit_view 退回到接口返回顺序），单独计数
    out["avail_ambiguous"] = out.groupby(["ts_code", "end_date", "avail"])["_v"].transform("nunique") > 1
    return out


# ---------------------------------------------------------------- D. 对冻结特征的影响
def overlay_lake(v: pd.DataFrame, lk: pd.DataFrame, api: str) -> pd.DataFrame:
    """与 lake 同一版本（同 ts_code、end_date、ann_date、版本键）的行改用 lake 的数值并标 is_lake。
    这样影响估计只反映「能用日期认定的晚到版本」，不混入派生字段口径（fcff 等）和接口返回顺序的变化。"""
    k = lk.assign(_v=vkey(lk, api))[["ts_code", "end_date", "ann_date", "_v"] + VALS[api]]
    k = k.drop_duplicates(["ts_code", "end_date", "ann_date", "_v"])
    out = v.drop(columns=VALS[api]).merge(k, on=["ts_code", "end_date", "ann_date", "_v"], how="left", indicator=True)
    out["is_lake"] = out.pop("_merge") == "both"
    first = v.set_index(out.index)[VALS[api]]
    for c in VALS[api]:
        out[c] = out[c].where(out.is_lake, first[c])
    return out


def pit_view(v: pd.DataFrame, avail_col: str, asof: str) -> pd.DataFrame:
    """as-of 时点可见的版本：avail ≤ asof，同一期取 avail 最大；可用日并列（分不出先后）时取 lake 那一行，再按接口返回顺序。"""
    d = v[v[avail_col] <= asof]
    d = d.sort_values(["ts_code", "end_date", avail_col, "is_lake", "_row"], ascending=[True, True, False, False, True])
    return d.drop_duplicates(["ts_code", "end_date"])


def feats(fi, inc, bs, cf, mv, asof: str) -> pd.DataFrame:
    """冻结构建器 one_day 的同一套函数（只算受财报影响的列，不缩尾）。"""
    snap ={"financial_indicators": fi, "financials_income": inc, "financials_balance_sheet": bs,
            "financials_cashflow": cf, "daily_basic": mv}
    as_of = pd.Timestamp(asof).date()
    out = {n: getattr(F, n)(snap, as_of) for n in FINA_FEATS + ["accruals", "ocf_to_revenue_ttm"]}
    cfa = cf.sort_values(["ticker", "report_date"], ascending=[True, False]).drop_duplicates("ticker").set_index("ticker")
    m = mv.set_index("ticker")["total_mv"]
    num = cfa["n_cashflow_act"] - cfa["c_paid_invest"]
    common = num.index.intersection(m.index)
    out["fcf_yield_cf"] = num.loc[common] / m.loc[common].replace(0, np.nan)
    return pd.DataFrame(out)


def impact(root: Path, sample: list[str], lakes: dict, vs: dict, vfi_av: pd.DataFrame) -> pd.DataFrame:
    ev_all = pd.read_parquet(REPO / "docs/work/evidence/T-030/p6/frozen_eval_rows_20d.parquet")
    ev = ev_all[ev_all.ticker.isin(sample)]
    dates = sorted(ev.as_of.unique())
    db = pq.read_table(root / "data/lake/daily_basic.parquet", columns=["ts_code", "trade_date", "total_mv"],
                       filters=[("ts_code", "in", sample)]).to_pandas()
    db["trade_date"] = pd.to_datetime(db["trade_date"])
    cols = FINA_FEATS + STMT_FEATS
    res, month, fzm = [], None, None
    for dt in dates:
        if f"{pd.Timestamp(dt):%Y-%m}" != month:
            month = f"{pd.Timestamp(dt):%Y-%m}"
            fzm = pd.read_parquet(root / f"data/gpu_v1/panel/fnd/{month}.parquet", columns=cols)
        fz = fzm.xs(pd.Timestamp(dt), level="as_of")
        # 缩尾边界 = 冻结全截面的 min/max；分位用当日可评估截面
        bounds = {c: (fz[c].min(), fz[c].max()) for c in cols}
        cs = fz.loc[fz.index.isin(set(ev_all.loc[ev_all.as_of == dt, "ticker"]))]
        srt = {c: np.sort(cs[c].dropna().values) for c in cols}
        asof = pd.Timestamp(dt).strftime("%Y%m%d")
        tick = set(ev.loc[ev.as_of == dt, "ticker"])
        ltd = db.loc[db.trade_date <= dt, "trade_date"].max()
        mv = db[db.trade_date == ltd].assign(ticker=lambda x: x.ts_code)
        L = {k: d[(d.ann_date <= asof) & d.ts_code.isin(tick)] for k, d in lakes.items()}
        fl = feats(L["fina_indicator"].assign(ticker=lambda x: x.ts_code, report_date=lambda x: x.end_date),
                   *(L[k].assign(ticker=lambda x: x.ts_code, report_date=lambda x: x.end_date,
                                 f_ann_date=lambda x: x.ann_date) for k in STMT), mv, asof)
        P = {k: pit_view(vs[k][vs[k].ts_code.isin(tick)], "f_ann_date", asof) for k in STMT}
        pfi = pit_view(vfi_av[vfi_av.ts_code.isin(tick)], "avail", asof)
        fp = feats(pfi.assign(ticker=lambda x: x.ts_code, report_date=lambda x: x.end_date, ann_date=lambda x: x.avail),
                   *(P[k].assign(ticker=lambda x: x.ts_code, report_date=lambda x: x.end_date) for k in STMT), mv, asof)
        # 选中的报告期是否不同
        sel_l = L["income"].groupby("ts_code").end_date.max()
        sel_p = P["income"].groupby("ts_code").end_date.max()
        self_l = L["fina_indicator"].groupby("ts_code").end_date.max()
        self_p = pfi.groupby("ts_code").end_date.max()

        def pct(c, x):
            if np.isnan(x) or not len(srt[c]):
                return np.nan
            return float(np.searchsorted(srt[c], np.clip(x, *bounds[c]), side="right") / len(srt[c]))

        for t in sorted(tick):
            rec = {"as_of": dt, "ticker": t, "sel_end_lake": sel_l.get(t), "sel_end_pit": sel_p.get(t),
                   "fina_end_lake": self_l.get(t), "fina_end_pit": self_p.get(t)}
            for c in cols:
                a = float(fl[c].get(t, np.nan)) if c in fl else np.nan
                b = float(fp[c].get(t, np.nan)) if c in fp else np.nan
                z = float(fz[c].get(t, np.nan)) if t in fz.index else np.nan
                lo, hi = bounds[c]
                rec[f"{c}_lake"], rec[f"{c}_pit"], rec[f"{c}_frozen"] = a, b, z
                rec[f"{c}_repro_ok"] = bool((np.isnan(a) and np.isnan(z)) or
                                            (not np.isnan(a) and not np.isnan(z)
                                             and abs(np.clip(a, lo, hi) - z) <= 1e-9 * max(1.0, abs(z))))
                rec[f"{c}_pct_lake"], rec[f"{c}_pct_pit"] = pct(c, a), pct(c, b)
            res.append(rec)
    return pd.DataFrame(res)


def impact_summary(im: pd.DataFrame) -> dict:
    out = {"rows": int(len(im)), "tickers": int(im.ticker.nunique()), "dates": int(im.as_of.nunique()),
           "sel_period_differs": float((im.sel_end_lake.fillna("") != im.sel_end_pit.fillna("")).mean()),
           "fina_period_differs": float((im.fina_end_lake.fillna("") != im.fina_end_pit.fillna("")).mean())}
    any_s = np.zeros(len(im), bool)
    any_f = np.zeros(len(im), bool)
    per = {}
    for c in FINA_FEATS + STMT_FEATS:
        a, b = im[f"{c}_lake"].astype(float), im[f"{c}_pit"].astype(float)
        both = a.notna() & b.notna()
        diff = (both & (rel(a, b) > 1e-9)) | (a.notna() != b.notna())
        dp = (im[f"{c}_pct_lake"] - im[f"{c}_pct_pit"]).abs()
        per[c] = {"differs": float(diff.mean()), "nan_status_differs": float((a.notna() != b.notna()).mean()),
                  "repro_ok": float(im[f"{c}_repro_ok"].mean()) if f"{c}_repro_ok" in im else None,
                  "abs_dpct_when_differs": q(dp[diff & both]), "abs_dpct_gt_10pp": float((dp[diff & both] > .10).mean())
                  if (diff & both).any() else 0.0}
        (any_s if c in STMT_FEATS else any_f)[:] |= diff.values
    out.update({"any_stmt_feature_differs": float(any_s.mean()), "any_fina_feature_differs": float(any_f.mean()),
                "any_feature_differs": float((any_s | any_f).mean()), "per_feature": per})
    yr = pd.Series(any_s | any_f, index=pd.to_datetime(im.as_of).dt.year).groupby(level=0).mean()
    out["any_feature_differs_by_year"] = {int(k): float(v) for k, v in yr.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--sub", default="", help="分析 pit_audit/<sub>/ 下的名单（如 T-038 案例），不做特征影响")
    ap.add_argument("--allow-partial", action="store_true", help="只用四个接口都已拉到的股票（试跑用）")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root = a.data_root.resolve()
    base = root / "data/research/pit_audit" / a.sub if a.sub else root / "data/research/pit_audit"
    res = base / a.out
    res.mkdir(parents=True, exist_ok=True)
    sample = json.loads((base / "sample.json").read_text())["tickers"]
    have = [t for t in sample if all((base / "raw" / api / f"{t}.parquet").exists() for api in VALS)]
    if len(have) < len(sample) and not a.allow_partial:
        raise SystemExit(f"{len(sample) - len(have)} 只股票的原始文件不全；试跑请加 --allow-partial")
    summary: dict = {"sample": len(sample), "analysed": len(have)}
    sample = have

    raws, vs, lakes = {}, {}, {}
    for api in VALS:
        raws[api] = load_raw(base, api, sample)
        vs[api] = versions(raws[api], api)
        lk = pd.read_parquet(root / f"data/lake/{LAKE[api]}.parquet")
        lk = lk[lk.ts_code.astype(str).isin(sample)].copy()
        for c in ("ts_code", "ann_date", "end_date"):
            lk[c] = lk[c].astype(object).where(lk[c].notna(), None)
        lk = lk[lk.ann_date.notna()]
        for c in VALS[api]:
            lk[c] = pd.to_numeric(lk[c], errors="coerce").astype("float64")
        lakes[api] = lk

        s, st = structure(vs[api], api)
        st["raw_rows"] = int(len(raws[api]))
        st["rows_per_group"] = {int(k): int(n) for k, n in raws[api].groupby(["ts_code", "end_date"]).size()
                                .value_counts().sort_index().items()}
        m = lake_match(lk, vs[api], api)
        m["frozen_window"] = m.ann_date <= FROZEN_END
        st["lake_rows"] = int(len(m))
        st["lake_cls"] = {k: float(x) for k, x in m.cls.value_counts(normalize=True).items()}
        mw = m[m.frozen_window]
        st["lake_cls_frozen_window"] = {k: float(x) for k, x in mw.cls.value_counts(normalize=True).items()}
        if api in STMT:
            mv_ = m[m.n_ver.fillna(1) > 1] if "n_ver" in m else m.iloc[:0]
            ok = len(mv_) and "eq_earliest" in mv_
            st["lake_in_multi_version_groups"] = {
                "n": int(len(mv_)), "eq_earliest": float(mv_.eq_earliest.eq(True).mean()) if ok else None,
                "eq_latest": float(mv_.eq_latest.eq(True).mean()) if ok else None,
                "neither": float((m.loc[mv_.index, "cls"] == "match_none").mean()) if len(mv_) else None}
            la = m[m.cls == "match_late_only"]
            st["lookahead_rows"] = {"n": int(len(la)), "share": float(len(la) / max(len(m), 1)),
                                    "tickers": int(la.ts_code.nunique()),
                                    "lag_days": q((pd.to_datetime(la.matched_fann) - pd.to_datetime(la.ann_date)).dt.days)
                                    if len(la) else {"n": 0},
                                    "orig_present": float(la.orig_present.mean()) if len(la) else None,
                                    "share_annual": float((m[m.end_date.str[4:] == "1231"].cls == "match_late_only").mean()),
                                    "share_interim": float((m[m.end_date.str[4:] != "1231"].cls == "match_late_only").mean())}
            st["lookahead_rel_diff_vs_orig"] = {c: q(la.get(f"rel_{c}", pd.Series(dtype=float))) for c in USED[api]}
            d = stmt_diff(vs[api], api)
            d.to_parquet(res / f"diff_{api}.parquet", index=False)
            st["orig_vs_latest_rel_diff"] = {c: q(d[f"rel_{c}"][d[f"rel_{c}"] > 0]) for c in KEYC[api]} if len(d) else {}
            st["orig_vs_latest_groups"] = int(len(d))
        else:
            st["lake_flag"] = {k: float(x) for k, x in m.flag.value_counts(normalize=True).items()} if "flag" in m else {}
            mv_ = m[(m.n_ver.fillna(1) > 1) & (m.cls == "match")] if "n_ver" in m else m.iloc[:0]
            st["lake_in_multi_version_groups"] = {"n": int(len(mv_)), "first_returned": float(mv_.first_returned.mean())
                                                  if len(mv_) else None}
            d = fina_diff(vs[api])
            d.to_parquet(res / f"diff_{api}.parquet", index=False)
            st["two_version_abs_diff"] = {c: q(d[f"abs_{c}"][d[f"abs_{c}"] > 0]) for c in USED[api]} if len(d) else {}
        if api in DERIVED:
            st["derived_field_diff"] = derived_diff(raws[api], lk, api)
        s.to_parquet(res / f"structure_{api}.parquet", index=False)
        m.to_parquet(res / f"lake_match_{api}.parquet", index=False)
        summary[api] = st
        print(f"[{api}] {json.dumps({k: st[k] for k in ('groups', 'multi_value_versions', 'lake_cls')}, ensure_ascii=False)}",
              flush=True)

    vfi = fina_avail(vs["fina_indicator"], vs["income"], vs["balancesheet"])
    vfi.drop(columns="_v").to_parquet(res / "fina_versions_avail.parquet", index=False)
    summary["fina_avail_how"] = {k: int(x) for k, x in vfi.avail_how.value_counts().items()}
    multi = vfi.groupby(["ts_code", "end_date"])["_v"].transform("nunique") > 1
    summary["fina_avail_how_multi_version"] = {k: int(x) for k, x in vfi[multi].avail_how.value_counts().items()}
    summary["fina_avail_ambiguous_multi_version"] = int(vfi[multi].avail_ambiguous.sum())

    if not a.sub:
        vo = {k: overlay_lake(vs[k], lakes[k], k) for k in STMT}
        vfo = overlay_lake(vfi, lakes["fina_indicator"], "fina_indicator")
        summary["overlay_is_lake_share"] = {k: float(x.is_lake.mean()) for k, x in {**vo, "fina_indicator": vfo}.items()}
        im = impact(root, sample, lakes, vo, vfo)
        im.to_parquet(res / "impact_eval_rows.parquet", index=False)
        summary["impact"] = impact_summary(im)
    (res / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summary.get("impact", {}), ensure_ascii=False, indent=1, default=str)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
