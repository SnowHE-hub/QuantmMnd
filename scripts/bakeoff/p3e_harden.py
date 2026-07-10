#!/usr/bin/env python
"""P3e：tabular 预测硬化（regime 条件 IC / 逐 fold 分布 / 早晚时段），不重训。

复用已存预测 + _diag_wf_v2 的 regime 切分（market_momentum_60d 三档）+ 中性化口径。
回答：Ridge(full) 的 +0.034 是真特异 alpha 还是 v2 的 regime tilt 换脸。
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from scripts.bakeoff.evaluate_bakeoff import _neutralize_date, _load_context
from quantmind.data.lake import load_trading_calendar
from quantmind.backtest.wf_split import PurgedWalkForwardSplit

OOS_START = pd.Timestamp("2022-01-01")
LATE_START = pd.Timestamp("2025-01-01")


def make_quarterly_cutoffs(cal, start, end):
    cs = sorted(cal); out = []
    for qd in pd.date_range(start, end, freq="QS-JAN"):
        a = [d for d in cs if d >= qd]
        if a: out.append(a[0])
    return out


def per_asof_ic(preds, panel, lmc, label):
    df = preds.copy(); df["as_of"] = pd.to_datetime(df["as_of"])
    df = df.merge(panel[label].rename("realized"), left_on=["as_of", "ticker"], right_index=True, how="left")
    df = df.merge(panel["exposure_industry"].rename("industry"), left_on=["as_of", "ticker"], right_index=True, how="left")
    rows = []
    for a, sub in df.groupby("as_of"):
        s = lmc.get(pd.Timestamp(a).normalize())
        sub = sub.assign(log_mktcap=sub["ticker"].map(s) if s is not None else np.nan)
        r = _neutralize_date(sub[["score", "realized", "industry", "log_mktcap"]].rename(columns={}))
        if r is not None:
            rows.append((pd.Timestamp(a), float(r["raw_ic"]), float(r["neu_ic"])))
    return pd.DataFrame(rows, columns=["as_of", "raw_ic", "neu_ic"]).set_index("as_of").sort_index()


def regime_labels(panel):
    mm = panel.groupby(level="as_of")["market_momentum_60d"].first()
    oos = mm[mm.index >= OOS_START].dropna()
    q33, q67 = oos.quantile([1/3, 2/3])
    def lab(v):
        return "unknown" if v != v else ("bear" if v <= q33 else ("bull" if v >= q67 else "neutral"))
    return mm.map(lab)


def fold_of(cal, as_of_list):
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=20, mode="rolling",
                               rolling_lookback_td=756, n_val=2)
    cut = make_quarterly_cutoffs(cal, OOS_START, pd.Timestamp("2025-10-01"))
    folds = sp.make_folds(as_of_list, cut, oos_start=OOS_START, oos_end=as_of_list[-1])
    m = {}
    for i, f in enumerate(folds):
        for d in f.test_dates:
            m[pd.Timestamp(d)] = i
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--period", default="12d")
    a = ap.parse_args()
    label = {"12d": "forward_return_12d", "63d": "forward_return_63d"}[a.period]
    panel, lmc = _load_context()
    cal = load_trading_calendar()
    as_of_list = sorted(panel.index.get_level_values("as_of").unique())
    preds = pd.read_parquet(a.preds)
    ic = per_asof_ic(preds, panel, lmc, label)
    reg = regime_labels(panel); fmap = fold_of(cal, as_of_list)
    ic["regime"] = ic.index.map(reg); ic["fold"] = ic.index.map(fmap)
    ic["seg"] = np.where(ic.index >= LATE_START, "late(2025-26)", "early(2022-24)")

    name = Path(a.preds).stem
    print(f"\n===== HARDENING: {name} ({a.period}) =====")
    print(f"overall neut IC = {ic['neu_ic'].mean():+.4f}  ICIR = {ic['neu_ic'].mean()/ic['neu_ic'].std():.3f}  "
          f"n_asof = {len(ic)}  posfrac = {(ic['neu_ic']>0).mean():.2f}")

    print("\n-- (1) regime-conditional neut IC --  (v2: bear +0.028 / bull +0.024 / neutral -0.007)")
    rt = ic.groupby("regime")["neu_ic"].agg(neut_ic="mean", std="std", n="count",
                                            posfrac=lambda s: (s > 0).mean())
    print(rt.round(4).to_string())

    print("\n-- (2) per-fold neut IC distribution --")
    ft = ic.groupby("fold")["neu_ic"].agg(neut_ic="mean", n="count").round(4)
    print(ft.to_string())
    fic = ft["neut_ic"]
    print(f"  folds: {len(fic)}  pos={int((fic>0).sum())}/{len(fic)}  "
          f"mean={fic.mean():+.4f}  std={fic.std():.4f}  min={fic.min():+.4f}  max={fic.max():+.4f}")
    # 去掉 top-1/top-2 fold 后的均值（集中度）
    s = fic.sort_values(ascending=False)
    print(f"  drop top-1 fold mean={s.iloc[1:].mean():+.4f}  drop top-2={s.iloc[2:].mean():+.4f}")

    print("\n-- (3) early vs late segment --")
    st = ic.groupby("seg")["neu_ic"].agg(neut_ic="mean", std="std", n="count",
                                         posfrac=lambda s: (s > 0).mean())
    print(st.round(4).to_string())

    ic.reset_index().to_csv(REPO / f"data/bakeoff/harden_{name}.csv", index=False)
    print(f"\n[save] data/bakeoff/harden_{name}.csv")


if __name__ == "__main__":
    main()
