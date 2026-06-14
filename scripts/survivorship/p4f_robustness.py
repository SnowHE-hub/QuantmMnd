"""P4 补1+补2：流动性分桶 IC（验证中小盘解释）+ 逐 fold 稳健性（quantmind env）.

补1：v6 预测按 PIT 流动性分桶（每 as_of adv20 排名，≤as_of）top500 / mid500-1500 / small1500+；
     每桶 neut IC（UNKNOWN 桶口径）+ ICIR + 三 regime。验证"IC 随流动性下降而升、top500≈v5"。
补2：16 fold neut IC（全市场 / PIT top-1500 两口径）分布 + 时序图 + 丢最好 2 fold 后均值。
输出 data/bakeoff/p4f_robustness.json + data/bakeoff/p4f_fold_ic.png。不调参，纯诊断。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from scripts.survivorship.p4d_eval_v6 import (
    _v6_context, _merge_ctx, _per_asof_neut, _regime_breakdown, V6_PRICES, PREDS)
from scripts._diag_wf_v2 import build_regime
from quantmind.data.lake import load_trading_calendar
from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from scripts.bakeoff.p3b_tabular import make_quarterly_cutoffs

V5_NEUT_IC = 0.034


def adv20_rank(as_of_list):
    """每 as_of：ticker→adv20 降序排名（1=最活跃；PIT 只用 ≤as_of）。"""
    px = pd.read_parquet(V6_PRICES, columns=["ts_code", "trade_date", "amount"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    amt = (px.drop_duplicates(["trade_date", "ts_code"])
           .pivot(index="trade_date", columns="ts_code", values="amount").sort_index())
    adv = amt.rolling(20, min_periods=10).mean()
    out = {}
    for a in as_of_list:
        a = pd.Timestamp(a)
        prior = adv.index[adv.index <= a]
        if len(prior) == 0:
            out[a] = pd.Series(dtype=float); continue
        out[a] = adv.loc[prior[-1]].dropna().rank(ascending=False, method="first")
    return out


def _agg(per, regime):
    ic = float(per["neu_ic"].mean()); sd = float(per["neu_ic"].std())
    reg, _ = _regime_breakdown(per, regime)
    return {"neut_ic": round(ic, 4), "neut_icir": round(ic / sd, 3) if sd else None,
            "posfrac": round(float((per["neu_ic"] > 0).mean()), 3),
            "n_asof": int(len(per)), "regime": reg}


def main():
    panel, lmc = _v6_context()
    df = _merge_ctx(pd.read_parquet(PREDS), panel, lmc)
    df["ticker"] = df["ticker"].astype(str)
    as_of_list = sorted(panel.index.get_level_values("as_of").unique())
    regime, _ = build_regime(panel, as_of_list)
    rk = adv20_rank(as_of_list)
    df["_rank"] = df.apply(lambda r: rk[pd.Timestamp(r["as_of"])].get(r["ticker"], np.nan), axis=1)

    res = {"v5_neut_ic_baseline": V5_NEUT_IC, "buckets": {}, "folds": {}}

    # ---- 补1：流动性分桶 ----
    buckets = {"top500 (rank≤500)": (df["_rank"] <= 500),
               "mid 500-1500": (df["_rank"] > 500) & (df["_rank"] <= 1500),
               "small 1500+": (df["_rank"] > 1500)}
    print("=== 补1：流动性分桶 neut IC（UNKNOWN 桶口径）===")
    print(f"{'bucket':22} {'neutIC':>8} {'ICIR':>6} {'posfrac':>8}  bear/neutral/bull")
    for name, mask in buckets.items():
        per = _per_asof_neut(df[mask])
        a = _agg(per, regime); res["buckets"][name] = a
        rg = a["regime"]
        print(f"{name:22} {a['neut_ic']:>8.4f} {a['neut_icir']:>6.2f} {a['posfrac']:>8.3f}  "
              f"{rg['bear']:+.4f}/{rg['neutral']:+.4f}/{rg['bull']:+.4f}")
    print(f"  (v5 基线 neut IC = {V5_NEUT_IC})")

    # ---- 补2：逐 fold 稳健性（两口径）----
    cal = load_trading_calendar()
    oos_start = pd.Timestamp("2022-01-01"); oos_end = as_of_list[-1]
    sp = PurgedWalkForwardSplit(cal, horizon=12, embargo=20, mode="rolling",
                                rolling_lookback_td=756, n_val=2)
    cut = make_quarterly_cutoffs(cal, oos_start, pd.Timestamp("2025-10-01"))
    folds = sp.make_folds(as_of_list, cut, oos_start=oos_start, oos_end=oos_end)
    asof2fold, fold_cut = {}, {}
    for i, f in enumerate(folds):
        fold_cut[i] = pd.Timestamp(f.cutoff)
        for dt in f.test_dates:
            asof2fold[pd.Timestamp(dt)] = i

    top1500 = {a: set(s[s <= 1500].index) for a, s in rk.items()}
    df_top = df[df.apply(lambda r: r["ticker"] in top1500.get(pd.Timestamp(r["as_of"]), set()), axis=1)]

    fig_series = {}
    print("\n=== 补2：逐 fold neut IC 分布 ===")
    for lens_name, d in [("full_market", df), ("pit_top1500", df_top)]:
        per = _per_asof_neut(d)
        per["fold"] = per["as_of"].map(lambda a: asof2fold.get(pd.Timestamp(a)))
        fic = per.dropna(subset=["fold"]).groupby("fold")["neu_ic"].mean().sort_index()
        vals = fic.values
        drop2 = float(np.sort(vals)[:-2].mean()) if len(vals) > 2 else float(vals.mean())
        q = np.quantile(vals, [0, .25, .5, .75, 1.0]).round(4)
        summary = {"n_folds": int(len(vals)), "min": float(q[0]), "p25": float(q[1]),
                   "median": float(q[2]), "p75": float(q[3]), "max": float(q[4]),
                   "n_folds_positive": int((vals > 0).sum()),
                   "mean_drop_best2": round(drop2, 4),
                   "robust_pass": bool(drop2 >= 0.02 and (vals > 0).sum() > len(vals) / 2)}
        res["folds"][lens_name] = summary
        fig_series[lens_name] = pd.Series(vals, index=[fold_cut[i] for i in fic.index])
        print(f"[{lens_name}] folds={summary['n_folds']} "
              f"min/p25/med/p75/max={q[0]}/{q[1]}/{q[2]}/{q[3]}/{q[4]} "
              f"正fold={summary['n_folds_positive']}/{summary['n_folds']} "
              f"丢最好2后均值={summary['mean_drop_best2']} robust_pass={summary['robust_pass']}")

    # ---- 时序图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for lens_name, s in fig_series.items():
            ax.plot(s.index, s.values, marker="o", label=lens_name)
        ax.axhline(0.034, ls="--", c="gray", lw=1, label="v5 baseline 0.034")
        ax.axhline(0.02, ls=":", c="red", lw=1, label="gate 0.02")
        ax.axhline(0, c="black", lw=0.6)
        ax.set_title("P4 v6 Ridge(full) 12d — per-fold neut IC (2022Q1→2025Q4)")
        ax.set_xlabel("fold cutoff"); ax.set_ylabel("neut IC"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        fig.tight_layout()
        out_png = REPO / "data/bakeoff/p4f_fold_ic.png"
        fig.savefig(out_png, dpi=120)
        print(f"\n[fig] saved {out_png}")
    except Exception as e:
        print(f"[fig] skip: {e!r}")

    (REPO / "data/bakeoff/p4f_robustness.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n[save] data/bakeoff/p4f_robustness.json")


if __name__ == "__main__":
    main()
