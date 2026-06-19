"""P63-3：63d 两口径评估 + ABCD leaderboard（复刻 P4d 口径，period=63d）.

复用 p4d_eval_v6 的 UNKNOWN 桶中性化 / adv20 top-1500 / regime / per-asof neut。
变体 B=full_fnd / C=full_nofnd(ablation) / D=fnd_only。leaderboard 含硬编码 A(P4 12d)。
输出 data/bakeoff/p63_3_verdict_v6.json。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import scripts.bakeoff.evaluate_bakeoff as EB
from scripts.bakeoff.evaluate_bakeoff import score_predictions
from scripts.survivorship.p4d_eval_v6 import (
    _v6_context, _adv20_top1500_mask, _per_asof_neut, _regime_breakdown)  # 这步会给 EB 打 UNKNOWN 补丁
from scripts._diag_wf_v2 import build_regime
from quantmind.data.lake import load_trading_calendar
from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from scripts.survivorship.p63_3_train_v6 import make_semiannual_cutoffs

PREDS = REPO / "data/bakeoff/preds"
LABEL, PERIOD = "forward_return_63d", "63d"
V6_PRICES = REPO / "data/raw/alpha_prices_panel_v6.parquet"
# A 行：P4 v6 12d Ridge(full) PIT top-1500（硬编码自 p4_verdict_v6）
ROW_A = {"row": "A", "desc": "P4 v6 12d Ridge(full) top1500", "neut_ic": 0.057,
         "net_excess_cost": 0.0275, "neut_icir": 0.988,
         "regime": {"bear": 0.067, "neutral": 0.050, "bull": 0.055}}


def _merge_ctx(preds, panel, lmc):
    df = preds.copy(); df["as_of"] = pd.to_datetime(df["as_of"])
    df = df.merge(panel[LABEL].rename("realized"), left_on=["as_of", "ticker"], right_index=True, how="left")
    df = df.merge(panel["exposure_industry"].rename("industry"), left_on=["as_of", "ticker"], right_index=True, how="left")
    df = df.merge(panel["amihud_illiquidity"].rename("amihud"), left_on=["as_of", "ticker"], right_index=True, how="left")
    rows = []
    for a, sub in df.groupby("as_of"):
        s = lmc.get(pd.Timestamp(a).normalize())
        rows.append(sub.assign(log_mktcap=sub["ticker"].map(s) if s is not None else np.nan))
    return pd.concat(rows, ignore_index=True)


def _eval(df, panel, lmc, regime, name, asof2fold=None):
    m = score_predictions(df[["as_of", "ticker", "score"]].copy(), period=PERIOD, panel=panel, lmc=lmc)
    per = _per_asof_neut(df)
    reg, reg_n = _regime_breakdown(per, regime)
    out = {"lens": name, "raw_ic": m["raw_ic"], "neut_ic": m["neut_ic"], "neut_icir": m["neut_icir"],
           "net_excess_cost": m["net_excess_cost"], "avg_oneway_turnover": m["avg_oneway_turnover"],
           "max_drawdown": m["max_drawdown"], "neut_ic_posfrac": m["neut_ic_posfrac"],
           "regime": reg, "regime_counts": reg_n, "n_asof": int(len(per))}
    if asof2fold is not None:
        per = per.copy(); per["fold"] = per["as_of"].map(lambda a: asof2fold.get(pd.Timestamp(a)))
        fic = per.dropna(subset=["fold"]).groupby("fold")["neu_ic"].mean()
        v = fic.values
        out["per_fold"] = {"n": int(len(v)), "min": round(float(v.min()), 4), "median": round(float(np.median(v)), 4),
                           "max": round(float(v.max()), 4), "n_pos": int((v > 0).sum()),
                           "drop_best2": round(float(np.sort(v)[:-2].mean()), 4) if len(v) > 2 else None}
    return out


def main():
    panel, lmc = _v6_context()
    as_of_list = sorted(panel.index.get_level_values("as_of").unique())
    regime, _ = build_regime(panel, as_of_list)
    top = _adv20_top1500_mask(as_of_list)
    # fold 映射（同训练配置）
    cal = load_trading_calendar()
    sp = PurgedWalkForwardSplit(cal, horizon=63, embargo=63, mode="rolling", rolling_lookback_td=756, n_val=2)
    cut = make_semiannual_cutoffs(cal, "2022-04-01", "2025-10-31")
    folds = sp.make_folds(as_of_list, cut, oos_start=pd.Timestamp("2022-01-01"), oos_end=as_of_list[-1])
    asof2fold = {pd.Timestamp(dt): i for i, f in enumerate(folds) for dt in f.test_dates}

    res = {"period": "63d", "leaderboard": [ROW_A], "details": {}}
    variants = {"full_fnd": ("B", "63d Ridge(full+fnd)"), "full_nofnd": ("C", "63d Ridge(full,no-fnd) ablation"),
                "fnd_only": ("D", "63d Ridge(fnd-only)")}
    for variant, (row, desc) in variants.items():
        fp = PREDS / f"ridge_{variant}_63d_quarterly_v6.parquet"
        if not fp.exists():
            print(f"[skip] {fp.name} 不存在"); continue
        df = _merge_ctx(pd.read_parquet(fp), panel, lmc)
        df["ticker"] = df["ticker"].astype(str)
        df2 = df[df.apply(lambda r: r["ticker"] in top.get(pd.Timestamp(r["as_of"]), set()), axis=1)]
        e_top = _eval(df2, panel, lmc, regime, "pit_top1500", asof2fold)
        detail = {"pit_top1500": e_top}
        if variant == "full_fnd":
            detail["full_market"] = _eval(df, panel, lmc, regime, "full_market", asof2fold)
        res["details"][variant] = detail
        res["leaderboard"].append({"row": row, "desc": desc, "neut_ic": e_top["neut_ic"],
                                   "net_excess_cost": e_top["net_excess_cost"], "neut_icir": e_top["neut_icir"],
                                   "regime": e_top["regime"], "per_fold": e_top.get("per_fold"),
                                   "avg_oneway_turnover": e_top["avg_oneway_turnover"], "max_drawdown": e_top["max_drawdown"]})
        print(f"[{row}] {desc}: top1500 neut_ic={e_top['neut_ic']} net={e_top['net_excess_cost']} "
              f"icir={e_top['neut_icir']} regime={e_top['regime']}", flush=True)

    (REPO / "data/bakeoff/p63_3_verdict_v6.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n=== ABCD leaderboard (PIT top-1500) ===")
    for r in res["leaderboard"]:
        print(f"  {r['row']} {r['desc']:36} neut_ic={r['neut_ic']:+.4f} net={r.get('net_excess_cost')} "
              f"icir={r.get('neut_icir')} regime={r.get('regime')}")
    print("\n[save] data/bakeoff/p63_3_verdict_v6.json")


if __name__ == "__main__":
    main()
