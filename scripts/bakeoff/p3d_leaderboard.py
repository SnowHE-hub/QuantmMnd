#!/usr/bin/env python
"""P3d：评估 data/bakeoff/preds/ 下所有预测 → leaderboard（quantmind env）.

每个 <model>_<featset/fmt>_<period>_<refit>[_sN].parquet 过 evaluate_bakeoff.score_predictions，
含 v2 对照行（_diag_wf_v2 预测）。按 (中性化 IC, 含成本净超额) 降序。
深度模型多 seed 时报均值±std。
"""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from scripts.bakeoff.evaluate_bakeoff import score_predictions, _load_context

PREDS = REPO / "data/bakeoff/preds"
panel, lmc = _load_context()


def parse(name):
    s = name.replace(".parquet", "")
    seed = None
    m = re.search(r"_s(\d+)$", s)
    if m:
        seed = int(m.group(1)); s = s[:m.start()]
    parts = s.split("_")
    # model_featset..._period_refit
    period = next((p for p in parts if p in ("12d", "63d")), "12d")
    refit = next((p for p in parts if p in ("quarterly", "monthly")), "quarterly")
    model = parts[0]
    fmt = "_".join(parts[1:parts.index(period)]) if period in parts else "?"
    return model, fmt, period, refit, seed


def main():
    rows = []
    # v2 对照行
    v2p = REPO / "data/loss_signals_v4/wf_v2_oos_predictions.parquet"
    if v2p.exists():
        v2 = pd.read_parquet(v2p).rename(columns={"pred": "score"})
        m = score_predictions(v2[["as_of", "ticker", "score", "realized", "industry", "log_mktcap"]],
                              period="12d", panel=panel, lmc=lmc)
        m.update({"model": "LGBM(v2对照)", "fmt": "tab35", "period": "12d", "refit": "季度", "seed": "-"})
        rows.append(m)
    # 所有预测
    by_key = {}
    for f in sorted(PREDS.glob("*.parquet")):
        model, fmt, period, refit, seed = parse(f.name)
        preds = pd.read_parquet(f)
        m = score_predictions(preds, period=period, panel=panel, lmc=lmc)
        m.update({"model": model, "fmt": fmt, "period": period, "refit": refit, "seed": seed})
        rows.append(m)
        by_key.setdefault((model, fmt, period, refit), []).append(m["neut_ic"])
    df = pd.DataFrame(rows)
    keep = ["model", "fmt", "period", "refit", "seed", "raw_ic", "neut_ic", "neut_icir",
            "neut_ic_posfrac", "net_excess_cost", "annualized_turnover", "max_drawdown",
            "win_rate", "n_rebalances"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.sort_values(["neut_ic", "net_excess_cost"], ascending=False)
    out = REPO / "data/bakeoff/leaderboard.csv"
    df.to_csv(out, index=False)
    print(f"=== BAKE-OFF LEADERBOARD (排序: neut_ic, net_excess) ===")
    print(df.to_string(index=False))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
