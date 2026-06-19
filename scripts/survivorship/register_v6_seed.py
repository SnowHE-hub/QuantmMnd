"""注册首条产品候选 Ridge_full_12d_v6_seed（gate=research_candidate_pending_nav）.

证据来自 P4 判定（docs/plans/survivorship_p4_verdict.md）。净超额 +2.75% < formal gate +5%，
故 gate_status=research_candidate_pending_nav（待 executable NAV 回测签收）。
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import json
from quantmind.contracts.model_registry import ModelRecord, register
from scripts.bakeoff.run_manifest import sha256_file

preds = REPO / "data/bakeoff/preds/ridge_full_12d_quarterly_v6.parquet"
verdict = json.loads((REPO / "data/bakeoff/p4_verdict_v6.json").read_text())
top = next(l for l in verdict["lenses"] if l["lens"] == "pit_top1500")
full = next(l for l in verdict["lenses"] if l["lens"] == "full_market")
robust = json.loads((REPO / "data/bakeoff/p4f_robustness.json").read_text())

rec = ModelRecord(
    model_id="ridge_full_12d_v6_seed",
    label="forward_return_12d",
    horizon="short",
    feature_set="full_35_16_158",
    data_version="v6",
    wf_config={"splitter": "PurgedWalkForwardSplit", "horizon": 12, "embargo": 20,
               "mode": "rolling", "rolling_lookback_td": 756, "n_val": 2, "refit": "quarterly",
               "oos_start": "2022-01-01", "decide_direction": "val_only", "alpha": 10.0,
               "n_folds": 16, "n_flips": 0},
    neutralization={"industry_dummies": True, "unknown_bucket": True,
                    "size": "log_circ_mv_zscore", "per_cross_section": True},
    gate_status="research_candidate_pending_nav",
    metrics={
        "pit_top1500_neut_ic": top["neut_ic"], "pit_top1500_net_excess_cost": top["net_excess_cost"],
        "pit_top1500_neut_icir": top["neut_icir"], "pit_top1500_regime": top["regime_neut_ic"],
        "full_market_neut_ic": full["neut_ic"], "full_market_net_excess_cost": full["net_excess_cost"],
        "robust_drop_best2_pit_top1500": robust["folds"]["pit_top1500"]["mean_drop_best2"],
        "folds_positive": f"{robust['folds']['pit_top1500']['n_folds_positive']}/16",
        "formal_gate_line_net_excess": 0.05, "gate_pass": False,
        "v5_baseline_neut_ic": 0.034, "v6_on_v5_universe_neut_ic": 0.032,
        "train_seconds": 208},
    artifacts={"preds": str(preds.relative_to(REPO)), "preds_sha256": sha256_file(preds),
               "verdict_doc": "docs/plans/survivorship_p4_verdict.md",
               "verdict_json": "data/bakeoff/p4_verdict_v6.json"})

r = register(rec)
print(f"registered {r.model_id} gate={r.gate_status}")
print(f"  pit_top1500 neut_ic={r.metrics['pit_top1500_neut_ic']} net={r.metrics['pit_top1500_net_excess_cost']} "
      f"gate_pass={r.metrics['gate_pass']}")
