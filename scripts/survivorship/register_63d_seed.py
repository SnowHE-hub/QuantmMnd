"""注册第二产品线种子 ridge_full_fnd_63d_v6_seed（gate=research_candidate_pending_nav）."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
import json
from quantmind.contracts.model_registry import ModelRecord, register
from scripts.bakeoff.run_manifest import sha256_file

preds = REPO / "data/bakeoff/preds/ridge_full_fnd_63d_quarterly_v6.parquet"
v = json.loads((REPO / "data/bakeoff/p63_3_verdict_v6.json").read_text())
B = next(r for r in v["leaderboard"] if r["row"] == "B")
hard = json.loads((REPO / "data/bakeoff/p63_3_hardening.json").read_text())["B"]

rec = ModelRecord(
    model_id="ridge_full_fnd_63d_v6_seed",
    label="forward_return_63d", horizon="long", feature_set="full_v6_63d_35_16sh_16fnd_158",
    data_version="v6",
    wf_config={"splitter": "PurgedWalkForwardSplit", "horizon": 63, "embargo": 63, "mode": "rolling",
               "rolling_lookback_td": 756, "n_val": 2, "refit": "semi-annual(Apr/Oct)",
               "oos_start": "2022-01-01", "decide_direction": "val_only", "alpha": 10.0, "n_folds": 8},
    neutralization={"industry_dummies": True, "unknown_bucket": True, "size": "log_circ_mv_zscore"},
    gate_status="research_candidate_pending_nav",
    metrics={
        "pit_top1500_neut_ic": B["neut_ic"], "pit_top1500_net_excess_cost": B["net_excess_cost"],
        "pit_top1500_neut_icir": B["neut_icir"], "pit_top1500_regime": B["regime"],
        "per_fold_drop_best2": hard["fold"]["drop_best2"], "per_fold_pos": f"{hard['fold']['n_pos']}/7",
        "liquidity_buckets": hard["liquidity_bucket"], "diagB_v5universe_neut_ic": hard["diagB_v5universe_neut_ic"],
        "ablation_marginal_ic_vs_C": round(B["neut_ic"] - 0.0788, 4),
        "ablation_marginal_net_vs_C": round(B["net_excess_cost"] - (-0.0415), 4),
        "formal_gate_line_net_excess": 0.05, "gate_pass_research": True, "needs_executable_nav": True,
        "oos_trustworthy_factors": ["book_to_market", "dividend_yield_ttm", "earnings_yield",
                                    "ocf_to_revenue_ttm", "accruals"]},
    artifacts={"preds": str(preds.relative_to(REPO)), "preds_sha256": sha256_file(preds),
               "verdict_doc": "docs/plans/survivorship_p63_3_verdict.md",
               "verdict_json": "data/bakeoff/p63_3_verdict_v6.json",
               "hardening_json": "data/bakeoff/p63_3_hardening.json"})
r = register(rec)
print(f"registered {r.model_id} gate={r.gate_status} neut_ic={r.metrics['pit_top1500_neut_ic']} "
      f"net={r.metrics['pit_top1500_net_excess_cost']}")
