"""3c：核对 ModelRegistry 两条 seed 齐全 + gate 一致。"""
from quantmind.contracts import list_models

REQUIRED = {"ridge_full_12d_v6_seed", "ridge_full_fnd_63d_v6_seed"}
recs = {r.model_id: r for r in list_models()}
print("registry model_ids:", sorted(recs))
missing = REQUIRED - set(recs)
print("missing required:", missing or "无 ✅")
for mid in sorted(REQUIRED):
    if mid not in recs:
        continue
    r = recs[mid]
    fields_ok = all([r.label, r.horizon, r.feature_set, r.data_version, r.wf_config, r.neutralization, r.metrics])
    gp = r.metrics.get("gate_pass", r.metrics.get("gate_pass_research"))
    print(f"\n{mid}:")
    print(f"  horizon={r.horizon} featset={r.feature_set} data={r.data_version} gate_status={r.gate_status}")
    print(f"  net_excess={r.metrics.get('pit_top1500_net_excess_cost')} neut_ic={r.metrics.get('pit_top1500_neut_ic')}")
    print(f"  needs_executable_nav={r.metrics.get('needs_executable_nav')} gate_pass(formal)=False(=pending_nav)")
    print(f"  字段齐全={'✅' if fields_ok else '❌'}  gate_status==pending_nav={'✅' if r.gate_status=='research_candidate_pending_nav' else '❌'}")
