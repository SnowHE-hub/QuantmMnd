from quantmind.contracts import list_models, HorizonRegistry

print("=== HorizonRegistry ===")
for h in HorizonRegistry.all():
    print("  {:7} days={} label={} product_line={} rebal_step={}".format(
        h.name, h.days, h.label_col, h.is_product_line, h.rebalance_step))

print("=== ModelRegistry ===")
for r in list_models():
    m = r.metrics
    print("  {}  horizon={} featset={} data={}".format(
        r.model_id, r.horizon, r.feature_set, r.data_version))
    print("    gate_status={}".format(r.gate_status))
    print("    pit_top1500: neut_ic={} net_excess={} icir={}".format(
        m.get("pit_top1500_neut_ic"), m.get("pit_top1500_net_excess_cost"),
        m.get("pit_top1500_neut_icir")))
    print("    gate_pass={} (formal line {}) robust_drop2={} folds+={}".format(
        m.get("gate_pass"), m.get("formal_gate_line_net_excess"),
        m.get("robust_drop_best2_pit_top1500"), m.get("folds_positive")))
    print("    v5_baseline={} v6_on_v5_universe={}".format(
        m.get("v5_baseline_neut_ic"), m.get("v6_on_v5_universe_neut_ic")))
