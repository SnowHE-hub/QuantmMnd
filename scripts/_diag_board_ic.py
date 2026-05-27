"""诊断各板块因子 IC 方向，与全量对比。"""
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr

panel = pd.read_parquet("data/panel/alpha_panel_v4.parquet")
label_col = "forward_return_63d"

def get_board(t):
    code = t.split(".")[0]
    if code.startswith("688"): return "STAR"
    if code.startswith("300"): return "GEM"
    return "MAIN"

panel["board"] = panel.index.get_level_values("ticker").map(
    {t: get_board(t) for t in panel.index.get_level_values("ticker").unique()}
)

# 加载混训模型的特征列表
m = pickle.load(open("models/lgbm_v6_alpha.pkl", "rb"))
feats = m._feature_names or []
key_features = [f for f in feats if f in panel.columns][:15]
print(f"检查 {len(key_features)} 个特征（lgbm_v6_alpha._feature_names）")
print(f"特征: {key_features}\n")

print(f"{'因子':30s} {'全量':>8} {'MAIN':>8} {'GEM':>8} {'STAR':>8} {'方向冲突'}")
print("-" * 80)

for feat in key_features:
    ics = {"全量": [], "MAIN": [], "GEM": [], "STAR": []}
    for as_of, g in panel.groupby("as_of"):
        sub = g[[feat, label_col, "board"]].dropna(subset=[feat, label_col])
        if len(sub) < 20:
            continue
        ic = spearmanr(sub[feat], sub[label_col])[0]
        if not np.isnan(ic):
            ics["全量"].append(ic)
        for board in ["MAIN", "GEM", "STAR"]:
            b_sub = sub[sub["board"] == board]
            if len(b_sub) >= 8:
                b_ic = spearmanr(b_sub[feat], b_sub[label_col])[0]
                if not np.isnan(b_ic):
                    ics[board].append(b_ic)

    means = {k: np.mean(v) if v else float("nan") for k, v in ics.items()}
    base_sign = np.sign(means["全量"]) if not np.isnan(means["全量"]) else 0
    conflicts = []
    for b in ["MAIN", "GEM", "STAR"]:
        bv = means[b]
        if not np.isnan(bv) and base_sign != 0 and np.sign(bv) != base_sign:
            conflicts.append(b)

    row = f"{feat:30s}"
    for k in ["全量", "MAIN", "GEM", "STAR"]:
        v = means[k]
        row += f" {v:>+8.3f}" if not np.isnan(v) else f" {'N/A':>8}"
    row += f"  {', '.join(conflicts) if conflicts else ''}"
    print(row)

# 统计整体 IC（LightGBM 会根据全量 walk-forward IC 决定方向）
print("\n\n=== LGBM walk-forward IC 模拟（全量 vs 板块子集）===")
print("这决定 auto_flip 是否触发\n")

# 估算 walk-forward IC（简化：用最后 N 个截面的 IC 均值）
dates_sorted = sorted(panel.index.get_level_values("as_of").unique())
test_dates = dates_sorted[-10:]  # 最近 10 个截面

for board_filter in [None, "MAIN", "GEM", "STAR"]:
    ics_wf = []
    for dt in test_dates:
        g = panel.loc[panel.index.get_level_values("as_of") == dt]
        if board_filter:
            g = g[g["board"] == board_filter]
        sub = g[key_features + [label_col]].dropna(subset=[label_col])
        if len(sub) < 10:
            continue
        # 综合打分（用均值代理模型预测）
        for feat in key_features[:5]:
            if feat in sub.columns:
                ic = spearmanr(sub[feat].fillna(0), sub[label_col])[0]
                if not np.isnan(ic):
                    ics_wf.append(ic)
                break
    label = board_filter or "全量"
    mean_ic = np.mean(ics_wf) if ics_wf else float("nan")
    direction = -1 if mean_ic < 0 else 1
    print(f"  {label:8s}: mean IC={mean_ic:+.4f}  → direction={direction:+d}")
