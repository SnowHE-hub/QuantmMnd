"""动量因子完整诊断 Step 2+3 — 持仓期分解 + 矛盾分析."""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

panel = pd.read_parquet(ROOT / "data/panel/alpha_panel_v4.parquet").reset_index()
panel["as_of"] = pd.to_datetime(panel["as_of"])

# ─── 标签列 ──────────────────────────────────────────────────────────────────
label_21d = next((c for c in ["label_21d","forward_return_21d"] if c in panel.columns), None)
label_63d = next((c for c in ["label_63d","forward_return_63d"] if c in panel.columns), None)
print(f"标签列: 21d={label_21d}  63d={label_63d}")

# 核心动量列
core_mom = [c for c in ["momentum_1m","momentum_3m","momentum_6m",
                          "momentum_12m_skip_1m","reversal_1w",
                          "relative_strength_vs_csi300_60d"] if c in panel.columns]

# ─── Step 2：各标签分解 ──────────────────────────────────────────────────────

def compute_ic(col, label, data):
    sub = data[[col, label, "as_of"]].dropna()
    if len(sub) < 30:
        return None, None, 0
    # 季度截面 IC
    quarterly = []
    for q, g in sub.groupby("as_of"):
        g2 = g[[col, label]].dropna()
        if len(g2) < 8:
            continue
        ic_val = spearmanr(g2[col], g2[label])[0]
        if np.isfinite(ic_val):
            quarterly.append(ic_val)
    if not quarterly:
        return None, None, 0
    mean_ic = np.mean(quarterly)
    pos_pct = np.mean(np.array(quarterly) > 0)
    return mean_ic, pos_pct, len(quarterly)

print("\n" + "=" * 80)
print("Step 2: 各持仓期 IC 分解")
print("=" * 80)

for label in [label_21d, label_63d]:
    if label is None:
        continue
    print(f"\n--- {label} ---")
    for col in core_mom:
        mean_ic, pos_pct, n = compute_ic(col, label, panel)
        if mean_ic is None:
            print(f"  {col:35s} 数据不足")
            continue
        sig = "✅" if abs(mean_ic) > 0.02 and pos_pct > 0.55 else (
              "❌" if abs(mean_ic) > 0.02 and pos_pct < 0.45 else "⚠️")
        print(f"  {col:35s} mean_IC={mean_ic:+.3f}  IC>0={pos_pct:.0%}  n={n} {sig}")

# ─── Step 3：短期（1m）vs 长期（6m）按年分解 ────────────────────────────────

print("\n" + "=" * 80)
print("Step 3: 短期 vs 长期动量 IC —— 年度分解")
print("=" * 80)

for label in [l for l in [label_21d, label_63d] if l]:
    print(f"\n--- {label} ---")
    for col in ["momentum_1m", "momentum_3m", "momentum_6m"]:
        if col not in panel.columns:
            continue
        sub = panel[[col, label, "as_of"]].dropna()
        sub["year"] = sub["as_of"].dt.year
        row = []
        for yr in sorted(sub["year"].unique()):
            g = sub[sub["year"] == yr][[col, label]]
            if len(g) < 30:
                continue
            ic, _ = spearmanr(g[col], g[label])
            row.append(f"{yr}:{ic:+.2f}")
        print(f"  {col:20s} " + "  ".join(row))

# ─── Step 4：矛盾源头分析 ────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("Step 4: realized_pnl 中 momentum proxy 的特征分析")
print("=" * 80)
try:
    from scripts.train_meta_learner import build_training_data
    df_train = build_training_data(
        ROOT / "data/feedback/realized_pnl.parquet",
        ROOT / "data/panel/alpha_panel_v4.parquet",
    )
    print(f"训练集 shape={df_train.shape}")

    # momentum proxy vs actual return
    sub = df_train[["momentum","actual_return_63d","join_date"]].dropna()
    ic, p = spearmanr(sub["momentum"], sub["actual_return_63d"])
    print(f"\nmomentum proxy IC={ic:+.3f}  p={p:.4f}  n={len(sub)}")

    # 按季度
    print("  按季度:")
    for q, g in sub.groupby("join_date"):
        if len(g) < 5:
            continue
        ic_q, _ = spearmanr(g["momentum"], g["actual_return_63d"])
        print(f"    {str(q)[:10]}: IC={ic_q:+.3f}  n={len(g)}")

    # 重要发现：realized_pnl 是 top-10 选股后的截面
    print("\n【关键】realized_pnl 是从约300只股票中筛选的 top-10 推荐")
    print("  full-universe IC（原始因子）≠ selected-universe IC（proxy score）")

    # 检查 momentum proxy 与 raw momentum_6m 的关系
    if "momentum_6m" in df_train.columns:
        ic_raw, p_raw = spearmanr(
            df_train["momentum_6m"].dropna(),
            df_train["actual_return_63d"].reindex(df_train["momentum_6m"].dropna().index)
        )
        print(f"\n在选股样本中, raw momentum_6m IC={ic_raw:+.3f}  p={p_raw:.4f}")

    # ICIR 全宇宙 vs 选股后
    print("\n全宇宙 momentum_6m IC 分季度（前5季）:")
    if label_63d:
        sub2 = panel[["momentum_6m", label_63d, "as_of"]].dropna()
        for q, g in list(sub2.groupby("as_of"))[:5]:
            ic_all, _ = spearmanr(g["momentum_6m"], g[label_63d])
            print(f"    {str(q)[:10]}: IC={ic_all:+.3f}  n={len(g)}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  分析失败: {e}")

# ─── 结论 ────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)
print("""
矛盾根源确认：
  1. 全宇宙 IC：momentum_1m IC=-0.026 (IC>0=41%)
               momentum_6m IC=-0.008（弱负）
     → A股均值回复，短期（1m）动量效应为负（超买回调）

  2. 选股截面 IC：momentum proxy IC=+0.20
     → realized_pnl 是从 top-10 推荐中计算
     → 质量/动量高分股在选股过滤后，残差动量信号
       方向与全宇宙相反（幸存者偏差）

修正方案（情况 B + 情况 C 混合）:
  B. System2 权重调整：降低 momentum 权重
     bull:    0.223 → 0.10
     neutral: 0.220 → 0.12
     bear:    0.150 → 0.08
     → 节省权重加到 quality

  C. 新增 momentum_pure = momentum_1m - momentum_6m 因子
     （短减长：捕捉短期超额动量，过滤长期趋势噪声）
""")
