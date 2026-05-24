"""动量因子完整诊断脚本 — Steps 1+2."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

panel = pd.read_parquet(ROOT / "data/panel/alpha_panel_v4.parquet")
panel = panel.reset_index()  # as_of, ticker → columns
panel["as_of"] = pd.to_datetime(panel["as_of"])

print("Panel shape:", panel.shape)
print("Columns sample:", panel.columns[:10].tolist())
print("as_of range:", panel["as_of"].min().date(), "→", panel["as_of"].max().date())

# ─── Step 1：找出所有动量相关列 ──────────────────────────────────────────────

mom_keywords = ["momentum", "reversal", "return", "ret", "strength", "relative"]
mom_cols = [c for c in panel.columns
            if any(k in c.lower() for k in mom_keywords)
            and c not in ("ticker", "as_of")]
print(f"\n动量相关列 ({len(mom_cols)}):", mom_cols)

# ─── Step 1：各动量因子季度平均 IC vs label_63d ──────────────────────────────

label_col = "label_63d" if "label_63d" in panel.columns else None
if label_col is None:
    # 尝试其他命名
    for c in panel.columns:
        if "label" in c or "forward" in c:
            label_col = c
            break

print(f"\n使用标签列: {label_col}")
print("\n" + "=" * 75)
print(f"各动量因子 季度截面 IC (vs {label_col}):")
print("=" * 75)

ic_results = {}
for col in mom_cols:
    if col not in panel.columns or label_col not in panel.columns:
        continue
    sub = panel[[col, label_col, "as_of"]].dropna()
    if len(sub) < 50:
        continue

    def _quarter_ic(g):
        data = g[[col, label_col]].dropna()
        if len(data) < 10:
            return np.nan
        return spearmanr(data[col], data[label_col])[0]

    quarterly_ic = sub.groupby("as_of").apply(_quarter_ic).dropna()
    mean_ic = quarterly_ic.mean()
    pos_pct = (quarterly_ic > 0).mean()
    ic_results[col] = {"mean_ic": mean_ic, "pos_pct": pos_pct, "n_quarters": len(quarterly_ic)}

    sig = "✅" if abs(mean_ic) > 0.02 and pos_pct > 0.55 else (
          "❌" if abs(mean_ic) > 0.02 and pos_pct < 0.45 else "⚠️")
    print(f"  {col:35s} mean_IC={mean_ic:+.3f}  IC>0={pos_pct:.0%}  n={len(quarterly_ic)} {sig}")

# ─── Step 2：按持仓期分解 IC ─────────────────────────────────────────────────

print("\n" + "=" * 75)
print("按持仓期分解：label_21d vs label_63d")
print("=" * 75)

core_mom = [c for c in ["momentum_1m", "momentum_3m", "momentum_6m",
                          "momentum_12m_skip_1m", "reversal_1w",
                          "relative_strength_vs_csi300_60d",
                          "momentum_12m"] if c in panel.columns]
print(f"核心动量列: {core_mom}")

for label in ["label_21d", "label_63d"]:
    if label not in panel.columns:
        print(f"\n  {label} 不存在")
        continue
    print(f"\n=== 标签: {label} ===")
    for col in core_mom:
        sub = panel[[col, label]].dropna()
        if len(sub) < 50:
            print(f"  {col:35s} 样本不足 n={len(sub)}")
            continue
        ic, p = spearmanr(sub[col], sub[label])
        sig = "✅" if abs(ic) > 0.02 and p < 0.05 else "⚠️"
        print(f"  {col:35s} IC={ic:+.3f}  p={p:.4f}  n={len(sub):5d}  {sig}")

# ─── Step 3：理解矛盾 ─────────────────────────────────────────────────────────

print("\n" + "=" * 75)
print("矛盾来源分析：regime IC（原始因子）vs realized_pnl IC（agent proxy）")
print("=" * 75)
print("""
  regime IC（dynamic_weights 文档）使用：
    原始 alpha_panel 因子 → 季度截面 IC vs label_63d
    bull: momentum IC = -0.033（持续为负）

  realized_pnl IC（meta-learner 诊断）使用：
    momentum_proxy_score = zscore(
      0.40×zscore(momentum_6m)
      + 0.35×zscore(momentum_3m)
      + 0.25×zscore(relative_strength_vs_csi300_60d)
    )
    IC = +0.20（正向）

  矛盾解释候选：
  A. 原始因子 IC 为负，但 proxy 经 z-score 后样本分布变化，
     同时 realized_pnl 是选股后（top-10）的截面，非全宇宙
  B. momentum_6m/3m 原始 IC 为负（均值回复），
     但 momentum_1m 短期 IC 为正（短期持续性）
  C. proxy score 里 relative_strength_vs_csi300_60d 是超额动量，
     与原始绝对动量方向不同
""")

# ─── Step 4：检验短期 vs 长期动量分化 ────────────────────────────────────────

print("=" * 75)
print("短期（1m）vs 长期（6m/12m）动量 IC 分化:")
print("=" * 75)
for col in ["momentum_1m", "momentum_3m", "momentum_6m"]:
    if col not in panel.columns or label_col not in panel.columns:
        continue
    sub = panel[[col, label_col, "as_of"]].dropna()

    # 按年份分段
    sub["year"] = sub["as_of"].dt.year
    for yr, grp in sub.groupby("year"):
        if len(grp) < 30:
            continue
        ic, _ = spearmanr(grp[col], grp[label_col])
        bar = "▲" if ic > 0 else "▼"
        print(f"  {yr}  {col:20s}  IC={ic:+.3f}  {bar}")
    print()

# ─── Step 5：realized_pnl momentum 样本特征 ──────────────────────────────────

print("=" * 75)
print("realized_pnl 中的 momentum 代理分数特征（选股后截面）:")
print("=" * 75)

try:
    from scripts.train_meta_learner import build_training_data, AGENT_COLS, _compute_proxies_fallback
    df_train = build_training_data(
        ROOT / "data/feedback/realized_pnl.parquet",
        ROOT / "data/panel/alpha_panel_v4.parquet",
    )
    print(f"训练集 shape: {df_train.shape}")
    print(f"momentum proxy 均值: {df_train['momentum'].mean():.4f}")
    print(f"momentum proxy 与 actual_return_63d IC:")
    sub = df_train[["momentum", "actual_return_63d"]].dropna()
    ic, p = spearmanr(sub["momentum"], sub["actual_return_63d"])
    print(f"  全样本: IC={ic:+.3f}  p={p:.4f}  n={len(sub)}")
    # 按季度
    df_train["year"] = pd.to_datetime(df_train["join_date"]).dt.year
    for yr, grp in df_train.groupby("year"):
        sub2 = grp[["momentum", "actual_return_63d"]].dropna()
        if len(sub2) < 5:
            continue
        ic2, _ = spearmanr(sub2["momentum"], sub2["actual_return_63d"])
        print(f"  {yr}: IC={ic2:+.3f}  n={len(sub2)}")
except Exception as e:
    print(f"  无法加载训练数据: {e}")

print("\n=== 诊断完成 ===")
