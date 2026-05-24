"""验证动量因子修正效果 — 修正前后 IC 对比."""
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

label_63d = next((c for c in ["label_63d","forward_return_63d"] if c in panel.columns), None)
label_21d = next((c for c in ["label_21d","forward_return_21d"] if c in panel.columns), None)

def mean_quarterly_ic(col, label, data):
    sub = data[[col, label, "as_of"]].dropna()
    ics = []
    for _, g in sub.groupby("as_of"):
        g2 = g[[col, label]].dropna()
        if len(g2) < 8:
            continue
        ic = spearmanr(g2[col], g2[label])[0]
        if np.isfinite(ic):
            ics.append(ic)
    return np.mean(ics) if ics else np.nan, np.mean(np.array(ics) > 0) if ics else np.nan, len(ics)

# ─── 1. 验证 dynamic_weights.py 权重更新 ────────────────────────────────────

from quantmind.regime.dynamic_weights import DynamicWeightManager
mgr = DynamicWeightManager()

print("=" * 70)
print("验证 1: dynamic_weights.py 新权重")
print("=" * 70)
print(f"{'Regime':10s} {'value':8s} {'momentum':10s} {'quality':9s} {'technical':10s} {'sum':6s}")
print("-" * 70)
for regime in ["bull", "neutral", "bear"]:
    w = mgr.get_system2_weights(regime)
    s = sum(w.values())
    ok = "✅" if abs(s - 1.0) < 1e-6 else "❌"
    print(f"{regime:10s} {w['value']:.3f}    {w['momentum']:.3f}      {w['quality']:.3f}    {w['technical']:.3f}      {s:.3f} {ok}")

# ─── 2. 验证 momentum_pure 因子 IC ──────────────────────────────────────────

print("\n" + "=" * 70)
print("验证 2: momentum_pure 因子 IC（修正前 vs 修正后）")
print("=" * 70)

# 修正前：原始动量组合（旧代理分的 mock）
# 模拟旧 proxy: 0.40×momentum_6m + 0.35×momentum_3m + 0.25×rel_str_csi300_60d

def zscore(s):
    std = s.std()
    return (s - s.mean()) / std if std > 1e-9 else s * 0

# 旧 proxy: 按季度计算
old_proxy_ics_63d = []
new_proxy_ics_63d = []

for q, g in panel.groupby("as_of"):
    g = g.copy()
    if label_63d not in g.columns:
        continue
    avail = g[[label_63d]].dropna().index
    if len(avail) < 15:
        continue

    # 旧 proxy
    old_cols = {
        "momentum_6m": 0.40,
        "momentum_3m": 0.35,
        "relative_strength_vs_csi300_60d": 0.25,
    }
    old_score = None
    for col, w in old_cols.items():
        if col not in g.columns:
            continue
        fz = zscore(g[col].fillna(g[col].median()))
        old_score = fz * w if old_score is None else old_score + fz * w
    if old_score is not None:
        old_score = zscore(old_score)

    # 新 proxy
    new_cols = {
        "momentum_12m_skip_1m":  0.50,
        "momentum_6m":            0.20,
        "momentum_1m":           -0.30,  # 负权：短期反转
    }
    new_score = None
    total_w = 0.0
    for col, w in new_cols.items():
        if col not in g.columns:
            continue
        fz = zscore(g[col].fillna(g[col].median()))
        new_score = fz * w if new_score is None else new_score + fz * w
        total_w += abs(w)
    if new_score is not None and total_w > 0:
        new_score = zscore(new_score / total_w)

    # 计算 IC
    sub_label = g[label_63d].dropna()
    if old_score is not None:
        sub = pd.concat([old_score, sub_label], axis=1).dropna()
        if len(sub) >= 8:
            ic_old = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])[0]
            if np.isfinite(ic_old):
                old_proxy_ics_63d.append(ic_old)

    if new_score is not None:
        sub = pd.concat([new_score, sub_label], axis=1).dropna()
        if len(sub) >= 8:
            ic_new = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])[0]
            if np.isfinite(ic_new):
                new_proxy_ics_63d.append(ic_new)

print(f"\n动量 proxy 修正前后对比（vs {label_63d}）:")
print(f"  旧 proxy (6m×0.40 + 3m×0.35 + rel_str×0.25):")
print(f"    mean IC = {np.mean(old_proxy_ics_63d):+.4f}  "
      f"IC>0 = {np.mean(np.array(old_proxy_ics_63d)>0):.0%}  "
      f"n = {len(old_proxy_ics_63d)}")
print(f"  新 proxy (12m×0.50 + 6m×0.20 - 1m×0.30):")
print(f"    mean IC = {np.mean(new_proxy_ics_63d):+.4f}  "
      f"IC>0 = {np.mean(np.array(new_proxy_ics_63d)>0):.0%}  "
      f"n = {len(new_proxy_ics_63d)}")

delta_ic = np.mean(new_proxy_ics_63d) - np.mean(old_proxy_ics_63d)
print(f"\n  IC 改善: {delta_ic:+.4f} {'✅ 改善' if delta_ic > 0 else '⚠️ 未改善'}")

# ─── 3. 验证 momentum_pure 表达式 ───────────────────────────────────────────

print("\n" + "=" * 70)
print("验证 3: momentum_pure 因子 IC")
print("=" * 70)

if "momentum_1m" in panel.columns and "momentum_12m_skip_1m" in panel.columns:
    # 计算 momentum_pure = 12m_skip_1m - 1m
    panel["momentum_pure_approx"] = panel["momentum_12m_skip_1m"] - panel["momentum_1m"]

    for label in [label_21d, label_63d]:
        if label is None:
            continue
        ic, pos_pct, n = mean_quarterly_ic("momentum_pure_approx", label, panel)
        sig = "✅" if (np.isfinite(ic) and ic > 0.01) else "⚠️"
        print(f"  momentum_pure vs {label}: mean_IC={ic:+.3f}  IC>0={pos_pct:.0%}  n={n} {sig}")

# ─── 4. 验证 AGENT_PROXY_CONFIG 更新 ────────────────────────────────────────

print("\n" + "=" * 70)
print("验证 4: meta_learner AGENT_PROXY_CONFIG 新动量配置")
print("=" * 70)

from quantmind.models.meta_learner import AGENT_PROXY_CONFIG
print("momentum proxy 新配置:")
for factor, weight in AGENT_PROXY_CONFIG["momentum"]:
    direction = "正（越高越好）" if weight > 0 else "负（反转：低越好）"
    print(f"  {factor:30s} w={weight:+.2f}  {direction}")

# ─── 5. 打印完整对比表 ───────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("修正前后对比总表")
print("=" * 70)
rows = [
    ("momentum_1m (21d)",  "IC=-0.026 ❌", "proxy weight -0.30 (反转修正)"),
    ("momentum_3m (63d)",  "IC=-0.017 ⚠️", "已从旧 proxy 中移除"),
    ("momentum_6m (63d)",  "IC=+0.007 ⚠️", "proxy weight +0.20（降权）"),
    ("momentum_12m_skip (63d)", "IC=+0.025 ✅", "proxy weight +0.50（新增）"),
    ("rel_str_csi300_60d (63d)", "IC=-0.029 ❌", "已从 proxy 中移除"),
    ("momentum_pure_approx", "新 IC（见上）", "新增 expr factor"),
]
print(f"  {'因子':35s} {'诊断 IC':25s} {'修正动作':35s}")
print("  " + "-" * 95)
for factor, diag, action in rows:
    print(f"  {factor:35s} {diag:25s} {action}")

print("\n权重调整汇总:")
print(f"  {'Regime':10s} {'momentum (旧→新)':20s} {'quality (旧→新)':20s}")
print("  " + "-" * 50)
print(f"  {'bull':10s} {'0.223 → 0.100':20s} {'0.333 → 0.456':20s}")
print(f"  {'neutral':10s} {'0.220 → 0.120':20s} {'0.300 → 0.400':20s}")
print(f"  {'bear':10s} {'0.150 → 0.080':20s} {'0.280 → 0.350':20s}")
print("\n=== 验证完成 ===")
