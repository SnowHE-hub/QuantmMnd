"""诊断 meta-learner 可用特征 + 训练 v3."""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── 1. 检查 realized_pnl ───────────────────────────────────────────────────

pnl = pd.read_parquet(ROOT / "data/feedback/realized_pnl.parquet")
pnl["as_of_date"] = pd.to_datetime(pnl["as_of_date"])
print(f"realized_pnl shape: {pnl.shape}")
print(f"有效样本: {pnl['actual_return_63d'].notna().sum()} 条")
print("\n季度分布:")
print(pnl["as_of_date"].dt.date.value_counts().sort_index())

# ─── 2. 检查 alpha_panel ────────────────────────────────────────────────────

panel_path = ROOT / "data/panel/alpha_panel_v4.parquet"
print(f"\nalpha_panel_v4 存在: {panel_path.exists()}")
if panel_path.exists():
    panel = pd.read_parquet(panel_path)
    print(f"panel shape: {panel.shape}")
    print(f"panel 索引: {panel.index.names}")
    print(f"panel 列（前30）: {panel.columns[:30].tolist()}")
    print(f"panel as_of 唯一值: {panel.index.get_level_values('as_of').unique()[:8].tolist() if 'as_of' in panel.index.names else '?'}")

# ─── 3. 运行 build_training_data 并诊断 IC ──────────────────────────────────

print("\n" + "=" * 60)
print("运行 build_training_data ...")
print("=" * 60)

try:
    from scripts.train_meta_learner import build_training_data, AGENT_COLS
    df = build_training_data(
        ROOT / "data/feedback/realized_pnl.parquet",
        ROOT / "data/panel/alpha_panel_v4.parquet",
    )
    print(f"\nbuild_training_data 结果: {df.shape}")
    print(f"列: {df.columns.tolist()}")
    print(f"unique quarters: {df['join_date'].nunique()}")

    # IC 诊断
    print("\n特征 vs actual_return_63d Spearman IC:")
    for col in AGENT_COLS:
        if col not in df.columns:
            print(f"  {col:20s} 列不存在")
            continue
        sub = df[[col, "actual_return_63d", "hit"]].dropna()
        if len(sub) < 10:
            print(f"  {col:20s} 样本不足 (n={len(sub)})")
            continue
        ic, p = spearmanr(sub[col], sub["actual_return_63d"])
        ic_hit, p_hit = spearmanr(sub[col], sub["hit"].astype(int))
        sig = "✅" if (abs(ic) > 0.05 and p < 0.1) else "⚠️"
        print(f"  {col:15s} IC={ic:+.3f} p={p:.3f} | hit_IC={ic_hit:+.3f} p={p_hit:.3f} n={len(sub):3d} {sig}")

except Exception as e:
    print(f"build_training_data 失败: {e}")
    import traceback
    traceback.print_exc()
