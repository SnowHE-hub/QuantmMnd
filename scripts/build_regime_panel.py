"""scripts/build_regime_panel.py — 将 Regime 指标和交互特征注入因子面板.

读取:
  data/panel/alpha_panel_v2.parquet
  data/features/regime_features.parquet

输出:
  data/panel/alpha_panel_v2_regime.parquet

新增列:
  1. regime_label             — 0=大盘占优 / 1=小盘占优（截面常数，每期相同）
  2. regime_small_prob        — 小盘溢价 soft signal，已归一化到 [0,1]
  3. csi500_csi300_63d        — CSI500 vs CSI300 63日超额（截面常数）
  4. csi300_20d_vol           — 市场已实现波动率（截面常数）
  5. breadth_20d              — 近20日正收益股票比例（截面常数）
  6. {factor}_x_regime        — 关键因子 × regime_label 交互项（约 10 个）
     在小盘占优期因子方向往往反转，交互项让模型自动学习这种反转
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PANEL_IN   = PROJECT_ROOT / "data" / "panel" / "alpha_panel_v3.parquet"
REGIME_IN  = PROJECT_ROOT / "data" / "features" / "regime_features.parquet"
PANEL_OUT  = PROJECT_ROOT / "data" / "panel" / "alpha_panel_v3_regime.parquet"

# 与 regime_label 做交互的关键因子（信号最强的前 10 个）
INTERACTION_FACTORS = [
    "turnover_3m_avg",
    "volatility_3m",
    "amplitude_quantile",
    "downside_volatility_3m",
    "margin_buy_intensity",
    "rsi_14",
    "bollinger_position",
    "momentum_1m",
    "reversal_1w",
    "pb",
    # 新增小盘因子交互项
    "price_to_52w_low",
    "turnover_acceleration",
    "volume_price_corr_20d",
    "relative_strength_vs_csi500_60d",
]

# 从 regime_features 里直接并入面板的截面常数列
REGIME_COLS = [
    "regime_label",
    "regime_small_prob",
    "csi500_csi300_63d",
    "csi300_20d_vol",
    "breadth_20d",
    "csi300_63d_return",
]


def main() -> None:
    print("=== 构建 Regime 增强面板 ===")

    panel = pd.read_parquet(PANEL_IN)
    regime = pd.read_parquet(REGIME_IN)
    regime.index = pd.to_datetime(regime.index)
    print(f"面板形状: {panel.shape}")
    print(f"Regime 指标: {regime.shape}")

    # ── 将 as_of index 对齐 ──────────────────────────────────────────────────
    as_of_ts = panel.index.get_level_values("as_of")
    panel = panel.copy()

    # 按 as_of 合并 regime 特征（每行相同，截面常数）
    regime_selected = regime[REGIME_COLS]
    panel_reset = panel.reset_index()
    panel_reset["as_of_ts"] = pd.to_datetime(panel_reset["as_of"])

    regime_reset = regime_selected.reset_index()
    regime_reset.rename(columns={"as_of": "as_of_ts"}, inplace=True)
    regime_reset["as_of_ts"] = pd.to_datetime(regime_reset["as_of_ts"])

    merged = panel_reset.merge(regime_reset, on="as_of_ts", how="left")
    merged = merged.drop(columns=["as_of_ts"]).set_index(["as_of", "ticker"])

    # ── 构建交互特征 ─────────────────────────────────────────────────────────
    print("\n构建交互特征:")
    for factor in INTERACTION_FACTORS:
        if factor not in merged.columns:
            print(f"  SKIP {factor} (列不存在)")
            continue
        col_name = f"{factor}_x_regime"
        merged[col_name] = merged[factor] * merged["regime_label"]
        print(f"  + {col_name}")

    print(f"\n最终面板形状: {merged.shape}")
    print(f"新增列数: {len(REGIME_COLS) + len([f for f in INTERACTION_FACTORS if f in panel.columns])}")

    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(PANEL_OUT)
    print(f"\n✅ 保存 → {PANEL_OUT}")

    # ── 验证 regime 分布 ──────────────────────────────────────────────────────
    r = merged["regime_label"].groupby(level="as_of").first()
    r.index = r.index.astype(str)
    print("\nRegime 分布（每期）:")
    for d, v in r.items():
        tag = "🔴 小盘" if v == 1 else "🔵 大盘"
        print(f"  {d}  {tag}")
    print(f"\n大盘占优: {(r==0).sum()} 期  |  小盘占优: {(r==1).sum()} 期")


if __name__ == "__main__":
    main()
