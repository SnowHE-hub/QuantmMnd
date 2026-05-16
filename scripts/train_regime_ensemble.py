"""scripts/train_regime_ensemble.py — 训练 Regime 感知集成模型.

策略：
1. 计算每个因子在大盘/小盘 regime 下的条件 IC
2. 为大盘期、小盘期各选一套最优因子集
3. 训练两个专用 LightGBM 子模型
4. 用 regime_small_prob 做软加权集成
5. 输出: models/lgbm_ensemble_regime.pkl + 评估报告

集成预测规则（推理时）:
  prediction = (1 - regime_small_prob) * pred_large + regime_small_prob * pred_small
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quantmind.core.logger import get_logger, setup_logger

log = get_logger(__name__)

PANEL_IN    = PROJECT_ROOT / "data" / "panel" / "alpha_panel_v3_regime.parquet"
REGIME_IN   = PROJECT_ROOT / "data" / "features" / "regime_features.parquet"
OUT_DIR     = PROJECT_ROOT / "models"
REPORT_DIR  = PROJECT_ROOT / "reports" / "wf_regime_ensemble"

LABEL            = "forward_return_63d"
MIN_IC_ABS_LARGE = 0.017   # 大盘子模型阈值（更严格，控制在 ~30 特征，与 v2 相近）
MIN_IC_ABS_SMALL = 0.015   # 小盘子模型阈值（略宽松，捕捉更多小盘信号）
MIN_IC_ABS       = MIN_IC_ABS_SMALL  # 保留向后兼容


# ──────────────────────────────────────────────────────────────────────────────
# 1. 计算 regime 条件 IC
# ──────────────────────────────────────────────────────────────────────────────

def compute_conditional_ic(panel: pd.DataFrame, label: str) -> pd.DataFrame:
    """计算每个因子在大盘/小盘期的平均截面 IC."""
    import warnings
    warnings.filterwarnings("ignore")

    skip = {label, "forward_return_21d", "forward_return_63d",
            "industry", "exposure_industry", "exposure_area",  # 分类列，非数值型
            "regime_label", "regime_small_prob",
            "csi500_csi300_63d", "csi300_20d_vol", "breadth_20d",
            "csi300_63d_return", "chiext_csi300_63d", "small_large_63d_spread"}
    factor_cols = [c for c in panel.columns if c not in skip]

    large_grps = {d: g for d, g in panel[panel["regime_label"] == 0].groupby(level="as_of")}
    small_grps = {d: g for d, g in panel[panel["regime_label"] == 1].groupby(level="as_of")}

    rows = []
    for feat in factor_cols:
        def mean_ic(grps):
            ics = []
            for g in grps.values():
                sub = g[[feat, label]].dropna()
                if len(sub) > 20:
                    rho, _ = stats.spearmanr(sub[feat].values, sub[label].values)
                    if not np.isnan(rho):
                        ics.append(rho)
            return float(np.mean(ics)) if ics else float("nan")

        ic_l = mean_ic(large_grps)
        ic_s = mean_ic(small_grps)
        rows.append({"factor": feat, "ic_large": ic_l, "ic_small": ic_s})

    df = pd.DataFrame(rows).set_index("factor").dropna()
    df["ic_diff"] = df["ic_small"] - df["ic_large"]
    df["flipped"] = (df["ic_large"] * df["ic_small"]) < 0
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. 构建 regime 专属特征集
# ──────────────────────────────────────────────────────────────────────────────

def select_regime_factors(
    ic_df: pd.DataFrame, regime: str, threshold: float = MIN_IC_ABS
) -> list[str]:
    """选取在指定 regime 下 IC 绝对值 >= threshold 且方向一致的因子."""
    col = "ic_large" if regime == "large" else "ic_small"
    selected = ic_df[ic_df[col].abs() >= threshold][col]
    return selected.sort_values(key=abs, ascending=False).index.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Walk-Forward 子模型训练
# ──────────────────────────────────────────────────────────────────────────────

def train_submodel(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    min_train: int,
    n_val: int,
    model_kwargs: dict[str, Any],
    name: str,
) -> tuple[Any, Any]:
    from quantmind.models.lgbm_ranker import LGBMRankerModel, walk_forward_evaluate
    result, model = walk_forward_evaluate(
        panel,
        feature_cols=[f for f in feature_cols if f in panel.columns],
        label_col=label,
        min_train_periods=min_train,
        n_val_periods=n_val,
        model_kwargs=model_kwargs,
        auto_flip=True,
        verbose=True,
    )
    print(f"[{name}] IC={result.effective_ic_mean:+.4f}  IR={result.effective_ic_ir:+.4f}"
          f"  WinRate={result.ic_win_rate:.1%}  Q5-Q1={result.avg_spread*100:+.3f}%"
          f"  direction={result.direction:+d}")
    return result, model


# ──────────────────────────────────────────────────────────────────────────────
# 4. Regime Ensemble 推理（写入元数据供 Agent 调用）
# ──────────────────────────────────────────────────────────────────────────────

def build_ensemble_meta(
    model_large_path: Path,
    model_small_path: Path,
    feat_large: list[str],
    feat_small: list[str],
    result_large: Any,
    result_small: Any,
    ic_df: pd.DataFrame,
) -> dict:
    """构建集成模型元数据，供推理时使用."""
    return {
        "ensemble_type": "regime_weighted",
        "blend_formula": "pred = (1-regime_small_prob)*pred_large + regime_small_prob*pred_small",
        "model_large": str(model_large_path),
        "model_small": str(model_small_path),
        "feature_cols_large": feat_large,
        "feature_cols_small": feat_small,
        "direction_large": result_large.direction,
        "direction_small": result_small.direction,
        "regime_features_required": [
            "regime_label", "regime_small_prob",
            "csi500_csi300_63d", "csi300_20d_vol", "breadth_20d",
        ],
        "submodel_stats": {
            "large": {
                "n_folds": result_large.n_folds,
                "ic_ir": round(result_large.effective_ic_ir, 4),
                "ic_mean": round(result_large.effective_ic_mean, 4),
                "win_rate": round(result_large.ic_win_rate, 4),
            },
            "small": {
                "n_folds": result_small.n_folds,
                "ic_ir": round(result_small.effective_ic_ir, 4),
                "ic_mean": round(result_small.effective_ic_mean, 4),
                "win_rate": round(result_small.ic_win_rate, 4),
            },
        },
        "top_flipped_factors": ic_df[ic_df["flipped"]].sort_values(
            "ic_diff", ascending=False
        )["ic_diff"].head(10).to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logger()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("QuantMind — Regime Ensemble 训练")
    print("=" * 65)

    panel = pd.read_parquet(PANEL_IN)
    print(f"面板: {panel.shape}  (大盘={( panel['regime_label']==0).sum()}行 / 小盘={(panel['regime_label']==1).sum()}行)")

    # ── 条件 IC ────────────────────────────────────────────────────────────────
    print("\n[1/4] 计算 regime 条件 IC …")
    ic_df = compute_conditional_ic(panel, LABEL)
    print(f"  共 {len(ic_df)} 个因子，其中 {ic_df['flipped'].sum()} 个方向反转")

    # ── 因子选择 ────────────────────────────────────────────────────────────────
    print("\n[2/4] 选择 regime 专属特征集 …")
    feat_large = select_regime_factors(ic_df, "large", threshold=MIN_IC_ABS_LARGE)
    feat_small = select_regime_factors(ic_df, "small", threshold=MIN_IC_ABS_SMALL)
    print(f"  大盘期因子: {len(feat_large)} 个")
    for f in feat_large[:8]:
        print(f"    {f:<40} IC_large={ic_df.loc[f,'ic_large']:+.4f}")
    print(f"  小盘期因子: {len(feat_small)} 个")
    for f in feat_small[:8]:
        print(f"    {f:<40} IC_small={ic_df.loc[f,'ic_small']:+.4f}")

    # ── 训练子模型 ─────────────────────────────────────────────────────────────
    model_kwargs = dict(n_estimators=300, num_leaves=15, learning_rate=0.05,
                        early_stopping_rounds=50)

    print("\n[3/4] 训练子模型 …")

    # 大盘子模型（用全量面板，让模型见过两种 regime 的数据，但特征是大盘专属）
    print("\n  ── 大盘专属子模型 ──")
    result_large, model_large = train_submodel(
        panel, feat_large, LABEL, min_train=8, n_val=2, model_kwargs=model_kwargs, name="large"
    )

    print("\n  ── 小盘专属子模型 ──")
    result_small, model_small = train_submodel(
        panel, feat_small, LABEL, min_train=8, n_val=2, model_kwargs=model_kwargs, name="small"
    )

    # ── 保存模型 ────────────────────────────────────────────────────────────────
    print("\n[4/4] 保存模型和集成元数据 …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    path_large = OUT_DIR / "lgbm_ensemble_large.pkl"
    path_small = OUT_DIR / "lgbm_ensemble_small.pkl"

    model_large._feature_names = feat_large
    model_small._feature_names = feat_small
    model_large.save(path_large)
    model_small.save(path_small)
    print(f"  ✅ 大盘子模型 → {path_large}")
    print(f"  ✅ 小盘子模型 → {path_small}")

    meta = build_ensemble_meta(
        path_large, path_small, feat_large, feat_small,
        result_large, result_small, ic_df,
    )
    meta_path = OUT_DIR / "lgbm_ensemble_regime.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  ✅ 集成元数据 → {meta_path}")

    # ── IC 报告 ──────────────────────────────────────────────────────────────
    ic_csv = REPORT_DIR / "conditional_ic.csv"
    ic_df.reset_index().to_csv(ic_csv, index=False)
    print(f"  ✅ 条件 IC 报告 → {ic_csv}")

    print("\n" + "=" * 65)
    print("🎉 Regime Ensemble 训练完成")
    print("=" * 65)
    print(f"  大盘子模型  IC IR={result_large.effective_ic_ir:+.3f}  WinRate={result_large.ic_win_rate:.1%}")
    print(f"  小盘子模型  IC IR={result_small.effective_ic_ir:+.3f}  WinRate={result_small.ic_win_rate:.1%}")
    print(f"  推理时按 regime_small_prob 软加权两子模型的排名分数")


if __name__ == "__main__":
    main()
