"""scripts/train_meta_learner.py

重训 MetaLearner v2：用因子分（stock_returns.parquet）预测 return_3m。

训练逻辑
--------
1. 以 data/sim30d/stock_returns.parquet 为主数据源（包含因子分 + return_3m）
2. 过滤 --features 中实际存在的列（缺失列会 WARN 并跳过）
3. hist_win_rate / hist_sharpe 等稀疏列用横截面中位数填充
4. 使用 StandardScaler + Ridge 作为基线模型
5. 若 train R² < 0.20：展示特征重要度，剔除最弱特征，重训一次
6. 保存模型到 --out，同时写 JSON 元数据

用法
----
python scripts/train_meta_learner.py \\
  --pnl  data/feedback/realized_pnl.parquet \\
  --out  data/meta_learner/meta_learner_v2.pkl \\
  --features composite_score value_score momentum_score quality_score \\
             technical_score lgbm_score hist_win_rate hist_sharpe \\
             log_market_cap pe_ttm pb
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
_STOCK_RETURNS = ROOT / "data" / "sim30d" / "stock_returns.parquet"
_V1_META       = ROOT / "data" / "meta_learner" / "meta_learner_meta.json"


# ─── 工具 ─────────────────────────────────────────────────────────────────────

def load_training_data(
    pnl_path: Path,
    requested_features: list[str],
) -> tuple[pd.DataFrame, list[str], str]:
    """加载并合并训练数据。

    优先从 stock_returns.parquet 拉取因子分和 return_3m；
    realized_pnl.parquet 的 actual_return_63d 作为补充目标（若有）。

    Returns
    -------
    df            : 清洗后的 DataFrame
    avail_features: 实际可用特征列列表
    target_col    : 目标列名
    """
    # ── 主数据源 ──
    if not _STOCK_RETURNS.exists():
        raise FileNotFoundError(f"主数据文件不存在：{_STOCK_RETURNS}")
    df = pd.read_parquet(_STOCK_RETURNS)
    logger.info(f"stock_returns 加载：{df.shape}  列={df.columns.tolist()}")

    target_col = "return_3m"
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在 stock_returns 中")

    # ── 可选：realized_pnl 合并（仅作日志参考，不作为主训练源）──
    if pnl_path.exists():
        pnl = pd.read_parquet(pnl_path)
        logger.info(f"realized_pnl 加载：{pnl.shape}  列={pnl.columns.tolist()}")
        # 目前 realized_pnl 不含因子分列，仅记录样本量
        logger.info(
            f"realized_pnl 样本量 {len(pnl)}（不含因子分，不并入训练集）"
        )
    else:
        logger.warning(f"realized_pnl 不存在，跳过：{pnl_path}")

    # ── 特征过滤 ──
    avail_features: list[str] = []
    for feat in requested_features:
        if feat in df.columns:
            avail_features.append(feat)
        else:
            logger.warning(f"特征列 '{feat}' 不在 stock_returns 中，跳过")

    if not avail_features:
        raise ValueError("没有可用特征列，请检查 --features 参数")

    logger.info(f"最终使用特征 ({len(avail_features)}): {avail_features}")
    return df, avail_features, target_col


def build_xy(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """构造特征矩阵 X 和标签向量 y。

    稀疏列（非空率 < 100%）用横截面中位数填充。
    仅保留 target 非空的行（stock_returns.return_3m 全非空）。
    """
    sub = df[features + [target]].copy()

    # 填充稀疏特征列
    for col in features:
        null_rate = sub[col].isna().mean()
        if null_rate > 0:
            median_val = sub[col].median()
            sub[col] = sub[col].fillna(median_val)
            logger.info(
                f"  {col}: {null_rate:.1%} 缺失 → 中位数填充 {median_val:.4f}"
            )

    # 丢弃 target 为空的行（理论上为 0）
    sub = sub.dropna(subset=[target])
    n_used = len(sub)
    logger.info(f"训练样本：{n_used}（target={target} 非空）")

    X = sub[features].values.astype(float)
    y = sub[target].values.astype(float)
    return X, y, features


def train_ridge(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    ridge_alpha: float = 1.0,
) -> tuple[Pipeline, float, float, dict[str, float]]:
    """训练 StandardScaler + Ridge 管道，返回 (pipeline, train_r2, cv_r2, coef_dict)."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=ridge_alpha)),
    ])
    pipe.fit(X, y)

    train_r2 = float(r2_score(y, pipe.predict(X)))

    # 5-fold CV（固定随机种子，结果可复现）
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    cv_r2 = float(cv_scores.mean())

    # 特征系数（标准化空间）
    coefs_raw = pipe.named_steps["ridge"].coef_
    coef_dict = {f: float(c) for f, c in zip(features, coefs_raw)}

    return pipe, train_r2, cv_r2, coef_dict


def drop_weakest_feature(
    coef_dict: dict[str, float],
    features: list[str],
) -> list[str]:
    """剔除绝对系数最小的特征，返回新特征列表。"""
    abs_coefs = {f: abs(c) for f, c in coef_dict.items() if f in features}
    weakest = min(abs_coefs, key=abs_coefs.get)
    logger.info(
        f"R² < 0.20：剔除最弱特征 '{weakest}' (|coef|={abs_coefs[weakest]:.4f})"
    )
    return [f for f in features if f != weakest]


def save_model(
    pipe: Pipeline,
    features: list[str],
    coef_dict: dict[str, float],
    train_r2: float,
    cv_r2: float,
    n_samples: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pipeline": pipe,
        "features": features,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    meta = {
        "version":   "v2",
        "fitted":    True,
        "n_samples": n_samples,
        "features":  features,
        "train_r2":  round(train_r2, 4),
        "cv_r2":     round(cv_r2, 4),
        "ridge_alpha": 1.0,
        "coefficients": {k: round(v, 6) for k, v in coef_dict.items()},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"模型已保存：{out_path}")
    logger.info(f"元数据已保存：{meta_path}")


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="重训 MetaLearner v2")
    p.add_argument(
        "--pnl",
        type=Path,
        default=ROOT / "data" / "feedback" / "realized_pnl.parquet",
        help="realized_pnl.parquet 路径（用于日志参考）",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "meta_learner" / "meta_learner_v2.pkl",
        help="输出模型路径",
    )
    p.add_argument(
        "--features",
        nargs="+",
        default=[
            "composite_score", "value_score", "momentum_score",
            "quality_score",   "technical_score", "lgbm_score",
            "hist_win_rate",   "hist_sharpe",
        ],
        help="候选特征列（不存在的列自动跳过）",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge 正则化强度（默认 1.0）",
    )
    args = p.parse_args(argv)

    logger.info("=" * 60)
    logger.info("MetaLearner v2 训练脚本启动")
    logger.info("=" * 60)

    # ── 读取 v1 基线 ──
    v1_r2 = 0.1078
    if _V1_META.exists():
        with open(_V1_META) as f:
            meta_v1 = json.load(f)
        v1_r2 = meta_v1.get("train_r2", v1_r2)
    logger.info(f"[基线] MetaLearner v1 train R² = {v1_r2:.4f}")

    # ── 加载数据 ──
    df, avail_features, target_col = load_training_data(
        pnl_path=args.pnl,
        requested_features=args.features,
    )

    # ── 构建 X, y ──
    X, y, features = build_xy(df, avail_features, target_col)
    n_samples = len(y)

    # ── 第一次训练 ──
    logger.info("\n[训练 Round 1]")
    pipe, train_r2, cv_r2, coef_dict = train_ridge(X, y, features, args.alpha)

    logger.info(f"  train R²  = {train_r2:+.4f}")
    logger.info(f"  5-fold CV R² = {cv_r2:+.4f}")
    logger.info("  特征系数（标准化空间）：")
    for feat, coef in sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True):
        logger.info(f"    {feat:<25s} {coef:+.4f}")

    # ── 若 R² < 0.20：剔除最弱特征重训一次 ──
    if train_r2 < 0.20:
        logger.warning(
            f"train R²={train_r2:.4f} < 0.20，尝试剔除最弱特征后重训"
        )
        features_v2 = drop_weakest_feature(coef_dict, features)
        X2 = df[features_v2].copy()
        for col in features_v2:
            null_rate = X2[col].isna().mean()
            if null_rate > 0:
                X2[col] = X2[col].fillna(X2[col].median())
        # align with y（y 已按 return_3m 非空 filter）
        valid_mask = df[target_col].notna()
        X2_arr = X2[valid_mask].values.astype(float)

        logger.info(f"\n[训练 Round 2]  特征: {features_v2}")
        pipe2, train_r2_2, cv_r2_2, coef_dict2 = train_ridge(
            X2_arr, y, features_v2, args.alpha
        )
        logger.info(f"  train R²  = {train_r2_2:+.4f}")
        logger.info(f"  5-fold CV R² = {cv_r2_2:+.4f}")

        if train_r2_2 >= train_r2:
            logger.info("Round 2 改善，采用 Round 2 模型")
            pipe, train_r2, cv_r2, coef_dict, features = (
                pipe2, train_r2_2, cv_r2_2, coef_dict2, features_v2
            )
        else:
            logger.info("Round 2 无改善，保持 Round 1 模型")

    # ── 保存 ──
    save_model(pipe, features, coef_dict, train_r2, cv_r2, n_samples, args.out)

    # ── 最终对比 ──
    logger.info("\n" + "=" * 60)
    logger.info("【训练结果汇总】")
    logger.info(f"  v1 train R²  = {v1_r2:.4f}  (n=58, 6 agent proxy scores)")
    logger.info(f"  v2 train R²  = {train_r2:.4f}  (n={n_samples}, {len(features)} factor scores)")
    logger.info(f"  v2 5-fold CV = {cv_r2:.4f}")
    delta = train_r2 - v1_r2
    sign  = "↑" if delta > 0 else "↓"
    logger.info(f"  Δ(v2-v1)     = {delta:+.4f} {sign}")
    if train_r2 >= 0.35:
        logger.info("  ✓  达到目标 R² ≥ 0.35")
    elif train_r2 >= 0.20:
        logger.info("  ≈  未达 0.35 目标，但优于 0.20 门槛")
    else:
        logger.warning("  ✗  R² < 0.20，模型预测力不足")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
