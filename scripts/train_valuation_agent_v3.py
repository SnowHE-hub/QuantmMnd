#!/usr/bin/env python3
"""训练 ValuationAgent v3：基于 alpha_panel_v3 + forward_return_63d.

升级要点（相比 v2）：
  1. 数据源：alpha_panel_v3.parquet（1374只 × 20期，内含所有特征，无需外部 fina/basic）
  2. 标签：forward_return_63d（季度调仓，与因子时效更匹配）
  3. 行业相对估值：sector_pe_ratio / sector_pb_ratio（截面内 pe_ttm / 中位数）
  4. 新增质量因子：fcf_yield, earnings_accel_q, revenue_accel_q, accruals, ocf_to_revenue_ttm
  5. Purged 时间序列 CV：embargo=90d（>63d 收益重叠期），5 折

运行：
  python scripts/train_valuation_agent_v3.py \\
    --panel data/panel/alpha_panel_v3.parquet \\
    --out   models/agents/valuation_lgbm_v3.pkl
"""
from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent

# ── 特征集 ─────────────────────────────────────────────────────────────────────
# 仅包含估值/质量/成长因子；动量、北向、融资等短期因子不纳入
VALUATION_FEATURES = [
    # 价格乘数
    "pe_ttm", "pb", "ps_ttm", "earnings_yield", "book_to_market",
    # 行业相对估值（截面计算）
    "sector_pe_ratio", "sector_pb_ratio",
    # 盈利质量
    "roe_ttm", "roa_ttm", "gross_margin", "net_margin",
    # 财务健康
    "debt_to_assets", "current_ratio",
    # 成长
    "revenue_yoy", "net_profit_yoy", "earnings_accel_q", "revenue_accel_q",
    # 现金流质量
    "accruals", "ocf_to_revenue_ttm", "fcf_yield",
    # 规模
    "size_rank", "log_market_cap",
]

LABEL_COL = "forward_return_63d"


# ── 特征工程 ───────────────────────────────────────────────────────────────────

def add_sector_relative(df: pd.DataFrame) -> pd.DataFrame:
    """截面内 pe_ttm / 行业中位PE，pb / 行业中位PB."""
    if "exposure_industry" not in df.columns:
        df["sector_pe_ratio"] = np.nan
        df["sector_pb_ratio"] = np.nan
        return df

    df = df.copy()
    for raw_col, new_col in [("pe_ttm", "sector_pe_ratio"), ("pb", "sector_pb_ratio")]:
        if raw_col not in df.columns:
            df[new_col] = np.nan
            continue
        valid = df[raw_col].replace(0, np.nan)
        sector_med = df.groupby(["as_of", "exposure_industry"])[raw_col].transform(
            lambda x: x[x > 0].median() if (x > 0).any() else np.nan
        )
        df[new_col] = valid / sector_med.replace(0, np.nan)
        # 截断极端值
        df[new_col] = df[new_col].clip(0.1, 10.0)
    return df


def cross_section_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """将每个特征替换为截面百分位秩（[0,1]），减少异常值影响."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby("as_of")[col].rank(pct=True, na_option="keep")
    return df


# ── IC 计算 ────────────────────────────────────────────────────────────────────

def mean_ic(pred: np.ndarray, y: np.ndarray, dates: pd.Series) -> float:
    """按日截面 Spearman IC，再对日期均值."""
    tmp = pd.DataFrame({"p": pred.ravel(), "y": y.ravel(), "d": dates.values})
    ics = []
    for _, g in tmp.groupby("d", observed=False):
        if len(g) < 8:
            continue
        ic = g["p"].corr(g["y"], method="spearman")
        if pd.notna(ic) and np.isfinite(ic):
            ics.append(float(ic))
    return float(np.nanmean(ics)) if ics else float("nan")


# ── Purged 时间序列 CV ─────────────────────────────────────────────────────────

def purged_ts_cv(
    df_pre: pd.DataFrame,
    feats: list[str],
    *,
    label_col: str,
    params: dict,
    n_estimators: int,
    cv_folds: int,
    embargo_days: int,
) -> tuple[list[float], list[int]]:
    import lightgbm as lgb

    dates = sorted(df_pre["as_of"].unique())
    n_dates = len(dates)
    K = max(2, min(cv_folds, max(2, n_dates // 2)))

    fold_ics, best_iters = [], []

    for k in range(K):
        i0 = int(k * n_dates / K)
        i1 = int((k + 1) * n_dates / K) if k < K - 1 else n_dates
        val_dates = dates[i0:i1]
        if not val_dates:
            continue
        embargo_boundary = val_dates[0] - pd.Timedelta(days=embargo_days)

        tr = df_pre[df_pre["as_of"] < embargo_boundary]
        va = df_pre[df_pre["as_of"].isin(val_dates)]

        if len(tr) < 300 or len(va) < 40:
            logger.warning(f"CV fold {k}: train={len(tr)} val={len(va)}，跳过")
            continue

        X_tr = tr[feats].fillna(0.5)
        y_tr = tr[label_col]
        X_va = va[feats].fillna(0.5)
        y_va = va[label_col]

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

        booster = lgb.train(
            params, dtr, num_boost_round=n_estimators,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        bi = booster.best_iteration or max(20, n_estimators // 4)
        best_iters.append(int(bi))

        ic_va = mean_ic(booster.predict(X_va, num_iteration=bi), y_va.values, va["as_of"])
        fold_ics.append(ic_va)
        logger.info(
            f"CV fold {k+1}/{K} | [{val_dates[0].date()}…{val_dates[-1].date()}] "
            f"| IC={ic_va:.4f} | iter={bi}"
        )

    return fold_ics, best_iters


# ── 主训练流程 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/panel/alpha_panel_v3.parquet")
    parser.add_argument("--out", default="models/agents/valuation_lgbm_v3.pkl")
    parser.add_argument("--test-cutoff", default="2024-01-01")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=90,
                        help="Purge embargo（应 > label 期数63d）")
    args = parser.parse_args()

    panel_path = ROOT / args.panel
    out_path   = ROOT / args.out
    test_cutoff = pd.Timestamp(args.test_cutoff)

    # ── 加载面板 ──────────────────────────────────────────────────────────────
    logger.info(f"加载面板: {panel_path}")
    raw = pd.read_parquet(panel_path)

    # 展平 MultiIndex → 列
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index()
        if "ticker" in raw.columns and "ts_code" not in raw.columns:
            raw = raw.rename(columns={"ticker": "ts_code"})
    elif raw.index.name == "as_of":
        raw = raw.reset_index()

    raw["as_of"] = pd.to_datetime(raw["as_of"], errors="coerce")
    raw = raw.dropna(subset=["as_of"]).sort_values("as_of")
    logger.info(f"面板: {raw.shape[0]} 行 | {raw['as_of'].nunique()} 期 | "
                f"{raw['ts_code'].nunique() if 'ts_code' in raw.columns else '?'} 只")

    # ── 特征工程 ──────────────────────────────────────────────────────────────
    logger.info("计算行业相对估值…")
    raw = add_sector_relative(raw)

    # 确定可用特征
    feats = [f for f in VALUATION_FEATURES if f in raw.columns]
    missing = [f for f in VALUATION_FEATURES if f not in raw.columns]
    if missing:
        logger.warning(f"缺失特征（将跳过）: {missing}")
    logger.info(f"可用特征 ({len(feats)}): {feats}")

    # 截面百分位秩
    logger.info("截面百分位秩变换…")
    raw = cross_section_rank(raw, feats)

    # 过滤有效行
    df_valid = raw[raw[LABEL_COL].notna()].copy()
    df_valid = df_valid.dropna(subset=[LABEL_COL])
    logger.info(f"有效行（label 非空）: {len(df_valid)}")

    # 同时对 label 做截面百分位秩（让模型在截面排名上学习）
    df_valid[LABEL_COL] = df_valid.groupby("as_of")[LABEL_COL].rank(pct=True)

    df_pre  = df_valid[df_valid["as_of"] < test_cutoff].copy()
    df_test = df_valid[df_valid["as_of"] >= test_cutoff].copy()
    logger.info(f"Pre-test: {len(df_pre)} 行 | Test: {len(df_test)} 行")

    # ── LightGBM 超参 ─────────────────────────────────────────────────────────
    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "learning_rate":    args.learning_rate,
        "num_leaves":       31,
        "min_child_samples": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "lambda_l1":        0.05,
        "lambda_l2":        0.1,
        "verbose":          -1,
        "n_jobs":           -1,
    }

    # ── Purged CV ─────────────────────────────────────────────────────────────
    logger.info(f"Purged CV ({args.cv_folds} 折, embargo={args.embargo_days}d)…")
    fold_ics, best_iters = purged_ts_cv(
        df_pre, feats,
        label_col=LABEL_COL,
        params=params,
        n_estimators=args.n_estimators,
        cv_folds=args.cv_folds,
        embargo_days=args.embargo_days,
    )

    if fold_ics:
        cv_mean = float(np.nanmean(fold_ics))
        cv_std  = float(np.nanstd(fold_ics))
        # 过滤掉 best_iter<=5 的异常折（early stopping 立即触发，说明该期无法学习）
        valid_iters = [bi for bi in best_iters if bi > 5]
        median_iter = int(np.median(valid_iters)) if valid_iters else 200
        # 至少 100 轮，避免欠拟合
        rounds = max(100, median_iter)
    else:
        logger.warning("CV 无有效折，使用默认轮数")
        cv_mean, cv_std, rounds = float("nan"), float("nan"), 200

    logger.info(
        f"Purged CV IC: mean={cv_mean:.4f} std={cv_std:.4f} "
        f"(n={len(fold_ics)}) | 最终轮数={rounds}"
    )

    # ── 保存特征分位映射（用于推理时将原始值转为近似百分位秩） ──────────────
    # 取 pre-test 数据训练前（cross-section rank 之前）的原始值分布
    raw_pre = raw[raw["as_of"] < test_cutoff].copy()
    feature_quantiles: dict[str, list[float]] = {}
    n_quantiles = 100
    for f in feats:
        if f not in raw_pre.columns:
            continue
        vals = pd.to_numeric(raw_pre[f], errors="coerce").dropna().values
        if len(vals) < 10:
            continue
        qs = np.nanpercentile(vals, np.linspace(0, 100, n_quantiles + 1))
        feature_quantiles[f] = qs.tolist()
    logger.info(f"存储 {len(feature_quantiles)} 个特征分位映射（n_quantiles={n_quantiles}）")

    # ── 全量训练（用 pre-test 所有数据） ──────────────────────────────────────
    import lightgbm as lgb
    logger.info("训练最终模型…")
    X_all = df_pre[feats].fillna(0.5)
    y_all = df_pre[LABEL_COL]
    model = lgb.train(params, lgb.Dataset(X_all, label=y_all), num_boost_round=rounds)

    # 训练集 IC
    train_ic = mean_ic(model.predict(X_all, num_iteration=rounds), y_all.values, df_pre["as_of"])

    # Hold-out 测试集 IC
    test_ic = float("nan")
    if len(df_test) > 0:
        X_te = df_test[feats].fillna(0.5)
        pred_te = model.predict(X_te, num_iteration=rounds)
        test_ic = mean_ic(pred_te, df_test[LABEL_COL].values, df_test["as_of"])

    logger.info(
        f"训练IC(pre-test)={train_ic:.4f} | "
        f"Hold-out IC(>={args.test_cutoff})={test_ic:.4f}"
    )

    # 特征重要性
    importance = sorted(
        zip(feats, model.feature_importance("gain")),
        key=lambda x: x[1], reverse=True,
    )
    logger.info("特征重要性 (Top 15):")
    for fname, imp in importance[:15]:
        logger.info(f"  {fname:<35} {imp:.0f}")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":             model,
        "features":          feats,
        "feature_quantiles": feature_quantiles,  # {feature: [101 quantile values]}
        "version":           "lgbm_v3",
        "kind":              "valuation_lgbm_v3",
        "label":             LABEL_COL,
        "test_cutoff":       str(test_cutoff.date()),
        "cv_folds":          args.cv_folds,
        "embargo_days":      args.embargo_days,
        "num_boost_round":   rounds,
        "metrics": {
            "cv_val_ic_mean":    round(cv_mean, 5) if np.isfinite(cv_mean) else None,
            "cv_val_ic_std":     round(cv_std, 5)  if np.isfinite(cv_std)  else None,
            "cv_fold_ics":       [round(x, 5) for x in fold_ics],
            "train_ic_pre_test": round(train_ic, 5) if np.isfinite(train_ic) else None,
            "holdout_test_ic":   round(test_ic, 5)  if np.isfinite(test_ic)  else None,
            "n_pre_test":        len(df_pre),
            "n_test":            len(df_test),
            "n_tickers":         raw["ts_code"].nunique() if "ts_code" in raw.columns else None,
            "n_features":        len(feats),
        },
        "created_at": datetime.now().isoformat(),
        "feature_importance": {k: int(v) for k, v in importance},
    }
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"保存: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
