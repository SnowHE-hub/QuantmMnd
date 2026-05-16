"""训练 QualityAgent v2 — 基于财务因子的 LGBM 二分类模型.

特征：20 个财务质量相关指标（来自 alpha_panel_v3）
标签：下季度收益是否跑赢横截面中位数（1=跑赢, 0=跑输）
方法：Walk-forward 训练（每期用历史所有期训练，预测下一期）

产出：models/agents/quality_lgbm_v2.pkl（含 feature_names + threshold）
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

QUALITY_FEATURES = [
    # 盈利能力
    "roe_ttm", "roa_ttm", "gross_margin", "net_margin",
    # 成长性
    "revenue_yoy", "operating_profit_yoy", "net_profit_yoy",
    "quarterly_revenue_yoy", "earnings_accel_q", "revenue_accel_q",
    # 现金流质量
    "accruals", "ocf_to_revenue_ttm", "fcf_yield",
    # 财务健康
    "debt_to_assets", "current_ratio", "asset_turnover", "equity_multiplier",
    # 其他
    "free_float_ratio", "list_age_years", "is_recent_ipo",
]

LABEL_COL = "forward_return_63d"

# IC 权重（基于因子实测，正号=高值为好，负号=低值为好）
_IC_WEIGHTS = {
    "roe_ttm":            +0.012,
    "roa_ttm":            +0.006,
    "ocf_to_revenue_ttm": +0.012,
    "accruals":           -0.020,   # 应计为负向
    "gross_margin":       -0.006,   # 行业差异大，IC 轻度负
    "net_margin":         -0.012,
    "debt_to_assets":     +0.003,
    "earnings_accel_q":   +0.008,
    "revenue_accel_q":    +0.005,
    "net_profit_yoy":     +0.004,
    "operating_profit_yoy": +0.005,
}


def load_panel(panel_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(panel_path).reset_index()
    if "as_of" not in df.columns:
        raise ValueError("panel 缺少 as_of 列")
    df = df.sort_values(["as_of", "ticker"] if "ticker" in df.columns else ["as_of", "ts_code"])
    return df


def _rank_norm(series: pd.Series) -> pd.Series:
    """截面内 rank 归一化到 [-1, +1]"""
    r = series.rank(pct=True, na_option="keep")
    return r * 2 - 1


def make_quality_label(df: pd.DataFrame) -> pd.Series:
    """
    自监督质量标签：截面内 IC 加权合成质量分 → 前 30% 为 1，后 30% 为 0，中间 40% 丢弃（但保留用于推理）。
    实际训练时仅使用 top/bottom 30% 行。
    """
    def _score_group(g: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=g.index)
        denom = 0.0
        for feat, w in _IC_WEIGHTS.items():
            if feat not in g.columns:
                continue
            col = g[feat].copy()
            if col.isna().all():
                continue
            ranked = _rank_norm(col)
            score += ranked * abs(w) * np.sign(w)
            denom += abs(w)
        if denom > 0:
            score /= denom
        n = len(g)
        low_q = score.quantile(0.30)
        high_q = score.quantile(0.70)
        label = pd.Series(np.nan, index=g.index)
        label[score >= high_q] = 1
        label[score <= low_q] = 0
        return label

    return df.groupby("as_of", group_keys=False).apply(_score_group)


def walk_forward_cv(df: pd.DataFrame, features: list[str]):
    """Walk-forward：每期累积所有历史 → 预测当期，仅使用 top/bottom 30% 样本训练."""
    periods = sorted(df["as_of"].unique())
    if len(periods) < 3:
        raise ValueError("至少需要 3 期数据才能做 walk-forward")

    import lightgbm as lgb

    oof_records = []

    for i, test_period in enumerate(periods[2:], start=2):
        # 训练集：历史期 top/bottom 30%（排除中间 40% 噪声）
        train_mask = df["as_of"].isin(periods[:i]) & df["label"].notna()
        # 测试集：当期全部（推理不受限）
        test_mask = df["as_of"] == test_period

        X_tr = df.loc[train_mask, features].copy()
        y_tr = df.loc[train_mask, "label"].copy()
        X_te = df.loc[test_mask, features].copy()

        fill = X_tr.median()
        X_tr = X_tr.fillna(fill)
        X_te = X_te.fillna(fill)

        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        idx = df.loc[test_mask].index
        oof_records.append(pd.Series(proba, index=idx, name="proba"))
        logger.info(
            f"  [{i}/{len(periods)-1}] {str(test_period)[:10]}  "
            f"train={train_mask.sum():4d}  test={test_mask.sum():4d}  "
            f"pos_rate={y_tr.mean():.2f}"
        )

    return pd.concat(oof_records)


def train_final_model(df: pd.DataFrame, features: list[str]):
    """用全部数据训练最终模型"""
    import lightgbm as lgb

    X = df[features].copy()
    y = df["label"].copy()
    for col in features:
        X[col] = X[col].fillna(X[col].median())

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def evaluate(df: pd.DataFrame, oof_proba: pd.Series):
    """计算 OOF 精度/AUC（only on labeled rows） + IC vs. 未来收益"""
    from sklearn.metrics import roc_auc_score, accuracy_score

    # AUC/ACC 只用 top/bottom 标签行
    merged_all = df.loc[oof_proba.index].copy()
    merged_all["proba"] = oof_proba
    merged = merged_all.dropna(subset=["label"])

    acc = accuracy_score(merged["label"], (merged["proba"] > 0.5).astype(int))
    auc = roc_auc_score(merged["label"], merged["proba"])

    # IC：各期内 proba vs. 63d 未来收益（所有行）
    ic_list = []
    for _, g in merged_all.groupby("as_of"):
        if len(g) < 10 or LABEL_COL not in g.columns:
            continue
        gv = g[[LABEL_COL, "proba"]].dropna()
        if len(gv) < 10:
            continue
        ic = gv["proba"].corr(gv[LABEL_COL], method="spearman")
        ic_list.append(ic)
    ic_mean = np.nanmean(ic_list) if ic_list else float("nan")
    ic_ir = (np.mean(ic_list) / np.std(ic_list)) * np.sqrt(4) if len(ic_list) > 1 else float("nan")

    logger.info(f"  OOF Acc={acc:.3f}  AUC={auc:.3f}  IC_vs_63d={ic_mean:.4f}  ICIR={ic_ir:.3f}")
    return {"accuracy": acc, "auc": auc, "ic_mean": ic_mean, "ic_ir": ic_ir}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="data/panel/alpha_panel_v3.parquet")
    parser.add_argument("--out", default="models/agents/quality_lgbm_v2.pkl")
    parser.add_argument("--label", default=LABEL_COL)
    args = parser.parse_args()

    panel_path = _ROOT / args.panel
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载因子面板: {panel_path}")
    df = load_panel(panel_path)

    # 检查特征覆盖
    avail = [f for f in QUALITY_FEATURES if f in df.columns]
    missing = [f for f in QUALITY_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"缺失特征（将跳过）: {missing}")
    features = avail
    if len(features) < 5:
        raise ValueError(f"可用特征太少: {features}")

    # 自监督质量标签（截面 IC 加权合成，top/bottom 30%）
    df = df.copy()
    logger.info("生成自监督质量标签（IC 加权合成，top/bottom 30%）...")
    df["label"] = make_quality_label(df)
    labeled_n = df["label"].notna().sum()
    logger.info(f"有效标签行数: {labeled_n}（共 {len(df)} 行）")

    periods = sorted(df["as_of"].unique())
    logger.info(f"可用期数: {len(periods)}，特征数: {len(features)}，总样本: {len(df)}")
    logger.info(f"特征列表: {features}")

    # Walk-forward OOF 评估
    logger.info("开始 Walk-Forward 交叉验证...")
    oof_proba = walk_forward_cv(df, features)

    # 评估
    logger.info("OOF 评估结果:")
    metrics = evaluate(df, oof_proba)

    # 训练最终模型（只用 top/bottom 30% 标签行）
    logger.info("训练最终模型（全量标签数据）...")
    df_labeled = df.dropna(subset=["label"])
    final_model = train_final_model(df_labeled, features)

    # 特征重要性
    fi = dict(zip(features, final_model.feature_importances_))
    top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]
    logger.info(f"Top-5 特征重要性: {top5}")

    # 计算截面中位数填充值（推理时用）
    fill_values = {col: float(df[col].median()) for col in features}

    bundle = {
        "model": final_model,
        "feature_names": features,
        "fill_values": fill_values,
        "label_col": args.label,
        "metrics": metrics,
        "n_periods": len(periods),
        "model_version": "quality_lgbm_v2",
        "feature_importances": fi,
    }

    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"模型已保存: {out_path}")
    logger.info(f"评估摘要: ACC={metrics['accuracy']:.3f}  AUC={metrics['auc']:.3f}  ICIR={metrics['ic_ir']:.3f}")


if __name__ == "__main__":
    main()
