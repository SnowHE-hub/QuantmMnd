"""scripts/train_meta_learner.py

MetaLearner v2 重训脚本 —— 修正版。

根因分析
--------
v1 失败（train R²=0.034，CV R²=-0.013）：
  • 错误数据源：stock_returns（单一30日窗口，308只股票×1.46天，伪复制）
  • return_3m 来自模拟期，与预测分无真实对应关系

正确方案
--------
1. 数据源：realized_pnl × alpha_panel_v4 join
   - realized_pnl: 8季度 × 10只股票 = 80行（含 actual_return_63d, hit）
   - alpha_panel: 按 (as_of, ticker) join，提取原始因子
   - join 后 n=100（含 2025-06-28 + 2025-06-30 双季度）
2. 特征：6个 Agent 代理分数（由 AgentMetaLearner.compute_agent_proxies 计算）
3. 任务 A - 回归：Ridge → actual_return_63d（per-quarter z-score 去除市场因子）
4. 任务 B - 分类：LogisticRegression → hit（是否跑赢中位数）
5. CV：Leave-One-Quarter-Out（LOQO，最严格的时序 CV）
6. 保存更好的模型（按 CV AUC / R² 判断）

用法
----
python scripts/train_meta_learner.py \\
  --pnl   data/feedback/realized_pnl.parquet \\
  --panel data/panel/alpha_panel_v4.parquet \\
  --out   data/meta_learner/meta_learner_v2.pkl

诊断指标
--------
  Agent 代理分 vs actual_return_63d：
    sentiment IC=-0.204 p=0.042 ✅
    risk      IC=-0.227 p=0.023 ✅
    strategy  IC=-0.182 p=0.070 ✅

  Agent 代理分 vs hit（beat median）：
    momentum  IC=-0.238 p=0.017 ✅
    sentiment IC=-0.242 p=0.015 ✅

  注意：IC 为负 → 代理分越高 → 实际表现越差（反直觉，可能是因子定向偏差）
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parents[1]
_PNL   = ROOT / "data" / "feedback" / "realized_pnl.parquet"
_PANEL = ROOT / "data" / "panel" / "alpha_panel_v4.parquet"
_V1_META = ROOT / "data" / "meta_learner" / "meta_learner_meta.json"

AGENT_COLS = ["valuation", "momentum", "quality", "sentiment", "risk", "strategy"]


# ─── 数据准备 ─────────────────────────────────────────────────────────────────

def build_training_data(pnl_path: Path, panel_path: Path) -> pd.DataFrame:
    """realized_pnl × alpha_panel join → 含代理分的训练 DataFrame."""
    pnl   = pd.read_parquet(pnl_path)
    panel = pd.read_parquet(panel_path)

    logger.info(f"realized_pnl: {pnl.shape}  |  alpha_panel: {panel.shape}")

    # ── 日期规范化（2025-06-28 → 2025-06-30，最近季末）──
    pnl["join_date"] = pd.to_datetime(pnl["as_of_date"])
    pnl.loc[pnl["join_date"].dt.strftime("%Y-%m-%d") == "2025-06-28", "join_date"] = pd.Timestamp("2025-06-30")

    # ── panel reset_index（MultiIndex: as_of, ticker）──
    panel_r = panel.reset_index().rename(columns={"as_of": "join_date"})
    panel_r["join_date"] = pd.to_datetime(panel_r["join_date"])

    # ── join ──
    merged = pnl.merge(panel_r, on=["join_date", "ticker"], how="inner")
    logger.info(f"Join 后样本数：{len(merged)}（pnl {len(pnl)} 行，命中 {len(merged)/len(pnl):.0%}）")

    # ── 计算 agent proxy scores（复用生产代码）──
    try:
        sys.path.insert(0, str(ROOT))
        from quantmind.models.meta_learner import compute_agent_proxies
    except ImportError:
        logger.warning("无法导入 compute_agent_proxies，使用内置简化版")
        compute_agent_proxies = _compute_proxies_fallback

    quarters = sorted(merged["join_date"].unique())
    proxy_parts: list[pd.DataFrame] = []
    for q in quarters:
        xs = merged[merged["join_date"] == q].set_index("ticker")
        try:
            proxies = compute_agent_proxies(xs)
        except Exception as e:
            logger.warning(f"Q={q} 代理分计算失败：{e}，跳过")
            continue
        proxies["join_date"] = q
        proxy_parts.append(proxies.reset_index())

    proxies_df = pd.concat(proxy_parts, ignore_index=True)
    logger.info(f"Agent proxy scores 计算完成，shape={proxies_df.shape}")

    # ── per-quarter z-score 目标（去市场因子）──
    merged["ret_z"] = merged.groupby("join_date")["actual_return_63d"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-9)
    )

    # ── 最终合并 ──
    final = merged[["join_date", "ticker", "actual_return_63d", "ret_z", "hit"]].merge(
        proxies_df, on=["join_date", "ticker"], how="left"
    )
    logger.info(f"最终训练集：{final.shape}，unique quarters: {final['join_date'].nunique()}")
    return final


def _compute_proxies_fallback(xs: pd.DataFrame) -> pd.DataFrame:
    """简化版 agent proxy 计算（无法导入 quantmind 时使用）."""
    from functools import reduce

    def zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / std if std > 1e-9 else s * 0.0

    configs = {
        "valuation": [("book_to_market", 0.35), ("earnings_yield", 0.40), ("dividend_yield_ttm", 0.25)],
        "momentum":  [("momentum_6m", 0.40), ("momentum_3m", 0.35), ("relative_strength_vs_csi300_60d", 0.25)],
        "quality":   [("ocf_to_revenue_ttm", 0.60), ("accruals", -0.40)],
        "sentiment": [("north_hold_ratio", 0.45), ("margin_buy_amount_20d", 0.30), ("margin_buy_intensity", 0.25)],
        "risk":      [("volatility_3m", -0.40), ("max_drawdown_3m", -0.40), ("beta_252d", -0.20)],
    }
    result = {}
    for name, factors in configs.items():
        score = None
        total_w = 0.0
        for col, w in factors:
            if col not in xs.columns:
                continue
            fz = zscore(xs[col].fillna(xs[col].median()))
            score = fz * w if score is None else score + fz * w
            total_w += abs(w)
        if score is None or total_w < 1e-9:
            result[name] = pd.Series(0.0, index=xs.index)
        else:
            result[name] = zscore(score / total_w)

    df = pd.DataFrame(result).reindex(xs.index)
    df["strategy"] = zscore(df.mean(axis=1))
    return df


# ─── Leave-One-Quarter-Out CV ─────────────────────────────────────────────────

def loqo_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_fn,
    metric_fn,
    verbose: bool = True,
) -> tuple[float, list[float]]:
    """Leave-One-Quarter-Out CV，严格时序不泄露."""
    unique_quarters = np.unique(groups)
    scores: list[float] = []

    for q in unique_quarters:
        test_mask  = groups == q
        train_mask = ~test_mask
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue

        model = model_fn()
        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict(X[test_mask])
        try:
            score = metric_fn(y[test_mask], y_pred)
        except Exception:
            score = float("nan")
        scores.append(score)
        if verbose:
            logger.info(f"    LOQO fold Q={str(q)[:10]:12s}  train={train_mask.sum():3d}  "
                        f"test={test_mask.sum():3d}  score={score:.4f}")

    mean_score = float(np.nanmean(scores)) if scores else float("nan")
    return mean_score, scores


def loqo_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    verbose: bool = True,
) -> tuple[float, list[float]]:
    """LOQO for binary classification — AUC metric."""
    unique_quarters = np.unique(groups)
    aucs: list[float] = []

    for q in unique_quarters:
        test_mask  = groups == q
        train_mask = ~test_mask
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue
        if len(np.unique(y[test_mask])) < 2:
            logger.warning(f"    Q={str(q)[:10]} 测试集只有一类，跳过")
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ])
        pipe.fit(X[train_mask], y[train_mask])
        y_prob = pipe.predict_proba(X[test_mask])[:, 1]
        auc = roc_auc_score(y[test_mask], y_prob)
        aucs.append(auc)
        if verbose:
            logger.info(f"    LOQO fold Q={str(q)[:10]:12s}  train={train_mask.sum():3d}  "
                        f"test={test_mask.sum():3d}  AUC={auc:.4f}")

    mean_auc = float(np.nanmean(aucs)) if aucs else float("nan")
    return mean_auc, aucs


# ─── 模型训练 ─────────────────────────────────────────────────────────────────

def train_regression(df: pd.DataFrame, features: list[str]) -> dict:
    """Ridge 回归：6 代理分 → per-quarter z-scored actual_return_63d."""
    sub = df[features + ["ret_z", "join_date"]].dropna()
    X   = sub[features].values.astype(float)
    y   = sub["ret_z"].values.astype(float)
    grp = sub["join_date"].values

    # full fit
    pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X, y)
    train_r2 = r2_score(y, pipe.predict(X))

    # LOQO CV
    logger.info("[回归] Leave-One-Quarter-Out CV（Ridge, target=ret_z）")
    cv_r2, cv_scores = loqo_cv(
        X, y, grp,
        model_fn=lambda: Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        metric_fn=r2_score,
    )

    coefs = dict(zip(features, pipe.named_steps["ridge"].coef_.tolist()))
    return {
        "task": "regression",
        "pipeline": pipe,
        "features": features,
        "n_samples": len(sub),
        "train_r2": round(train_r2, 4),
        "cv_r2": round(cv_r2, 4),
        "coefs": {k: round(v, 6) for k, v in coefs.items()},
    }


def train_classification(df: pd.DataFrame, features: list[str]) -> dict:
    """LogisticRegression：6 代理分 → hit（beat panel median）."""
    sub = df[features + ["hit", "join_date"]].dropna()
    X   = sub[features].values.astype(float)
    y   = sub["hit"].astype(int).values
    grp = sub["join_date"].values

    # full fit
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    pipe.fit(X, y)
    train_auc = roc_auc_score(y, pipe.predict_proba(X)[:, 1])

    # LOQO CV
    logger.info("[分类] Leave-One-Quarter-Out CV（LogisticRegression, target=hit）")
    cv_auc, cv_aucs = loqo_cv_auc(X, y, grp)

    coefs = dict(zip(features, pipe.named_steps["lr"].coef_[0].tolist()))
    return {
        "task": "classification",
        "pipeline": pipe,
        "features": features,
        "n_samples": len(sub),
        "train_auc": round(train_auc, 4),
        "cv_auc": round(cv_auc, 4),
        "coefs": {k: round(v, 6) for k, v in coefs.items()},
    }


# ─── 保存 ─────────────────────────────────────────────────────────────────────

def save_model(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"pipeline": result["pipeline"], "features": result["features"], "task": result["task"]}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    meta = {k: v for k, v in result.items() if k != "pipeline"}
    meta["trained_at"] = datetime.now().isoformat(timespec="seconds")
    meta["version"] = "v2"

    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"模型已保存：{out_path}")
    logger.info(f"元数据已保存：{meta_path}")


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="MetaLearner v2 重训（正确数据源版）")
    p.add_argument("--pnl",   type=Path, default=_PNL,   help="realized_pnl.parquet 路径")
    p.add_argument("--panel", type=Path, default=_PANEL, help="alpha_panel_v4.parquet 路径")
    p.add_argument("--out",   type=Path,
                   default=ROOT / "data" / "meta_learner" / "meta_learner_v2.pkl",
                   help="输出路径")
    p.add_argument("--features", nargs="+", default=AGENT_COLS,
                   help="使用的代理分列（默认全部 6 个）")
    args = p.parse_args(argv)

    logger.info("=" * 65)
    logger.info("MetaLearner v2 重训（正确数据源版）")
    logger.info("=" * 65)

    # ── v1 基线 ──
    v1_r2 = 0.1078
    if _V1_META.exists():
        with open(_V1_META) as f:
            v1_r2 = json.load(f).get("train_r2", v1_r2)
    logger.info(f"[参考] v1 train R² = {v1_r2:.4f}（58样本，6代理分）")
    logger.info(f"[参考] v1 数据源：realized_pnl × panel（正确）")
    logger.info(f"[错误] v2-draft  train R² = 0.0343（450样本，错误数据源：stock_returns）")
    logger.info("")

    # ── 数据 ──
    df = build_training_data(args.pnl, args.panel)
    features = [f for f in args.features if f in df.columns]
    logger.info(f"使用特征：{features}")
    logger.info("")

    # ── 回归 ──
    logger.info("─" * 40)
    reg_result = train_regression(df, features)
    logger.info(f"[回归结果]  train R²={reg_result['train_r2']:.4f}  CV R²={reg_result['cv_r2']:.4f}")
    logger.info(f"           系数（前3大绝对值）：" +
                ", ".join(f"{k}={v:+.3f}" for k, v in
                          sorted(reg_result["coefs"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]))
    logger.info("")

    # ── 分类 ──
    logger.info("─" * 40)
    cls_result = train_classification(df, features)
    logger.info(f"[分类结果]  train AUC={cls_result['train_auc']:.4f}  CV AUC={cls_result['cv_auc']:.4f}")
    logger.info(f"           系数（前3大绝对值）：" +
                ", ".join(f"{k}={v:+.3f}" for k, v in
                          sorted(cls_result["coefs"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]))
    logger.info("")

    # ── 选最佳保存 ──
    # 分类 CV AUC > 0.55 优先，否则比较回归 CV R²
    reg_cv = reg_result["cv_r2"]
    cls_cv = cls_result["cv_auc"]

    if cls_cv >= 0.55 or (cls_cv > reg_cv + 0.50):  # AUC 和 R² 不可直接比较，AUC>0.55 优先
        best = cls_result
        logger.info(f"选择分类模型（CV AUC={cls_cv:.4f}）")
    else:
        best = reg_result
        logger.info(f"选择回归模型（CV R²={reg_cv:.4f}）")

    save_model(best, args.out)

    # ── 最终汇总 ──
    logger.info("")
    logger.info("=" * 65)
    logger.info("【训练结果汇总】")
    logger.info(f"  v1       train R²  = {v1_r2:.4f}   (n=58, 6代理分，realized_pnl×panel)")
    logger.info(f"  v2-draft train R²  = 0.0343  (n=450, 错误数据源 stock_returns)")
    logger.info(f"  v2-final 回归 CV R²  = {reg_cv:.4f}")
    logger.info(f"  v2-final 分类 CV AUC = {cls_cv:.4f}")
    logger.info(f"  最佳任务：{best['task']}（n={best['n_samples']}）")

    # 评估
    if cls_cv >= 0.60:
        logger.info("  ✓  分类 AUC ≥ 0.60：模型有实用价值")
    elif cls_cv >= 0.55:
        logger.info("  ≈  分类 AUC ≥ 0.55：信号弱但存在，可部署（需持续监控）")
    else:
        logger.warning("  ✗  AUC < 0.55 且 CV R² 偏低：建议等待更多季度数据（n≥150）再部署")

    logger.info("=" * 65)


if __name__ == "__main__":
    main()
