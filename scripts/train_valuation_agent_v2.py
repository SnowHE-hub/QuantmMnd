#!/usr/bin/env python
"""训练 ValuationAgent LightGBM v2：截面估值因子 → 相对收益回归."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


FEATURE_BASE = [
    "pe_ttm",
    "pb",
    "ps_ttm",
    "roe",
    "roe_yoy",
    "revenue_growth",
    "net_profit_margin",
    "asset_turnover",
    "debt_to_assets",
    "pe_rank",
    "pb_rank",
    "size_rank",
]


def add_group_rank(
    df: pd.DataFrame,
    cols: list[str],
    *,
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """按日期截面计算百分位秩：pe_ttm→pe_rank，pb→pb_rank."""
    out = df.copy()
    if date_col not in out.columns:
        logger.warning(f"[valuation_v2] 缺少 {date_col}，跳过截面秩")
        return out
    gb = out.groupby(date_col, observed=True)
    for c in cols:
        if c not in out.columns:
            continue
        rk = gb[c].rank(pct=True, method="average")
        if c == "pe_ttm":
            out["pe_rank"] = rk
        elif c == "pb":
            out["pb_rank"] = rk
        else:
            out[f"{c}_rank"] = rk
    if "market_cap" in out.columns and "size_rank" not in out.columns:
        out["size_rank"] = gb["market_cap"].rank(pct=True, method="average")
    return out


def _resolve_label(df: pd.DataFrame) -> str:
    if "forward_return_21d" in df.columns:
        return "forward_return_21d"
    if "forward_return_1m" in df.columns:
        return "forward_return_1m"
    raise ValueError("找不到标签列 forward_return_21d / forward_return_1m")


def _normalize_roe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """兼容 roe / roe_ttm 等命名."""
    out = df.copy()
    if "roe" not in out.columns and "roe_ttm" in out.columns:
        out["roe"] = out["roe_ttm"]
    return out


def cross_sectional_ic(pred: np.ndarray, y: np.ndarray, dates: pd.Series) -> tuple[float, float]:
    """按 trade_date 分组计算 Spearman IC，返回 mean / std."""
    tmp = pd.DataFrame({"p": pred, "y": y, "d": dates})
    ics: list[float] = []
    for _, g in tmp.groupby("d"):
        if len(g) < 15:
            continue
        ic, _ = spearmanr(g["p"].values, g["y"].values)
        if ic == ic and np.isfinite(ic):
            ics.append(float(ic))
    if not ics:
        ic_all, _ = spearmanr(pred, y)
        return float(ic_all) if ic_all == ic_all else 0.0, 0.0
    return float(np.mean(ics)), float(np.std(ics))


def train_valuation_lgbm_v2(
    panel_path: Path,
    output_model: Path,
    metrics_path: Path,
) -> None:
    import lightgbm as lgb

    if not panel_path.exists():
        raise FileNotFoundError(f"面板不存在: {panel_path}")

    df = pd.read_parquet(panel_path)
    df = _normalize_roe_columns(df)
    label_col = _resolve_label(df)

    for c in ("is_train", "is_val", "is_test"):
        if c not in df.columns:
            raise ValueError(f"面板缺少切分列 {c}，当前列: {list(df.columns)[:30]}...")

    if ("pe_rank" not in df.columns or "pb_rank" not in df.columns) or (
        "size_rank" not in df.columns and "market_cap" in df.columns
    ):
        df = add_group_rank(df, ["pe_ttm", "pb"], date_col="trade_date")

    avail = [f for f in FEATURE_BASE if f in df.columns]
    if len(avail) < 8:
        logger.warning(f"[valuation_v2] 可用特征较少 ({len(avail)}): {avail}")

    train_df = df[df["is_train"] == True].copy()  # noqa: E712
    val_df = df[df["is_val"] == True].copy()  # noqa: E712
    test_df = df[df["is_test"] == True].copy()  # noqa: E712

    train_df = train_df.dropna(subset=[label_col])
    val_df = val_df.dropna(subset=[label_col])
    test_df = test_df.dropna(subset=[label_col])

    if train_df.empty or val_df.empty:
        raise ValueError("训练集或验证集为空；请检查 is_train / is_val 与标签缺失情况")

    X_train = train_df[avail].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y_train = train_df[label_col].values.astype(float)
    X_val = val_df[avail].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y_val = val_df[label_col].values.astype(float)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    bi = getattr(model, "best_iteration_", None)
    best_iter = int(bi) if bi is not None else int(getattr(model, "n_estimators_", 500))

    pred_val = model.predict(X_val, num_iteration=best_iter)
    val_ic_mean, val_ic_std = cross_sectional_ic(
        pred_val, y_val, val_df["trade_date"] if "trade_date" in val_df.columns else pd.Series(0, index=val_df.index)
    )

    pred_test = None
    test_ic_mean = 0.0
    if not test_df.empty:
        X_test = test_df[avail].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        y_test = test_df[label_col].values.astype(float)
        pred_test = model.predict(X_test, num_iteration=best_iter)
        test_ic_mean, _ = cross_sectional_ic(
            pred_test,
            y_test,
            test_df["trade_date"] if "trade_date" in test_df.columns else pd.Series(0, index=test_df.index),
        )

    train_preds = model.predict(X_train, num_iteration=best_iter)
    ref = np.sort(train_preds.astype(np.float64))
    max_ref = 100_000
    if len(ref) > max_ref:
        rng = np.random.default_rng(42)
        ref = np.sort(rng.choice(ref, size=max_ref, replace=False))

    bundle = {
        "kind": "valuation_lgbm_v2",
        "model": model,
        "feature_cols": avail,
        "label_col": label_col,
        "pred_reference_sorted": ref.astype(np.float32),
        "best_iteration": best_iter,
    }

    output_model.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"[valuation_v2] 模型已保存: {output_model}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "val_IC_mean": round(val_ic_mean, 6),
        "val_IC_std": round(val_ic_std, 6),
        "test_IC_mean": round(float(test_ic_mean), 6),
        "best_iteration": best_iter,
        "features_used": avail,
        "label_col": label_col,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[valuation_v2] 指标: {metrics}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ValuationAgent lgbm_v2")
    ap.add_argument(
        "--panel",
        type=Path,
        default=_ROOT / "data" / "panel" / "quarterly_panel_final.parquet",
    )
    ap.add_argument(
        "--output-model",
        type=Path,
        default=_ROOT / "models" / "agents" / "valuation_lgbm_v2.pkl",
    )
    ap.add_argument(
        "--metrics-json",
        type=Path,
        default=_ROOT / "reports" / "model_training" / "valuation_lgbm_v2_metrics.json",
    )
    args = ap.parse_args()
    train_valuation_lgbm_v2(args.panel, args.output_model, args.metrics_json)


if __name__ == "__main__":
    main()
