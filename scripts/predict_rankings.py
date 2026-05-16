#!/usr/bin/env python3
"""对 predict.parquet 最新截面打分并输出排名 CSV。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.models.factor_model import FactorModel
from quantmind.utils.score_order import order_preserving_pct_rank

_CANDIDATE_MODELS = (
    _ROOT / "models" / "lgbm_v1_final.pkl",
    _ROOT / "models" / "lgbm_v2_base_features.pkl",
)


def _predict_raw_scores(model_path: Path, xs: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    """加载模型并对截面 xs 打分；缺失特征时返回 (None, err_msg)."""
    model = FactorModel.load(model_path)
    feat_names = getattr(model, "_feature_names", None) or model.feature_names
    if not feat_names:
        return None, "no feature names"
    missing = [c for c in feat_names if c not in xs.columns]
    if missing:
        return None, f"missing {len(missing)} features"
    X = xs[list(feat_names)].to_numpy(dtype=np.float32, copy=True)
    return model.predict(X), None


def _pick_model_by_raw_uniqueness(xs: pd.DataFrame) -> tuple[Path, np.ndarray]:
    """在 v1_final / v2_base_features（若存在且特征齐全）中选原始分数唯一值更多的模型。"""
    best_path: Path | None = None
    best_scores: np.ndarray | None = None
    best_u = -1
    for p in _CANDIDATE_MODELS:
        if not p.is_file():
            continue
        scores, err = _predict_raw_scores(p, xs)
        if scores is None:
            print(f"[predict_rankings] skip {p.name}: {err}")
            continue
        u = int(len(np.unique(scores.astype(np.float64))))
        print(f"[predict_rankings] candidate {p.name}: unique_raw_scores={u}")
        if u > best_u:
            best_u, best_path, best_scores = u, p, scores
    if best_path is None or best_scores is None:
        raise SystemExit("auto: no usable candidate model under models/")
    print(f"[predict_rankings] auto-pick → {best_path.name} unique_raw_scores={best_u}")
    return best_path, best_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=str,
        default=str(_ROOT / "models" / "lgbm_v1_final.pkl"),
        help="因子模型 pickle 路径，或填 auto：在 v1_final 与 v2_base_features 间按原始分数唯一数自动选择（需特征齐全）。",
    )
    p.add_argument("--predict", type=Path, default=Path("data/panel/predict.parquet"))
    p.add_argument("--output", type=Path, default=Path("reports/model_training/rankings_2024Q4.csv"))
    p.add_argument("--top-n", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predict.is_file():
        raise SystemExit(f"predict panel not found: {args.predict}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.predict)
    dates = sorted(panel.index.get_level_values("as_of").unique())
    latest = dates[-1]
    xs = panel.xs(latest, level="as_of")

    model_arg = args.model.strip()
    if model_arg.lower() == "auto":
        model_path, scores = _pick_model_by_raw_uniqueness(xs)
    else:
        model_path = Path(model_arg)
        if not model_path.is_file():
            alt = _ROOT / model_arg
            if alt.is_file():
                model_path = alt
        if not model_path.is_file():
            raise SystemExit(f"model not found: {model_arg}")
        scores, err = _predict_raw_scores(model_path, xs)
        if scores is None:
            raise SystemExit(err)

    n_unique_raw = int(len(np.unique(scores.astype(np.float64))))
    tickers = xs.index.tolist()

    if "company_profile" in xs.columns:
        industry = xs["company_profile"].astype(str).tolist()
    elif "exposure_industry" in xs.columns:
        industry = xs["exposure_industry"].astype(str).tolist()
    else:
        industry = [""] * len(tickers)

    ser = pd.Series(scores, index=tickers, dtype=np.float64)
    score_norm = order_preserving_pct_rank(ser).to_numpy(dtype=np.float64)
    n_unique_norm = int(len(np.unique(score_norm)))

    df = pd.DataFrame({
        "ticker": tickers,
        "as_of": [str(latest)[:10]] * len(tickers),
        "score": scores,
        "score_normalized": score_norm,
        "industry": industry,
    })
    df = df.sort_values(
        ["score_normalized", "score"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    df["rank_normalized"] = np.arange(1, len(df) + 1)
    n_unique_rank = int(df["rank_normalized"].nunique())

    df = df[["ticker", "as_of", "score", "score_normalized", "rank_normalized", "industry"]]
    df.to_csv(args.output, index=False)

    print(f"model={model_path.resolve()}")
    print(f"latest as_of={str(latest)[:10]}  n={len(df)}")
    print(f"unique_raw_scores={n_unique_raw}  unique_score_normalized={n_unique_norm}  unique_rank_normalized={n_unique_rank}")
    print(f"saved → {args.output.resolve()}")
    print(f"\nTop {args.top_n}:")
    print(df.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
