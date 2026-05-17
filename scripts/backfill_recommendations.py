"""backfill_recommendations.py — 从历史面板数据回填推荐记录.

功能
====
无需 Tushare API，直接从 alpha_panel_v4.parquet 中提取各季末因子截面，
用当前冠军模型打分，生成 Top-10 推荐 JSON，写入 data/recommendations/。

用途：扩充 realized_pnl 训练样本（当前仅 3 期），为 meta-learner 重训提供数据。

用法
====
  python scripts/backfill_recommendations.py \\
      --panel   data/panel/alpha_panel_v4.parquet \\
      --model   models/lgbm_v6_alpha.pkl \\
      --recs    data/recommendations/ \\
      --dates   2024-09-30 2025-03-31 2025-06-30 2025-09-30 \\
      [--top-n 10] [--skip-existing]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]


DEFAULT_DATES = [
    "2024-09-30",
    "2025-03-31",
    "2025-06-30",
    "2025-09-30",
]


def load_model(model_path: Path):
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_period(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    model,
    top_n: int = 10,
) -> list[dict] | None:
    """从面板截取单期，用模型打分，返回 Top-N 推荐."""
    # 找最近的面板日期
    all_dates = sorted(panel.index.get_level_values("as_of").unique())
    nearest = min(all_dates, key=lambda d: abs((d - as_of).days))
    gap_days = abs((nearest - as_of).days)
    if gap_days > 45:
        logger.warning(f"{as_of.date()}: 最近面板期 {nearest.date()} 相差 {gap_days} 天，跳过")
        return None

    xs = panel.xs(nearest, level="as_of").copy()
    if len(xs) < 50:
        logger.warning(f"{as_of.date()}: 截面股票数过少（{len(xs)}），跳过")
        return None

    # 提取模型特征
    feat_names = getattr(model, "_feature_names", None) or getattr(model, "feature_names", None)
    if feat_names is None:
        # 尝试从内部 lgbm model 获取
        try:
            feat_names = model._model.feature_name()
        except Exception:
            pass
    if not feat_names:
        logger.error("无法获取模型特征名")
        return None

    missing = [c for c in feat_names if c not in xs.columns]
    if len(missing) > len(feat_names) * 0.3:
        logger.error(f"缺失特征过多（{len(missing)}/{len(feat_names)}），跳过 {as_of.date()}")
        return None

    # 填充缺失特征为 0
    for c in missing:
        xs[c] = 0.0

    X = xs[list(feat_names)].fillna(0.0).to_numpy(dtype=np.float32)

    # 打分
    scores_raw = model.predict(X)
    scores = pd.Series(scores_raw, index=xs.index, dtype=np.float64)

    # 百分位归一化
    score_pct = scores.rank(pct=True)

    top_idx = score_pct.sort_values(ascending=False).head(top_n).index

    top_list = []
    for rank, ticker in enumerate(top_idx, start=1):
        raw_score = float(scores.loc[ticker])
        norm_score = float(score_pct.loc[ticker])
        row = xs.loc[ticker]
        key_factors = {}
        for col in ["pe_ttm", "pb", "roe_ttm", "momentum_6m", "accruals",
                    "book_to_market", "earnings_yield", "north_hold_ratio"]:
            if col in row.index:
                v = row[col]
                if pd.notna(v):
                    key_factors[col] = round(float(v), 6)
        top_list.append({
            "rank": rank,
            "ticker": str(ticker),
            "lgbm_rank": rank,
            "lgbm_score": round(norm_score, 6),
            "lgbm_score_raw": round(raw_score, 6),
            "reason": f"面板回填（panel_date={nearest.date()}）",
            "key_factors": key_factors,
        })

    logger.info(
        f"  {as_of.date()} (panel={nearest.date()}，gap={gap_days}d): "
        f"截面 {len(xs)} 只 → Top-{top_n} 已生成"
    )
    return top_list


def run(
    panel_path: Path,
    model_path: Path,
    recs_dir: Path,
    dates: list[str],
    top_n: int = 10,
    skip_existing: bool = True,
) -> int:
    recs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载面板：{panel_path}")
    panel = pd.read_parquet(panel_path)
    n_periods = panel.index.get_level_values("as_of").nunique()
    logger.info(f"  面板 {panel.shape}，{n_periods} 期")

    logger.info(f"加载模型：{model_path}")
    model = load_model(model_path)
    feat_names = (
        getattr(model, "_feature_names", None)
        or getattr(model, "feature_names", None)
    )
    if feat_names is None:
        try:
            feat_names = model._model.feature_name()
        except Exception:
            pass
    logger.info(f"  模型特征数：{len(feat_names) if feat_names else 'unknown'}")

    n_written = 0
    for date_str in dates:
        out_path = recs_dir / f"{date_str}.json"
        if skip_existing and out_path.exists():
            logger.info(f"  {date_str}: 推荐已存在，跳过（--skip-existing）")
            continue

        as_of_ts = pd.Timestamp(date_str)
        top_list = score_period(panel, as_of_ts, model, top_n=top_n)
        if not top_list:
            continue

        payload = {
            "as_of": date_str,
            "generated_at": datetime.now().isoformat(),
            "source": "backfill_from_panel",
            "top10": top_list,
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 同时写子目录格式
        subdir = recs_dir / date_str
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "top10.json").write_text(
            json.dumps({"as_of": date_str, "top10": top_list}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"  ✅ {date_str}: 推荐已写入 {out_path}")
        n_written += 1

    logger.info(f"\n回填完成：新写入 {n_written} 条推荐记录")
    return n_written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",  type=Path, default=Path("data/panel/alpha_panel_v4.parquet"))
    p.add_argument("--model",  type=Path, default=Path("models/lgbm_v6_alpha.pkl"))
    p.add_argument("--recs",   type=Path, default=Path("data/recommendations/"))
    p.add_argument("--dates",  nargs="*", default=DEFAULT_DATES,
                   help="要回填的日期列表（YYYY-MM-DD）")
    p.add_argument("--top-n",  type=int, default=10)
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="已存在的推荐跳过（默认 True）")
    p.add_argument("--force", action="store_true",
                   help="强制重写已有推荐")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    skip = not args.force
    run(
        panel_path=args.panel,
        model_path=args.model,
        recs_dir=args.recs,
        dates=args.dates,
        top_n=args.top_n,
        skip_existing=skip,
    )


if __name__ == "__main__":
    main()
