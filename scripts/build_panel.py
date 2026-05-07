"""scripts/build_panel.py — 多时点 × 多 ticker 训练集 panel CLI.

示例
====

::

    # 季度面板（推荐 — 22 期 × 300 票 ≈ 5000 行）
    python scripts/build_panel.py --start 2019-12-31 --end 2024-06-30 --freq Q --max-tickers 300

    # 月度面板（更多样本，~66 期）
    python scripts/build_panel.py --start 2019-01-31 --end 2024-06-30 --freq M --max-tickers 300

    # 小样本 smoke test（4 期 × 20 票 ≈ 80 行）
    python scripts/build_panel.py --start 2023-12-31 --end 2024-06-30 --freq Q --max-tickers 20
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from quantmind.core.logger import get_logger, setup_logger
from quantmind.features.panel import (
    build_panel,
    month_end_dates,
    quarter_end_dates,
)

log = get_logger(__name__)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    setup_logger()
    parser = argparse.ArgumentParser(description="QuantMind multi-date training panel builder")
    parser.add_argument("--start", type=_parse_date, required=True, help="第一个 rebalance 日期")
    parser.add_argument("--end", type=_parse_date, required=True, help="最后一个 rebalance 日期")
    parser.add_argument(
        "--freq",
        choices=["Q", "M"],
        default="Q",
        help="再平衡频率：Q=每季末 SSE 最后交易日 / M=每月末 SSE 最后交易日",
    )
    parser.add_argument(
        "--universe",
        default="csi300",
        help="历史 PIT-correct 指数成分（csi300/csi500/csi800）",
    )
    parser.add_argument(
        "--max-tickers", type=int, default=None, help="每个时点限取前 N 只权重最大股"
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[21, 63],
        help="forward return 天数（默认 21d/3M、63d/3M）",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="不做横截面标准化（输出 raw 因子值）",
    )
    parser.add_argument(
        "--rebuild-snapshot",
        action="store_true",
        help="强制重建 snapshot（即使已存在）",
    )
    parser.add_argument("--name", default=None, help="自定义 panel 文件名")
    args = parser.parse_args()

    if args.freq == "Q":
        dates = quarter_end_dates(args.start, args.end)
    else:
        dates = month_end_dates(args.start, args.end)

    if not dates:
        log.error(f"no rebalance dates between {args.start} and {args.end}")
        return 1

    log.info(f"will build {len(dates)} rebalance dates: {dates[0]} ... {dates[-1]}")

    panel = build_panel(
        rebalance_dates=dates,
        universe_name=args.universe,
        max_tickers=args.max_tickers,
        do_standardize=not args.no_standardize,
        forward_horizons_days=tuple(args.horizons),
        rebuild_snapshot=args.rebuild_snapshot,
        save=True,
        panel_name=args.name,
    )

    print("\n=== Panel summary ===")
    print(f"shape: {panel.shape}")
    print(f"index: {panel.index.names}")
    print(f"date range: {panel.index.get_level_values(0).min()} ~ {panel.index.get_level_values(0).max()}")
    print(f"unique tickers: {panel.index.get_level_values(1).nunique()}")
    print(f"unique dates: {panel.index.get_level_values(0).nunique()}")
    label_cols = [c for c in panel.columns if c.startswith("forward_return_")]
    print(f"label columns: {label_cols}")
    print("label coverage:")
    for c in label_cols:
        cov = panel[c].notna().mean()
        print(f"  {c}: {cov:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
