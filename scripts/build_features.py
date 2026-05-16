"""scripts/build_features.py — 因子计算与持久化 CLI.

用法
====

::

    # 用最新 snapshot 计算 csi300_top20 的因子
    python scripts/build_features.py --as-of 2024-06-30 --universe csi300_top20

    # 多时点
    python scripts/build_features.py --as-of 2024-03-31 2024-06-30 2024-09-30

    # 仅 raw（不标准化）
    python scripts/build_features.py --as-of 2024-06-30 --raw

    # 关掉中性化
    python scripts/build_features.py --as-of 2024-06-30 --no-neutralize
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from quantmind.core.logger import get_logger, setup_logger
from quantmind.features import FeaturePipeline

log = get_logger(__name__)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    setup_logger()
    parser = argparse.ArgumentParser(description="QuantMind feature builder")
    parser.add_argument(
        "--as-of",
        nargs="+",
        type=_parse_date,
        required=True,
        help="一个或多个 ISO 日期（必须已存在对应 snapshot）",
    )
    parser.add_argument("--universe", default="csi300", help="存盘文件名前缀")
    parser.add_argument("--raw", action="store_true", help="不做横截面标准化")
    parser.add_argument("--no-neutralize", action="store_true", help="跳过行业/市值中性化")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        choices=["fundamental", "technical", "sentiment", "expansion"],
        help="只计算指定组（默认全部）",
    )
    parser.add_argument("--winsorize-sigma", type=float, default=3.0)
    args = parser.parse_args()

    pipe = FeaturePipeline(
        groups=args.groups,
        do_standardize=not args.raw,
        do_neutralize=not args.no_neutralize,
        winsorize_sigma=args.winsorize_sigma,
    )

    failures: list[tuple[date, str]] = []
    for as_of in args.as_of:
        log.info(f"==== building features for {as_of} ====")
        try:
            df = pipe.run_single(as_of)
            log.info(f"  shape: {df.shape}")
            path = pipe.save(df, as_of, universe=args.universe)
            log.info(f"  saved: {path}")
        except Exception as e:  # noqa: BLE001
            log.error(f"FAILED for {as_of}: {e}")
            failures.append((as_of, str(e)))

    if failures:
        print("\n=== Failures ===")
        for d, msg in failures:
            print(f"  {d}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
