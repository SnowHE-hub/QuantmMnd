"""scripts/download_data.py — 批量下载并构建 PIT snapshot.

用法
====

::

    # 1. 拉沪深 300 当前快照（含财报，慢，~30 分钟）
    python scripts/download_data.py --as-of 2024-06-30 --universe csi300

    # 2. 仅价格（快，~10 分钟）
    python scripts/download_data.py --as-of 2024-06-30 --no-financials

    # 3. 多个时点连续构建
    python scripts/download_data.py --as-of 2023-06-30 2023-12-31 2024-06-30

    # 4. 强制覆盖
    python scripts/download_data.py --as-of 2024-06-30 --overwrite
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from quantmind.core.logger import get_logger, setup_logger
from quantmind.data import build_snapshot, list_snapshots

log = get_logger(__name__)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    setup_logger()
    parser = argparse.ArgumentParser(description="QuantMind data download / snapshot builder")
    parser.add_argument(
        "--as-of",
        nargs="+",
        type=_parse_date,
        required=True,
        help="一个或多个 ISO 日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--universe",
        default="csi300",
        choices=["csi300", "csi500", "csi800", "csi1000", "sse50"],
    )
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--no-financials", action="store_true")
    parser.add_argument("--no-indicators", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"既有 snapshots: {[d.isoformat() for d in list_snapshots()]}")

    failures = []
    for as_of in args.as_of:
        log.info(f"==== building snapshot for {as_of} ({args.universe}) ====")
        try:
            meta = build_snapshot(
                as_of=as_of,
                universe_name=args.universe,
                price_lookback_days=args.lookback_days,
                include_financials=not args.no_financials,
                include_indicators=not args.no_indicators,
                overwrite=args.overwrite,
            )
            log.info(f"OK: {meta['snapshot_dir']}  rows={meta['rows_per_table']}")
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
