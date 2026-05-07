"""scripts/download_data.py — 批量下载并构建 PIT snapshot.

用法
====

::

    # 1. 快速验证（前 20 只权重最大股，~5 分钟）
    python scripts/download_data.py --as-of 2024-06-30 --max-tickers 20

    # 2. 拉沪深 300 当前快照（含财报，~30-45 分钟）
    python scripts/download_data.py --as-of 2024-06-30 --universe csi300

    # 3. 仅价格（快，~5-10 分钟）
    python scripts/download_data.py --as-of 2024-06-30 --no-financials --no-indicators

    # 4. 多个时点连续构建
    python scripts/download_data.py --as-of 2023-06-30 2023-12-31 2024-06-30

    # 5. 强制覆盖 + 校验
    python scripts/download_data.py --as-of 2024-06-30 --overwrite --validate

    # 6. 长线：季线末 SSE × 全 CSI300（推荐 2020–2024，约 20 期；可 nohup）
    python scripts/download_data.py --rebalance-quarterly-range 2020-01-01 2024-12-31 --universe csi300

    # 7. 月线末 SSE（期数多、耗时长）
    python scripts/download_data.py --rebalance-monthly-range 2023-06-01 2024-06-01 --universe csi300
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from quantmind.core.logger import get_logger, setup_logger
from quantmind.data import build_snapshot, list_snapshots, validate_snapshot
from quantmind.data.sse_calendar import monthly_last_trade_days, quarterly_last_trade_days

log = get_logger(__name__)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    setup_logger()
    parser = argparse.ArgumentParser(description="QuantMind data download / snapshot builder")
    parser.add_argument(
        "--as-of",
        nargs="*",
        type=_parse_date,
        default=[],
        help="一个或多个 ISO 日期；或使用下面 rebalance-range 自动展开",
    )
    parser.add_argument(
        "--rebalance-monthly-range",
        nargs=2,
        metavar=("START", "END"),
        type=str,
        help="月线末：区间内每月最后一个 SSE 交易日",
    )
    parser.add_argument(
        "--rebalance-quarterly-range",
        nargs=2,
        metavar=("START", "END"),
        type=str,
        help="季线末：区间内每季最后一个 SSE 交易日（3/6/9/12 月历季末）",
    )
    parser.add_argument(
        "--universe",
        default="csi300",
        choices=["csi300", "csi500", "csi800", "csi1000", "sse50"],
    )
    parser.add_argument("--lookback-days", type=int, default=280)
    parser.add_argument("--max-tickers", type=int, default=None,
                        help="只拉前 N 只权重最大股（验证用）")
    parser.add_argument("--no-financials", action="store_true")
    parser.add_argument("--no-indicators", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true",
                        help="构建后立即跑 validate_snapshot 检查")
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    parser.set_defaults(validate=True)
    args = parser.parse_args()

    if args.rebalance_monthly_range and args.rebalance_quarterly_range:
        parser.error("--rebalance-monthly-range 与 --rebalance-quarterly-range 只能二选一")

    if args.rebalance_quarterly_range:
        s = _parse_date(args.rebalance_quarterly_range[0])
        e = _parse_date(args.rebalance_quarterly_range[1])
        args.as_of = quarterly_last_trade_days(s, e)
        log.info(
            f"--rebalance-quarterly-range expanded to {len(args.as_of)} dates "
            f"({args.as_of[0]!r} … {args.as_of[-1]!r})"
        )
    elif args.rebalance_monthly_range:
        s = _parse_date(args.rebalance_monthly_range[0])
        e = _parse_date(args.rebalance_monthly_range[1])
        args.as_of = monthly_last_trade_days(s, e)
        log.info(
            f"--rebalance-monthly-range expanded to {len(args.as_of)} dates "
            f"({args.as_of[0]} … {args.as_of[-1]})"
        )

    if not args.as_of:
        parser.error(
            "必须提供 --as-of … 和/或 --rebalance-monthly-range 或 --rebalance-quarterly-range"
        )

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
                max_tickers=args.max_tickers,
                overwrite=args.overwrite,
            )
            log.info(f"OK: {meta['snapshot_dir']}  rows={meta['rows_per_table']}")

            if args.validate:
                log.info(f"---- validating snapshot {as_of} ----")
                report = validate_snapshot(as_of, strict=False)
                print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
                if not report["ok"]:
                    failures.append((as_of, f"validation failed: {report['failed_checks']}"))
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
