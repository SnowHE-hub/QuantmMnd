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

    # 8. Alpha 全市场（~1374 只每期都要有因子）：用 replace 把 universe 完全替换为列表，
    #    不按 CSI300 求交。耗时长，建议 nohup + --overwrite 重建各季末日。
    python scripts/download_data.py \\
        --rebalance-quarterly-range 2020-01-01 2024-12-31 \\
        --universe csi300 \\
        --tickers-file data/alpha_universe/alpha_universe.txt \\
        --tickers-override-policy replace \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import date

from quantmind.core.logger import get_logger, setup_logger
from quantmind.data import build_snapshot, list_snapshots, validate_snapshot
from quantmind.data.sse_calendar import monthly_last_trade_days, quarterly_last_trade_days

log = get_logger(__name__)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _load_tickers_file(path: Path) -> list[str]:
    lines: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


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
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="逆序处理日期（从最近季度往最远跑）。Token B 从末端倒跑时使用，与 Token A 正向进程自然收敛于中间。",
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        default=None,
        help="每行一只 ts_code；与 --tickers-override-policy 配合：filter=与指数求交，replace=完全替换 universe",
    )
    parser.add_argument(
        "--tickers-override-policy",
        choices=["filter", "replace"],
        default="filter",
        help="filter：universe∩列表（默认）；replace：列表即 universe（Alpha 全市场用 replace）",
    )
    parser.set_defaults(validate=True)
    args = parser.parse_args()

    tickers_override: list[str] | None = None
    if args.tickers_file is not None:
        p = args.tickers_file.resolve()
        if not p.is_file():
            parser.error(f"--tickers-file 不是文件: {p}")
        tickers_override = _load_tickers_file(p)
        if not tickers_override:
            parser.error(f"--tickers-file 读完为空: {p}")
        log.info(
            "loaded %s tickers from %s (policy=%s)",
            len(tickers_override),
            p,
            args.tickers_override_policy,
        )

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

    if args.reverse:
        args.as_of = list(reversed(args.as_of))
        log.info(f"--reverse: 日期已逆序，从 {args.as_of[0]} 往前跑")

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
                tickers_override=tickers_override,
                tickers_override_policy=args.tickers_override_policy,
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
