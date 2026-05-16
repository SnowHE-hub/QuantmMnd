#!/usr/bin/env python3
"""Data Expansion v1：构建 snapshot（小样本或完整 CSI300）。

小样本：用于验证 Tushare 接口（默认不关财报以节省时间，可用 --with-financials）。
完整：与生产一致 — 财报三表 + fina_indicator + v1 四表；需 --overwrite。

用法::

    # 小样本（勿在已要保留的日期上使用 --overwrite）
    python scripts/backfill_snapshot_v1.py --as-of 2024-09-30 --smoke --max-tickers 5 --overwrite

    # 单日完整 CSI300（2024-12-31）
    python scripts/backfill_snapshot_v1.py --as-of 2024-12-31 --full --overwrite

说明：不打印任何密钥。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantmind.core.logger import setup_logger
from quantmind.data.snapshot import build_snapshot


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def main() -> int:
    setup_logger()
    # 重定向到文件时便于 tail；不影响功能
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    p = argparse.ArgumentParser(description="Backfill snapshot Data Expansion v1")
    p.add_argument("--as-of", type=_parse_date, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="仅 max_tickers 只票；默认不含财报（快）")
    g.add_argument("--full", action="store_true", help="完整 universe，含财报三表+fina_indicator")
    p.add_argument("--max-tickers", type=int, default=5)
    p.add_argument("--universe", type=str, default="csi300")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--with-financials",
        action="store_true",
        help="与 --smoke 合用：仍限制只数但拉财报与 fina_indicator（较慢）",
    )
    p.add_argument("--strict", action="store_true", help="snapshot_strict（默认 False）")
    args = p.parse_args()

    if args.smoke:
        meta = build_snapshot(
            args.as_of,
            universe_name=args.universe,
            max_tickers=args.max_tickers,
            include_financials=bool(args.with_financials),
            include_indicators=bool(args.with_financials),
            overwrite=args.overwrite,
            strict=args.strict,
        )
    else:
        meta = build_snapshot(
            args.as_of,
            universe_name=args.universe,
            max_tickers=None,
            include_financials=True,
            include_indicators=True,
            overwrite=args.overwrite,
            strict=args.strict if args.strict else None,
        )

    print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
