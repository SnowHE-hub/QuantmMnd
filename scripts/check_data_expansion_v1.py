#!/usr/bin/env python3
"""检查 Data Expansion v1：snapshot 中 stock_basic / hk_hold / margin / index_daily 及 manifest.

用法::

    python scripts/check_data_expansion_v1.py --as-of 2024-06-28
    python scripts/check_data_expansion_v1.py --as-of 2024-06-28 --build-smoke --max-tickers 5

说明：不打印任何密钥；依赖环境变量由 get_settings()/Tushare 自行解析。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# 项目根：scripts/ -> parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantmind.core.config import get_settings
from quantmind.core.logger import setup_logger
from quantmind.data.snapshot import build_snapshot, load_snapshot


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _pit_ok_trade_date(df: pd.DataFrame | None, as_of: date, col: str = "trade_date") -> tuple[bool, str]:
    if df is None or df.empty or col not in df.columns:
        return True, "n/a (empty or no column)"
    ts = pd.to_datetime(df[col], errors="coerce")
    mx = ts.max()
    if pd.isna(mx):
        return True, "n/a (all NaT)"
    ok = bool(mx <= pd.Timestamp(as_of))
    return ok, f"max={mx.date()} <= as_of={as_of}"


def _print_module(name: str, df: pd.DataFrame | None, meta_mod: dict | None, as_of: date) -> None:
    print(f"\n--- {name} ---")
    if df is None:
        print("  [缺失] 未在 snapshot 中加载（无 parquet 或未列入 meta.files）")
        if meta_mod:
            print(f"  manifest 记录: rows={meta_mod.get('row_count')}")
        return
    print(f"  rows: {len(df)}")
    print(f"  columns: {list(df.columns)}")
    pit_col = "trade_date"
    if name == "stock_basic":
        pit_col = "list_date"
    if pit_col in df.columns:
        s = pd.to_datetime(df[pit_col], errors="coerce")
        print(f"  {pit_col} min: {s.min()}  max: {s.max()}")
    if name == "stock_basic":
        ok, msg = (
            True,
            "static metadata; list_date bounds only",
        )
        print(f"  PIT (trade_date): {msg}")
    else:
        ok, msg = _pit_ok_trade_date(df, as_of, "trade_date")
        print(f"  PIT (trade_date <= as_of): {'PASS' if ok else 'FAIL'}  ({msg})")
    if meta_mod:
        print(
            f"  meta.modules[{name}]: rows={meta_mod.get('row_count')}, "
            f"schema={meta_mod.get('schema_version')}, "
            f"pit_col={meta_mod.get('pit_date_column')}"
        )


def main() -> int:
    setup_logger()
    parser = argparse.ArgumentParser(description="Data Expansion v1 snapshot 检查报告")
    parser.add_argument("--as-of", type=_parse_date, required=True, help="snapshot 日期 YYYY-MM-DD")
    parser.add_argument(
        "--build-smoke",
        action="store_true",
        help="若目录不存在则构建小样本 snapshot（限 max_tickers，不关财报）",
    )
    parser.add_argument("--max-tickers", type=int, default=5, help="与 --build-smoke 合用")
    parser.add_argument("--universe", type=str, default="csi300")
    parser.add_argument("--overwrite", action="store_true", help="构建时覆盖已存在目录")
    args = parser.parse_args()

    as_of = args.as_of
    snap_dir = Path(get_settings().data.dir) / "snapshots" / as_of.isoformat()

    if args.build_smoke and (not (snap_dir / "meta.json").exists() or args.overwrite):
        print(">>> 构建 smoke snapshot（不拉全市场成分，仅 max_tickers）...")
        meta = build_snapshot(
            as_of,
            universe_name=args.universe,
            max_tickers=args.max_tickers,
            include_financials=False,
            include_indicators=False,
            overwrite=args.overwrite,
            strict=False,
        )
        print(f">>> 完成: {meta.get('snapshot_dir')}  rows_per_table={meta.get('rows_per_table')}")

    snap = load_snapshot(as_of)
    meta = snap["meta"]
    assert isinstance(meta, dict)

    print("=" * 72)
    print("QuantMind Data Expansion v1 — Snapshot 检查报告")
    print("=" * 72)
    print(f"as_of:           {as_of}")
    print(f"snapshot_dir:    {meta.get('snapshot_dir')}")
    print(f"data_expansion:  {meta.get('data_expansion_version', '(legacy meta)')}")
    print(f"snapshot_strict: {meta.get('snapshot_strict', meta.get('include_flags') is not None)}")
    if "include_flags" in meta:
        print(f"include_flags:   {meta['include_flags']}")
    if "modules" in meta and meta["modules"]:
        print("modules (manifest):", ", ".join(meta["modules"].keys()))

    modules_meta = meta.get("modules") or {}
    for key in ("stock_basic", "hk_hold", "margin", "index_daily"):
        _print_module(key, snap.get(key), modules_meta.get(key), as_of)

    print("\n" + "=" * 72)
    print("提示: stock_basic 为构建时静态基本信息；行业非历史沿革 PIT。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
