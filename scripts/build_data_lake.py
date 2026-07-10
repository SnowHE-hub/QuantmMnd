"""scripts/build_data_lake.py — 从 data/snapshots/* 去重并集建数据湖表.

建表（5 张，daily_basic 不在此处——它无快照，由 backfill_tushare.py 纯回补）:
    index_daily / north_bound / margin / hk_hold / stock_basic

用法:
    python scripts/build_data_lake.py            # 建全部并自检
    python scripts/build_data_lake.py --check    # 仅打印现有湖表覆盖&缺口

幂等：重跑安全（write_lake 按 key 去重 merge）。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

# 允许直接 python scripts/xxx.py 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantmind.data.lake import (  # noqa: E402
    DATA_LAKE_DIR,
    coverage,
    lake_key,
    lake_path,
    write_lake,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_GLOB = os.path.join(REPO_ROOT, "data", "snapshots", "*")

# 从快照建的序列（daily_basic 排除：历史上 merge 在 prices.parquet，无独立文件）
SNAPSHOT_SERIES = ("index_daily", "north_bound", "margin", "hk_hold", "stock_basic")


def _snapshot_dirs() -> list[str]:
    """升序的快照目录（升序 → write_lake keep='last' 取最新快照版本）。"""
    return sorted(d for d in glob.glob(SNAPSHOT_GLOB) if os.path.isdir(d))


def build_series(series: str) -> dict:
    """读所有快照中该序列文件，并集去重写入湖表。"""
    frames = []
    n_files = 0
    for d in _snapshot_dirs():
        fp = os.path.join(d, f"{series}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] 读取失败 {fp}: {e}")
            continue
        if df is None or len(df) == 0:
            continue
        frames.append(df)
        n_files += 1
    if not frames:
        print(f"[{series}] 无快照文件，跳过")
        return {"series": series, "snapshot_files": 0}

    # 统一列并集（不同快照列可能略有差异），保留原生列
    union = pd.concat(frames, ignore_index=True, sort=False)
    rows = write_lake(series, union)
    print(f"[{series}] 快照文件={n_files} 并集行={len(union)} → 湖表行={rows} (key={lake_key(series)})")
    return {"series": series, "snapshot_files": n_files, "lake_rows": rows}


def print_coverage(series: str) -> None:
    cov = coverage(series)
    if not cov.get("exists"):
        print(f"  [{series}] 湖表不存在")
        return
    if "unique_days" not in cov:  # stock_basic
        print(f"  [{series}] rows={cov['rows']} unique_ts_code={cov.get('unique_ts_code')}（静态表，无 trade_date）")
        return
    print(
        f"  [{series}] rows={cov['rows']} days={cov['unique_days']} "
        f"ts_code={cov.get('unique_ts_code','-')} "
        f"{cov.get('start')}→{cov.get('end')} "
        f"覆盖={cov.get('coverage_pct_vs_calendar')}% "
        f"缺失交易日={cov.get('missing_trading_days')} "
        f"(区间内日历={cov.get('calendar_days_in_span')}) "
        f"最大自然日间隔={cov.get('max_gap_calendar_days')}d"
    )
    if cov.get("missing_trading_days"):
        print(f"      缺口范围: {cov.get('missing_first')} … {cov.get('missing_last')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="仅打印覆盖，不重建")
    ap.add_argument("--series", nargs="*", default=list(SNAPSHOT_SERIES))
    args = ap.parse_args()

    DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
    series_list = [s for s in args.series if s in SNAPSHOT_SERIES]

    if not args.check:
        print("=== 建湖：从快照并集 ===")
        for s in series_list:
            build_series(s)

    print("\n=== 自检：覆盖 & 缺口（日历以 alpha_prices_panel 为准）===")
    for s in series_list:
        print_coverage(s)
    # 顺带显示 daily_basic（若已回补）
    if os.path.exists(lake_path("daily_basic")):
        print_coverage("daily_basic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
