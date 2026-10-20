"""quantmind.line_a.snapshot — F0 数据快照与时点闸门（03_final_plan_v2 第 5 节 F0）.

对一个决策日 as_of：
1. 列出本次允许使用的数据文件，记录 sha256、行数、水位（日期列的最大值）；
2. 关键数据（行情、复权因子、每日指标、涨跌停价、停复牌）必须覆盖 as_of，否则 data_ready = False；
3. 其余来源报告水位滞后（例如事件表还停在更早的日期），不阻断，但写进清单；
4. 清单写到 data/line_a_v2/snapshots/<as_of>/snapshot_<内容哈希前 16 位>.json（原子写）。
   输入相同就复用已有清单；数据补齐后再跑，会得到另一个内容哈希的新清单，旧清单保留作记录。

用法：
    PYTHONPATH=. python -m quantmind.line_a.snapshot --data-root /home/lenovo/projects/quantmind --as-of 2026-10-19
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

from quantmind.line_a.atomic import atomic_write_json
from quantmind.line_a.contracts import SnapshotManifest, snapshot_input

TZ = ZoneInfo("Asia/Shanghai")

# 名称 → (相对 data/ 的路径或 glob, 日期列, 是否关键)
LAKE_TABLES: dict[str, tuple[str, str, bool]] = {
    "trade_cal": ("lake/trade_cal_sse.parquet", "cal_date", True),
    "daily": ("lake/daily.parquet", "trade_date", True),
    "adj_factor": ("lake/adj_factor.parquet", "trade_date", True),
    "daily_basic": ("lake/daily_basic.parquet", "trade_date", True),
    "index_daily": ("lake/index_daily.parquet", "trade_date", False),
    "margin": ("lake/margin.parquet", "trade_date", False),
    "north_bound": ("lake/north_bound.parquet", "trade_date", False),
    "income": ("lake/income.parquet", "ann_date", False),
    "balancesheet": ("lake/balancesheet.parquet", "ann_date", False),
    "cashflow": ("lake/cashflow.parquet", "ann_date", False),
    "fina_indicator_full": ("lake/fina_indicator_full.parquet", "ann_date", False),
    "stock_basic_full": ("lake/stock_basic_full_2*.parquet", "", False),
}
# 事件表：取最新的 <api>_<YYYYMMDD>.parquet；关键的两张单独标出
EVENT_DATE_COLS = {
    "stk_limit": ("trade_date", True), "suspend_d": ("trade_date", True),
    "index_member_all": ("in_date", False), "index_weight": ("trade_date", False), "namechange": ("ann_date", False),
    "forecast": ("ann_date", False), "express": ("ann_date", False), "disclosure_date": ("end_date", False),
    "stk_holdernumber": ("ann_date", False), "stk_holdertrade": ("ann_date", False), "repurchase": ("ann_date", False),
    "share_float": ("ann_date", False), "moneyflow": ("trade_date", False), "top_list": ("trade_date", False),
    "top_inst": ("trade_date", False), "block_trade": ("trade_date", False),
    "pledge_stat": ("end_date", False), "fina_audit": ("ann_date", False), "dividend": ("ann_date", False),
}


def _latest(data: Path, pattern: str) -> Path | None:
    files = sorted(data.glob(pattern))
    return files[-1] if files else None


def _scan_dates(path: Path, col: str, ymd: str) -> tuple[str | None, int, bool]:
    """一次读取日期列：返回 (水位, 行数, 是否包含 ymd)。"""
    pf = pq.ParquetFile(path)
    rows = pf.metadata.num_rows
    if not col or col not in pf.schema_arrow.names:
        return None, rows, False
    s = pd.Series(pd.read_parquet(path, columns=[col])[col].dropna().astype(str).unique())
    s = s.str.replace("-", "").str[:8]
    s = s[s.str.fullmatch(r"\d{8}")]
    return (s.max() if len(s) else None), rows, bool((s == ymd).any())


def build(data_root: Path, as_of: str, cutoff: time = time(21, 0)) -> tuple[SnapshotManifest, dict]:
    data = Path(data_root) / "data"
    ymd = as_of.replace("-", "")
    inputs, report = [], {"critical_missing": [], "stale": {}, "absent": []}
    entries: list[tuple[str, Path | None, str, bool]] = []
    for name, (rel, col, critical) in LAKE_TABLES.items():
        p = _latest(data, rel) if "*" in rel else (data / rel if (data / rel).exists() else None)
        entries.append((name, p, col, critical))
    for api, (col, critical) in EVENT_DATE_COLS.items():
        entries.append((api, _latest(data, f"lake/events/{api}_2*.parquet"), col, critical))
    for name, p, col, critical in entries:
        if p is None:
            report["absent"].append(name)
            if critical:
                report["critical_missing"].append(f"{name}: file absent")
            continue
        wm, rows, has_day = _scan_dates(p, col, ymd)
        inputs.append(snapshot_input(name, p, rows=rows, watermark=wm))
        if critical and col and not has_day:
            report["critical_missing"].append(f"{name}: no rows for {ymd} (watermark {wm})")
        elif wm and col in ("trade_date", "cal_date") and wm < ymd:
            report["stale"][name] = wm
    manifest = SnapshotManifest(
        as_of=as_of, data_cutoff_ts=datetime.combine(pd.Timestamp(as_of).date(), cutoff, tzinfo=TZ),
        created_utc=datetime.now(UTC), inputs=inputs,
        notes=[f"data_ready={not report['critical_missing']}"] + [f"critical: {m}" for m in report["critical_missing"]]
              + [f"stale: {k} watermark {v}" for k, v in sorted(report["stale"].items())]
              + [f"absent: {n}" for n in report["absent"]])
    report["data_ready"] = not report["critical_missing"]
    return manifest, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--as-of", required=True)
    a = ap.parse_args()
    manifest, report = build(a.data_root, a.as_of)
    ch = manifest.content_hash()
    out = Path(a.data_root) / "data/line_a_v2/snapshots" / a.as_of / f"snapshot_{ch[:16]}.json"
    reused = out.exists()          # 内容相同（输入文件与哈希都相同）的清单已存在：直接复用，不覆盖
    sha = None if reused else atomic_write_json(out, json.loads(manifest.model_dump_json()))
    print(json.dumps({"out": str(out), "reused": reused, "sha256": sha, "content_hash": ch, **report},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
