"""scripts/data_pipeline/update_lake_latest.py — 把 data/lake 行情类序列补到最新交易日（T-015）.

为什么需要：lake 的回补脚本按 ``lake.load_trading_calendar()`` 决定要拉哪些日期，而该日历由
``alpha_prices_panel.parquet`` 反推——那个文件因 T-012 的下载 bug 缺 18 个真实交易日，且止于 2026-05-11。
于是 daily / adj_factor / daily_basic 等全部缺同样 18 天，并停在 5 月。

本脚本：
  1. 交易日历直接取 Tushare ``trade_cal``（SSE, is_open=1），另存 ``data/lake/trade_cal_sse.parquet``
  2. 对每个序列，目标日期 = 日历 ∩ [起点, 终点] − lake 中**已有数据的日期**（不看 checkpoint）
  3. 只补缺失日期；新行对齐 lake 现有列与类型；``write_lake`` 的 keep='last' 不会替换任何已有日期的行
  4. 交易日返回空表 = 失败（记录、不算完成，下次运行自动重试；T-012 的教训）
  5. 每个序列第一次写入前，把原文件复制到 ``data/lake/_backup_<YYYYMMDD>/``
  6. 运行报告写 ``data/lake/_checkpoints/update_latest_<时间>.json``

安全：token 由 ``backfill_tushare._load_token_into_env`` 从 api_key.txt 读入环境变量，走官方端点，
全程不回显、不落盘；loguru sink 全部移除。

用法（仓库根目录）：
  python scripts/data_pipeline/update_lake_latest.py --plan          # 只取日历并列出待补日期，不拉数据
  python scripts/data_pipeline/update_lake_latest.py                 # 全部序列补到最新
  python scripts/data_pipeline/update_lake_latest.py --series daily adj_factor --end 2026-10-09
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.backfill_tushare as bf  # noqa: E402  (import 时已屏蔽日志，防 token 泄漏)
from quantmind.data import lake  # noqa: E402

START = "2019-01-02"


def set_data_root(root: Path) -> None:
    """让 lake 与 token 读取都指向 ``root``（在 worktree 里运行、数据在主 checkout 时使用）."""
    lake.DATA_LAKE_DIR = root / "data" / "lake"
    lake.CHECKPOINT_DIR = lake.DATA_LAKE_DIR / "_checkpoints"
    lake.PRICE_PANEL_PATH = root / "data" / "raw" / "alpha_prices_panel.parquet"
    bf.REPO_ROOT = str(root)


def cal_path() -> Path:
    return lake.DATA_LAKE_DIR / "trade_cal_sse.parquet"

# series → (tushare api, 取数方式)。by_date：每个交易日一次全市场；by_code：每个代码一次区间
SERIES: dict[str, tuple[str, str]] = {
    "daily": ("daily", "by_date"),
    "adj_factor": ("adj_factor", "by_date"),
    "daily_basic": ("daily_basic", "by_date"),
    "margin": ("margin_detail", "by_date"),
    "north_bound": ("moneyflow_hsgt", "by_date"),
    "index_daily": ("index_daily", "by_code"),
}
# 这些序列在任何交易日都应有数据：空表一律当失败
MUST_HAVE_ROWS = {"daily", "adj_factor", "daily_basic"}


def _tp():
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger as _lg
        _lg.remove()  # provider import 会重注册 sink，再清一次
    except Exception:  # noqa: BLE001
        pass
    return tp


def fetch_calendar(tp, end: str) -> pd.DatetimeIndex:
    raw = tp._call("trade_cal", exchange="SSE", start_date="20190101",
                   end_date=end.replace("-", ""), is_open="1")
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(raw["cal_date"].astype(str)).unique()))
    if len(cal) == 0:
        raise SystemExit("trade_cal 返回空，停止")
    return cal


def present_dates(series: str) -> set[pd.Timestamp]:
    p = lake.lake_path(series)
    if not p.exists():
        return set()
    return set(pd.to_datetime(pd.read_parquet(p, columns=["trade_date"])["trade_date"]).unique())


def missing_dates(cal: pd.DatetimeIndex, present: set, start: str, end: str) -> list[pd.Timestamp]:
    sel = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    return [d for d in sel if d not in present]


def align_to_lake(series: str, df: pd.DataFrame) -> pd.DataFrame:
    """新数据对齐到 lake 现有 schema：同列、同顺序、同类型；多余列丢弃，缺列补 NA."""
    import pyarrow.parquet as pq

    df = df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    sch = pq.read_schema(lake.lake_path(series))
    if "ticker" in sch.names and "ticker" not in df.columns and "ts_code" in df.columns:
        df["ticker"] = df["ts_code"].map(bf._safe_ticker)
    out = pd.DataFrame(index=df.index)
    for name, typ in zip(sch.names, sch.types, strict=True):
        col = df[name] if name in df.columns else pd.Series(pd.NA, index=df.index)
        t = str(typ)
        if t == "string":
            out[name] = col.map(lambda v: None if pd.isna(v) else str(v))   # moneyflow_hsgt 本身返回字符串
        elif t.startswith("timestamp"):
            out[name] = pd.to_datetime(col)
        else:
            out[name] = pd.to_numeric(col, errors="coerce").astype("float64" if t == "double" else t)
    return out.reset_index(drop=True)


def backup_once(series: str, backup_dir: Path) -> None:
    src = lake.lake_path(series)
    dst = backup_dir / src.name
    if src.exists() and not dst.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  [backup] {src.name} → {dst}")


def write_append_arrow(series: str, df: pd.DataFrame, path: Path | None = None) -> int:
    """只追加新日期时的省内存写法（2026-10-22：daily_basic 约 900 万行，lake.write_lake 用 pandas 整表读入、合并、
    去重、排序，峰值超过 6 GB，每日影子运行被 cgroup 杀掉）。结果与 write_lake 相同：
    - 新数据先在自身内部按主键去重（keep="last"）；
    - 新日期与表里已有日期有重叠时，退回 lake.write_lake（那时才需要整表去重）；
    - 合并后按主键排序，原子写（临时文件 + os.replace），保留原文件的 schema 元数据。"""
    import os

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    p = Path(path) if path else lake.lake_path(series)
    key = lake.lake_key(series)
    if not p.exists() or "trade_date" not in key:
        return lake.write_lake(series, df)
    new = df.copy()
    new["trade_date"] = pd.to_datetime(new["trade_date"])
    new = new.drop_duplicates(subset=key, keep="last")
    old = pq.read_table(p)
    old_dates = pd.to_datetime(pc.unique(old["trade_date"]).to_pandas())
    if pd.Series(new["trade_date"].unique()).isin(old_dates).any():
        del old
        return lake.write_lake(series, df)
    nt = pa.Table.from_pandas(new[old.schema.names], schema=old.schema, preserve_index=False)
    merged = pa.concat_tables([old, nt]).sort_by([(k, "ascending") for k in key])
    del old, nt
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(merged, tmp)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()
    return merged.num_rows


def update_by_date(tp, series: str, api: str, todo: list[pd.Timestamp], backup_dir: Path,
                   flush_every: int) -> dict:
    report = {"series": series, "todo": len(todo), "filled": [], "empty": [], "failed": [], "rows": 0}
    buf: list[pd.DataFrame] = []

    def flush():
        if buf:
            backup_once(series, backup_dir)
            total = write_append_arrow(series, pd.concat(buf, ignore_index=True))
            print(f"  [{series}] flush → lake 行数 {total:,}", flush=True)
            buf.clear()

    for i, d in enumerate(todo, 1):
        ds = d.strftime("%Y%m%d")
        try:
            raw = tp._call(api, trade_date=ds)
        except Exception as e:  # noqa: BLE001
            report["failed"].append({"date": ds, "error": repr(e)[:160]})
            print(f"  [{series}] FAIL {ds}: {repr(e)[:100]}", flush=True)
            continue
        if raw is None or len(raw) == 0:
            report["empty"].append(ds)
            continue
        buf.append(align_to_lake(series, raw))
        report["filled"].append(ds)
        report["rows"] += len(raw)
        if len(buf) >= flush_every:
            flush()
        if i % 20 == 0:
            print(f"  [{series}] {i}/{len(todo)}", flush=True)
    flush()
    if series in MUST_HAVE_ROWS and report["empty"]:
        print(f"  [{series}] ⚠ {len(report['empty'])} 个交易日返回空表，视为失败、未写入，下次运行会重试", flush=True)
    return report


def update_index_daily(tp, todo: list[pd.Timestamp], backup_dir: Path) -> dict:
    report = {"series": "index_daily", "todo": len(todo), "filled": [], "empty": [], "failed": [], "rows": 0}
    if not todo:
        return report
    codes = sorted(pd.read_parquet(lake.lake_path("index_daily"), columns=["ts_code"])["ts_code"].unique())
    lo, hi = min(todo).strftime("%Y%m%d"), max(todo).strftime("%Y%m%d")
    want = set(todo)
    parts = []
    for c in codes:
        try:
            raw = tp._call("index_daily", ts_code=c, start_date=lo, end_date=hi)
        except Exception as e:  # noqa: BLE001
            report["failed"].append({"code": c, "error": repr(e)[:160]})
            continue
        if raw is None or len(raw) == 0:
            report["empty"].append(c)
            continue
        df = align_to_lake("index_daily", raw)
        df = df[df["trade_date"].isin(want)]
        parts.append(df)
        report["rows"] += len(df)
    if parts:
        backup_once("index_daily", backup_dir)
        lake.write_lake("index_daily", pd.concat(parts, ignore_index=True))
        report["filled"] = sorted({d.strftime("%Y%m%d") for p in parts for d in p["trade_date"]})
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", nargs="*", default=list(SERIES))
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--plan", action="store_true", help="只取交易日历、列出待补日期，不拉序列数据")
    ap.add_argument("--rate-interval", type=float, default=0.5)
    ap.add_argument("--flush-every", type=int, default=40)
    ap.add_argument("--data-root", type=Path, default=REPO)
    a = ap.parse_args(argv)
    unknown = set(a.series) - set(SERIES)
    if unknown:
        print(f"unknown series: {sorted(unknown)}")
        return 2
    set_data_root(a.data_root.resolve())

    bf._load_token_into_env()
    tp = _tp()
    tp._MIN_INTERVAL = max(tp._MIN_INTERVAL, a.rate_interval)
    cal = fetch_calendar(tp, a.end)
    print(f"[calendar] trade_cal SSE {cal[0].date()} → {cal[-1].date()}，{len(cal)} 个交易日")
    if not a.plan:
        pd.DataFrame({"cal_date": cal}).to_parquet(cal_path(), index=False)
        print(f"[calendar] saved {cal_path()}")

    stamp = datetime.now().strftime("%Y%m%d")
    backup_dir = lake.DATA_LAKE_DIR / f"_backup_{stamp}"
    reports = []
    for s in a.series:
        api, mode = SERIES[s]
        todo = missing_dates(cal, present_dates(s), START, a.end)
        gaps = [d for d in todo if d <= max(present_dates(s) or {pd.Timestamp(START)})]
        print(f"[{s}] 待补 {len(todo)} 天（其中历史空洞 {len(gaps)} 天："
              f"{[d.strftime('%Y-%m-%d') for d in gaps][:20]}）", flush=True)
        if a.plan:
            reports.append({"series": s, "todo": len(todo), "gaps": [d.strftime("%Y%m%d") for d in gaps]})
            continue
        if mode == "by_date":
            reports.append(update_by_date(tp, s, api, todo, backup_dir, a.flush_every))
        else:
            reports.append(update_index_daily(tp, todo, backup_dir))
        r = reports[-1]
        print(f"[{s}] 完成：补入 {len(r['filled'])} 天、空表 {len(r['empty'])}、失败 {len(r['failed'])}", flush=True)

    out = lake.CHECKPOINT_DIR / f"update_latest_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"end": a.end, "plan": a.plan, "calendar_days": len(cal),
                               "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
