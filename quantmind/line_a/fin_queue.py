"""quantmind.line_a.fin_queue — 财报增量的三个更新队列（03_final_plan_v2 第 4.2 节；用户 2026-10-21 决定）.

不再按公告日一次拉全市场财报，而是"多个来源触发、再定期对账"：
- normal（常规）：窗口 (since, end] 内有下列任一事件的股票，按 ``since`` 起拉：
  财报实际披露（disclosure_date.actual_date）、业绩预告（forecast.ann_date）、业绩快报（express.ann_date）、
  审计意见（fina_audit.ann_date）、巨潮公告标题含定期报告（年度 / 半年度 / 季度报告，排除摘要以外的说明会等）。
- late_revision（迟到修订）：巨潮标题含"更正 / 补充 / 修订"且涉及报告或财务数据的股票，从 ``end`` 往前
  ``revision_lookback_days`` 天起拉，覆盖被更正的旧报告期。
- reconciliation（对账）：
  - 按股票代码哈希分成 ``buckets`` 桶，每个交易日轮到一桶（约一个月覆盖全市场一遍），从 ``end`` 往前
    ``recon_lookback_days`` 天起拉；
  - 法定披露截止日（4-30、8-31、10-31）的次日落在窗口内时，对全市场做一次对账。
同一只股票落在多个队列时取最早的起点，原因全部保留。写入仍由 update_fundamentals_latest.py 完成
（旧表一行不动、主键去重、ann_date 晚于截止日不写）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

PERIODIC = r"(?:年度报告|半年度报告|季度报告|年报|半年报|季报)"
NOT_PERIODIC = r"(?:说明会|业绩会|提示性公告|披露的提示|预约|延期披露|问询函)"
REVISION = r"(?:更正|补充|修订)"
DEADLINES = ("0430", "0831", "1031")


def _ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _latest(lake: Path, api: str) -> Path | None:
    fs = sorted((lake / "events").glob(f"{api}_2*.parquet"))
    return fs[-1] if fs else None


def _in_window(s: pd.Series, lo: str, hi: str) -> pd.Series:
    v = s.astype("string").str.slice(0, 8)
    return (v > lo) & (v <= hi)


def _bucket(ts: str, buckets: int) -> int:
    return int(hashlib.sha256(ts.encode()).hexdigest()[:8], 16) % buckets


def cninfo_titles(lake: Path, lo: str, hi: str) -> pd.DataFrame:
    """巨潮公告元数据（按日分片）在 (lo, hi] 内的 (ts_code, title)。按代码首位补交易所后缀（6 → SH，0/3 → SZ，4/8/9 → BJ）。"""
    parts = sorted((lake / "text/cninfo/_parts").glob("*.parquet"))
    parts = [p for p in parts if lo < p.stem <= hi]
    if not parts:
        return pd.DataFrame(columns=["ts_code", "title"])
    df = pd.concat([pd.read_parquet(p, columns=["secCode", "announcementTitle"]) for p in parts])
    code = df["secCode"].astype(str).str.zfill(6)
    suf = code.str[0].map({"6": ".SH", "0": ".SZ", "3": ".SZ", "4": ".BJ", "8": ".BJ", "9": ".BJ"})
    ok = suf.notna().to_numpy()
    return pd.DataFrame({"ts_code": (code + suf)[ok], "title": df["announcementTitle"].astype(str)[ok]})


def build_queues(lake: Path, since, end, universe: list[str], *, run_index: int, buckets: int = 20,
                 revision_lookback_days: int = 400, recon_lookback_days: int = 400) -> pd.DataFrame:
    """返回每只要拉的股票：ts_code、fetch_since（YYYYMMDD）、queues、reasons。run_index 决定轮到哪一桶。"""
    lo, hi = _ymd(since), _ymd(end)
    rows: list[tuple[str, str, str, str]] = []          # (ts_code, queue, reason, fetch_since)
    for api, col in (("disclosure_date", "actual_date"), ("forecast", "ann_date"), ("express", "ann_date"),
                     ("fina_audit", "ann_date")):
        p = _latest(lake, api)
        if p is None:
            continue
        df = pd.read_parquet(p, columns=["ts_code", col])
        hit = df.loc[_in_window(df[col], lo, hi), "ts_code"].unique()
        rows += [(t, "normal", f"{api}.{col}", lo) for t in hit]
    ti = cninfo_titles(lake, lo, hi)
    if len(ti):
        per = ti["title"].str.contains(PERIODIC) & ~ti["title"].str.contains(NOT_PERIODIC)
        revised = ti["title"].str.contains(REVISION)
        rev = (per & revised) | (revised & ti["title"].str.contains("财务|会计|数据"))
        rows += [(t, "normal", "cninfo.periodic_report", lo) for t in ti.loc[per & ~rev, "ts_code"].unique()]
        rev_since = _ymd(pd.Timestamp(hi) - pd.Timedelta(days=revision_lookback_days))
        rows += [(t, "late_revision", "cninfo.revision", rev_since) for t in ti.loc[rev, "ts_code"].unique()]
    rec_since = _ymd(pd.Timestamp(hi) - pd.Timedelta(days=recon_lookback_days))
    after = [_ymd(pd.Timestamp(f"{y}{md}") + pd.Timedelta(days=1)) for y in {lo[:4], hi[:4]} for md in DEADLINES]
    full_sweep = any(lo < d <= hi for d in after)
    k = run_index % buckets
    for t in universe:
        if full_sweep:
            rows.append((t, "reconciliation", "statutory_deadline_sweep", rec_since))
        elif _bucket(t, buckets) == k:
            rows.append((t, "reconciliation", f"bucket_{k}_of_{buckets}", rec_since))
    if not rows:
        return pd.DataFrame(columns=["ts_code", "fetch_since", "queues", "reasons"])
    df = pd.DataFrame(rows, columns=["ts_code", "queue", "reason", "fetch_since"])
    df = df[df["ts_code"].isin(set(universe))]
    return (df.groupby("ts_code")
              .agg(fetch_since=("fetch_since", "min"),
                   queues=("queue", lambda s: ",".join(sorted(set(s)))),
                   reasons=("reason", lambda s: ",".join(sorted(set(s)))))
              .reset_index())
