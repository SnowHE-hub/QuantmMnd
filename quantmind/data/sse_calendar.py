"""quantmind.data.sse_calendar — SSE 交易日历（Tushare trade_cal）。

用于生成月度调仓日、回看窗口等训练集编排逻辑。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from quantmind.core.cache import cached
from quantmind.core.logger import get_logger

log = get_logger(__name__)


def _yyyymmdd(d: date | str) -> str:
    if isinstance(d, str):
        return d.replace("-", "")[:8]
    return d.strftime("%Y%m%d")


@cached(ttl_hours=168)  # 7d
def _raw_sse_trade_cal(start_yyyyymmdd: str, end_yyyyymmdd: str) -> pd.DataFrame:
    from quantmind.data.tushare_provider import _call  # noqa: PLC0415

    log.debug(f"[sse_calendar] trade_cal SSE {start_yyyyymmdd}~{end_yyyyymmdd}")
    return _call("trade_cal", exchange="SSE", start_date=start_yyyyymmdd, end_date=end_yyyyymmdd)


def list_sse_trade_dates(start: date | str, end: date | str) -> list[date]:
    """返回 ``[start, end]`` 内所有 SSE 交易日的 ``date`` 列表（有序）."""
    s = datetime.strptime(str(start)[:10], "%Y-%m-%d").date()
    e = datetime.strptime(str(end)[:10], "%Y-%m-%d").date()
    if e < s:
        return []
    raw = _raw_sse_trade_cal(_yyyymmdd(s), _yyyymmdd(e))
    if raw is None or raw.empty:
        return []
    open_days = raw[raw["is_open"] == 1].copy()
    open_days["_d"] = pd.to_datetime(open_days["cal_date"], format="%Y%m%d").dt.date
    return sorted(open_days["_d"].tolist())


def monthly_last_trade_days(start: date, end: date) -> list[date]:
    """每个自然月选取「不超过该月最后日历日」的最后一笔 SSE 交易日（常见月频调仓）.

    若某月在 ``[start, end]`` 内无交集则跳过。
    """
    if end < start:
        return []
    buf_start = start - timedelta(days=7)
    buf_end = end + timedelta(days=7)
    days = list_sse_trade_dates(buf_start, buf_end)
    if not days:
        return []

    months = pd.period_range(start=start, end=end, freq="M")
    out: list[date] = []
    for pm in months:
        me = pm.to_timestamp(how="end").date()
        # last trading day ≤ month-end calendar date
        candidates = [x for x in days if start <= x <= end and x <= me]
        if not candidates:
            continue
        out.append(max(candidates))
    # consecutive months should not dup; hedge
    uniq: list[date] = []
    seen = set()
    for d in sorted(set(out)):
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


__all__ = ["list_sse_trade_dates", "monthly_last_trade_days"]
