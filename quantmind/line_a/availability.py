"""quantmind.line_a.availability — 按来源计算数据的可用时点（configs/line_a_v2/source_semantics.yaml）.

- 精确到分钟的来源（巨潮发布时间）：可用时点 = 发布时间；
- 只有日期的来源（财报公告日、研报日期、事件表的公告日）：同日夜间决策不用，
  可用时点 = 该日期之后第一个交易日的数据截止时刻；
- 交易日行情：可用时点 = 当日的数据截止时刻（收盘后批量发布）。
所有时间都按北京时间处理。
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_CUTOFF = time(21, 0)


class TradingCalendar:
    def __init__(self, trade_days: list[date] | pd.DatetimeIndex, cutoff: time = DEFAULT_CUTOFF):
        self.days = sorted(pd.Timestamp(d).date() for d in trade_days)
        self.cutoff = cutoff

    def cutoff_at(self, d: date) -> datetime:
        return datetime.combine(d, self.cutoff, tzinfo=TZ)

    def next_trading_day_after(self, d: date) -> date:
        i = bisect_right(self.days, d)
        if i >= len(self.days):
            raise ValueError(f"calendar ends before the trading day after {d}")
        return self.days[i]

    def is_trading_day(self, d: date) -> bool:
        i = bisect_right(self.days, d)
        return i > 0 and self.days[i - 1] == d


def available_at_minute(published_at: datetime) -> datetime:
    """有精确发布时间的数据。没有时区的时间按北京时间解释。"""
    return published_at if published_at.tzinfo else published_at.replace(tzinfo=TZ)


def available_at_date_only(d: date, cal: TradingCalendar) -> datetime:
    """只有日期的数据：该日期之后第一个交易日的截止时刻。"""
    return cal.cutoff_at(cal.next_trading_day_after(d))


def available_at_trade_bar(d: date, cal: TradingCalendar) -> datetime:
    """交易日 d 的行情：当日截止时刻。"""
    if not cal.is_trading_day(d):
        raise ValueError(f"{d} is not a trading day")
    return cal.cutoff_at(d)
