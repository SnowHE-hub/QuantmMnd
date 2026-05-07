"""quantmind.data — 数据层（Provider + Universe + Snapshot）.

公开 API
========

- ``DataProvider``：抽象基类（所有数据源都继承）
- ``AkshareProvider`` / ``TushareProvider``：两个具体实现
- ``get_universe`` / ``get_universe_with_weights`` / ``cross_validate_universe``
- ``build_snapshot`` / ``load_snapshot`` / ``list_snapshots``
- 几个工具函数：``normalize_ticker`` / ``to_akshare_symbol`` 等

约定
====
1. 所有 ticker 内部都标准化为 ``XXXXXX.SH/SZ/BJ``（用 ``normalize_ticker``）。
2. 所有「带 PIT 风险」的方法都接受 ``as_of: date | None``。
3. 财务数据 PIT 用 ``f_ann_date``（实际公告日，含修订）。
4. 行情数据 PIT 用 ``trade_date``。
"""

from quantmind.data.akshare_provider import AkshareProvider
from quantmind.data.base import (
    Adjustment,
    DataProvider,
    DataProviderError,
    Frequency,
    PITViolationError,
    StatementType,
    normalize_ticker,
    to_akshare_em_symbol,
    to_akshare_symbol,
    to_tushare_code,
)
from quantmind.data.snapshot import (
    build_snapshot,
    list_snapshots,
    load_snapshot,
    validate_snapshot,
)
from quantmind.data.sse_calendar import list_sse_trade_dates, monthly_last_trade_days
from quantmind.data.tushare_provider import TushareProvider
from quantmind.data.universe import (
    INDEX_ALIASES,
    cross_validate_universe,
    get_universe,
    get_universe_with_weights,
)

__all__ = [
    "INDEX_ALIASES",
    "Adjustment",
    "AkshareProvider",
    "DataProvider",
    "DataProviderError",
    "Frequency",
    "PITViolationError",
    "StatementType",
    "TushareProvider",
    "build_snapshot",
    "cross_validate_universe",
    "get_universe",
    "get_universe_with_weights",
    "list_snapshots",
    "list_sse_trade_dates",
    "load_snapshot",
    "monthly_last_trade_days",
    "validate_snapshot",
    "normalize_ticker",
    "to_akshare_em_symbol",
    "to_akshare_symbol",
    "to_tushare_code",
]
