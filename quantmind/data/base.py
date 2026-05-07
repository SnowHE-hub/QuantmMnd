"""quantmind.data.base — DataProvider 抽象基类与 PIT 防护工具.

设计原则
========

1. **PIT 是第一公民**：所有方法必须接受 ``as_of`` 参数，返回前断言所有数据日期 <= ``as_of``
2. **统一 ticker 格式**：内部统一用 tushare 风格 ``XXXXXX.SH/SZ/BJ``，provider 自行转换
3. **统一列名**：所有 DataFrame 列名为英文 snake_case，通过 ``_COLUMN_MAP`` 标准化
4. **附加元信息**：返回的 DataFrame 必带 ``df.attrs = {data_source, fetched_at, as_of, ticker}``
5. **失败可追溯**：所有 API 调用经 ``operation_logger`` 记录

DataProvider 子类（akshare / tushare / yfinance）只需实现具体方法，
基类负责 PIT 校验、列名标准化、错误包装等共通逻辑。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd

from quantmind.core.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# Ticker 标准化
# ============================================================================

# 格式约定：内部统一 6 位数字 + . + 交易所后缀（SH/SZ/BJ）
_TICKER_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$", re.IGNORECASE)


def normalize_ticker(ticker: str) -> str:
    """把任意常见格式标准化为 ``XXXXXX.SH/SZ/BJ``.

    支持输入::

        '600519'              -> '600519.SH'
        '300750'              -> '300750.SZ'
        '600519.SH'           -> '600519.SH'
        '600519.sh' / 'SH600519'/ 'sh.600519' / 'sh600519' -> '600519.SH'
    """
    t = ticker.strip().upper().replace(" ", "").replace(".", "").replace("/", "")
    if t.startswith(("SH", "SZ", "BJ")):
        ex, num = t[:2], t[2:]
    elif t.endswith(("SH", "SZ", "BJ")):
        num, ex = t[:6], t[-2:]
    else:
        num = t[:6]
        if not num.isdigit() or len(num) != 6:
            raise ValueError(f"Invalid ticker: {ticker!r}")
        # 按规则推断交易所
        if num.startswith(("60", "68", "9", "5")):
            ex = "SH"
        elif num.startswith(("00", "30", "20")):
            ex = "SZ"
        elif num.startswith(("4", "8", "43", "83", "87", "92")):
            ex = "BJ"
        else:
            ex = "SZ"  # 兜底
    return f"{num}.{ex}"


def to_akshare_symbol(ticker: str) -> str:
    """转换为 akshare 习惯：纯 6 位数字（无交易所）."""
    return normalize_ticker(ticker).split(".")[0]


def to_akshare_em_symbol(ticker: str) -> str:
    """转换为 akshare 东财接口习惯：``SH600519`` / ``SZ300750``."""
    t = normalize_ticker(ticker)
    num, ex = t.split(".")
    return f"{ex}{num}"


def to_tushare_code(ticker: str) -> str:
    """转换为 tushare 标准：``600519.SH``（与本项目内部一致）."""
    return normalize_ticker(ticker)


# ============================================================================
# 通用 statement_type 枚举
# ============================================================================

StatementType = Literal["balance_sheet", "income", "cashflow"]
Frequency = Literal["D", "W", "M", "Q", "Y"]
Adjustment = Literal["none", "qfq", "hfq"]


# ============================================================================
# 列名映射表
# ============================================================================
# 把不同数据源的中文/英文混杂列名统一成项目内英文 snake_case。
# 各 provider 自行扩展。

PRICE_COLUMN_MAP: dict[str, str] = {
    # akshare
    "日期": "trade_date",
    "股票代码": "ticker",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover_rate",
    # tushare 已是英文，但需补几个
    "trade_date": "trade_date",
    "ts_code": "ticker",
    "vol": "volume",
    "pct_chg": "pct_change",
    "pre_close": "pre_close",
    "amount": "amount",
}


FINANCIAL_COLUMN_MAP: dict[str, str] = {
    # tushare 财务接口（income/balancesheet/cashflow）
    "ts_code": "ticker",
    "ann_date": "ann_date",          # 公告日期（首次）
    "f_ann_date": "f_ann_date",      # 实际公告日期（含修订）— PIT 用此字段
    "end_date": "report_date",       # 报告期
    "report_type": "report_type",
    # akshare 东财财报
    "SECUCODE": "ticker_with_ex",
    "SECURITY_CODE": "ticker",
    "SECURITY_NAME_ABBR": "name",
    "REPORT_DATE": "report_date",
    "REPORT_DATE_NAME": "report_period_name",
    "NOTICE_DATE": "f_ann_date",     # akshare 的披露日 → 统一为 f_ann_date
    "UPDATE_DATE": "update_date",
}


# ============================================================================
# DataProvider 抽象基类
# ============================================================================


class DataProviderError(Exception):
    """数据获取错误的基类."""


class PITViolationError(DataProviderError):
    """检测到 look-ahead bias：返回数据中存在 > as_of 的记录."""


class DataProvider(ABC):
    """所有数据源的抽象基类.

    子类必须实现：
        - ``name`` 类属性
        - ``get_price``、``get_financials``、``get_index_constituents``、``is_tradable``

    可选实现（基类有默认抛 NotImplementedError）：
        - ``get_news``、``get_corporate_actions``、``get_consensus_estimate``、
          ``get_north_bound_flow`` 等
    """

    name: str = "base"

    # ---- 元信息附加 ----
    def _stamp(
        self,
        df: pd.DataFrame,
        as_of: date | None,
        ticker: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """把元数据写到 ``df.attrs``."""
        df.attrs.update(
            {
                "data_source": self.name,
                "fetched_at": datetime.now().isoformat(),
                "as_of": as_of.isoformat() if isinstance(as_of, date) else None,
                "ticker": ticker,
                **(extra or {}),
            }
        )
        return df

    # ---- 列名标准化 ----
    @staticmethod
    def _rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
        """重命名列，仅处理出现的列；不存在的 key 自动跳过."""
        present = {k: v for k, v in mapping.items() if k in df.columns}
        return df.rename(columns=present)

    # ---- PIT 防护（核心） ----
    @staticmethod
    def _assert_pit(
        df: pd.DataFrame,
        as_of: date | None,
        date_cols: str | list[str],
        strict: bool = True,
    ) -> None:
        """断言 DataFrame 任何 date_cols 列均 <= as_of.

        Args:
            df: 待检查的 DataFrame
            as_of: PIT 时点；None 则跳过检查（但记 warning）
            date_cols: 单列名或列名列表；列内容应可被 ``pd.to_datetime`` 解析
            strict: True 抛 PITViolationError；False 仅 log warning

        Raises:
            PITViolationError: strict=True 且发现 date > as_of
        """
        if as_of is None:
            log.debug("_assert_pit skipped (as_of=None, full-history slice)")
            return
        if df is None or df.empty:
            return
        cols = [date_cols] if isinstance(date_cols, str) else date_cols
        for col in cols:
            if col not in df.columns:
                continue
            # 转 datetime；非法值变 NaT，跳过
            ser = pd.to_datetime(df[col], errors="coerce").dropna()
            if ser.empty:
                continue
            cutoff = pd.Timestamp(as_of)
            violations = ser[ser > cutoff]
            if len(violations) > 0:
                msg = (
                    f"[{type(df).__name__}] PIT violation in column '{col}': "
                    f"{len(violations)} rows have date > {as_of}; "
                    f"max violating date = {violations.max().date()}"
                )
                if strict:
                    raise PITViolationError(msg)
                log.warning(msg)

    @staticmethod
    def _filter_pit(
        df: pd.DataFrame,
        as_of: date | None,
        date_col: str,
    ) -> pd.DataFrame:
        """按 ``date_col <= as_of`` 过滤数据.

        典型用法：财务数据用 ``f_ann_date`` 过滤；行情数据用 ``trade_date`` 过滤。
        """
        if as_of is None or df is None or df.empty or date_col not in df.columns:
            return df
        cutoff = pd.Timestamp(as_of)
        ser = pd.to_datetime(df[date_col], errors="coerce")
        mask = ser.notna() & (ser <= cutoff)
        return df.loc[mask].copy()

    # ============================================================================
    # 抽象方法（子类必须实现）
    # ============================================================================

    @abstractmethod
    def get_price(
        self,
        ticker: str,
        start: str | date,
        end: str | date,
        as_of: date | None = None,
        freq: Frequency = "D",
        adjust: Adjustment = "qfq",
    ) -> pd.DataFrame:
        """日线行情（前复权默认）.

        返回列：
            ``trade_date, ticker, open, high, low, close, volume, amount,
            pct_change, turnover_rate``
        """
        ...

    @abstractmethod
    def get_financials(
        self,
        ticker: str,
        statement_type: StatementType,
        as_of: date | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """财务报表（PIT 严格：用披露日 ``f_ann_date`` 过滤）.

        返回必含 ``ticker, ann_date, f_ann_date, report_date, report_type`` + 报表字段
        """
        ...

    @abstractmethod
    def get_index_constituents(
        self,
        index_code: str,
        as_of: date | None = None,
    ) -> list[str]:
        """指数成分股（PIT 严格：返回 as_of 时点的真实成分股，含后退市的）."""
        ...

    @abstractmethod
    def is_tradable(self, ticker: str, on_date: date) -> bool:
        """是否在 on_date 可交易（停牌/退市 -> False）."""
        ...

    # ============================================================================
    # 可选方法（子类有就实现，没有就保留 NotImplementedError）
    # ============================================================================

    def get_news(
        self,
        ticker: str,
        start: str | date,
        end: str | date,
        as_of: date | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not implement get_news")

    def get_corporate_actions(
        self,
        ticker: str,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """分红、配股、拆股、停复牌."""
        raise NotImplementedError(f"{self.name} does not implement get_corporate_actions")

    def get_consensus_estimate(
        self,
        ticker: str,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """券商一致预期."""
        raise NotImplementedError(f"{self.name} does not implement get_consensus_estimate")

    def get_research_reports(
        self,
        ticker: str | None = None,
        as_of: date | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """研报列表."""
        raise NotImplementedError(f"{self.name} does not implement get_research_reports")

    def get_north_bound_flow(
        self,
        ticker: str | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """北向资金."""
        raise NotImplementedError(f"{self.name} does not implement get_north_bound_flow")

    def get_financial_indicators(
        self,
        ticker: str,
        as_of: date | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """已计算好的财务比率（ROE/ROA/毛利等）— tushare 独有."""
        raise NotImplementedError(f"{self.name} does not implement get_financial_indicators")

    def get_disclosure_schedule(
        self,
        period_end: str | date,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """全市场披露时间表（pre_date 预约 + actual_date 实际）."""
        raise NotImplementedError(f"{self.name} does not implement get_disclosure_schedule")

    def get_adj_factor(
        self,
        ticker: str,
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        """复权因子时间序列."""
        raise NotImplementedError(f"{self.name} does not implement get_adj_factor")

    # ============================================================================
    # 辅助：常用日期格式化
    # ============================================================================

    @staticmethod
    def _to_yyyymmdd(d: str | date | datetime) -> str:
        """统一为 ``YYYYMMDD`` 字符串（akshare/tushare 都用这个格式）."""
        if isinstance(d, str):
            return d.replace("-", "").replace("/", "")
        if isinstance(d, (date, datetime)):
            return d.strftime("%Y%m%d")
        raise TypeError(f"Cannot convert {type(d)} to YYYYMMDD")

    @staticmethod
    def _to_yyyy_dash(d: str | date | datetime) -> str:
        """统一为 ``YYYY-MM-DD`` 字符串."""
        if isinstance(d, str):
            if len(d) == 8 and d.isdigit():
                return f"{d[:4]}-{d[4:6]}-{d[6:]}"
            return d
        if isinstance(d, (date, datetime)):
            return d.strftime("%Y-%m-%d")
        raise TypeError(f"Cannot convert {type(d)} to YYYY-MM-DD")


__all__ = [
    "FINANCIAL_COLUMN_MAP",
    "PRICE_COLUMN_MAP",
    "Adjustment",
    "DataProvider",
    "DataProviderError",
    "Frequency",
    "PITViolationError",
    "StatementType",
    "normalize_ticker",
    "to_akshare_em_symbol",
    "to_akshare_symbol",
    "to_tushare_code",
]


if __name__ == "__main__":
    # 简单自测
    print("--- ticker normalization ---")
    for t in ["600519", "300750", "600519.SH", "sh600519", "SZ300750", "sz.000001"]:
        print(f"  {t:20s} -> {normalize_ticker(t)}")

    print("\n--- akshare symbol ---")
    print(f"  300750.SZ -> {to_akshare_symbol('300750.SZ')}")
    print(f"  300750.SZ -> {to_akshare_em_symbol('300750.SZ')}")

    print("\n--- PIT check (positive case) ---")
    df_ok = pd.DataFrame({"f_ann_date": ["2024-04-15", "2024-07-30"]})
    DataProvider._assert_pit(df_ok, date(2024, 8, 1), "f_ann_date")
    print("  no violations (as_of=2024-08-01)")

    print("\n--- PIT check (violation) ---")
    try:
        df_bad = pd.DataFrame({"f_ann_date": ["2024-04-15", "2025-01-01"]})
        DataProvider._assert_pit(df_bad, date(2024, 8, 1), "f_ann_date")
    except PITViolationError as e:
        print(f"  ✓ Raised: {e}")
