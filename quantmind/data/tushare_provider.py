"""quantmind.data.tushare_provider — Tushare DataProvider 实现.

特点：
    - 付费（积分制）：本项目使用 2000 分套餐
    - PIT：财报数据用 ``f_ann_date``（实际公告日期，含修订），优于 akshare 的 NOTICE_DATE
    - 速度：比 akshare 快 5-30 倍
    - 适合：财报、披露日、历史成分股、北向资金、复权因子、财务比率

可用接口（2000 分内）：
    stock_basic, trade_cal, daily, daily_basic, income, balancesheet, cashflow,
    fina_indicator, fina_audit, disclosure_date, index_weight, index_basic,
    index_daily, moneyflow_hsgt, hsgt_top10, top_list, adj_factor, suspend_d,
    namechange, stk_holdernumber, forecast, express
"""

from __future__ import annotations

import os
import threading
import time
from datetime import date

import pandas as pd

from quantmind.core.cache import cached
from quantmind.core.logger import get_logger, operation_logger
from quantmind.data.base import (
    FINANCIAL_COLUMN_MAP,
    PRICE_COLUMN_MAP,
    Adjustment,
    DataProvider,
    DataProviderError,
    Frequency,
    StatementType,
    normalize_ticker,
    to_tushare_code,
)

log = get_logger(__name__)


# ============================================================================
# Tushare 客户端单例（懒加载）
# ============================================================================

_tushare_lock = threading.Lock()
_tushare_pro = None  # type: ignore[var-annotated]


def _get_tushare_pro():
    """懒加载并缓存 tushare pro_api 客户端."""
    global _tushare_pro
    if _tushare_pro is not None:
        return _tushare_pro
    with _tushare_lock:
        if _tushare_pro is not None:
            return _tushare_pro
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            raise DataProviderError("TUSHARE_TOKEN env var is empty")
        try:
            import tushare as ts  # type: ignore[import-untyped]
        except ImportError as e:
            raise DataProviderError("tushare not installed; pip install tushare") from e
        ts.set_token(token)
        _tushare_pro = ts.pro_api()
        log.info("tushare pro_api initialized")
        return _tushare_pro


# ============================================================================
# 限频与重试
# ============================================================================

# 2000 分账号通用限频：每分钟 ~500 次（保守取 200 次/分钟 = 0.3s/call）
_MIN_INTERVAL = 0.25  # 秒
_last_call_ts = 0.0
_rate_lock = threading.Lock()


def _rate_limit() -> None:
    """简单的最小间隔限频."""
    global _last_call_ts
    with _rate_lock:
        elapsed = time.monotonic() - _last_call_ts
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _last_call_ts = time.monotonic()


_TRANSIENT_KEYWORDS = (
    "频率",
    "rate",
    "limit",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "503",
    "504",
    "502",
)


def _call(api_name: str, max_attempts: int = 4, **kwargs):
    """调用 tushare API（带限频 + 重试 + 超时退避）."""
    pro = _get_tushare_pro()
    fn = getattr(pro, api_name)
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        _rate_limit()
        try:
            return fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            msg = str(e).lower()
            transient = any(k in msg for k in _TRANSIENT_KEYWORDS)
            if transient and attempt < max_attempts - 1:
                wait = 2 ** attempt + 1  # 1, 3, 5, 9 ...
                log.warning(
                    f"tushare {api_name} transient error "
                    f"(attempt {attempt + 1}/{max_attempts}): {str(e)[:100]}; sleep {wait}s"
                )
                time.sleep(wait)
                continue
            raise DataProviderError(f"tushare {api_name} failed: {e}") from e
    raise DataProviderError(
        f"tushare {api_name} failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


# ============================================================================
# Raw fetchers (cached)
# ============================================================================


@cached(ttl_hours=24)
def _raw_daily(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch daily {ts_code} {start}~{end}")
    return _call("daily", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=24)
def _raw_daily_basic(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch daily_basic {ts_code} {start}~{end}")
    return _call("daily_basic", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=24)
def _raw_adj_factor(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch adj_factor {ts_code} {start}~{end}")
    return _call("adj_factor", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=72)
def _raw_income(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch income {ts_code} {start}~{end}")
    return _call("income", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=72)
def _raw_balancesheet(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch balancesheet {ts_code} {start}~{end}")
    return _call("balancesheet", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=72)
def _raw_cashflow(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch cashflow {ts_code} {start}~{end}")
    return _call("cashflow", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=72)
def _raw_fina_indicator(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch fina_indicator {ts_code} {start}~{end}")
    return _call("fina_indicator", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=24)
def _raw_disclosure_date(period_end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch disclosure_date end={period_end}")
    return _call("disclosure_date", end_date=period_end)


@cached(ttl_hours=72)
def _raw_index_weight(index_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch index_weight {index_code} {start}~{end}")
    return _call("index_weight", index_code=index_code, start_date=start, end_date=end)


@cached(ttl_hours=24)
def _raw_index_daily(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch index_daily {ts_code} {start}~{end}")
    return _call("index_daily", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=24)
def _raw_moneyflow_hsgt(start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch moneyflow_hsgt {start}~{end}")
    return _call("moneyflow_hsgt", start_date=start, end_date=end)


@cached(ttl_hours=72)
def _raw_suspend_d(ts_code: str | None, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch suspend_d {ts_code} {start}~{end}")
    if ts_code:
        return _call("suspend_d", ts_code=ts_code, start_date=start, end_date=end)
    return _call("suspend_d", suspend_type="S", start_date=start, end_date=end)


@cached(ttl_hours=24 * 7)
def _raw_namechange(ts_code: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch namechange {ts_code}")
    return _call("namechange", ts_code=ts_code)


@cached(ttl_hours=24 * 30)
def _raw_stock_basic() -> pd.DataFrame:
    log.info("[tushare] fetch stock_basic (all)")
    return _call(
        "stock_basic",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,fullname,enname,market,exchange,curr_type,list_status,list_date,delist_date,is_hs",
    )


@cached(ttl_hours=24 * 30)
def _raw_stock_basic_all() -> pd.DataFrame:
    """所有上市/已退市股票（含历史）— 用于 universe 判定."""
    log.info("[tushare] fetch stock_basic (L+D+P)")
    parts = []
    for status in ("L", "D", "P"):
        df = _call(
            "stock_basic",
            list_status=status,
            fields="ts_code,symbol,name,list_date,delist_date,list_status",
        )
        if df is not None and not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ============================================================================
# Provider 实现
# ============================================================================


class TushareProvider(DataProvider):
    """付费数据源 Tushare 的封装（2000 分套餐）."""

    name = "tushare"

    # ---------- 行情 ----------

    def get_price(
        self,
        ticker: str,
        start: str | date,
        end: str | date,
        as_of: date | None = None,
        freq: Frequency = "D",
        adjust: Adjustment = "qfq",
    ) -> pd.DataFrame:
        if freq != "D":
            raise NotImplementedError("tushare only D supported here")
        ts_code = to_tushare_code(ticker)
        start_str = self._to_yyyymmdd(start)
        end_str = self._to_yyyymmdd(end)

        with operation_logger("tushare.get_price", ticker=ticker):
            raw = _raw_daily(ts_code, start_str, end_str)
            # 复权处理
            if adjust != "none":
                adj = _raw_adj_factor(ts_code, start_str, end_str)
                if adj is not None and not adj.empty:
                    raw = raw.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
                    last_factor = adj["adj_factor"].iloc[0] if adjust == "qfq" else 1.0
                    for col in ("open", "high", "low", "close"):
                        if col in raw.columns:
                            if adjust == "qfq":
                                raw[col] = raw[col] * raw["adj_factor"] / last_factor
                            else:  # hfq
                                raw[col] = raw[col] * raw["adj_factor"]

        if raw is None or raw.empty:
            return pd.DataFrame()

        df = self._rename(raw, PRICE_COLUMN_MAP)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["ticker"] = normalize_ticker(ticker)
        df = self._filter_pit(df, as_of, "trade_date")
        keep = [
            "trade_date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "volume",
            "amount",
            "pct_change",
        ]
        df = df[[c for c in keep if c in df.columns]].sort_values("trade_date").reset_index(drop=True)
        self._assert_pit(df, as_of, "trade_date")
        return self._stamp(df, as_of, ticker, extra={"adjust": adjust})

    # ---------- 财报（PIT 关键，f_ann_date）----------

    def get_financials(
        self,
        ticker: str,
        statement_type: StatementType,
        as_of: date | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        ts_code = to_tushare_code(ticker)
        # tushare 默认 5 年范围；我们扩到 10 年防止漏
        start_str = self._to_yyyymmdd(start) if start else "20100101"
        end_str = self._to_yyyymmdd(end) if end else "20991231"

        fetcher = {
            "balance_sheet": _raw_balancesheet,
            "income": _raw_income,
            "cashflow": _raw_cashflow,
        }.get(statement_type)
        if fetcher is None:
            raise ValueError(f"Unknown statement_type: {statement_type}")

        with operation_logger(
            "tushare.get_financials", ticker=ticker, statement=statement_type
        ):
            raw = fetcher(ts_code, start_str, end_str)

        if raw is None or raw.empty:
            return pd.DataFrame()

        df = self._rename(raw, FINANCIAL_COLUMN_MAP)
        df["ticker"] = normalize_ticker(ticker)
        for col in ("ann_date", "f_ann_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # PIT 过滤（用 f_ann_date，最准确）
        df = self._filter_pit(df, as_of, "f_ann_date")
        df = df.sort_values("report_date", ascending=False).reset_index(drop=True)
        self._assert_pit(df, as_of, ["f_ann_date", "ann_date"])
        return self._stamp(df, as_of, ticker, extra={"statement_type": statement_type})

    # ---------- 财务比率（独家）----------

    def get_financial_indicators(
        self,
        ticker: str,
        as_of: date | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        """ROE/ROA/EPS/毛利率/资产负债率 等 108 个指标（含 TTM）."""
        ts_code = to_tushare_code(ticker)
        start_str = self._to_yyyymmdd(start) if start else "20100101"
        end_str = self._to_yyyymmdd(end) if end else "20991231"
        with operation_logger("tushare.get_fina_indicator", ticker=ticker):
            raw = _raw_fina_indicator(ts_code, start_str, end_str)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = self._rename(raw, FINANCIAL_COLUMN_MAP)
        df["ticker"] = normalize_ticker(ticker)
        for col in ("ann_date", "report_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        # fina_indicator 没有 f_ann_date，用 ann_date
        df = self._filter_pit(df, as_of, "ann_date")
        df = df.sort_values("report_date", ascending=False).reset_index(drop=True)
        self._assert_pit(df, as_of, "ann_date")
        return self._stamp(df, as_of, ticker)

    # ---------- 披露时间表（独家）----------

    def get_disclosure_schedule(
        self,
        period_end: str | date,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        period_str = self._to_yyyymmdd(period_end)
        with operation_logger("tushare.disclosure_date", period=period_str):
            raw = _raw_disclosure_date(period_str)
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        rename = {
            "ts_code": "ticker",
            "ann_date": "ann_date",
            "end_date": "report_date",
            "pre_date": "pre_announce_date",
            "actual_date": "actual_announce_date",
            "modify_date": "modify_date",
        }
        df = self._rename(df, rename)
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].apply(normalize_ticker)
        for col in ("ann_date", "report_date", "pre_announce_date", "actual_announce_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # PIT 过滤（用 actual_announce_date，已披露的）
        if as_of is not None and "actual_announce_date" in df.columns:
            df = df.copy()
            df["actual_announce_date"] = df["actual_announce_date"].where(
                df["actual_announce_date"] <= pd.Timestamp(as_of), pd.NaT
            )
        return self._stamp(df, as_of, None)

    # ---------- 指数成分股（PIT 关键）----------

    def get_index_constituents(
        self,
        index_code: str,
        as_of: date | None = None,
    ) -> list[str]:
        """返回 ``as_of`` 时点的指数成分股（PIT 严格，含后退市的）.

        Args:
            index_code: 如 ``000300.SH`` (沪深 300), ``000905.SH`` (中证 500)
            as_of: 时点；None = 最新

        Tushare ``index_weight`` 返回近一个月的成分股权重，
        我们取 ``trade_date <= as_of`` 的最新一条。
        """
        ts_code = self._normalize_index_code(index_code)
        if as_of is None:
            as_of = date.today()
        # 取过去 60 天数据（保证至少有一次月底权重快照）
        end_str = self._to_yyyymmdd(as_of)
        start_str = self._to_yyyymmdd(
            (pd.Timestamp(as_of) - pd.Timedelta(days=60)).date()
        )

        with operation_logger("tushare.index_weight", index=ts_code, as_of=str(as_of)):
            raw = _raw_index_weight(ts_code, start_str, end_str)

        if raw is None or raw.empty:
            log.warning(f"index_weight returned empty for {ts_code} ~{as_of}")
            return []

        df = raw.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # 过滤 <= as_of
        df = df[df["trade_date"] <= pd.Timestamp(as_of)]
        if df.empty:
            return []
        # 取最新一个调整日的全部成分股
        latest = df["trade_date"].max()
        df = df[df["trade_date"] == latest]
        return sorted(set(df["con_code"].apply(normalize_ticker).tolist()))

    @staticmethod
    def _normalize_index_code(code: str) -> str:
        """统一指数代码格式为 tushare 风格（``XXXXXX.SH`` / ``XXXXXX.SZ``）."""
        c = code.strip().upper().replace(" ", "")
        if c.endswith((".SH", ".SZ", ".CSI")):
            num = c.split(".")[0]
            ex = c.split(".")[1]
            # tushare 用 SH/SZ 后缀
            if ex == "CSI":
                ex = "SH" if num.startswith("000") else "SZ"
            return f"{num}.{ex}"
        # 纯数字，按惯例补
        if c.isdigit() and len(c) == 6:
            ex = "SH" if c.startswith(("000", "9")) else "SZ"
            return f"{c}.{ex}"
        return c

    # ---------- 是否可交易 ----------

    def is_tradable(self, ticker: str, on_date: date) -> bool:
        ts_code = to_tushare_code(ticker)
        date_str = self._to_yyyymmdd(on_date)
        try:
            df = _raw_suspend_d(ts_code, date_str, date_str)
            return df is None or df.empty
        except Exception as e:  # noqa: BLE001
            log.warning(f"is_tradable failed for {ticker}: {e}")
            return True

    # ---------- 北向资金 ----------

    def get_north_bound_flow(
        self,
        ticker: str | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        if ticker is not None:
            log.info("tushare moneyflow_hsgt is market aggregate; ticker arg ignored")
        if start is None or end is None:
            raise ValueError("start and end required")
        start_str = self._to_yyyymmdd(start)
        end_str = self._to_yyyymmdd(end)

        with operation_logger("tushare.moneyflow_hsgt"):
            raw = _raw_moneyflow_hsgt(start_str, end_str)
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        rename = {
            "trade_date": "trade_date",
            "ggt_ss": "ggt_ss",       # 港股通(上海)
            "ggt_sz": "ggt_sz",       # 港股通(深圳)
            "hgt": "hgt",             # 沪股通(亿元)
            "sgt": "sgt",             # 深股通(亿元)
            "north_money": "north_money",
            "south_money": "south_money",
        }
        df = self._rename(df, rename)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.sort_values("trade_date").reset_index(drop=True)
        df = self._filter_pit(df, as_of, "trade_date")
        self._assert_pit(df, as_of, "trade_date")
        return self._stamp(df, as_of, None)

    # ---------- 复权因子 ----------

    def get_adj_factor(
        self,
        ticker: str,
        start: str | date,
        end: str | date,
    ) -> pd.DataFrame:
        ts_code = to_tushare_code(ticker)
        with operation_logger("tushare.adj_factor", ticker=ticker):
            raw = _raw_adj_factor(ts_code, self._to_yyyymmdd(start), self._to_yyyymmdd(end))
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["ticker"] = normalize_ticker(ticker)
        df = df.sort_values("trade_date").reset_index(drop=True)
        return self._stamp(df, None, ticker)

    # ---------- 公司更名 ----------

    def get_corporate_actions(
        self,
        ticker: str,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        ts_code = to_tushare_code(ticker)
        with operation_logger("tushare.namechange", ticker=ticker):
            raw = _raw_namechange(ts_code)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        for col in ("start_date", "end_date", "ann_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df = self._filter_pit(df, as_of, "ann_date")
        return self._stamp(df, as_of, ticker)

    # ---------- 股票基本信息（list_date, delist_date）----------

    def get_stock_basic(self, include_delisted: bool = True) -> pd.DataFrame:
        with operation_logger("tushare.stock_basic"):
            raw = _raw_stock_basic_all() if include_delisted else _raw_stock_basic()
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df["ticker"] = df["ts_code"].apply(normalize_ticker)
        for col in ("list_date", "delist_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return self._stamp(df, None, None)


__all__ = ["TushareProvider"]


if __name__ == "__main__":
    from datetime import date as _date

    p = TushareProvider()
    print(f"Provider: {p.name}")

    print("\n--- get_price (300750.SZ, 2024-01) ---")
    df = p.get_price("300750.SZ", "2024-01-01", "2024-01-31", as_of=_date(2024, 2, 1))
    print(df.head())

    print("\n--- get_financials (income, as_of=2024-04-01) ---")
    df = p.get_financials("300750.SZ", "income", as_of=_date(2024, 4, 1))
    if not df.empty:
        print(df[["ticker", "report_date", "ann_date", "f_ann_date"]].head())
        print(f"  Latest f_ann_date <= 2024-04-01: {df['f_ann_date'].max()}")

    print("\n--- get_index_constituents (csi300, as_of=2023-06-30) ---")
    cons = p.get_index_constituents("000300.SH", as_of=_date(2023, 6, 30))
    print(f"  count: {len(cons)}, first 5: {cons[:5]}")

    print("\n--- get_disclosure_schedule (period=20231231) ---")
    df = p.get_disclosure_schedule("20231231")
    print(f"  shape: {df.shape}, sample:")
    if not df.empty:
        print(df.head())
