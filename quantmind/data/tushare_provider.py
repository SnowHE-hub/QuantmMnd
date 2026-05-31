"""quantmind.data.tushare_provider — Tushare DataProvider 实现.

特点：
    - 付费（积分制）：本项目使用 2000 分套餐
    - PIT：财报数据用 ``f_ann_date``（实际公告日期，含修订），优于 akshare 的 NOTICE_DATE
    - 速度：比 akshare 快 5-30 倍
    - 适合：财报、披露日、历史成分股、北向资金、复权因子、财务比率

可用接口（2000 分内）：
    stock_basic, trade_cal, daily, daily_basic, income, balancesheet, cashflow,
    fina_indicator, fina_audit, disclosure_date, index_weight, index_basic,
    index_daily, hk_hold, margin_detail, moneyflow_hsgt, hsgt_top10, top_list,
    adj_factor, suspend_d, namechange, stk_holdernumber, forecast, express
"""

from __future__ import annotations

import os
import socket
import threading
import time
from datetime import date

# 全局 socket 超时兜底：防止底层 TCP 连接无限期挂起
# DataApi 的 timeout 参数只控制 requests 库的读超时，
# 但 TCP 三次握手本身不受 requests timeout 保护；
# socket.setdefaulttimeout 在套接字层面兜底。
socket.setdefaulttimeout(30)

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
            from tushare.pro.client import DataApi  # type: ignore[import-untyped]
        except ImportError as e:
            raise DataProviderError("tushare not installed; pip install tushare") from e
        # 直接用 DataApi(token=token) 而非 ts.pro_api()：
        # ts.pro_api() 内部调用 get_token()，会优先读 TUSHARE_TOKEN 环境变量，
        # 在双 Token 场景中可能取到错误的 token。
        # timeout=15：每次 HTTP 请求最多等 15 秒；
        # 配合全局 socket.setdefaulttimeout(30) 双重保障，
        # 避免像 60.205.198.20:80 这类后端节点无限期挂起。
        _tushare_pro = DataApi(token=token, timeout=15)
        # 若设置了 TUSHARE_HI_URL，切换到高频代理（Token B 专用）
        hi_url = os.getenv("TUSHARE_HI_URL", "").strip()
        if hi_url:
            _tushare_pro._DataApi__http_url = hi_url  # type: ignore[attr-defined]
            log.info(f"tushare pro_api initialized (proxy={hi_url}, token={token[:8]}...)")
        else:
            log.info(f"tushare pro_api initialized (official, token={token[:8]}...)")
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
    "timed_out",
    "connection",
    "temporarily",
    "503",
    "504",
    "502",
    "socket",
    "read timed",
    "connect timed",
    "remote end closed",
    "reset by peer",
    "broken pipe",
)


def _call(api_name: str, max_attempts: int = 4, **kwargs):
    """调用 tushare API（带限频 + 重试 + 超时退避）.

    超时策略：
    - DataApi timeout=15s（HTTP 读超时）
    - socket.setdefaulttimeout(30)（TCP 层兜底）
    - 网络抖动时最多重试 4 次，退避 2/4/8s（合计最坏 15*4+14=74s/ticker）
    """
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
            # socket.timeout 不继承自 OSError 在部分版本，额外检查类型
            is_socket_timeout = isinstance(e, (TimeoutError, socket.timeout))
            transient = is_socket_timeout or any(k in msg for k in _TRANSIENT_KEYWORDS)
            if transient and attempt < max_attempts - 1:
                wait = 2 ** (attempt + 1)  # 2, 4, 8s
                log.warning(
                    f"tushare {api_name} transient error "
                    f"(attempt {attempt + 1}/{max_attempts}): {str(e)[:120]}; sleep {wait}s"
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
#
# 设计原则
# --------
# 为了让多时点（月度/季度 panel 构建）能共享 cache，
# 每个 ``_raw_*_full(ts_code)`` 都按 **固定宽范围** 取数（含 ts_code 全历史），
# 然后由上层 ``get_*`` 在内存里切窗。
#
# 兼容性：保留原 ``_raw_*(ts_code, start, end)`` 接口，但内部直接调 _full 后切片，
# 已经写好的代码无需改动；旧 cache 会被新 cache 替代（首次调用各自 ticker 时
# 会重新拉一次全历史，之后所有时点共享）。

# 全市场最早日期（A 股深市起步是 1990 年，作为兜底起点）
_FULL_START = "19900101"
# 远未来日期，Tushare 接受任何此类大日期，等价于 “直到当前为止”
_FULL_END = "20991231"


@cached(ttl_hours=24)
def _raw_daily_full(ts_code: str) -> pd.DataFrame:
    """日线全历史，由上层切片。Tushare daily 单次最多 6000 行，足够 24+ 年."""
    log.info(f"[tushare] fetch daily FULL {ts_code}")
    return _call("daily", ts_code=ts_code, start_date=_FULL_START, end_date=_FULL_END)


@cached(ttl_hours=24)
def _raw_adj_factor_full(ts_code: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch adj_factor FULL {ts_code}")
    return _call("adj_factor", ts_code=ts_code, start_date=_FULL_START, end_date=_FULL_END)


@cached(ttl_hours=24)
def _raw_daily_basic_full(ts_code: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch daily_basic FULL {ts_code}")
    return _call("daily_basic", ts_code=ts_code, start_date=_FULL_START, end_date=_FULL_END)


@cached(ttl_hours=72)
def _raw_income_full(ts_code: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch income FULL {ts_code}")
    return _call("income", ts_code=ts_code, start_date=_FULL_START, end_date=_FULL_END)


@cached(ttl_hours=72)
def _raw_balancesheet_full(ts_code: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch balancesheet FULL {ts_code}")
    return _call("balancesheet", ts_code=ts_code, start_date=_FULL_START, end_date=_FULL_END)


@cached(ttl_hours=72)
def _raw_cashflow_full(ts_code: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch cashflow FULL {ts_code}")
    return _call("cashflow", ts_code=ts_code, start_date=_FULL_START, end_date=_FULL_END)


@cached(ttl_hours=72)
def _raw_fina_indicator_full(ts_code: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch fina_indicator FULL {ts_code}")
    return _call("fina_indicator", ts_code=ts_code, start_date=_FULL_START, end_date=_FULL_END)


# ----- 兼容层（旧接口，调 _full 后切片）-----


def _slice_by_date(df: pd.DataFrame, col: str, start: str, end: str) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    s = df[col].astype(str)
    return df[(s >= start) & (s <= end)].reset_index(drop=True)


def _raw_daily(ts_code: str, start: str, end: str) -> pd.DataFrame:
    return _slice_by_date(_raw_daily_full(ts_code), "trade_date", start, end)


def _raw_daily_basic(ts_code: str, start: str, end: str) -> pd.DataFrame:
    return _slice_by_date(_raw_daily_basic_full(ts_code), "trade_date", start, end)


def _raw_daily_basic_history(ts_code: str, start: str, end: str) -> pd.DataFrame:
    return _slice_by_date(_raw_daily_basic_full(ts_code), "trade_date", start, end)


@cached(ttl_hours=24)
def _raw_daily_basic_market(trade_date: str) -> pd.DataFrame:
    """单日全市场 daily_basic（一次拉所有股票，比逐只快 1000x）."""
    log.info(f"[tushare] fetch daily_basic full market trade_date={trade_date}")
    return _call("daily_basic", trade_date=trade_date)


def _raw_adj_factor(ts_code: str, start: str, end: str) -> pd.DataFrame:
    return _slice_by_date(_raw_adj_factor_full(ts_code), "trade_date", start, end)


def _raw_income(ts_code: str, start: str, end: str) -> pd.DataFrame:
    df = _raw_income_full(ts_code)
    if df is None or df.empty:
        return df
    # 财报按 ann_date 切（PIT 对齐）
    if "ann_date" in df.columns:
        s = df["ann_date"].astype(str)
        return df[(s >= start) & (s <= end)].reset_index(drop=True)
    return df


def _raw_balancesheet(ts_code: str, start: str, end: str) -> pd.DataFrame:
    df = _raw_balancesheet_full(ts_code)
    if df is None or df.empty:
        return df
    if "ann_date" in df.columns:
        s = df["ann_date"].astype(str)
        return df[(s >= start) & (s <= end)].reset_index(drop=True)
    return df


def _raw_cashflow(ts_code: str, start: str, end: str) -> pd.DataFrame:
    df = _raw_cashflow_full(ts_code)
    if df is None or df.empty:
        return df
    if "ann_date" in df.columns:
        s = df["ann_date"].astype(str)
        return df[(s >= start) & (s <= end)].reset_index(drop=True)
    return df


def _raw_fina_indicator(ts_code: str, start: str, end: str) -> pd.DataFrame:
    df = _raw_fina_indicator_full(ts_code)
    if df is None or df.empty:
        return df
    if "ann_date" in df.columns:
        s = df["ann_date"].astype(str)
        return df[(s >= start) & (s <= end)].reset_index(drop=True)
    return df


@cached(ttl_hours=24)
def _raw_disclosure_date(period_end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch disclosure_date end={period_end}")
    return _call("disclosure_date", end_date=period_end)


@cached(ttl_hours=72)
def _raw_index_weight_year(index_code: str, year: int) -> pd.DataFrame:
    """按年缓存 index_weight，多个 as_of 共享缓存."""
    start = f"{year}0101"
    end = f"{year}1231"
    log.info(f"[tushare] fetch index_weight {index_code} year={year}")
    return _call("index_weight", index_code=index_code, start_date=start, end_date=end)


def _raw_index_weight(index_code: str, start: str, end: str) -> pd.DataFrame:
    """按起止年份分片调用 ``_raw_index_weight_year`` 后合并."""
    sy, ey = int(start[:4]), int(end[:4])
    parts = []
    for y in range(sy, ey + 1):
        df = _raw_index_weight_year(index_code, y)
        if df is not None and not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    full = pd.concat(parts, ignore_index=True)
    if "trade_date" in full.columns:
        s = full["trade_date"].astype(str)
        full = full[(s >= start) & (s <= end)].reset_index(drop=True)
    return full


@cached(ttl_hours=24)
def _raw_index_daily(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch index_daily {ts_code} {start}~{end}")
    return _call("index_daily", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=24)
def _raw_moneyflow_hsgt(start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch moneyflow_hsgt {start}~{end}")
    return _call("moneyflow_hsgt", start_date=start, end_date=end)


@cached(ttl_hours=24)
def _raw_hk_hold(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch hk_hold {ts_code} {start}~{end}")
    return _call("hk_hold", ts_code=ts_code, start_date=start, end_date=end)


@cached(ttl_hours=24)
def _raw_margin_detail(ts_code: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"[tushare] fetch margin_detail {ts_code} {start}~{end}")
    return _call("margin_detail", ts_code=ts_code, start_date=start, end_date=end)


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
        with_basic: bool = True,
    ) -> pd.DataFrame:
        """日线行情，可选 ``with_basic=True`` 合并 ``turnover_rate / pe / pb / total_mv``."""
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
            # 合并 daily_basic 时间序列
            if with_basic:
                try:
                    db = _raw_daily_basic_history(ts_code, start_str, end_str)
                    if db is not None and not db.empty:
                        keep_cols = [
                            "trade_date",
                            "turnover_rate",
                            "turnover_rate_f",
                            "pe",
                            "pe_ttm",
                            "pb",
                            "ps_ttm",
                            "total_mv",
                            "circ_mv",
                        ]
                        keep_cols = [c for c in keep_cols if c in db.columns]
                        raw = raw.merge(db[keep_cols], on="trade_date", how="left")
                except Exception as e:  # noqa: BLE001
                    log.warning(f"daily_basic merge skipped for {ticker}: {e}")

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
            # daily_basic 字段
            "turnover_rate",
            "turnover_rate_f",
            "pe",
            "pe_ttm",
            "pb",
            "ps_ttm",
            "total_mv",
            "circ_mv",
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

    # ---------- 沪深股通持股（个股）----------

    def get_hk_hold(
        self,
        ticker: str,
        start: str | date,
        end: str | date,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """港股通 / 股通持股明细，按交易日 PIT."""
        ts_code = to_tushare_code(ticker)
        start_str = self._to_yyyymmdd(start)
        end_str = self._to_yyyymmdd(end)
        with operation_logger("tushare.hk_hold", ticker=ticker):
            raw = _raw_hk_hold(ts_code, start_str, end_str)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        # 与文档对齐的常用别名（列存在才重命名）
        alias = {"vol": "hold_vol", "ratio": "hold_ratio"}
        for a, b in alias.items():
            if a in df.columns and b not in df.columns:
                df = df.rename(columns={a: b})
        if "hold_amount" not in df.columns and "amount" in df.columns:
            df = df.rename(columns={"amount": "hold_amount"})
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        if "ts_code" in df.columns:
            df["ticker"] = df["ts_code"].apply(normalize_ticker)
        df = df.sort_values("trade_date", na_position="last").reset_index(drop=True) if "trade_date" in df.columns else df
        df = self._filter_pit(df, as_of, "trade_date")
        if "trade_date" in df.columns:
            self._assert_pit(df, as_of, "trade_date")
        return self._stamp(df, as_of, ticker)

    # ---------- 融资融券明细（个股）----------

    def get_margin_detail(
        self,
        ticker: str,
        start: str | date,
        end: str | date,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        ts_code = to_tushare_code(ticker)
        start_str = self._to_yyyymmdd(start)
        end_str = self._to_yyyymmdd(end)
        with operation_logger("tushare.margin_detail", ticker=ticker):
            raw = _raw_margin_detail(ts_code, start_str, end_str)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        if "ts_code" in df.columns:
            df["ticker"] = df["ts_code"].apply(normalize_ticker)
        df = (
            df.sort_values("trade_date", na_position="last").reset_index(drop=True)
            if "trade_date" in df.columns
            else df
        )
        df = self._filter_pit(df, as_of, "trade_date")
        if "trade_date" in df.columns:
            self._assert_pit(df, as_of, "trade_date")
        return self._stamp(df, as_of, ticker)

    # ---------- 指数日线 ----------

    def get_index_daily(
        self,
        index_code: str,
        start: str | date,
        end: str | date,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        ts_code = self._normalize_index_code(index_code)
        start_str = self._to_yyyymmdd(start)
        end_str = self._to_yyyymmdd(end)
        with operation_logger("tushare.index_daily", ticker=ts_code):
            raw = _raw_index_daily(ts_code, start_str, end_str)
        if raw is None or raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.sort_values("trade_date").reset_index(drop=True)
        df = self._filter_pit(df, as_of, "trade_date")
        self._assert_pit(df, as_of, "trade_date")
        return self._stamp(df, as_of, None, extra={"index_code": ts_code})

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

    # ---------- 估值/市值（daily_basic）----------

    def get_daily_basic_market(
        self,
        trade_date: date,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """单日**全市场** daily_basic（pe / pb / ps / 市值 / 换手率 等 18 列）.

        如果 ``trade_date`` 当日非交易日（周末/节假日），会自动回退到上一个交易日。
        """
        # 防御：as_of 不能小于 trade_date
        if as_of is not None and trade_date > as_of:
            raise ValueError(f"trade_date {trade_date} > as_of {as_of} is forbidden by PIT")

        with operation_logger("tushare.daily_basic_market", trade_date=str(trade_date)):
            # 自动回退到最近交易日（最多回退 7 天）
            for offset in range(8):
                candidate = (pd.Timestamp(trade_date) - pd.Timedelta(days=offset)).date()
                date_str = self._to_yyyymmdd(candidate)
                raw = _raw_daily_basic_market(date_str)
                if raw is not None and not raw.empty:
                    log.info(
                        f"daily_basic_market: requested={trade_date} actual={candidate} rows={len(raw)}"
                    )
                    break
            else:
                return pd.DataFrame()

        df = raw.copy()
        df["ticker"] = df["ts_code"].apply(normalize_ticker)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # 按 PIT 校验
        if as_of is not None:
            df = self._filter_pit(df, as_of, "trade_date")
        self._assert_pit(df, as_of, "trade_date")
        keep = [
            "ticker",
            "trade_date",
            "close",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ratio",
            "dv_ttm",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",  # 总市值（万元）
            "circ_mv",   # 流通市值（万元）
        ]
        df = df[[c for c in keep if c in df.columns]]
        return self._stamp(df, as_of, None)

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
