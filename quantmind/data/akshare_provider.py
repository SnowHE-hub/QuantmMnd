"""quantmind.data.akshare_provider — akshare DataProvider 实现.

特点：
    - 免费、覆盖广，但慢（按报告期财报 38-51s/股票，业绩快报全 A 226s）
    - PIT：财报数据用 ``NOTICE_DATE`` 字段（首次披露日）映射为 ``f_ann_date``
    - 适合：日线、新闻、研报、业绩快报全量
"""

from __future__ import annotations

from datetime import date

import akshare as ak  # type: ignore[import-untyped]
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
    to_akshare_em_symbol,
    to_akshare_symbol,
)

log = get_logger(__name__)


# ============================================================================
# 模块级 raw fetcher（缓存层）—— 不带 as_of，只缓存外部 API 原始返回
# 这样多个 as_of 调用复用同一份数据，PIT 过滤在 provider 方法里做
# ============================================================================


@cached(ttl_hours=24)
def _raw_price(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    log.info(f"[akshare] fetch price {symbol} {start}~{end} adj={adjust!r}")
    return ak.stock_zh_a_hist(
        symbol=symbol, period="daily", start_date=start, end_date=end, adjust=adjust
    )


@cached(ttl_hours=72)  # 财报变更慢，72h
def _raw_balance_sheet(symbol_em: str) -> pd.DataFrame:
    log.info(f"[akshare] fetch balance_sheet {symbol_em}")
    return ak.stock_balance_sheet_by_report_em(symbol=symbol_em)


@cached(ttl_hours=72)
def _raw_income_statement(symbol_em: str) -> pd.DataFrame:
    log.info(f"[akshare] fetch income {symbol_em}")
    return ak.stock_profit_sheet_by_report_em(symbol=symbol_em)


@cached(ttl_hours=72)
def _raw_cashflow_statement(symbol_em: str) -> pd.DataFrame:
    log.info(f"[akshare] fetch cashflow {symbol_em}")
    return ak.stock_cash_flow_sheet_by_report_em(symbol=symbol_em)


@cached(ttl_hours=24)
def _raw_news(symbol: str) -> pd.DataFrame:
    log.info(f"[akshare] fetch news {symbol}")
    return ak.stock_news_em(symbol=symbol)


@cached(ttl_hours=24)
def _raw_research_reports(symbol: str) -> pd.DataFrame:
    log.info(f"[akshare] fetch research_reports {symbol}")
    return ak.stock_research_report_em(symbol=symbol)


@cached(ttl_hours=24)
def _raw_individual_info(symbol: str) -> pd.DataFrame:
    log.info(f"[akshare] fetch individual_info {symbol}")
    return ak.stock_individual_info_em(symbol=symbol)


@cached(ttl_hours=24)
def _raw_csi300_constituents() -> pd.DataFrame:
    log.info("[akshare] fetch csi300 constituents (csindex)")
    return ak.index_stock_cons_csindex(symbol="000300")


@cached(ttl_hours=24)
def _raw_csi300_constituents_sina() -> pd.DataFrame:
    log.info("[akshare] fetch csi300 constituents (sina)")
    return ak.index_stock_cons_sina(symbol="000300")


@cached(ttl_hours=24)
def _raw_north_bound_hist() -> pd.DataFrame:
    log.info("[akshare] fetch north bound hist")
    return ak.stock_hsgt_hist_em(symbol="北向资金")


@cached(ttl_hours=24)
def _raw_suspend_today(date_str: str) -> pd.DataFrame:
    log.info(f"[akshare] fetch suspend list date={date_str}")
    return ak.stock_tfp_em(date=date_str)


# ============================================================================
# Provider 实现
# ============================================================================


class AkshareProvider(DataProvider):
    """免费数据源 akshare 的封装."""

    name = "akshare"

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
            raise NotImplementedError("akshare only D supported here; use other tools for W/M")
        symbol = to_akshare_symbol(ticker)
        start_str = self._to_yyyymmdd(start)
        end_str = self._to_yyyymmdd(end)
        adj_str = "" if adjust == "none" else adjust

        with operation_logger("akshare.get_price", ticker=ticker):
            try:
                raw = _raw_price(symbol, start_str, end_str, adj_str)
            except Exception as e:
                raise DataProviderError(f"akshare get_price failed for {ticker}: {e}") from e

        if raw is None or raw.empty:
            return pd.DataFrame()

        df = self._rename(raw, PRICE_COLUMN_MAP)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["ticker"] = normalize_ticker(ticker)
        # PIT 过滤
        df = self._filter_pit(df, as_of, "trade_date")
        # 选标准列
        keep = [
            "trade_date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "pct_change",
            "turnover_rate",
        ]
        df = df[[c for c in keep if c in df.columns]].sort_values("trade_date").reset_index(drop=True)

        # 防御断言
        self._assert_pit(df, as_of, "trade_date")
        return self._stamp(df, as_of, ticker, extra={"adjust": adjust})

    # ---------- 财报（PIT 关键）----------

    def get_financials(
        self,
        ticker: str,
        statement_type: StatementType,
        as_of: date | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        symbol_em = to_akshare_em_symbol(ticker)
        fetcher = {
            "balance_sheet": _raw_balance_sheet,
            "income": _raw_income_statement,
            "cashflow": _raw_cashflow_statement,
        }.get(statement_type)
        if fetcher is None:
            raise ValueError(f"Unknown statement_type: {statement_type}")

        with operation_logger(
            "akshare.get_financials", ticker=ticker, statement=statement_type
        ):
            try:
                raw = fetcher(symbol_em)
            except Exception as e:
                raise DataProviderError(
                    f"akshare get_financials({statement_type}) failed for {ticker}: {e}"
                ) from e

        if raw is None or raw.empty:
            return pd.DataFrame()

        df = self._rename(raw, FINANCIAL_COLUMN_MAP)
        df["ticker"] = normalize_ticker(ticker)
        # 时间戳列转换
        for col in ("report_date", "f_ann_date", "update_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        # akshare 没有 ann_date 字段，用 f_ann_date 代替
        if "ann_date" not in df.columns and "f_ann_date" in df.columns:
            df["ann_date"] = df["f_ann_date"]

        # PIT 过滤（关键）
        df = self._filter_pit(df, as_of, "f_ann_date")

        # 报告期范围过滤
        if start is not None:
            df = df[df["report_date"] >= pd.Timestamp(self._to_yyyy_dash(start))]
        if end is not None:
            df = df[df["report_date"] <= pd.Timestamp(self._to_yyyy_dash(end))]

        df = df.sort_values("report_date", ascending=False).reset_index(drop=True)

        # 防御断言
        self._assert_pit(df, as_of, "f_ann_date")
        return self._stamp(df, as_of, ticker, extra={"statement_type": statement_type})

    # ---------- 指数成分股 ----------

    def get_index_constituents(
        self,
        index_code: str,
        as_of: date | None = None,
    ) -> list[str]:
        # akshare 只能拿当前成分股；历史 PIT 由 tushare 负责
        if index_code not in {"000300", "000300.SH", "000300.CSI"}:
            log.warning(f"akshare get_index_constituents only tested for csi300; got {index_code}")

        if as_of is not None:
            log.warning(
                "akshare can only return CURRENT csi300 constituents (survivorship bias risk!); "
                "use TushareProvider for historical PIT correctness"
            )

        with operation_logger("akshare.get_index_constituents", index=index_code):
            try:
                raw = _raw_csi300_constituents()
                # 列名查找优先级：成分券代码 > 含'代码'且非'指数代码' > 任何 'code'
                code_col = None
                for c in raw.columns:
                    if c == "成分券代码":
                        code_col = c
                        break
                if code_col is None:
                    for c in raw.columns:
                        if "代码" in c and "指数" not in c:
                            code_col = c
                            break
                if code_col is None:
                    code_col = next((c for c in raw.columns if "code" in c.lower() and "index" not in c.lower()), None)
                if code_col is None:
                    raise DataProviderError(f"Cannot find code column in {raw.columns.tolist()}")
                tickers = [normalize_ticker(str(c).zfill(6)) for c in raw[code_col].astype(str)]
                return sorted(set(tickers))
            except Exception as e:
                raise DataProviderError(f"akshare get_index_constituents failed: {e}") from e

    # ---------- 可交易性 ----------

    def is_tradable(self, ticker: str, on_date: date) -> bool:
        try:
            date_str = self._to_yyyymmdd(on_date)
            df = _raw_suspend_today(date_str)
            if df is None or df.empty:
                return True  # 当日没人停牌
            symbol = to_akshare_symbol(ticker)
            # 列名可能是 "代码" 或 "股票代码"
            code_col = next((c for c in df.columns if "代码" in c), None)
            if code_col is None:
                return True
            return symbol not in df[code_col].astype(str).values
        except Exception as e:  # noqa: BLE001
            log.warning(f"is_tradable check failed: {e}; assuming True")
            return True

    # ---------- 新闻 ----------

    def get_news(
        self,
        ticker: str,
        start: str | date,
        end: str | date,
        as_of: date | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        symbol = to_akshare_symbol(ticker)
        with operation_logger("akshare.get_news", ticker=ticker):
            try:
                raw = _raw_news(symbol)
            except Exception as e:
                raise DataProviderError(f"akshare get_news failed for {ticker}: {e}") from e
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        # 列名标准化（akshare 给 ['关键词','新闻标题','新闻内容','发布时间','文章来源','新闻链接']）
        rename = {
            "关键词": "keyword",
            "新闻标题": "title",
            "新闻内容": "content",
            "发布时间": "published_at",
            "文章来源": "source",
            "新闻链接": "url",
        }
        df = self._rename(df, rename)
        df["ticker"] = normalize_ticker(ticker)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
        df = df.dropna(subset=["published_at"])

        # 范围过滤
        start_ts = pd.Timestamp(self._to_yyyy_dash(start))
        end_ts = pd.Timestamp(self._to_yyyy_dash(end)) + pd.Timedelta(days=1)
        df = df[(df["published_at"] >= start_ts) & (df["published_at"] < end_ts)]

        # PIT 过滤
        df = self._filter_pit(df, as_of, "published_at")
        df = df.sort_values("published_at", ascending=False).head(limit).reset_index(drop=True)
        self._assert_pit(df, as_of, "published_at")
        return self._stamp(df, as_of, ticker)

    # ---------- 研报 ----------

    def get_research_reports(
        self,
        ticker: str | None = None,
        as_of: date | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        if ticker is None:
            raise NotImplementedError("akshare research_report_em requires ticker")
        symbol = to_akshare_symbol(ticker)
        with operation_logger("akshare.get_research_reports", ticker=ticker):
            try:
                raw = _raw_research_reports(symbol)
            except Exception as e:
                raise DataProviderError(f"akshare get_research_reports failed: {e}") from e
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        rename = {
            "报告名称": "title",
            "东财评级": "rating",
            "机构": "institution",
            "近一月个股研报数": "report_count_1m",
            "2024-盈利预测-收益": "eps_2024",
            "2025-盈利预测-收益": "eps_2025",
            "行业": "industry",
            "日期": "published_at",
        }
        df = self._rename(df, rename)
        df["ticker"] = normalize_ticker(ticker)
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
            df = df.dropna(subset=["published_at"])
            df = self._filter_pit(df, as_of, "published_at")
        df = df.head(limit).reset_index(drop=True)
        self._assert_pit(df, as_of, "published_at")
        return self._stamp(df, as_of, ticker)

    # ---------- 北向资金 ----------

    def get_north_bound_flow(
        self,
        ticker: str | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        if ticker is not None:
            log.info("akshare hsgt_hist_em returns market aggregate, ticker arg ignored")
        with operation_logger("akshare.get_north_bound_flow"):
            try:
                raw = _raw_north_bound_hist()
            except Exception as e:
                raise DataProviderError(f"akshare get_north_bound_flow failed: {e}") from e
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        # akshare 列：['日期','当日成交净买额','买入成交额','卖出成交额','历史累计净买额','当日资金流入','当日余额','持股市值','领涨股-代码','领涨股','领涨股-涨跌幅','沪深300','沪深300-涨跌幅']
        rename = {
            "日期": "trade_date",
            "当日成交净买额": "net_buy_amount",
            "买入成交额": "buy_amount",
            "卖出成交额": "sell_amount",
            "历史累计净买额": "cum_net_buy",
            "当日资金流入": "fund_inflow",
            "持股市值": "holding_market_value",
        }
        df = self._rename(df, rename)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df.dropna(subset=["trade_date"])

        if start is not None:
            df = df[df["trade_date"] >= pd.Timestamp(self._to_yyyy_dash(start))]
        if end is not None:
            df = df[df["trade_date"] <= pd.Timestamp(self._to_yyyy_dash(end))]

        df = self._filter_pit(df, as_of, "trade_date")
        df = df.sort_values("trade_date").reset_index(drop=True)
        self._assert_pit(df, as_of, "trade_date")
        return self._stamp(df, as_of, ticker)


__all__ = ["AkshareProvider"]


if __name__ == "__main__":
    from datetime import date as _date

    p = AkshareProvider()
    print(f"Provider: {p.name}")

    print("\n--- get_price (300750.SZ, 2024-01) ---")
    df = p.get_price("300750.SZ", "2024-01-01", "2024-01-31", as_of=_date(2024, 2, 1))
    print(df.head())
    print(f"  attrs: {dict(df.attrs)}")

    print("\n--- get_financials (income, as_of=2024-04-01) ---")
    df = p.get_financials("300750.SZ", "income", as_of=_date(2024, 4, 1))
    if not df.empty:
        print(df[["report_date", "f_ann_date", "ticker"]].head())
        print(f"  Latest f_ann_date <= 2024-04-01: {df['f_ann_date'].max()}")

    print("\n--- get_index_constituents (csi300, current) ---")
    cons = p.get_index_constituents("000300")
    print(f"  count: {len(cons)}, first 5: {cons[:5]}")

    print("\n--- is_tradable ---")
    print(f"  300750.SZ on 2024-01-05: {p.is_tradable('300750.SZ', _date(2024, 1, 5))}")
