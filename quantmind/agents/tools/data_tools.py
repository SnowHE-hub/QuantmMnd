"""quantmind.agents.tools.data_tools — 数据获取工具.

从已构建的 snapshot 中读取股票基本面、价格、同行等数据（严格 PIT）。
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

__all__ = [
    "fetch_stock_basics",
    "fetch_financials_pit",
    "fetch_price_history",
    "fetch_industry_peers",
    "fetch_recent_news",
]

# snapshot 缓存，避免重复加载
_SNAPSHOT_CACHE: dict[date, dict] = {}


def _load_snapshot(as_of: date) -> dict:
    if as_of not in _SNAPSHOT_CACHE:
        try:
            from quantmind.data.snapshot import load_snapshot
            _SNAPSHOT_CACHE[as_of] = load_snapshot(as_of)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"load_snapshot({as_of}) failed: {e}")
            _SNAPSHOT_CACHE[as_of] = {}
    return _SNAPSHOT_CACHE[as_of]


def fetch_stock_basics(ticker: str, as_of: date) -> dict[str, Any]:
    """获取股票基本信息（名称、行业、市值、PE、PB 等）.

    Args:
        ticker: 股票代码，如 "600519.SH"
        as_of:  数据截止日期（PIT）

    Returns:
        dict with keys: ticker, name, industry, sector, market_cap,
                        pe_ttm, pb, latest_close, is_tradable
    """
    snap = _load_snapshot(as_of)

    # 从 stock_basic 或 universe 取基本信息
    result: dict[str, Any] = {
        "ticker": ticker,
        "name": None,
        "industry": None,
        "sector": None,
        "market_cap": None,
        "pe_ttm": None,
        "pb": None,
        "latest_close": None,
        "is_tradable": True,
    }

    # stock_basic 表
    stock_basic: pd.DataFrame | None = snap.get("stock_basic")
    if stock_basic is not None and not stock_basic.empty:
        ts_code = _normalize_ticker(ticker)
        row = stock_basic[stock_basic.get("ts_code", stock_basic.index) == ts_code]
        if len(row) > 0:
            r = row.iloc[0]
            result["name"] = _safe_get(r, "name")
            result["industry"] = _safe_get(r, "industry")
            result["sector"] = _safe_get(r, "sector")

    # daily_basic 表（最新市值/PE/PB）
    daily_basic: pd.DataFrame | None = snap.get("daily_basic")
    if daily_basic is not None and not daily_basic.empty:
        ts_code = _normalize_ticker(ticker)
        col = "ts_code" if "ts_code" in daily_basic.columns else None
        if col:
            sub = daily_basic[daily_basic[col] == ts_code]
            if "trade_date" in daily_basic.columns:
                sub = sub[pd.to_datetime(sub["trade_date"]) <= pd.Timestamp(as_of)]
                sub = sub.sort_values("trade_date").tail(1)
            if len(sub) > 0:
                r = sub.iloc[0]
                result["pe_ttm"] = _safe_float(r, "pe_ttm")
                result["pb"] = _safe_float(r, "pb")
                result["market_cap"] = _safe_float(r, "total_mv")
                result["latest_close"] = _safe_float(r, "close")

    return result


def fetch_financials_pit(ticker: str, as_of: date) -> dict[str, Any]:
    """获取截至 as_of 的最近一期财务数据（PIT 严格）.

    Returns:
        dict with income/balance/cashflow 核心字段
    """
    snap = _load_snapshot(as_of)
    result: dict[str, Any] = {"ticker": ticker, "as_of": str(as_of)}

    ts_code = _normalize_ticker(ticker)
    as_of_ts = pd.Timestamp(as_of)

    for table_key in ["financials_income", "financials_balance_sheet", "financials_cashflow",
                      "financial_indicators"]:
        df: pd.DataFrame | None = snap.get(table_key)
        if df is None or df.empty:
            continue
        col = "ts_code" if "ts_code" in df.columns else None
        date_col = "end_date" if "end_date" in df.columns else (
            "ann_date" if "ann_date" in df.columns else None
        )
        if col is None:
            continue
        sub = df[df[col] == ts_code]
        if date_col and date_col in sub.columns:
            sub = sub[pd.to_datetime(sub[date_col]) <= as_of_ts]
            sub = sub.sort_values(date_col).tail(1)
        if len(sub) > 0:
            row = sub.iloc[0].to_dict()
            # 去掉 NaN，只保留数值字段
            for k, v in row.items():
                if k not in ("ts_code",) and not pd.isna(v) if not isinstance(v, str) else True:
                    result[f"{table_key}__{k}"] = v

    return result


def fetch_price_history(
    ticker: str,
    start: date,
    end: date,
    as_of: date,
) -> pd.DataFrame:
    """获取 [start, min(end, as_of)] 区间内的日线价格（PIT）.

    Returns:
        DataFrame with columns: trade_date, open, high, low, close, volume, pct_chg
    """
    if end > as_of:
        end = as_of

    snap = _load_snapshot(as_of)
    prices: pd.DataFrame | None = snap.get("prices")
    if prices is None or prices.empty:
        logger.warning(f"No price data in snapshot({as_of})")
        return pd.DataFrame()

    ts_code = _normalize_ticker(ticker)
    col = "ts_code" if "ts_code" in prices.columns else None
    if col is None:
        return pd.DataFrame()

    sub = prices[prices[col] == ts_code].copy()
    if "trade_date" in sub.columns:
        sub["trade_date"] = pd.to_datetime(sub["trade_date"])
        sub = sub[
            (sub["trade_date"] >= pd.Timestamp(start)) &
            (sub["trade_date"] <= pd.Timestamp(end))
        ].sort_values("trade_date")

    return sub.reset_index(drop=True)


def fetch_industry_peers(
    ticker: str,
    as_of: date,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """获取同行业股票列表（按市值排序取前 top_n）.

    Returns:
        list of dicts: [{"ticker": ..., "name": ..., "market_cap": ..., "pe_ttm": ...}]
    """
    # 先获取目标股票行业
    basics = fetch_stock_basics(ticker, as_of)
    industry = basics.get("industry")
    if not industry:
        return []

    snap = _load_snapshot(as_of)
    stock_basic: pd.DataFrame | None = snap.get("stock_basic")
    if stock_basic is None or stock_basic.empty:
        return []

    peers_df = stock_basic[stock_basic.get("industry", "") == industry] if "industry" in stock_basic.columns else pd.DataFrame()
    ts_code = _normalize_ticker(ticker)
    if "ts_code" in peers_df.columns:
        peers_df = peers_df[peers_df["ts_code"] != ts_code]

    peers: list[dict[str, Any]] = []
    for _, row in peers_df.head(top_n * 2).iterrows():
        tc = row.get("ts_code", "")
        b = fetch_stock_basics(tc, as_of)
        peers.append(b)
        if len(peers) >= top_n:
            break

    # 按市值降序
    peers.sort(key=lambda x: x.get("market_cap") or 0, reverse=True)
    return peers[:top_n]


def fetch_recent_news(
    ticker: str,
    days: int = 30,
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    """获取近 days 天的新闻（当前 snapshot 中无新闻数据，返回空列表）.

    Phase 5 知识库初始化后此函数将调用 kb_tools.search_news 补全。
    """
    logger.debug(f"fetch_recent_news({ticker}) — news data not in snapshot, returning []")
    return []


# ── 内部工具 ─────────────────────────────────────────────────────────────────

def _normalize_ticker(ticker: str) -> str:
    """统一 ticker 格式：600519.SH → 600519.SH（已是 Tushare 格式则不变）."""
    return ticker


def _safe_get(row: pd.Series, col: str) -> Any:
    try:
        v = row.get(col)
        return None if pd.isna(v) else v
    except Exception:  # noqa: BLE001
        return None


def _safe_float(row: pd.Series, col: str) -> float | None:
    try:
        v = row.get(col)
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return None
        return float(v)
    except Exception:  # noqa: BLE001
        return None
