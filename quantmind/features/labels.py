"""quantmind.features.labels — 训练用前向收益标签（离线构建）。

截面因子仍严格来自 snapshot PIT；标签从**完整日线**截取未来 ``n`` 个交易日收益，
不向特征截面泄露——仅在有全部历史回填的前提下用于监督学习标签列。
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def _close_series_sorted(prices_chunk: pd.DataFrame, *, date_col: str, price_col: str) -> pd.Series:
    px = prices_chunk[[date_col, price_col]].dropna(subset=[price_col]).copy()
    if px.empty:
        return pd.Series(dtype="float64")
    px["_ts"] = pd.to_datetime(px[date_col], errors="coerce").dt.normalize()
    px = px.dropna(subset=["_ts"]).sort_values("_ts").drop_duplicates(subset=["_ts"], keep="last")
    return px.set_index("_ts")[price_col].astype("float64")


def forward_return_by_position(close_ser: pd.Series, anchor_trade_date: date, n_bars: int) -> float:
    """``close_ser`` 索引为递增 ``DatetimeIndex``（同一天至多一条）。"""
    if close_ser.empty or n_bars <= 0:
        return float("nan")
    anchor_ts = pd.Timestamp(anchor_trade_date).normalize()
    ipos = int(close_ser.index.searchsorted(anchor_ts, side="right") - 1)
    if ipos < 0:
        return float("nan")
    jpos = ipos + n_bars
    if jpos >= len(close_ser):
        return float("nan")
    p0 = float(close_ser.iloc[ipos])
    p1 = float(close_ser.iloc[jpos])
    if p0 <= 0 or pd.isna(p0) or pd.isna(p1):
        return float("nan")
    return p1 / p0 - 1.0


def forward_return_n_bars(
    prices_long: pd.DataFrame,
    *,
    ticker: str,
    anchor_trade_date: date,
    n_bars: int,
    date_col: str = "trade_date",
    price_col: str = "close",
    _cache: dict[str, pd.Series] | None = None,
) -> float:
    """从 ``anchor_trade_date`` 收盘价持有 ``n_bars`` 个**交易日**的简单收益."""
    if _cache is not None:
        ser = _cache.get(ticker)
        if ser is None:
            ch = prices_long[prices_long["ticker"] == ticker]
            ser = _close_series_sorted(ch, date_col=date_col, price_col=price_col)
            _cache[ticker] = ser
        return forward_return_by_position(ser, anchor_trade_date, n_bars)

    px = prices_long[prices_long["ticker"] == ticker][[date_col, price_col]]
    ser = _close_series_sorted(px, date_col=date_col, price_col=price_col)
    return forward_return_by_position(ser, anchor_trade_date, n_bars)


def attach_forward_returns(
    panel_factors: pd.DataFrame,
    prices_long: pd.DataFrame,
    *,
    horizons: list[int],
    date_col: str = "trade_date",
    price_col: str = "close",
) -> pd.DataFrame:
    """给 ``panel_factors`` 挂上 ``fwd_ret_{h}`` 列。

    ``panel_factors`` 须有多级索引 ``(as_of, ticker)`` （有序即可）。
    """
    _cache: dict[str, pd.Series] = {}

    df = panel_factors.copy()
    for h in horizons:
        col = f"fwd_ret_{h}"
        srs: list[float] = []
        for (as_raw, ticker) in df.index:
            ad = pd.Timestamp(as_raw).date()
            srs.append(
                forward_return_n_bars(
                    prices_long,
                    ticker=str(ticker),
                    anchor_trade_date=ad,
                    n_bars=h,
                    date_col=date_col,
                    price_col=price_col,
                    _cache=_cache,
                )
            )
        df[col] = srs
    return df


__all__ = ["attach_forward_returns", "forward_return_n_bars"]
