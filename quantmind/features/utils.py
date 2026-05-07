"""quantmind.features.utils — 因子计算的通用辅助函数.

功能
====

1. TTM 计算（Trailing Twelve Months）
2. CAGR 计算（年复合增长率）
3. 横截面 rank 与 z-score（避免 sklearn 重复依赖）
4. 安全除法（分母 0 / NaN 自动返回 NaN）
5. 选取 ``as_of`` 之前最近一期的财报行
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# ============================================================================
# 安全运算
# ============================================================================


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """安全除法：分母为 0 / NaN 时返回 NaN."""
    out = num / den.where(den != 0)
    return out.replace([np.inf, -np.inf], np.nan)


def safe_log(x: pd.Series) -> pd.Series:
    """log(x)；x ≤ 0 返回 NaN."""
    return np.log(x.where(x > 0))


# ============================================================================
# 财报选取（PIT-correct latest report）
# ============================================================================


def latest_report_per_ticker(
    fin: pd.DataFrame,
    *,
    date_col: str = "f_ann_date",
    ticker_col: str = "ticker",
    report_col: str = "report_date",
) -> pd.DataFrame:
    """对每个 ticker 选 ``date_col`` 已披露的最新报告（按 ``report_col`` 排序）.

    要求：``fin`` 已按 PIT 过滤（即 ``date_col`` ≤ as_of），
    本函数仅做「每只票取最新一期」的 reduce。
    """
    if fin is None or fin.empty:
        return pd.DataFrame()
    df = fin.dropna(subset=[date_col, report_col]).copy()
    df = df.sort_values([ticker_col, report_col], ascending=[True, False])
    return df.drop_duplicates(subset=[ticker_col], keep="first").set_index(ticker_col)


def lookup_period(
    fin: pd.DataFrame,
    period_end: pd.Timestamp,
    *,
    ticker_col: str = "ticker",
    report_col: str = "report_date",
) -> pd.DataFrame:
    """选取每个 ticker 在某个 ``period_end`` 的报告行（若存在）."""
    if fin is None or fin.empty:
        return pd.DataFrame()
    df = fin[pd.to_datetime(fin[report_col]) == period_end]
    return df.drop_duplicates(subset=[ticker_col], keep="first").set_index(ticker_col)


# ============================================================================
# YTD → TTM 转换
# ============================================================================


def ytd_to_ttm(
    fin: pd.DataFrame,
    value_col: str,
    *,
    ticker_col: str = "ticker",
    report_col: str = "report_date",
    ann_col: str = "f_ann_date",
    as_of: date | None = None,
) -> pd.Series:
    """把 Tushare 的 YTD 字段（年初至报告期）换算为 TTM（最近 12 个月）.

    公式：
        TTM_t = current_period_YTD + prior_year_annual - prior_year_same_period_YTD

    如果当期是年报（Q4），则 TTM = 当期 YTD（即 annual）。

    Args:
        fin: 财报 DataFrame，必含 ticker / report_date / value_col 列
        value_col: 要转换的字段，如 ``total_revenue`` / ``n_income``
        as_of: 仅取 ``ann_col`` ≤ as_of 的记录（防御 PIT）

    Returns:
        Series 索引为 ticker，值为该字段的 TTM
    """
    if fin is None or fin.empty or value_col not in fin.columns:
        return pd.Series(dtype="float64")

    df = fin.copy()
    df[report_col] = pd.to_datetime(df[report_col])
    if ann_col in df.columns:
        df[ann_col] = pd.to_datetime(df[ann_col])
        if as_of is not None:
            df = df[df[ann_col] <= pd.Timestamp(as_of)]

    df = df[[ticker_col, report_col, value_col]].dropna(subset=[report_col])
    if df.empty:
        return pd.Series(dtype="float64")

    out: dict[str, float] = {}
    for ticker, g in df.groupby(ticker_col):
        g = g.sort_values(report_col, ascending=False)
        if g.empty:
            continue
        # 当期
        cur = g.iloc[0]
        cur_period = cur[report_col]
        cur_ytd = cur[value_col]
        if pd.isna(cur_ytd):
            out[ticker] = np.nan
            continue

        # 当期是年报（12-31）→ TTM = 年报值
        if cur_period.month == 12 and cur_period.day == 31:
            out[ticker] = float(cur_ytd)
            continue

        # 找上一年年报（Q4）
        prev_year_q4 = pd.Timestamp(year=cur_period.year - 1, month=12, day=31)
        prev_q4_row = g[g[report_col] == prev_year_q4]

        # 找上一年同期 YTD
        prev_same_period = pd.Timestamp(
            year=cur_period.year - 1, month=cur_period.month, day=cur_period.day
        )
        prev_same_row = g[g[report_col] == prev_same_period]

        if prev_q4_row.empty or prev_same_row.empty:
            out[ticker] = np.nan
            continue

        prev_q4_v = float(prev_q4_row[value_col].iloc[0])
        prev_same_v = float(prev_same_row[value_col].iloc[0])
        if pd.isna(prev_q4_v) or pd.isna(prev_same_v):
            out[ticker] = np.nan
            continue

        out[ticker] = float(cur_ytd) + prev_q4_v - prev_same_v

    return pd.Series(out, dtype="float64")


# ============================================================================
# CAGR / Growth 计算
# ============================================================================


def cagr(start: pd.Series, end: pd.Series, periods: float) -> pd.Series:
    """计算复合年增长率：(end/start)^(1/periods) - 1.

    起点 ≤ 0 或 终点 ≤ 0 / NaN 返回 NaN。
    """
    s = start.where(start > 0)
    e = end.where(end > 0)
    ratio = e / s
    return ratio.pow(1.0 / periods) - 1.0


# ============================================================================
# Rolling 时序辅助（行情用）
# ============================================================================


def pivot_prices(
    prices: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "trade_date",
    value_col: str = "close",
) -> pd.DataFrame:
    """把长格式 prices DataFrame 转成宽格式（index=date, columns=ticker, values=close）."""
    if prices is None or prices.empty:
        return pd.DataFrame()
    df = prices[[date_col, ticker_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.pivot(index=date_col, columns=ticker_col, values=value_col).sort_index()


__all__ = [
    "cagr",
    "latest_report_per_ticker",
    "lookup_period",
    "pivot_prices",
    "safe_divide",
    "safe_log",
    "ytd_to_ttm",
]
