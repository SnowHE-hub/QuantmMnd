"""quantmind.features.sentiment — 情绪 / 流动性微观因子.

注：LLM-extracted 情绪因子（管理层语调、研报情绪等）放在
``quantmind/features/llm_features.py``（Phase 4 实现），本模块仅含
**结构化数值**情绪信号，无需 LLM。

包含
====

1. 北向资金（市场级，所有 ticker 共享）
2. 换手率分位（个股，相对自身 1 年历史）
3. 振幅分位
4. （后续可加：融资融券余额变化、大单净流入）
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantmind.features.utils import pivot_prices

# ============================================================================
# 市场级情绪
# ============================================================================


def _north_bound_30d_net(snapshot: dict) -> float:
    """近 30 个交易日北向资金净流入合计（亿元）— 全市场单值."""
    nb = snapshot.get("north_bound")
    if nb is None or nb.empty or "north_money" not in nb.columns:
        return float("nan")
    s = nb.sort_values("trade_date").tail(30)["north_money"].astype("float64")
    return float(s.sum())


def north_bound_30d_net_inflow(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """北向资金 30 日净流入合计（亿元）— 市场级，对所有 ticker 同值.

    高 = 外资乐观；为对回归与 ranker 友好，作为「市场环境特征」给所有股票同样值，
    后续可与个股北向持仓变化叉乘。
    """
    db = snapshot.get("daily_basic")
    universe_idx = db.set_index("ticker").index if (db is not None and not db.empty) else pd.Index([])
    val = _north_bound_30d_net(snapshot)
    return pd.Series(val, index=universe_idx, name="north_bound_30d_net_inflow", dtype="float64")


# ============================================================================
# 个股微观情绪
# ============================================================================


def turnover_rate_quantile(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """近 5 日换手率均值在过去 252 日分布中的分位数（0~1）.

    高 = 换手活跃（资金关注度高）；低 = 冷门股。
    """
    px = snapshot.get("prices")
    if px is None or px.empty or "turnover_rate" not in px.columns:
        return pd.Series(dtype="float64")
    tr = pivot_prices(px, value_col="turnover_rate")
    if len(tr) < 252:
        return pd.Series(dtype="float64")
    recent5 = tr.iloc[-5:].mean()
    hist252 = tr.iloc[-252:]
    out = (hist252 < recent5).mean()  # 当前值在历史中的分位
    return out.rename("turnover_rate_quantile")


def amplitude_quantile(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """近 5 日振幅 (high-low)/close 均值在 252 日分布中的分位数."""
    px = snapshot.get("prices")
    if px is None or px.empty:
        return pd.Series(dtype="float64")
    needed = {"high", "low", "close"}
    if not needed.issubset(set(px.columns)):
        return pd.Series(dtype="float64")
    high = pivot_prices(px, value_col="high")
    low = pivot_prices(px, value_col="low")
    close = pivot_prices(px, value_col="close")
    amp = (high - low) / close
    if len(amp) < 252:
        return pd.Series(dtype="float64")
    recent5 = amp.iloc[-5:].mean()
    hist252 = amp.iloc[-252:]
    out = (hist252 < recent5).mean()
    return out.rename("amplitude_quantile")


def free_float_ratio(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """自由流通股 / 总股本 — 越低越「有概念」越易被资金推升."""
    db = snapshot.get("daily_basic")
    if db is None or db.empty:
        return pd.Series(dtype="float64")
    if not {"free_share", "total_share"}.issubset(set(db.columns)):
        return pd.Series(dtype="float64")
    df = db.set_index("ticker")
    out = df["free_share"] / df["total_share"].where(df["total_share"] > 0)
    return out.replace([np.inf, -np.inf], np.nan).rename("free_float_ratio")


# ============================================================================
# 公开 API
# ============================================================================


SENTIMENT_FACTORS = [
    ("north_bound_30d_net_inflow", north_bound_30d_net_inflow),
    ("turnover_rate_quantile", turnover_rate_quantile),
    ("amplitude_quantile", amplitude_quantile),
    ("free_float_ratio", free_float_ratio),
]


def compute_all_sentiment_factors(
    snapshot: dict, as_of: date
) -> pd.DataFrame:
    out = {}
    for name, fn in SENTIMENT_FACTORS:
        try:
            s = fn(snapshot, as_of)
        except Exception:  # noqa: BLE001
            s = pd.Series(dtype="float64", name=name)
        out[name] = s
    df = pd.DataFrame(out)
    df.index.name = "ticker"
    return df


__all__ = [
    "SENTIMENT_FACTORS",
    "amplitude_quantile",
    "compute_all_sentiment_factors",
    "free_float_ratio",
    "north_bound_30d_net_inflow",
    "turnover_rate_quantile",
]
