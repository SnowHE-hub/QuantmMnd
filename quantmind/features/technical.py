"""quantmind.features.technical — 量价因子.

设计
====

每个因子签名::

    def factor_name(snapshot: dict, as_of: date) -> pd.Series  # index=ticker

输入 ``snapshot`` 的 ``prices`` 表是长格式（每只票一段时间序列），
内部一律先 ``pivot`` 成宽格式（index=date, columns=ticker），用 numpy/pandas
向量化计算，避免 per-ticker 循环。

参考文献
========

- Jegadeesh & Titman (1993) — Momentum
- Carhart (1997) — UMD
- Asness et al. (2013) — Quality minus Junk
- Amihud (2002) — Illiquidity
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quantmind.features.utils import pivot_prices, safe_divide

# ============================================================================
# 动量因子（Momentum）
# ============================================================================


def _trailing_return(close: pd.DataFrame, days: int) -> pd.Series:
    """最后一个交易日相对 ``days`` 个交易日前的累计收益."""
    if close is None or close.empty or len(close) <= days:
        return pd.Series(dtype="float64")
    end = close.iloc[-1]
    start = close.iloc[-days - 1]
    return safe_divide(end, start) - 1.0


def momentum_1m(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """近 1 个月动量（约 21 个交易日）."""
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    return _trailing_return(close, 21).rename("momentum_1m")


def momentum_3m(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    return _trailing_return(close, 63).rename("momentum_3m")


def momentum_6m(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    return _trailing_return(close, 126).rename("momentum_6m")


def momentum_12m_skip_1m(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """剔除最近 1 个月的 12 个月动量（Carhart 经典定义，排除短期反转污染）.

    = (P_{-21} / P_{-252}) - 1
    """
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    if close.empty or len(close) <= 252:
        return pd.Series(dtype="float64")
    end = close.iloc[-22]   # 21 days ago
    start = close.iloc[-253]  # 252 days ago
    return (safe_divide(end, start) - 1.0).rename("momentum_12m_skip_1m")


# ============================================================================
# 反转因子（Reversal）
# ============================================================================


def reversal_1w(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """近 1 周收益率（5 日）— 短期反转因子，预期负相关于未来收益."""
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    return _trailing_return(close, 5).rename("reversal_1w")


# ============================================================================
# 波动率因子（Volatility）
# ============================================================================


def _daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change()


def volatility_3m(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """3 个月日收益率的年化波动率（√252 缩放）."""
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    if close.empty or len(close) < 21:
        return pd.Series(dtype="float64")
    rets = _daily_returns(close).iloc[-63:]
    return (rets.std() * np.sqrt(252)).rename("volatility_3m")


def volatility_1y(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """1 年日收益率的年化波动率."""
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    if close.empty:
        return pd.Series(dtype="float64")
    rets = _daily_returns(close).iloc[-252:]
    return (rets.std() * np.sqrt(252)).rename("volatility_1y")


def downside_volatility_3m(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """下行波动率（仅负收益）— 风险厌恶投资者更关心."""
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    if close.empty:
        return pd.Series(dtype="float64")
    rets = _daily_returns(close).iloc[-63:]
    neg = rets.where(rets < 0)
    return (neg.std() * np.sqrt(252)).rename("downside_volatility_3m")


def max_drawdown_3m(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """近 3 个月最大回撤（绝对值，越大越差）."""
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    if close.empty:
        return pd.Series(dtype="float64")
    sub = close.iloc[-63:]
    cummax = sub.cummax()
    dd = (sub - cummax) / cummax
    return dd.min().abs().rename("max_drawdown_3m")


# ============================================================================
# 流动性因子（Liquidity）
# ============================================================================


def amihud_illiquidity(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """Amihud (2002) 非流动性 = mean(|return| / 成交额) × 1e9.

    高值 = 流动性差。乘数仅为可读性放大，不影响排序。
    """
    px = snapshot.get("prices")
    if px is None or px.empty or "amount" not in px.columns:
        return pd.Series(dtype="float64")

    close = pivot_prices(px)
    amount = pivot_prices(px, value_col="amount")
    rets = _daily_returns(close).iloc[-63:].abs()
    amt = amount.iloc[-63:].where(amount.iloc[-63:] > 0)
    illiq = (rets / amt).mean() * 1e9
    return illiq.rename("amihud_illiquidity")


def turnover_3m_avg(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """近 3 个月平均换手率（%）."""
    px = snapshot.get("prices")
    if px is None or px.empty or "turnover_rate" not in px.columns:
        return pd.Series(dtype="float64")
    tr = pivot_prices(px, value_col="turnover_rate")
    return tr.iloc[-63:].mean().rename("turnover_3m_avg")


def volume_spike_5_30(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """成交量异常 = 近 5 日均量 / 前 30 日均量."""
    px = snapshot.get("prices")
    if px is None or px.empty or "volume" not in px.columns:
        return pd.Series(dtype="float64")
    vol = pivot_prices(px, value_col="volume")
    if len(vol) < 35:
        return pd.Series(dtype="float64")
    recent = vol.iloc[-5:].mean()
    prior = vol.iloc[-35:-5].mean()
    return safe_divide(recent, prior).rename("volume_spike_5_30")


# ============================================================================
# 技术形态（Technical patterns）
# ============================================================================


def rsi_14(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """14 日相对强弱指数（Wilder 平滑版近似，用 EMA）."""
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    if close.empty or len(close) < 30:
        return pd.Series(dtype="float64")
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = safe_divide(gain, loss)
    rsi = 100 - 100 / (1 + rs)
    return rsi.iloc[-1].rename("rsi_14")


def bollinger_position(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """布林带位置 = (close - lower_band) / (upper_band - lower_band).

    20 日均线 ± 2 倍标准差。值在 [0, 1] 表示在带内，超界表示突破。
    """
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    if close.empty or len(close) < 20:
        return pd.Series(dtype="float64")
    sub = close.iloc[-20:]
    mid = sub.mean()
    std = sub.std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    cur = close.iloc[-1]
    pos = safe_divide(cur - lower, upper - lower)
    return pos.rename("bollinger_position")


def distance_to_52w_high(snapshot: dict, as_of: date) -> pd.Series:  # noqa: ARG001
    """52 周最高价距离 = (current - 52w_high) / 52w_high.

    更接近 0 表示越靠近 52 周高点（动量延续指标）。
    """
    close = pivot_prices(snapshot.get("prices", pd.DataFrame()))
    if close.empty or len(close) < 252:
        return pd.Series(dtype="float64")
    high_52w = close.iloc[-252:].max()
    cur = close.iloc[-1]
    return ((cur - high_52w) / high_52w).rename("distance_to_52w_high")


# ============================================================================
# 公开 API
# ============================================================================


TECHNICAL_FACTORS = [
    # 动量 (4)
    ("momentum_1m", momentum_1m),
    ("momentum_3m", momentum_3m),
    ("momentum_6m", momentum_6m),
    ("momentum_12m_skip_1m", momentum_12m_skip_1m),
    # 反转 (1)
    ("reversal_1w", reversal_1w),
    # 波动 (4)
    ("volatility_3m", volatility_3m),
    ("volatility_1y", volatility_1y),
    ("downside_volatility_3m", downside_volatility_3m),
    ("max_drawdown_3m", max_drawdown_3m),
    # 流动性 (3)
    ("amihud_illiquidity", amihud_illiquidity),
    ("turnover_3m_avg", turnover_3m_avg),
    ("volume_spike_5_30", volume_spike_5_30),
    # 技术形态 (3)
    ("rsi_14", rsi_14),
    ("bollinger_position", bollinger_position),
    ("distance_to_52w_high", distance_to_52w_high),
]


def compute_all_technical_factors(
    snapshot: dict, as_of: date
) -> pd.DataFrame:
    """计算所有量价因子，返回 DataFrame(index=ticker, columns=factor_name)."""
    out = {}
    for name, fn in TECHNICAL_FACTORS:
        try:
            s = fn(snapshot, as_of)
        except Exception:  # noqa: BLE001
            s = pd.Series(dtype="float64", name=name)
        out[name] = s
    df = pd.DataFrame(out)
    df.index.name = "ticker"
    return df


__all__ = [
    "TECHNICAL_FACTORS",
    "amihud_illiquidity",
    "bollinger_position",
    "compute_all_technical_factors",
    "distance_to_52w_high",
    "downside_volatility_3m",
    "max_drawdown_3m",
    "momentum_12m_skip_1m",
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "reversal_1w",
    "rsi_14",
    "turnover_3m_avg",
    "volatility_1y",
    "volatility_3m",
    "volume_spike_5_30",
]
