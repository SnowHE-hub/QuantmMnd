"""quantmind/features/expr_factors.py

轻量级 Qlib 风格表达式因子引擎（纯 Python / pandas，无需安装 pyqlib）。

设计
----
- 每个因子用 Qlib 风格表达式字符串文档化（可读性 + 可移植性）
- 执行层使用 pandas rolling + EWM，无依赖外部因子引擎
- 支持从 daily_prices_panel.parquet 直接计算，输出与 alpha_panel 格式对齐
- 7 个内置因子，覆盖动量/波动/流动性/技术形态

表达式语法（文档用，非运行时解析）
-------------------------------------
::

    $field              原始数据字段（close / adj_close / vol / amount / pct_chg）
    Ref($x, n)          n 期滞后
    Mean($x, n)         n 期滚动均值
    Std($x, n)          n 期滚动标准差
    Abs($x)             绝对值
    Greater($x, y)      element-wise max(x, y)
    Log($x)             自然对数
    Rank($x)            截面百分位排名 [0, 1]

一致性验证（与 alpha_panel_v4 同名因子）：
    momentum_21d   ↔ momentum_1m        corr > 0.99
    reversal_5d    ↔ reversal_1w        corr > 0.99
    volatility_63d ↔ volatility_3m      corr > 0.99
    amihud_63d     ↔ amihud_illiquidity corr > 0.95
    rsi_14_expr    ↔ rsi_14             corr > 0.99
    bollinger_pos  ↔ bollinger_position corr > 0.90
    volume_ratio   ↔ volume_spike_5_30  corr > 0.80  (窗口不同)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─── 低层时序算子（操作宽格式 DataFrame: date × ticker）────────────────────────

def Ref(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Ref($x, n)：n 期滞后值。"""
    return df.shift(n)


def Mean(df: pd.DataFrame, n: int, min_periods: Optional[int] = None) -> pd.DataFrame:
    """Mean($x, n)：n 期滚动均值。"""
    mp = min_periods if min_periods is not None else max(1, n // 2)
    return df.rolling(n, min_periods=mp).mean()


def Std(df: pd.DataFrame, n: int, min_periods: Optional[int] = None) -> pd.DataFrame:
    """Std($x, n)：n 期滚动标准差（ddof=1）。"""
    mp = min_periods if min_periods is not None else max(2, n // 2)
    return df.rolling(n, min_periods=mp).std()


def RollingMax(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 期滚动最大值。"""
    return df.rolling(n, min_periods=1).max()


def RollingMin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 期滚动最小值。"""
    return df.rolling(n, min_periods=1).min()


def Abs(df: pd.DataFrame) -> pd.DataFrame:
    """Abs($x)：绝对值。"""
    return df.abs()


def Log(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Log($x)：自然对数（自动 clip 防止 log(0)）。"""
    return np.log(df.clip(lower=eps))


def Greater(df: pd.DataFrame, val) -> pd.DataFrame:
    """Greater($x, val)：clip(lower=val)，即 max(x, val)。"""
    return df.clip(lower=val)


def Rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank($x)：截面百分位排名 [0, 1]（每行内排名）。"""
    return df.rank(axis=1, pct=True)


# ─── RSI 实现（EWM Wilder 平滑，与 technical.py 一致）────────────────────────

def _ewm_rsi(close_wide: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder EWM RSI，alpha = 1/period。与 quantmind.features.technical.rsi_14 一致。"""
    delta = close_wide.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs    = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ─── 布林带位置（与 technical.bollinger_position 一致）───────────────────────

def _bollinger_position(close_wide: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """(close - lower) / (upper - lower)  where upper/lower = mean ± 2·std over n days."""
    mid   = Mean(close_wide, n, min_periods=n // 2)
    std_  = Std(close_wide, n, min_periods=n // 2)
    upper = mid + 2.0 * std_
    lower = mid - 2.0 * std_
    band  = (upper - lower).replace(0.0, np.nan)
    return (close_wide - lower) / band


# ─── 数据加载 ─────────────────────────────────────────────────────────────────

DAILY_PRICES_FIELDS = ["close", "adj_close", "vol", "amount", "pct_chg",
                       "high", "low", "open"]


def load_prices_context(prices_path: str) -> dict[str, pd.DataFrame]:
    """daily_prices_panel.parquet → 宽格式 dict（date × ticker）。

    Parameters
    ----------
    prices_path : str  e.g. "data/raw/daily_prices_panel.parquet"

    Returns
    -------
    dict mapping field name → pd.DataFrame(index=trade_date, columns=ts_code)
    """
    df = pd.read_parquet(prices_path)
    df = df.sort_values(["trade_date", "ts_code"])
    ctx: dict[str, pd.DataFrame] = {}
    for field in DAILY_PRICES_FIELDS:
        if field in df.columns:
            wide = df.pivot(index="trade_date", columns="ts_code", values=field)
            ctx[field] = wide.sort_index()
    log.info(
        "load_prices_context: %d dates × %d tickers  fields=%s",
        ctx["close"].shape[0] if "close" in ctx else 0,
        ctx["close"].shape[1] if "close" in ctx else 0,
        list(ctx.keys()),
    )
    return ctx


# ─── 因子数据类 ───────────────────────────────────────────────────────────────

@dataclass
class ExprFactor:
    """Qlib 风格表达式因子定义。

    Attributes
    ----------
    name     : 因子名称（与 alpha_panel 列名对应时便于验证）
    expr_str : Qlib 表达式字符串（文档/可移植性用）
    panel_equiv : alpha_panel 对应列名（None 表示无直接对应）
    eval_fn  : Callable[dict[str, pd.DataFrame]] → pd.DataFrame(date × ticker)
    """
    name: str
    expr_str: str
    panel_equiv: Optional[str]
    eval_fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]

    def eval(self, ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """在给定 price context 上计算全时序因子（wide format）。"""
        return self.eval_fn(ctx)

    def __repr__(self) -> str:
        return (f"ExprFactor(name={self.name!r}, "
                f"expr={self.expr_str!r}, "
                f"equiv={self.panel_equiv!r})")


# ─── 7 个内置因子定义 ─────────────────────────────────────────────────────────

EXPR_FACTORS: dict[str, ExprFactor] = {

    # 1. 21 日动量（= momentum_1m）
    "momentum_21d": ExprFactor(
        name       = "momentum_21d",
        expr_str   = "$close / Ref($close, 21) - 1",
        panel_equiv= "momentum_1m",
        eval_fn    = lambda ctx: (
            ctx["close"] / Ref(ctx["close"], 21) - 1.0
        ),
    ),

    # 2. 5 日反转（= reversal_1w）
    "reversal_5d": ExprFactor(
        name       = "reversal_5d",
        expr_str   = "$close / Ref($close, 5) - 1",
        panel_equiv= "reversal_1w",
        eval_fn    = lambda ctx: (
            ctx["close"] / Ref(ctx["close"], 5) - 1.0
        ),
    ),

    # 3. 63 日年化波动率（= volatility_3m）
    "volatility_63d": ExprFactor(
        name       = "volatility_63d",
        expr_str   = "Std($pct_chg / 100, 63) * Sqrt(252)",
        panel_equiv= "volatility_3m",
        eval_fn    = lambda ctx: (
            Std(ctx["pct_chg"] / 100.0, 63, min_periods=20) * np.sqrt(252)
        ),
    ),

    # 4. Amihud 非流动性（= amihud_illiquidity，63 日窗口）
    "amihud_63d": ExprFactor(
        name       = "amihud_63d",
        expr_str   = "Mean(Abs($pct_chg) / Greater($amount, 1e-9), 63) * 1e9",
        panel_equiv= "amihud_illiquidity",
        eval_fn    = lambda ctx: (
            (Abs(ctx["pct_chg"]) / Greater(ctx["amount"], 1e-9))
            .rolling(63, min_periods=20).mean() * 1e9
        ),
    ),

    # 5. 成交量短长期比（≈ volume_spike_5_30，窗口改为 5/20）
    "volume_ratio_5_20": ExprFactor(
        name       = "volume_ratio_5_20",
        expr_str   = "Mean($vol, 5) / Greater(Mean($vol, 20), 1e-9)",
        panel_equiv= "volume_spike_5_30",
        eval_fn    = lambda ctx: (
            Mean(ctx["vol"], 5) / Greater(Mean(ctx["vol"], 20), 1e-9)
        ),
    ),

    # 6. RSI(14) EWM（= rsi_14，与 technical.py 完全一致）
    "rsi_14_expr": ExprFactor(
        name       = "rsi_14_expr",
        expr_str   = "100 - 100 / (1 + EWM_gain($close,14) / EWM_loss($close,14))",
        panel_equiv= "rsi_14",
        eval_fn    = lambda ctx: _ewm_rsi(ctx["close"], period=14),
    ),

    # 7. 布林带位置（= bollinger_position，(close-lower)/(upper-lower)）
    "bollinger_pos_20": ExprFactor(
        name       = "bollinger_pos_20",
        expr_str   = "(close - (Mean($close,20) - 2*Std($close,20))) / (4*Std($close,20))",
        panel_equiv= "bollinger_position",
        eval_fn    = lambda ctx: _bollinger_position(ctx["close"], n=20),
    ),
}

EXPR_FACTOR_NAMES: list[str] = list(EXPR_FACTORS.keys())


# ─── 截面提取 API ─────────────────────────────────────────────────────────────

def eval_factor_at_dates(
    factor: ExprFactor,
    ctx: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """在给定日期列表提取因子截面，返回 DataFrame(as_of × ts_code)。

    Parameters
    ----------
    factor : ExprFactor
    ctx    : load_prices_context() 输出
    dates  : 提取日期（如季末日列表），会对齐到实际存在的最近日期

    Returns
    -------
    pd.DataFrame  index=dates（对齐后），columns=ts_code
    """
    wide = factor.eval(ctx)           # date × ticker，全时序
    avail = wide.index

    rows = []
    for dt in dates:
        # 取 ≤ dt 的最近可用日期
        idx = avail.searchsorted(dt, side="right") - 1
        if idx < 0:
            rows.append(pd.Series(dtype=float, name=dt))
        else:
            rows.append(wide.iloc[idx].rename(dt))

    result = pd.DataFrame(rows)
    result.index.name = "as_of"
    result.columns.name = "ts_code"
    return result


def build_expr_panel(
    prices_path: str = "data/raw/daily_prices_panel.parquet",
    quarter_dates: Optional[pd.DatetimeIndex] = None,
    factors: Optional[list[str]] = None,
) -> pd.DataFrame:
    """计算全部（或指定）表达式因子，返回 MultiIndex(as_of, ts_code) panel。

    Parameters
    ----------
    prices_path   : daily_prices_panel.parquet 路径
    quarter_dates : 提取日期（默认取 prices 内所有季末日）
    factors       : 因子名称列表（默认全部 7 个）

    Returns
    -------
    pd.DataFrame  MultiIndex(as_of, ts_code) × factor_columns
    """
    ctx = load_prices_context(prices_path)

    if quarter_dates is None:
        all_dates = ctx["close"].index
        quarter_dates = all_dates[all_dates.is_quarter_end]

    factor_names = factors if factors is not None else EXPR_FACTOR_NAMES
    missing = [n for n in factor_names if n not in EXPR_FACTORS]
    if missing:
        raise KeyError(f"未知因子: {missing}")

    slices: list[pd.DataFrame] = []
    for name in factor_names:
        wide = eval_factor_at_dates(EXPR_FACTORS[name], ctx, quarter_dates)  # (dates, tickers)
        slices.append(wide.stack(dropna=False).rename(name))

    panel = pd.concat(slices, axis=1)
    panel.index.names = ["as_of", "ts_code"]
    log.info(
        "build_expr_panel: %d rows × %d factors  (dates=%d, tickers≈%d)",
        len(panel), len(factor_names),
        len(quarter_dates), len(panel) // max(len(quarter_dates), 1),
    )
    return panel


# ─── 参考 Python 实现（验证用，与 expr 在同一数据上比较）────────────────────

def _ref_momentum_21d(ctx: dict) -> pd.DataFrame:
    """参考实现：close / close.shift(21) - 1（同 momentum_1m 公式）。"""
    c = ctx["close"]
    return c / c.shift(21) - 1.0


def _ref_reversal_5d(ctx: dict) -> pd.DataFrame:
    """参考实现：close / close.shift(5) - 1（同 reversal_1w 公式）。"""
    c = ctx["close"]
    return c / c.shift(5) - 1.0


def _ref_volatility_63d(ctx: dict) -> pd.DataFrame:
    """参考实现：std(pct_chg/100, 63) * sqrt(252)（同 volatility_3m）。"""
    r = ctx["pct_chg"] / 100.0
    return r.rolling(63, min_periods=20).std() * np.sqrt(252)


def _ref_amihud_63d(ctx: dict) -> pd.DataFrame:
    """参考实现：mean(|pct_chg| / amount, 63) * 1e9。"""
    ratio = ctx["pct_chg"].abs() / ctx["amount"].clip(lower=1e-9)
    return ratio.rolling(63, min_periods=20).mean() * 1e9


def _ref_volume_ratio_5_20(ctx: dict) -> pd.DataFrame:
    """参考实现：mean(vol, 5) / mean(vol, 20)。"""
    v5  = ctx["vol"].rolling(5, min_periods=1).mean()
    v20 = ctx["vol"].rolling(20, min_periods=5).mean().clip(lower=1e-9)
    return v5 / v20


def _ref_rsi_14(ctx: dict) -> pd.DataFrame:
    """参考实现：EWM RSI(14)，alpha=1/14（同 technical.rsi_14）。"""
    return _ewm_rsi(ctx["close"], period=14)


def _ref_bollinger_pos_20(ctx: dict) -> pd.DataFrame:
    """参考实现：(close - lower) / (upper - lower)，窗口 20（同 bollinger_position）。"""
    return _bollinger_position(ctx["close"], n=20)


# 参考实现注册表（与 EXPR_FACTORS 名称对应）
_REFERENCE_IMPLS: dict[str, Callable] = {
    "momentum_21d"    : _ref_momentum_21d,
    "reversal_5d"     : _ref_reversal_5d,
    "volatility_63d"  : _ref_volatility_63d,
    "amihud_63d"      : _ref_amihud_63d,
    "volume_ratio_5_20": _ref_volume_ratio_5_20,
    "rsi_14_expr"     : _ref_rsi_14,
    "bollinger_pos_20": _ref_bollinger_pos_20,
}


def validate_expr_consistency(
    ctx: dict[str, pd.DataFrame],
    dates: Optional[pd.DatetimeIndex] = None,
    min_corr: float = 0.99,
) -> pd.DataFrame:
    """验证表达式因子与等效参考 Python 实现的截面 Spearman 相关性。

    在 **相同** 底层数据（daily_prices_panel）上，比较
    ExprFactor.eval_fn 与 _REFERENCE_IMPLS 的输出排名一致性。

    Parameters
    ----------
    ctx      : load_prices_context() 输出
    dates    : 验证日期（默认取 ctx["close"].index 中季末日）
    min_corr : 期望最低均值相关性（默认 0.99）

    Returns
    -------
    pd.DataFrame  columns=[factor, mean_corr, min_corr_date, max_corr_date,
                            n_dates, pass]
    """
    if dates is None:
        all_d = ctx["close"].index
        dates = all_d[all_d.is_quarter_end]

    rows = []
    for name, fac in EXPR_FACTORS.items():
        ref_fn = _REFERENCE_IMPLS.get(name)
        if ref_fn is None:
            log.warning("跳过 %s：无参考实现", name)
            continue

        # 计算全时序
        expr_wide = fac.eval(ctx)    # date × ticker
        ref_wide  = ref_fn(ctx)      # date × ticker

        corrs = []
        for dt in dates:
            idx = expr_wide.index.searchsorted(dt, side="right") - 1
            if idx < 0:
                continue
            expr_row = expr_wide.iloc[idx]
            ref_row  = ref_wide.iloc[idx]

            # 对齐，去 NaN
            merged = pd.concat([expr_row.rename("expr"), ref_row.rename("ref")],
                               axis=1).dropna()
            if len(merged) < 10:
                continue
            c = merged["expr"].corr(merged["ref"], method="spearman")
            if not np.isnan(c):
                corrs.append(c)

        if not corrs:
            continue
        arr = np.array(corrs)
        rows.append({
            "factor"        : name,
            "mean_corr"     : float(np.mean(arr)),
            "min_corr_date" : float(np.min(arr)),
            "max_corr_date" : float(np.max(arr)),
            "n_dates"       : len(arr),
            "pass"          : bool(np.mean(arr) >= min_corr),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("mean_corr", ascending=False)
    log.info("=== 表达式引擎一致性验证（expr vs reference Python）===")
    for _, row in result.iterrows():
        status = "✓ PASS" if row["pass"] else "✗ FAIL"
        log.info(
            "  %s  %-22s  mean_corr=%.6f  (%d 期)",
            status, row["factor"], row["mean_corr"], int(row["n_dates"]),
        )
    n_pass = int(result["pass"].sum())
    log.info("总计 %d/%d 因子通过 (min_corr=%.2f)", n_pass, len(result), min_corr)
    return result


__all__ = [
    "DAILY_PRICES_FIELDS",
    "EXPR_FACTOR_NAMES",
    "EXPR_FACTORS",
    "ExprFactor",
    # operators
    "Abs",
    "Greater",
    "Log",
    "Mean",
    "Rank",
    "Ref",
    "RollingMax",
    "RollingMin",
    "Std",
    # API
    "build_expr_panel",
    "eval_factor_at_dates",
    "load_prices_context",
    "validate_expr_consistency",
]
