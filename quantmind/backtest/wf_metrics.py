"""quantmind.backtest.wf_metrics — Walk-forward 闸门专属指标.

依据 ``docs/plans/walkforward_design_plan.md`` §5 / §12-E。复用 ``PerformanceMetrics``
做 MDD/Sharpe/alpha；本模块**新建**三项口径敏感的指标：

- **每换仓期对基准胜率（H-D）**：组合每 12 日换仓期收益 > 同期基准收益的比例。
  **不是** ``metrics.py`` 的逐日正收益口径。
- **分位单调性**：按预测分 5 组，组均实现收益的单调度（组序 vs 组均收益 Spearman + 单调步数）。
- **截面 IC**：每截面 ``Spearman(pred, realized)``。
- **净超额年化**：组合 NAV vs 基准 NAV 的年化超额。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def cross_sectional_ic(pred: pd.Series, realized: pd.Series) -> float:
    """单截面 rank IC = Spearman(pred, realized)，对齐 index、丢 NaN。"""
    df = pd.concat([pred.rename("p"), realized.rename("r")], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(df["p"].values, df["r"].values)
    return float(rho) if rho == rho else float("nan")


@dataclass
class QuantileMonotonicity:
    group_means: list[float]
    order_corr: float          # Spearman(组序, 组均收益)
    monotone_steps: int        # 相邻组递增的步数（满分 n_groups-1）
    n_groups: int

    @property
    def strictly_monotone(self) -> bool:
        return self.monotone_steps == self.n_groups - 1


def quantile_monotonicity(
    pred: pd.Series, realized: pd.Series, n_groups: int = 5
) -> QuantileMonotonicity:
    """按 ``pred`` 分 ``n_groups`` 组，算各组 ``realized`` 均值的单调性。"""
    df = pd.concat([pred.rename("p"), realized.rename("r")], axis=1).dropna()
    if len(df) < n_groups:
        return QuantileMonotonicity([float("nan")] * n_groups, float("nan"), 0, n_groups)
    # 按预测分位切组（0=最低分..n-1=最高分）
    df["g"] = pd.qcut(df["p"].rank(method="first"), n_groups, labels=False)
    means = df.groupby("g")["r"].mean()
    means = means.reindex(range(n_groups))
    gm = means.values.astype(float)
    valid = ~np.isnan(gm)
    if valid.sum() < 2:
        return QuantileMonotonicity(list(gm), float("nan"), 0, n_groups)
    rho, _ = stats.spearmanr(np.arange(n_groups)[valid], gm[valid])
    steps = int(np.sum(np.diff(gm[valid]) > 0))
    return QuantileMonotonicity(list(gm), float(rho) if rho == rho else float("nan"),
                                steps, n_groups)


def per_rebalance_win_rate(
    port_period_returns: pd.Series, bench_period_returns: pd.Series
) -> float:
    """**每换仓期** 组合收益 > 同期基准收益的比例（H-D 口径）.

    输入是【每个换仓期】的收益序列（非逐日）。两序列按 index 对齐。
    """
    df = pd.concat(
        [port_period_returns.rename("p"), bench_period_returns.rename("b")], axis=1
    ).dropna()
    if df.empty:
        return float("nan")
    return float((df["p"] > df["b"]).mean())


def excess_period_returns(
    port_period_returns: pd.Series, bench_period_returns: pd.Series
) -> pd.Series:
    """每换仓期超额 = 组合 − 基准（对齐 index）。"""
    df = pd.concat(
        [port_period_returns.rename("p"), bench_period_returns.rename("b")], axis=1
    ).dropna()
    return (df["p"] - df["b"]).rename("excess")


def net_excess_annualized(
    port_period_returns: pd.Series,
    bench_period_returns: pd.Series,
    *,
    periods_per_year: float,
) -> float:
    """年化净超额 = (组合年化几何收益) − (基准年化几何收益).

    用每换仓期收益做几何累乘后年化（非重叠 12 日换仓，无重复计数）。
    """
    df = pd.concat(
        [port_period_returns.rename("p"), bench_period_returns.rename("b")], axis=1
    ).dropna()
    if df.empty:
        return float("nan")
    n = len(df)

    def _ann(r: pd.Series) -> float:
        growth = float(np.prod(1.0 + r.values))
        if growth <= 0:
            return -1.0
        return growth ** (periods_per_year / n) - 1.0

    return _ann(df["p"]) - _ann(df["b"])


def aggregate_ic(ic_series: pd.Series) -> dict:
    """IC 序列 → 均值 / 标准差 / ICIR / 年化 ICIR。"""
    ics = ic_series.dropna()
    if len(ics) < 2:
        return {"ic_mean": float(ics.mean()) if len(ics) else float("nan"),
                "ic_std": float("nan"), "ic_ir": float("nan"), "n": int(len(ics))}
    mean = float(ics.mean())
    std = float(ics.std(ddof=1))
    icir = mean / std if std > 1e-12 else float("nan")
    return {"ic_mean": mean, "ic_std": std, "ic_ir": icir, "n": int(len(ics))}


__all__ = [
    "cross_sectional_ic", "QuantileMonotonicity", "quantile_monotonicity",
    "per_rebalance_win_rate", "excess_period_returns", "net_excess_annualized",
    "aggregate_ic",
]
