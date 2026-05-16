"""quantmind.backtest.metrics — 回测性能指标.

指标体系：
  收益类  : total_return, annualized_return, cagr
  风险类  : volatility, max_drawdown, drawdown_duration, VaR_95, CVaR_95
  风险调整: sharpe_ratio(rf=2.5%), sortino_ratio, calmar_ratio, information_ratio
  交易类  : turnover_rate, win_rate, profit_loss_ratio, avg_holding_period
  归因类  : alpha, beta(vs CSI300), Jensen_alpha

参考：
  - Sharpe (1966): Risk-Adjusted Measure of Performance
  - Sortino & Price (1994): Performance Measurement in a Downside Risk Framework
  - Jensen (1968): The Performance of Mutual Funds in the Period 1945–1964
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from quantmind.backtest.execution import Fill

__all__ = ["PerformanceMetrics"]

_TRADING_DAYS_PER_YEAR = 252
_RISK_FREE_RATE = 0.025  # 年化无风险利率 2.5%（中国国债）
_RF_DAILY = (1 + _RISK_FREE_RATE) ** (1 / _TRADING_DAYS_PER_YEAR) - 1


class PerformanceMetrics:
    """回测性能指标计算器."""

    def compute_all(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        fills: list["Fill"] | None = None,
        nav_series: pd.Series | None = None,
    ) -> dict[str, Any]:
        """计算全套指标.

        Args:
            returns:           策略日收益率序列（index = datetime）
            benchmark_returns: 基准日收益率序列（index = datetime），可选
            fills:             成交记录列表，用于计算换手率、持仓期等
            nav_series:        策略净值序列（index 与 returns 对齐），用于 NAV 口径年化换手率

        Returns:
            dict，含所有指标及 interpretation
        """
        if returns is None or len(returns) < 2:
            return self._empty_metrics()

        returns = returns.dropna()
        n = len(returns)

        m: dict[str, Any] = {}

        # ── 收益类 ────────────────────────────────────────────────────────────
        # total_return: 累计收益率 = prod(1+r) - 1
        nav = (1 + returns).cumprod()
        m["total_return"] = float(nav.iloc[-1] - 1)

        # annualized_return: 年化收益率（算术平均 × 252）
        m["annualized_return"] = float(returns.mean() * _TRADING_DAYS_PER_YEAR)

        # cagr: 复合年增长率 = (end_nav/1)^(252/n) - 1
        years = n / _TRADING_DAYS_PER_YEAR
        m["cagr"] = float(nav.iloc[-1] ** (1 / max(years, 1e-6)) - 1) if years > 0 else 0.0

        # ── 风险类 ────────────────────────────────────────────────────────────
        # volatility: 年化波动率 = std(r) × sqrt(252)
        m["volatility"] = float(returns.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))

        # max_drawdown: 最大回撤 = max((peak - trough) / peak)
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        m["max_drawdown"] = float(drawdown.min())

        # drawdown_duration: 最长回撤持续天数
        m["drawdown_duration"] = int(self._max_drawdown_duration(drawdown))

        # VaR_95: 95% 置信度 1 日历史 VaR（负值）
        m["VaR_95"] = float(np.percentile(returns, 5))

        # CVaR_95: 95% 置信度条件 VaR（期望损失）= E[r | r < VaR_95]
        var_threshold = m["VaR_95"]
        tail_returns = returns[returns <= var_threshold]
        m["CVaR_95"] = float(tail_returns.mean()) if len(tail_returns) > 0 else var_threshold

        # ── 风险调整类 ────────────────────────────────────────────────────────
        # sharpe_ratio: (E[r] - rf) / std(r) × sqrt(252)
        # 经济含义：每承担 1 单位总风险，获得的超额收益
        excess_returns = returns - _RF_DAILY
        m["sharpe_ratio"] = float(
            excess_returns.mean() / (returns.std() + 1e-10) * np.sqrt(_TRADING_DAYS_PER_YEAR)
        )

        # sortino_ratio: 只考虑下行风险
        # 经济含义：每承担 1 单位下行风险，获得的超额收益
        downside_returns = returns[returns < _RF_DAILY]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else returns.std()
        m["sortino_ratio"] = float(
            excess_returns.mean() / (downside_std + 1e-10) * np.sqrt(_TRADING_DAYS_PER_YEAR)
        )

        # calmar_ratio: 年化收益 / |最大回撤|
        # 经济含义：每承担 1% 最大回撤，获得的年化收益
        m["calmar_ratio"] = float(
            m["cagr"] / max(abs(m["max_drawdown"]), 1e-6)
        )

        # information_ratio: (策略收益 - 基准收益) / tracking_error
        # 经济含义：相对基准，每承担 1 单位主动风险，获得的超额收益
        if benchmark_returns is not None and len(benchmark_returns) > 1:
            bench_aligned = benchmark_returns.reindex(returns.index).fillna(0)
            active_returns = returns - bench_aligned
            m["information_ratio"] = float(
                active_returns.mean() / (active_returns.std() + 1e-10) * np.sqrt(_TRADING_DAYS_PER_YEAR)
            )
        else:
            m["information_ratio"] = np.nan

        # ── 归因类（alpha / beta / Jensen_alpha）────────────────────────────
        if benchmark_returns is not None and len(benchmark_returns) > 10:
            bench_aligned = benchmark_returns.reindex(returns.index).fillna(0)
            # 线性回归：r_strategy = alpha + beta × r_benchmark + epsilon
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                bench_aligned.values, returns.values
            )
            m["beta"] = float(slope)   # 系统性风险暴露
            m["alpha"] = float(intercept * _TRADING_DAYS_PER_YEAR)  # 年化 alpha（超额截距）
            m["r_squared"] = float(r_value ** 2)

            # Jensen_alpha: E[r] - rf - beta × (E[r_b] - rf)
            # 经济含义：扣除市场风险后的纯粹 alpha
            m["Jensen_alpha"] = float(
                m["annualized_return"] - _RISK_FREE_RATE
                - m["beta"] * (bench_aligned.mean() * _TRADING_DAYS_PER_YEAR - _RISK_FREE_RATE)
            )
        else:
            m["beta"] = np.nan
            m["alpha"] = np.nan
            m["r_squared"] = np.nan
            m["Jensen_alpha"] = np.nan

        # ── 交易类 ────────────────────────────────────────────────────────────
        if fills:
            m.update(self._compute_trading_metrics(fills, returns, nav_series))
        else:
            m["turnover_rate"] = np.nan
            m["win_rate"] = np.nan
            m["profit_loss_ratio"] = np.nan
            m["avg_holding_period"] = np.nan

        # ── 综合解读 ──────────────────────────────────────────────────────────
        m["interpretation"] = self._interpret(m)

        return m

    # ── 交易统计 ─────────────────────────────────────────────────────────────

    def _compute_trading_metrics(
        self,
        fills: list["Fill"],
        returns: pd.Series,
        nav_series: pd.Series | None = None,
    ) -> dict[str, Any]:
        """计算交易相关指标."""
        t: dict[str, Any] = {}

        buy_fills = [f for f in fills if f.is_buy]
        sell_fills = [f for f in fills if not f.is_buy]

        # win_rate: 盈利交易（卖出成交价 > 买入成本）占比
        # 用近似方法：卖出时 fill_price > avg_cost（取决于组合持仓记录）
        if sell_fills:
            # 简化：基于日收益计算胜率
            wins = sum(1 for r in returns if r > 0)
            t["win_rate"] = float(wins / len(returns)) if len(returns) > 0 else np.nan
        else:
            t["win_rate"] = np.nan

        # profit_loss_ratio: 盈利日均收益 / 亏损日均收益绝对值
        pos_returns = returns[returns > 0]
        neg_returns = returns[returns < 0]
        if len(pos_returns) > 0 and len(neg_returns) > 0:
            t["profit_loss_ratio"] = float(pos_returns.mean() / abs(neg_returns.mean()))
        else:
            t["profit_loss_ratio"] = np.nan

        # turnover_rate（年化，NAV 基准）：
        #   全样本单边买入金额之和 / (日均 NAV × 回测年数)
        #   季度满仓换仓约 (4×100%) 量级 → 常见区间约 100%–400%（以小数存储，如 2.5=250%）
        total_buy_amount = sum(f.gross_amount for f in buy_fills)
        total_sell_amount = sum(f.gross_amount for f in sell_fills)
        n = len(returns)
        years = max(n / _TRADING_DAYS_PER_YEAR, 1e-9)
        if nav_series is not None and len(nav_series) > 0:
            nav_a = nav_series.reindex(returns.index)
            nav_a = nav_a.ffill().bfill()
            avg_nav = float(nav_a.mean()) if nav_a.notna().any() else np.nan
            if avg_nav and avg_nav > 0 and np.isfinite(avg_nav):
                t["turnover_rate"] = float(total_buy_amount / (avg_nav * years))
            else:
                t["turnover_rate"] = np.nan
        elif n > 0:
            # 兼容旧口径（无 NAV 时退化）
            t["turnover_rate"] = float(total_buy_amount / (n + 1e-6) * _TRADING_DAYS_PER_YEAR)
        else:
            t["turnover_rate"] = np.nan

        t["turnover_rate_buy_notional"] = float(total_buy_amount)
        t["turnover_rate_sell_notional"] = float(total_sell_amount)

        # avg_holding_period: 平均持仓天数（买入到卖出）
        t["avg_holding_period"] = np.nan  # 需要匹配买卖对，简化跳过

        return t

    # ── 辅助 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        """计算最长回撤持续天数."""
        if drawdown.empty:
            return 0
        in_dd = drawdown < 0
        max_dur = 0
        cur_dur = 0
        for v in in_dd:
            if v:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        return max_dur

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        keys = [
            "total_return", "annualized_return", "cagr", "volatility",
            "max_drawdown", "drawdown_duration", "VaR_95", "CVaR_95",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "information_ratio",
            "beta", "alpha", "r_squared", "Jensen_alpha",
            "turnover_rate", "win_rate", "profit_loss_ratio", "avg_holding_period",
        ]
        return {k: np.nan for k in keys}

    @staticmethod
    def _interpret(m: dict[str, Any]) -> str:
        """一句话总结策略表现."""
        sr = m.get("sharpe_ratio", 0) or 0
        mdd = m.get("max_drawdown", 0) or 0
        ret = m.get("total_return", 0) or 0
        alpha = m.get("Jensen_alpha", np.nan)

        quality = "优秀" if sr > 1.5 else ("良好" if sr > 0.8 else ("一般" if sr > 0 else "较差"))
        alpha_str = f"，Jensen Alpha={alpha:.1%}" if not np.isnan(alpha) else ""
        return (
            f"策略表现{quality}：总收益={ret:.1%}，Sharpe={sr:.2f}，"
            f"最大回撤={mdd:.1%}{alpha_str}"
        )

    def generate_report(self, result: Any) -> str:
        """生成 Markdown 格式的回测报告摘要.

        Args:
            result: BacktestResult 对象

        Returns:
            Markdown 字符串
        """
        m = result.metrics if hasattr(result, "metrics") else result
        lines = [
            f"# 回测报告：{getattr(result, 'strategy_name', '策略')}",
            f"",
            f"**回测区间**：{getattr(result, 'start_date', 'N/A')} ~ {getattr(result, 'end_date', 'N/A')}",
            f"**交易天数**：{getattr(result, 'n_trading_days', 'N/A')}",
            f"",
            f"## 收益指标",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总收益率 | {m.get('total_return', 0):.2%} |",
            f"| 年化收益率（算术）| {m.get('annualized_return', 0):.2%} |",
            f"| CAGR | {m.get('cagr', 0):.2%} |",
            f"",
            f"## 风险指标",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 年化波动率 | {m.get('volatility', 0):.2%} |",
            f"| 最大回撤 | {m.get('max_drawdown', 0):.2%} |",
            f"| 最长回撤天数 | {m.get('drawdown_duration', 0)} |",
            f"| 95% VaR（日） | {m.get('VaR_95', 0):.2%} |",
            f"| 95% CVaR（日）| {m.get('CVaR_95', 0):.2%} |",
            f"",
            f"## 风险调整指标",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| Sharpe Ratio | {m.get('sharpe_ratio', 0):.3f} |",
            f"| Sortino Ratio | {m.get('sortino_ratio', 0):.3f} |",
            f"| Calmar Ratio | {m.get('calmar_ratio', 0):.3f} |",
            f"| Information Ratio | {m.get('information_ratio', float('nan')):.3f} |",
            f"",
            f"## 归因指标",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| Beta | {m.get('beta', float('nan')):.3f} |",
            f"| Alpha（年化）| {m.get('alpha', float('nan')):.3f} |",
            f"| Jensen Alpha | {m.get('Jensen_alpha', float('nan')):.3f} |",
            f"",
            f"## 综合评价",
            f"{m.get('interpretation', 'N/A')}",
        ]
        return "\n".join(lines)
