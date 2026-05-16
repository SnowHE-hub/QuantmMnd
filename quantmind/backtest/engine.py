"""quantmind.backtest.engine — A股回测引擎核心.

特性：
- T+1 限制：当日买入明日才能卖
- 涨跌停限制：涨停不能买入（无量），跌停不能卖出
- 停牌处理：停牌股票跳过订单
- 严格 PIT：每个决策时点只使用 <= 该时刻的数据
- 撮合模式：open_price / close_price / vwap（可配置）
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quantmind.backtest.execution import ExecutionSimulator, Order, Fill
from quantmind.backtest.portfolio import Portfolio
from quantmind.backtest.metrics import PerformanceMetrics

__all__ = [
    "Strategy",
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
]

# 涨跌停比例（A股默认 10%，科创板 20%）
_UP_LIMIT = 0.0995    # 实际上涨超过 9.95% 视为涨停
_DOWN_LIMIT = -0.0995  # 下跌超过 9.95% 视为跌停


# ============================================================================
# 配置
# ============================================================================

@dataclass
class BacktestConfig:
    """回测引擎配置."""

    initial_capital: float = 10_000_000.0    # 初始资金（元）
    execution_mode: str = "open_price"       # open_price | close_price | vwap
    commission_rate: float = 3e-4            # 佣金率（万3）
    stamp_duty: float = 1e-3                 # 印花税（卖出千1）
    transfer_fee: float = 2e-5              # 过户费（沪市万0.2）
    slippage_bp: float = 10.0               # 滑点（bp）
    max_volume_pct: float = 0.05            # 单笔最大成交量占比
    benchmark: str = "000300.SH"            # 基准指数代码
    t1_restriction: bool = True             # T+1 限制


# ============================================================================
# Strategy 基类
# ============================================================================

class Strategy(ABC):
    """策略基类，子类实现信号生成逻辑."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        risk_manager: Any | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.portfolio: Portfolio | None = None  # 由 Engine 注入
        # risk_manager 可选：DrawdownController 或任意实现了 check_and_adjust() 的对象
        self.risk_manager = risk_manager

    def on_start(self, start_date: date, end_date: date, universe: list[str]) -> None:
        """回测开始时回调（可选）."""
        pass

    def on_end(self) -> None:
        """回测结束时回调（可选）."""
        pass

    @abstractmethod
    def on_market_open(
        self,
        current_date: date,
        prices: pd.DataFrame,
        universe: list[str],
    ) -> list[Order]:
        """开盘前生成订单.

        Args:
            current_date: 当前日期
            prices:       当日及历史价格 DataFrame（open/high/low/close/volume）
            universe:     当日可交易股票列表

        Returns:
            Order 列表
        """
        ...

    def on_market_close(
        self,
        current_date: date,
        prices: pd.DataFrame,
        fills: list[Fill],
    ) -> None:
        """收盘后回调（记录日志、更新状态等，可选）."""
        pass

    def on_signal(
        self,
        current_date: date,
        signals: dict[str, float],
    ) -> list[Order]:
        """收到外部信号时生成订单（可选）."""
        return []


# ============================================================================
# 回测结果
# ============================================================================

@dataclass
class BacktestResult:
    """回测结果容器."""

    # 基础信息
    strategy_name: str
    start_date: date
    end_date: date
    universe_size: int
    config: BacktestConfig

    # 净值序列（索引为日期）
    nav_series: pd.Series = field(default_factory=pd.Series)
    benchmark_series: pd.Series = field(default_factory=pd.Series)

    # 收益序列
    returns: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)

    # 交易记录
    fills: list[Fill] = field(default_factory=list)

    # 每日持仓快照
    daily_positions: list[dict] = field(default_factory=list)

    # 性能指标
    metrics: dict[str, Any] = field(default_factory=dict)

    # 运行状态
    n_trading_days: int = 0
    n_rebalance_days: int = 0
    n_orders: int = 0
    n_fills: int = 0

    def summary(self) -> str:
        """一行汇总."""
        m = self.metrics
        return (
            f"{self.strategy_name} | "
            f"Return={m.get('total_return', 0):.1%} | "
            f"Sharpe={m.get('sharpe_ratio', 0):.2f} | "
            f"MaxDD={m.get('max_drawdown', 0):.1%} | "
            f"Days={self.n_trading_days}"
        )


# ============================================================================
# 回测引擎
# ============================================================================

class BacktestEngine:
    """A股历史回测引擎.

    Args:
        config:       BacktestConfig 实例
        prices_df:    多股票 OHLCV DataFrame，MultiIndex (date, ticker)
                      或宽格式 (index=date, columns=MultiIndex(field, ticker))
        benchmark_df: 基准指数价格 Series（index=date, close price）
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        prices_df: pd.DataFrame | None = None,
        benchmark_df: pd.Series | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self._prices_df = prices_df
        self._benchmark_df = benchmark_df
        self._execution = ExecutionSimulator(self.config)

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def run(
        self,
        strategy: Strategy,
        start_date: date,
        end_date: date,
        universe: list[str],
        prices_df: pd.DataFrame | None = None,
        benchmark_df: pd.Series | None = None,
    ) -> BacktestResult:
        """运行回测.

        Args:
            strategy:     策略实例
            start_date:   回测开始日
            end_date:     回测结束日
            universe:     股票代码列表
            prices_df:    可选：覆盖构造函数传入的价格数据
            benchmark_df: 可选：覆盖构造函数传入的基准数据

        Returns:
            BacktestResult
        """
        prices = prices_df if prices_df is not None else self._prices_df
        benchmark = benchmark_df if benchmark_df is not None else self._benchmark_df

        if prices is None:
            raise ValueError("prices_df 不能为空")

        # 标准化价格 DataFrame：确保 MultiIndex (date, ticker)
        prices = self._normalize_prices(prices)

        # 统一日期类型：允许传入字符串或 date/datetime
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        # 获取交易日序列
        all_dates = sorted(set(prices.index.get_level_values("date")))
        trading_dates = [d for d in all_dates if start_date <= d <= end_date]

        if not trading_dates:
            raise ValueError(f"区间内无交易日数据: {start_date}~{end_date}")

        logger.info(
            f"[BacktestEngine] {strategy.__class__.__name__} | "
            f"{start_date}~{end_date} | {len(trading_dates)} days | "
            f"{len(universe)} stocks"
        )

        # 初始化组合
        portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            t1_restriction=self.config.t1_restriction,
        )
        strategy.portfolio = portfolio
        strategy.on_start(start_date, end_date, universe)

        # 记录容器
        nav_records: dict[date, float] = {}
        all_fills: list[Fill] = []
        daily_positions: list[dict] = []

        # ── 日循环 ────────────────────────────────────────────────────────────
        for i, current_date in enumerate(trading_dates):
            # 当日价格（历史 PIT 数据：只到 current_date）
            pit_prices = self._get_pit_prices(prices, current_date)
            held = list(portfolio.positions.keys())
            today_prices = self._get_today_bar(
                prices, current_date, universe, extra_tickers=held
            )

            if today_prices.empty:
                logger.debug(f"[Engine] {current_date}: 无价格数据，跳过")
                nav_records[current_date] = portfolio.nav
                continue

            # 当日可交易股票（剔除停牌、涨跌停）
            tradeable = self._get_tradeable_universe(today_prices, universe)

            # ── 风险管理：回撤检查（可选）────────────────────────────────────
            position_scale = 1.0  # 默认满仓
            if strategy.risk_manager is not None:
                # 计算当前回撤
                if len(nav_records) >= 2:
                    nav_vals = list(nav_records.values())
                    peak_nav = max(nav_vals)
                    cur_nav = nav_vals[-1]
                    cur_dd = abs((peak_nav - cur_nav) / peak_nav) if peak_nav > 0 else 0.0
                else:
                    cur_dd = 0.0

                position_scale = strategy.risk_manager.check_and_adjust(cur_dd)

                # 回撤 > 30%（position_scale == 0）→ 强制清仓
                if position_scale == 0.0 and portfolio.n_positions > 0:
                    logger.warning(
                        f"[Engine] {current_date} 回撤触发清仓，强制平所有持仓"
                    )
                    liquidation_orders: list[Order] = []
                    for tkr, pos in list(portfolio.positions.items()):
                        sell_shares = portfolio.sellable_shares(tkr, current_date)
                        if sell_shares > 0:
                            liquidation_orders.append(
                                Order(ticker=tkr, shares=sell_shares, direction="sell")
                            )
                    fills = self._execution.execute(
                        orders=liquidation_orders,
                        today_bar=today_prices,
                        portfolio=portfolio,
                        current_date=current_date,
                    )
                    all_fills.extend(fills)
                    portfolio.update(current_date, today_prices, fills)
                    nav = portfolio.compute_nav(today_prices)
                    nav_records[current_date] = nav
                    daily_positions.append({
                        "date": current_date,
                        "nav": nav,
                        "n_positions": len(portfolio.positions),
                        "cash": portfolio.cash,
                    })
                    continue  # 跳过本日策略信号

            # ── 策略生成订单 ──────────────────────────────────────────────────
            try:
                orders = strategy.on_market_open(current_date, pit_prices, tradeable)
                # 应用仓位缩放（position_scale < 1.0 时按比例缩减买入股数）
                if position_scale < 1.0 and orders:
                    scaled_orders: list[Order] = []
                    for o in orders:
                        if o.is_buy:
                            scaled_shares = max(int(o.abs_shares * position_scale // 100) * 100, 0)
                            if scaled_shares > 0:
                                scaled_orders.append(
                                    Order(ticker=o.ticker, shares=scaled_shares, direction="buy",
                                          order_type=o.order_type, limit_price=o.limit_price)
                                )
                        else:
                            scaled_orders.append(o)  # 卖出订单不缩减
                    orders = scaled_orders
            except Exception as e:
                logger.warning(f"[Engine] {current_date} on_market_open 异常: {e}")
                orders = []

            # ── 撮合执行 ──────────────────────────────────────────────────────
            fills = self._execution.execute(
                orders=orders,
                today_bar=today_prices,
                portfolio=portfolio,
                current_date=current_date,
            )
            all_fills.extend(fills)

            # ── 更新组合（T+1：更新可卖出日）────────────────────────────────
            portfolio.update(current_date, today_prices, fills)

            # ── 收盘回调 ──────────────────────────────────────────────────────
            try:
                strategy.on_market_close(current_date, pit_prices, fills)
            except Exception as e:
                logger.warning(f"[Engine] {current_date} on_market_close 异常: {e}")

            # ── 记录 NAV ──────────────────────────────────────────────────────
            nav = portfolio.compute_nav(today_prices)
            nav_records[current_date] = nav

            daily_positions.append({
                "date": current_date,
                "nav": nav,
                "n_positions": len(portfolio.positions),
                "cash": portfolio.cash,
            })

        strategy.on_end()

        # ── 构建结果 ──────────────────────────────────────────────────────────
        nav_series = pd.Series(nav_records, name="nav")
        nav_series.index = pd.to_datetime(nav_series.index)
        returns = nav_series.pct_change().dropna()

        # 基准收益
        bench_returns = pd.Series(dtype=float)
        bench_series = pd.Series(dtype=float)
        if benchmark is not None:
            bench_aligned = benchmark.reindex(nav_series.index).ffill()
            bench_series = bench_aligned / bench_aligned.iloc[0] * self.config.initial_capital
            bench_returns = bench_aligned.pct_change().dropna()

        # 计算性能指标
        metrics = PerformanceMetrics().compute_all(
            returns=returns,
            benchmark_returns=bench_returns,
            fills=all_fills,
            nav_series=nav_series,
        )

        result = BacktestResult(
            strategy_name=strategy.__class__.__name__,
            start_date=start_date,
            end_date=end_date,
            universe_size=len(universe),
            config=self.config,
            nav_series=nav_series,
            benchmark_series=bench_series,
            returns=returns,
            benchmark_returns=bench_returns,
            fills=all_fills,
            daily_positions=daily_positions,
            metrics=metrics,
            n_trading_days=len(trading_dates),
            n_rebalance_days=len([f for f in all_fills if f]),
            n_orders=len([o for o in all_fills]),
            n_fills=len(all_fills),
        )

        logger.info(f"[BacktestEngine] Done. {result.summary()}")
        return result

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    def _normalize_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """标准化价格 DataFrame 为 MultiIndex(date, ticker)."""
        if isinstance(prices.index, pd.MultiIndex):
            idx_names = list(prices.index.names)
            # 确保 level 0 = date, level 1 = ticker
            if "date" not in idx_names:
                prices.index.names = ["date", "ticker"]
            return prices

        # 宽格式：index=date, columns=ticker（单字段）
        if not isinstance(prices.columns, pd.MultiIndex):
            # 假设单字段（close），添加 ticker 维度
            prices = prices.stack()
            prices.index.names = ["date", "ticker"]
            prices = prices.to_frame("close")
            return prices

        # columns 为 MultiIndex(field, ticker)
        prices = prices.stack(level=1, future_stack=True)
        prices.index.names = ["date", "ticker"]
        return prices

    def _get_pit_prices(self, prices: pd.DataFrame, as_of: date) -> pd.DataFrame:
        """PIT 价格：只返回 <= as_of 的数据."""
        as_of_ts = pd.Timestamp(as_of)
        dates = prices.index.get_level_values("date")
        mask = dates <= as_of_ts
        return prices[mask]

    def _get_today_bar(
        self,
        prices: pd.DataFrame,
        current_date: date,
        universe: list[str],
        extra_tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """获取当日各股票 bar 数据.

        ``extra_tickers``：当前持仓等（可能被调出基准池但仍需在面板中有收盘价以便盯市）。
        """
        try:
            today_ts = pd.Timestamp(current_date)
            bar = prices.xs(today_ts, level="date")
            tickers = set(universe)
            if extra_tickers:
                tickers.update(t for t in extra_tickers if t)
            bar = bar.loc[bar.index.isin(tickers)]
            return bar
        except KeyError:
            return pd.DataFrame()

    def _get_tradeable_universe(
        self,
        today_bar: pd.DataFrame,
        universe: list[str],
    ) -> list[str]:
        """返回当日可交易股票（剔除停牌、严格涨跌停）."""
        if today_bar.empty:
            return []

        tradeable = []
        for ticker in universe:
            if ticker not in today_bar.index:
                continue  # 停牌/未上市
            row = today_bar.loc[ticker]
            close = row.get("close", row.get("c", np.nan))
            prev_close = row.get("pre_close", np.nan)
            volume = row.get("volume", row.get("vol", 1.0))

            # 停牌：成交量为 0
            if pd.notna(volume) and volume == 0:
                continue

            # 涨跌停检测（需要前收盘价）
            if pd.notna(close) and pd.notna(prev_close) and prev_close > 0:
                chg = (close - prev_close) / prev_close
                if abs(chg) >= _UP_LIMIT:
                    continue  # 涨停或跌停（不允许成交）

            tradeable.append(ticker)
        return tradeable


# ============================================================================
# 内置简单策略：等权组合（用于测试）
# ============================================================================

class EqualWeightStrategy(Strategy):
    """等权组合策略（每月再平衡，用于基准对比）."""

    def __init__(
        self,
        rebalance_freq: str = "M",  # M=月, Q=季
        top_n: int | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        super().__init__(config)
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self._last_rebalance: date | None = None

    def on_market_open(
        self,
        current_date: date,
        prices: pd.DataFrame,
        universe: list[str],
    ) -> list[Order]:
        if not self._should_rebalance(current_date):
            return []

        self._last_rebalance = current_date
        n = self.top_n or len(universe)
        selected = universe[:n]
        if not selected:
            return []

        target_value = (self.portfolio.nav if self.portfolio else self.config.initial_capital)
        weight = 1.0 / len(selected)
        target_amount = target_value * weight

        orders = []
        # 获取今日收盘价近似值
        today_bar = prices.xs(pd.Timestamp(current_date), level="date") if isinstance(prices.index, pd.MultiIndex) else prices.loc[prices.index == pd.Timestamp(current_date)]

        for ticker in selected:
            try:
                if isinstance(prices.index, pd.MultiIndex):
                    price = today_bar.loc[ticker, "close"] if ticker in today_bar.index else None
                else:
                    price = None
                if price is None or price <= 0:
                    continue
                shares = int(target_amount / price / 100) * 100  # 整手
                if shares > 0:
                    orders.append(Order(
                        ticker=ticker,
                        shares=shares,
                        direction="buy",
                        order_type="target_value",
                    ))
            except Exception:
                continue
        return orders

    def _should_rebalance(self, current_date: date) -> bool:
        if self._last_rebalance is None:
            return True
        if self.rebalance_freq == "M":
            return current_date.month != self._last_rebalance.month
        if self.rebalance_freq == "Q":
            return (current_date.month - 1) // 3 != (self._last_rebalance.month - 1) // 3
        return False
