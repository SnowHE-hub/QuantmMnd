"""quantmind.backtest.portfolio — 组合持仓管理.

特性：
- T+1 支持：买入当日记录 buy_date，卖出时检查
- Position 含：ticker / shares / avg_cost / market_value / unrealized_pnl / weight
- rebalance_to_weights：目标权重 → 调仓订单列表
- get_returns：返回日收益序列
- get_turnover：返回换手率
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from quantmind.backtest.execution import Fill, Order

__all__ = ["Portfolio", "Position"]


def _to_trade_date(d: Any) -> date:
    """Normalize engine dates (``Timestamp`` / ``date`` / str) to ``datetime.date``."""
    return pd.Timestamp(d).date()


@dataclass
class Position:
    """单股持仓."""

    ticker: str
    shares: int                     # 持股数（股）
    avg_cost: float                 # 平均成本（元/股）
    buy_date: date | None = None    # 最近一次买入日期（T+1 用）
    market_value: float = 0.0       # 当前市值
    unrealized_pnl: float = 0.0     # 浮动盈亏
    weight: float = 0.0             # 占净值比例
    last_close: float = 0.0        # 最近可用收盘价（当日缺失时用上次收盘价盯市）

    @property
    def cost_basis(self) -> float:
        return self.avg_cost * self.shares


class Portfolio:
    """组合持仓管理器.

    Args:
        initial_capital: 初始资金（元）
        t1_restriction:  是否启用 T+1（默认 True）
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,
        t1_restriction: bool = True,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.t1_restriction = t1_restriction
        self.positions: dict[str, Position] = {}

        # 历史记录
        self._nav_history: list[tuple[date, float]] = []
        self._returns_history: list[tuple[date, float]] = []

    # ── 基础属性 ──────────────────────────────────────────────────────────────

    @property
    def nav(self) -> float:
        """组合净值（现金 + 持仓市值）."""
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def total_market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    # ── T+1 接口 ──────────────────────────────────────────────────────────────

    def can_sell(self, ticker: str, current_date: date) -> bool:
        """是否可以卖出（T+1 检查）."""
        cur = _to_trade_date(current_date)
        if not self.t1_restriction:
            return ticker in self.positions and self.positions[ticker].shares > 0
        if ticker not in self.positions:
            return False
        pos = self.positions[ticker]
        if pos.shares <= 0:
            return False
        if pos.buy_date is None:
            return True
        # T+1：买入日之后才能卖出
        bd = pos.buy_date if isinstance(pos.buy_date, date) else _to_trade_date(pos.buy_date)
        return cur > bd

    def sellable_shares(self, ticker: str, current_date: date) -> int:
        """可卖股数（T+1 约束后）."""
        if not self.can_sell(ticker, current_date):
            return 0
        return self.positions.get(ticker, Position(ticker, 0, 0.0)).shares

    # ── 更新方法 ──────────────────────────────────────────────────────────────

    def update(
        self,
        current_date: date,
        today_bar: pd.DataFrame,
        fills: list["Fill"],
    ) -> None:
        """根据成交记录和当日价格更新持仓.

        Args:
            current_date: 当前日期
            today_bar:    当日收盘价 DataFrame（index=ticker）
            fills:        当日成交记录
        """
        cur = _to_trade_date(current_date)
        # 先处理成交
        for fill in fills:
            self._apply_fill(fill, cur)

        # 更新市值
        self._update_market_values(today_bar)

        # 清除空仓
        self.positions = {k: v for k, v in self.positions.items() if v.shares > 0}

        # 更新权重
        total_nav = self.nav
        if total_nav > 0:
            for pos in self.positions.values():
                pos.weight = pos.market_value / total_nav

    def _apply_fill(self, fill: "Fill", current_date: date) -> None:
        """将成交应用到持仓."""
        if fill.is_buy:
            if fill.ticker in self.positions:
                pos = self.positions[fill.ticker]
                total_cost = pos.avg_cost * pos.shares + fill.fill_price * fill.shares
                pos.shares += fill.shares
                pos.avg_cost = total_cost / pos.shares
                pos.buy_date = current_date  # 更新买入日
                if np.isfinite(fill.fill_price) and fill.fill_price > 0:
                    pos.last_close = float(fill.fill_price)
            else:
                self.positions[fill.ticker] = Position(
                    ticker=fill.ticker,
                    shares=fill.shares,
                    avg_cost=fill.fill_price,
                    buy_date=current_date,
                    last_close=float(fill.fill_price),
                )
            self.cash -= fill.net_amount
        else:
            if fill.ticker in self.positions:
                pos = self.positions[fill.ticker]
                pos.shares -= fill.shares
                if pos.shares <= 0:
                    del self.positions[fill.ticker]
            self.cash += fill.net_amount

    def _update_market_values(self, today_bar: pd.DataFrame) -> None:
        """更新各持仓市值."""
        for ticker, pos in self.positions.items():
            px = np.nan
            if ticker in today_bar.index:
                bar = today_bar.loc[ticker]
                px = float(bar.get("close", bar.get("adj_close", bar.get("c", np.nan))))
            if px == px and px > 0:
                pos.last_close = px
                pos.market_value = px * pos.shares
                pos.unrealized_pnl = (px - pos.avg_cost) * pos.shares
            elif pos.last_close > 0 and pos.shares > 0:
                # 当日无有效收盘价（退市边缘数据缺口）：沿用上一有效收盘价盯市，避免 NAV 冻结
                pos.market_value = pos.last_close * pos.shares
                pos.unrealized_pnl = (pos.last_close - pos.avg_cost) * pos.shares

    def compute_nav(self, today_bar: pd.DataFrame) -> float:
        """计算当日净值."""
        self._update_market_values(today_bar)
        return self.nav

    # ── 调仓接口 ──────────────────────────────────────────────────────────────

    def rebalance_to_weights(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        current_date: date,
    ) -> list["Order"]:
        """生成从当前持仓调整到目标权重的订单列表.

        Args:
            target_weights: {ticker: weight}（权重之和应 <= 1.0）
            prices:         {ticker: 当前价格}
            current_date:   当前日期（T+1 检查用）

        Returns:
            Order 列表（先卖后买）
        """
        from quantmind.backtest.execution import Order

        total_nav = self.nav
        orders: list[Order] = []

        # 先生成卖出订单（T+1 检查）
        for ticker, pos in list(self.positions.items()):
            target_w = target_weights.get(ticker, 0.0)
            target_shares = int(total_nav * target_w / prices.get(ticker, 1.0) / 100) * 100
            diff = pos.shares - target_shares
            if diff > 0 and self.can_sell(ticker, current_date):
                sell_shares = min(diff, self.sellable_shares(ticker, current_date))
                sell_shares = (sell_shares // 100) * 100
                if sell_shares > 0:
                    orders.append(Order(
                        ticker=ticker,
                        shares=sell_shares,
                        direction="sell",
                    ))

        # 再生成买入订单
        for ticker, target_w in target_weights.items():
            if target_w <= 0:
                continue
            price = prices.get(ticker)
            if not price or price <= 0:
                continue
            target_shares = int(total_nav * target_w / price / 100) * 100
            current_shares = self.positions.get(ticker, Position(ticker, 0, 0.0)).shares
            diff = target_shares - current_shares
            if diff > 0:
                orders.append(Order(
                    ticker=ticker,
                    shares=diff,
                    direction="buy",
                ))

        return orders

    # ── 历史指标 ──────────────────────────────────────────────────────────────

    def get_returns(self) -> pd.Series:
        """返回日收益率序列."""
        if len(self._nav_history) < 2:
            return pd.Series(dtype=float)
        df = pd.DataFrame(self._nav_history, columns=["date", "nav"])
        df = df.set_index("date").sort_index()
        return df["nav"].pct_change().dropna()

    def get_turnover(self) -> float:
        """换手率（当期买入金额 / 组合净值）."""
        # 简单实现，详细版本在 metrics 中
        return 0.0

    def snapshot(self) -> dict[str, Any]:
        """持仓快照（用于日志/调试）."""
        return {
            "cash": self.cash,
            "nav": self.nav,
            "n_positions": self.n_positions,
            "positions": {
                t: {
                    "shares": p.shares,
                    "avg_cost": p.avg_cost,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "weight": p.weight,
                }
                for t, p in self.positions.items()
            },
        }
