"""quantmind.backtest.execution — A股交易撮合模拟器.

A股标准交易成本：
  - 佣金：万3（买卖双向）
  - 印花税：千1（仅卖出方向）
  - 过户费：万0.2（沪市 .SH 股票买卖双向）
  - 滑点：默认 10bp

成交量约束：
  - 单笔成交不超过当日成交额的 5%
  - 涨跌停时不能成交（涨停不能买，跌停不能卖）
  - 停牌（volume=0）不成交
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from quantmind.backtest.portfolio import Portfolio

__all__ = ["Order", "Fill", "ExecutionSimulator"]

_UP_LIMIT_THRESHOLD = 0.0995
_DOWN_LIMIT_THRESHOLD = -0.0995


@dataclass
class Order:
    """交易订单."""

    ticker: str
    shares: int          # 股数（正=买，负=卖；也可配合 direction 使用）
    direction: str = "buy"   # "buy" | "sell"
    order_type: str = "market"  # "market" | "limit" | "target_value"
    limit_price: float | None = None
    # 注：target_value 类型时 shares 为目标持仓金额

    @property
    def is_buy(self) -> bool:
        return self.direction == "buy"

    @property
    def abs_shares(self) -> int:
        return abs(self.shares)


@dataclass
class Fill:
    """成交记录."""

    ticker: str
    direction: str          # "buy" | "sell"
    shares: int             # 成交股数（绝对值）
    fill_price: float       # 成交价格（含滑点）
    gross_amount: float     # 成交金额（税前）
    commission: float       # 佣金
    stamp_duty: float       # 印花税
    transfer_fee: float     # 过户费
    net_amount: float       # 净成交金额（= gross ± fees）
    fill_date: date | None = None
    order_ref: str = ""

    @property
    def total_cost(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee

    @property
    def is_buy(self) -> bool:
        return self.direction == "buy"


class ExecutionSimulator:
    """A股交易撮合模拟器.

    Args:
        config: BacktestConfig（含佣金率、印花税等参数）
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def execute(
        self,
        orders: list[Order],
        today_bar: pd.DataFrame,
        portfolio: "Portfolio",
        current_date: date,
    ) -> list[Fill]:
        """按照 orders 在 today_bar 上撮合成交.

        Args:
            orders:       策略生成的订单列表
            today_bar:    当日 OHLCV（index=ticker）
            portfolio:    当前持仓（用于校验 T+1、可卖股数等）
            current_date: 当前日期

        Returns:
            Fill 列表
        """
        fills: list[Fill] = []
        available_cash = portfolio.cash

        for order in orders:
            ticker = order.ticker

            # 检查价格数据
            if ticker not in today_bar.index:
                logger.debug(f"[Execution] {ticker} 无当日价格，跳过")
                continue

            bar = today_bar.loc[ticker]
            fill = self._try_fill(
                order=order,
                bar=bar,
                portfolio=portfolio,
                current_date=current_date,
                available_cash=available_cash,
            )

            if fill is not None:
                fills.append(fill)
                # 更新可用资金
                if fill.is_buy:
                    available_cash -= fill.net_amount
                else:
                    available_cash += fill.net_amount

        return fills

    def _try_fill(
        self,
        order: Order,
        bar: pd.Series,
        portfolio: "Portfolio",
        current_date: date,
        available_cash: float,
    ) -> Fill | None:
        """尝试撮合单笔订单."""
        ticker = order.ticker
        close = _get_price(bar, "close")
        # 兼容 pre_close（tushare/akshare）和 prev_close（内部 mock 数据）两种字段名
        prev_close = _get_price(bar, "pre_close") or _get_price(bar, "prev_close")
        open_p = _get_price(bar, "open")
        volume = _get_price(bar, "volume") or _get_price(bar, "vol") or 0.0
        amount = _get_price(bar, "amount") or (close * volume if close and volume else 0.0)

        if close is None or close <= 0:
            return None

        # ── 停牌检测 ──────────────────────────────────────────────────────────
        if volume == 0:
            logger.debug(f"[Execution] {ticker} 停牌，跳过")
            return None

        # ── 涨跌停检测 ────────────────────────────────────────────────────────
        if prev_close and prev_close > 0:
            chg = (close - prev_close) / prev_close
            if chg >= _UP_LIMIT_THRESHOLD and order.is_buy:
                logger.debug(f"[Execution] {ticker} 涨停，买入被拒绝")
                return None
            if chg <= _DOWN_LIMIT_THRESHOLD and not order.is_buy:
                logger.debug(f"[Execution] {ticker} 跌停，卖出被拒绝")
                return None

        # ── T+1 检查（卖出）──────────────────────────────────────────────────
        if not order.is_buy:
            if not portfolio.can_sell(ticker, current_date):
                logger.debug(f"[Execution] {ticker} T+1 限制，今日买入不可卖出")
                return None
            # 可卖数量
            max_sell = portfolio.sellable_shares(ticker, current_date)
            shares = min(order.abs_shares, max_sell)
        else:
            shares = order.abs_shares

        if shares <= 0:
            return None

        # 整手处理（A股 100 股/手）
        shares = (shares // 100) * 100
        if shares <= 0:
            return None

        # ── 成交量约束（单笔不超过当日成交额5%）────────────────────────────
        if amount > 0:
            max_amount_by_volume = amount * self.config.max_volume_pct
            max_shares_by_volume = int(max_amount_by_volume / close / 100) * 100
            if max_shares_by_volume > 0:
                shares = min(shares, max_shares_by_volume)

        if shares <= 0:
            return None

        # ── 成交价（含滑点）──────────────────────────────────────────────────
        execution_mode = self.config.execution_mode
        if execution_mode == "open_price" and open_p and open_p > 0:
            base_price = open_p
        elif execution_mode == "vwap":
            vwap = _get_price(bar, "vwap") or (amount / volume if volume > 0 else close)
            base_price = vwap if vwap and vwap > 0 else close
        else:  # close_price or fallback
            base_price = close

        slippage = self.config.slippage_bp / 10000.0
        if order.is_buy:
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)

        # ── 资金检查（买入）────────────────────────────────────────────────
        gross_amount = fill_price * shares
        if order.is_buy and gross_amount > available_cash * 1.01:
            # 按可用资金调整股数
            affordable = int(available_cash / fill_price / 100) * 100
            if affordable <= 0:
                return None
            shares = affordable
            gross_amount = fill_price * shares

        # ── 费用计算 ──────────────────────────────────────────────────────────
        commission = gross_amount * self.config.commission_rate
        commission = max(commission, 5.0)  # 最低佣金 5 元

        stamp_duty_fee = gross_amount * self.config.stamp_duty if not order.is_buy else 0.0

        # 过户费：仅沪市
        transfer_fee_rate = self.config.transfer_fee if ticker.endswith(".SH") else 0.0
        transfer_fee = gross_amount * transfer_fee_rate

        total_fees = commission + stamp_duty_fee + transfer_fee

        if order.is_buy:
            net_amount = gross_amount + total_fees
        else:
            net_amount = gross_amount - total_fees

        return Fill(
            ticker=ticker,
            direction=order.direction,
            shares=shares,
            fill_price=fill_price,
            gross_amount=gross_amount,
            commission=commission,
            stamp_duty=stamp_duty_fee,
            transfer_fee=transfer_fee,
            net_amount=net_amount,
            fill_date=current_date,
        )


def _get_price(bar: pd.Series, field: str) -> float | None:
    """安全从 bar 取价格字段."""
    v = bar.get(field)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
