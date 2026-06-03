"""quantmind/execution/stop_loss_engine.py — 退出规则引擎.

每日扫描所有 OPEN 订单，按优先级判定是否平仓。

规则优先级（高→低）：
  1. 止损触发: close < stop_loss_price → 'stop_loss'
  2. 止盈触发: close >= target_price → 'target_hit'
  3. 追踪止损: 从持仓期最高点回撤 ≥ trailing_stop_pct → 'trailing_stop'
  4. 持仓到期: 已持仓天数 >= holding_period → 'time_expired'
  5. Regime 突变: bull→bear 且持有热门股（可选，默认关）

设计原则：
  * 引擎只负责"决定平仓"，不真正执行（Manager 才能写 DB）
  * 输入是订单 dict + 当日价格 + regime，输出 ExitDecision 列表
  * 纯函数风格，便于测试
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Iterable

from quantmind.execution.exit_decision import ExitDecision

log = logging.getLogger(__name__)


def _parse_date(d: Any) -> date | None:
    """容错日期解析。"""
    if d is None:
        return None
    if isinstance(d, date):
        return d
    if isinstance(d, datetime):
        return d.date()
    try:
        return datetime.fromisoformat(str(d)).date()
    except Exception:  # noqa: BLE001
        try:
            return datetime.strptime(str(d), "%Y-%m-%d").date()
        except Exception:  # noqa: BLE001
            return None


class StopLossEngine:
    """订单退出规则引擎（纯函数风格）。"""

    def __init__(
        self,
        trailing_stop_pct: float = 0.15,
        use_regime_exit: bool = False,
        regime_exit_industries: set[str] | None = None,
    ) -> None:
        """
        Args:
            trailing_stop_pct: 追踪止损阈值（从持仓期最高点回撤超过此比例）
            use_regime_exit: 是否启用 regime 突变退出（默认关）
            regime_exit_industries: bull→bear 时强制平仓的行业集合
        """
        assert 0 < trailing_stop_pct < 1, "trailing_stop_pct must be in (0, 1)"
        self.trailing_stop_pct = trailing_stop_pct
        self.use_regime_exit = use_regime_exit
        self.regime_exit_industries = regime_exit_industries or {
            "小金属", "证券", "电池", "光伏设备"
        }

    # ── 单笔订单评估 ─────────────────────────────────────────────────────────

    def check_single(
        self,
        order: dict[str, Any],
        current_price: float,
        as_of: date | str,
        regime: str | None = None,
        regime_prev: str | None = None,
    ) -> ExitDecision | None:
        """评估单笔订单，返回需要平仓的决策（不需平仓返回 None）。"""
        if order.get("status") != "OPEN":
            return None
        if current_price is None or current_price <= 0:
            return None

        order_id = int(order["order_id"])
        ticker = str(order["ticker"])
        stop_loss = order.get("stop_loss_price")
        target = order.get("target_price")
        high_price = order.get("high_price") or order.get("open_price")
        open_date = _parse_date(order.get("open_date"))
        holding_period = order.get("holding_period")
        today = _parse_date(as_of) or date.today()

        # 规则 1：止损（最高优先级）
        if stop_loss is not None and stop_loss > 0 and current_price < stop_loss:
            return ExitDecision(
                order_id=order_id, ticker=ticker,
                exit_reason="stop_loss",
                exit_price=current_price,
                rule_triggered="price < stop_loss",
                note=f"current={current_price:.3f} < stop_loss={stop_loss:.3f}",
            )

        # 规则 2：止盈
        if target is not None and target > 0 and current_price >= target:
            return ExitDecision(
                order_id=order_id, ticker=ticker,
                exit_reason="target_hit",
                exit_price=current_price,
                rule_triggered="price >= target_price",
                note=f"current={current_price:.3f} >= target={target:.3f}",
            )

        # 规则 3：追踪止损（持仓期高点回撤）
        if high_price and high_price > 0:
            drawdown = (high_price - current_price) / high_price
            if drawdown >= self.trailing_stop_pct:
                return ExitDecision(
                    order_id=order_id, ticker=ticker,
                    exit_reason="trailing_stop",
                    exit_price=current_price,
                    rule_triggered=f"drawdown >= {self.trailing_stop_pct:.0%}",
                    note=f"high={high_price:.3f} → current={current_price:.3f} "
                         f"(drop {drawdown:.2%})",
                )

        # 规则 4：到期
        if open_date and holding_period:
            days_held = (today - open_date).days
            if days_held >= int(holding_period):
                return ExitDecision(
                    order_id=order_id, ticker=ticker,
                    exit_reason="time_expired",
                    exit_price=current_price,
                    rule_triggered="days_held >= holding_period",
                    note=f"held {days_held}d >= plan {holding_period}d",
                )

        # 规则 5：Regime 突变（可选）
        if self.use_regime_exit and regime_prev == "bull" and regime == "bear":
            industry = order.get("industry", "")
            if industry in self.regime_exit_industries:
                return ExitDecision(
                    order_id=order_id, ticker=ticker,
                    exit_reason="regime_change",
                    exit_price=current_price,
                    rule_triggered="bull→bear & hot industry",
                    note=f"industry={industry}",
                )

        return None

    # ── 批量评估 ─────────────────────────────────────────────────────────────

    def evaluate_orders(
        self,
        orders: Iterable[dict[str, Any]],
        current_prices: dict[str, float],
        as_of: date | str,
        regime: str | None = None,
        regime_prev: str | None = None,
    ) -> list[ExitDecision]:
        """批量评估订单列表，返回所有需要平仓的决策。

        Args:
            orders: OPEN 订单的 dict 列表
            current_prices: {ticker: close_price}
            as_of: 评估日期
            regime: 当日 regime 标签
            regime_prev: 前日 regime（用于检测突变）
        """
        decisions: list[ExitDecision] = []
        for order in orders:
            ticker = str(order.get("ticker", ""))
            px = current_prices.get(ticker)
            if px is None:
                continue
            d = self.check_single(order, px, as_of, regime, regime_prev)
            if d is not None:
                decisions.append(d)
        return decisions
