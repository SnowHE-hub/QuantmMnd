"""退出决策数据类。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExitReason = Literal[
    "stop_loss",       # 价格跌破止损线
    "target_hit",      # 价格达到目标
    "trailing_stop",   # 从持仓期最高点回撤超过阈值
    "time_expired",    # 持仓到期
    "regime_change",   # Regime 切换（如 bull → bear）
    "manual",          # 手动平仓
]


@dataclass(frozen=True)
class ExitDecision:
    """单笔订单的退出决策。"""
    order_id: int
    ticker: str
    exit_reason: ExitReason
    exit_price: float
    rule_triggered: str   # 简短规则名，便于日志
    note: str = ""        # 详细说明（含触发数值）

    def to_dict(self) -> dict:
        return {
            "order_id":       self.order_id,
            "ticker":         self.ticker,
            "exit_reason":    self.exit_reason,
            "exit_price":     self.exit_price,
            "rule_triggered": self.rule_triggered,
            "note":           self.note,
        }
