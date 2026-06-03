"""quantmind/execution — E3 模拟执行层.

模块结构:
  stop_loss_engine.py  规则引擎：止损 / 止盈 / 追踪止损 / 到期 / Regime
  manager.py           ExecutionManager：开仓 / 每日维护 / 平仓 / DB 写入
  exit_decision.py     ExitDecision dataclass
"""
from quantmind.execution.exit_decision import ExitDecision
from quantmind.execution.stop_loss_engine import StopLossEngine
from quantmind.execution.manager import ExecutionManager

__all__ = ["ExitDecision", "StopLossEngine", "ExecutionManager"]
