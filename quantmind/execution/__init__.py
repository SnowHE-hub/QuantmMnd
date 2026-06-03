"""quantmind/execution — E3 模拟执行层 + E3.5 参数优化.

模块结构:
  stop_loss_engine.py  规则引擎：止损 / 止盈 / 追踪止损 / 到期 / Regime
  manager.py           ExecutionManager：开仓 / 每日维护 / 平仓 / DB 写入
  exit_decision.py     ExitDecision dataclass
  replay_engine.py     HistoricalReplayEngine：参数化历史回放
  optimizer.py         ExecutionParamOptimizer：网格搜索 + Pareto 前沿
"""
from quantmind.execution.exit_decision import ExitDecision
from quantmind.execution.stop_loss_engine import StopLossEngine
from quantmind.execution.manager import ExecutionManager
from quantmind.execution.replay_engine import (
    HistoricalReplayEngine, ReplayParams, HistoricalRecommendation,
    load_historical_recommendations, preload_price_history, replay_single_order,
)
from quantmind.execution.optimizer import ExecutionParamOptimizer

__all__ = [
    "ExitDecision", "StopLossEngine", "ExecutionManager",
    "HistoricalReplayEngine", "ReplayParams", "HistoricalRecommendation",
    "load_historical_recommendations", "preload_price_history", "replay_single_order",
    "ExecutionParamOptimizer",
]
