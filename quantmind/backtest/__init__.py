"""quantmind.backtest — Phase 6 回测引擎.

组件：
  BacktestEngine:       A股历史回测引擎（T+1/涨跌停/停牌/PIT）
  Portfolio:            持仓管理（T+1, rebalance_to_weights）
  ExecutionSimulator:   A股撮合模拟（佣金/印花税/过户费/滑点）
  PerformanceMetrics:   全套性能指标（收益/风险/风险调整/归因）
  WalkForwardValidator: Rolling Walk-Forward OOS 验证
  statistical_tests:    DSR / PSR / Bootstrap CI / IC Newey-West
  AgentBacktester:      Agent 决策历史回测（LGBM + LLM Rerank）
"""

from quantmind.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    Strategy,
    EqualWeightStrategy,
)
from quantmind.backtest.portfolio import Portfolio, Position
from quantmind.backtest.execution import ExecutionSimulator, Order, Fill
from quantmind.backtest.metrics import PerformanceMetrics
from quantmind.backtest.walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
)
from quantmind.backtest.statistical_tests import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    bootstrap_ci,
    ic_ttest_newey_west,
)
from quantmind.backtest.agent_backtest import AgentBacktester, AgentBacktestResult

__all__ = [
    # engine
    "BacktestEngine", "BacktestConfig", "BacktestResult",
    "Strategy", "EqualWeightStrategy",
    # portfolio
    "Portfolio", "Position",
    # execution
    "ExecutionSimulator", "Order", "Fill",
    # metrics
    "PerformanceMetrics",
    # walk_forward
    "WalkForwardValidator", "WalkForwardConfig", "WalkForwardResult", "WalkForwardWindow",
    # statistical_tests
    "deflated_sharpe_ratio", "probabilistic_sharpe_ratio",
    "bootstrap_ci", "ic_ttest_newey_west",
    # agent_backtest
    "AgentBacktester", "AgentBacktestResult",
]
