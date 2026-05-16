"""quantmind.risk — Phase 7 风险与组合管理.

模块：
  FactorRiskModel:    Barra 简化版因子风险模型（因子收益/特质风险/组合风险分解/归因）
  PositionSizer:      多种仓位管理方法（等权/逆波动率/最小方差/风险平价/HRP/Kelly）
  DrawdownController: 动态回撤风控（回撤阈值/波动率目标/CPPI）
  DrawdownRule:       单条回撤控制规则
"""

from quantmind.risk.factor_risk import FactorRiskModel
from quantmind.risk.position_sizing import PositionSizer
from quantmind.risk.drawdown import DrawdownController, DrawdownRule

__all__ = [
    "FactorRiskModel",
    "PositionSizer",
    "DrawdownController",
    "DrawdownRule",
]
