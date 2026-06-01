"""quantmind/iteration — 模拟迭代优化闭环.

流程：
  SimulationAnalyzer  →  ParameterOptimizer  →  IterationComparator
  (诊断30d结果)           (生成/应用参数建议)     (对比两轮结果)
"""
from quantmind.iteration.analyzer import SimulationAnalyzer, SimDiagnosis
from quantmind.iteration.optimizer import ParameterOptimizer, ParameterSuggestion
from quantmind.iteration.comparator import IterationComparator, ComparisonReport

__all__ = [
    "SimulationAnalyzer",
    "SimDiagnosis",
    "ParameterOptimizer",
    "ParameterSuggestion",
    "IterationComparator",
    "ComparisonReport",
]
