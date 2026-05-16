"""quantmind.selection — 全市场漏斗选股模块."""

from quantmind.selection.funnel_selector import FunnelSelector, FunnelResult, FunnelStats
from quantmind.selection.lazy_data_engine import LazyDataEngine

__all__ = ["FunnelSelector", "FunnelResult", "FunnelStats", "LazyDataEngine"]
