"""quantmind.contracts —— 产品化契约层（P0：定义 schema + 注册表，不接生产链路）.

解封依据：Phase 1 闸门通过（P4 Tier 1 + 补1/补2/Diag 定因）。详见 docs/plans/productization_backlog.md。
本层只做"最小可用契约"：HorizonRegistry / ModelRegistry / RecommendationContract / OutcomeContract。
前端/API/Agent 迁移属 P1，排在 executable NAV 回测之后。
"""
from quantmind.contracts.horizons import Horizon, HorizonRegistry
from quantmind.contracts.model_registry import (
    ModelRecord, register, get, list_models, set_gate_status, register_from_manifest)
from quantmind.contracts.recommendation import RecommendationContract, OutcomeContract

__all__ = [
    "Horizon", "HorizonRegistry",
    "ModelRecord", "register", "get", "list_models", "set_gate_status", "register_from_manifest",
    "RecommendationContract", "OutcomeContract",
]
