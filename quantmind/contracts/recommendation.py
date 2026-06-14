"""RecommendationContract + OutcomeContract —— 推荐/反馈统一 schema（定义 ≠ 接生产）.

字段照录 Codex 报告 §A.2 / §A.3（productization_backlog.md 附录）。本模块**只定义 schema +
校验 + 序列化**，不接入任何生产链路（前端/API/Agent 迁移属 P1，排在 executable NAV 之后）。
每条推荐必须能溯源：来自哪个 horizon / 模型 / 特征 / 数据版本 / gate。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from quantmind.contracts.horizons import HorizonRegistry


@dataclass
class RecommendationContract:
    """单条推荐输出（前后端共用）。score 已含方向；gate_status 来自 ModelRegistry。"""
    as_of: str                      # YYYY-MM-DD
    ticker: str                     # 000001.SZ
    horizon_name: str               # short | robust | long
    horizon_days: int               # 12 | 21 | 63
    label_col: str                  # forward_return_12d ...
    model_id: str
    feature_set_id: str
    score_raw: float
    score_neutralized: float
    rank: int
    weight: float
    liquidity_bucket: str           # b0(最活跃)..b2 / unknown
    cost_model_id: str              # wf_costs_v1
    gate_status: str                # research_candidate / ..._pending_nav / production
    agent_summary_id: str | None = None

    def validate(self) -> "RecommendationContract":
        h = HorizonRegistry.get(self.horizon_name)   # 抛错若 horizon 非法
        if self.horizon_days != h.days:
            raise ValueError(f"horizon_days {self.horizon_days} != registry {h.days} for {self.horizon_name}")
        if self.label_col != h.label_col:
            raise ValueError(f"label_col {self.label_col!r} != registry {h.label_col!r}")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"weight {self.weight} out of [0,1]")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RecommendationContract":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class OutcomeContract:
    """统一收集 12d/21d/63d 真实收益 + 成本 + 成交状态（feedback schema）。"""
    as_of: str
    ticker: str
    horizon_days: int
    model_id: str
    entry_price: float
    exit_price: float
    actual_return_12d: float | None = None
    actual_return_21d: float | None = None
    actual_return_63d: float | None = None
    slippage_bps: float | None = None
    fees_bps: float | None = None
    net_return: float | None = None
    fill_status: str = "pending"     # pending | filled | partial | rejected
    liquidity_bucket: str | None = None

    def validate(self) -> "OutcomeContract":
        if self.fill_status not in {"pending", "filled", "partial", "rejected"}:
            raise ValueError(f"bad fill_status {self.fill_status!r}")
        HorizonRegistry.by_days(self.horizon_days)   # 抛错若 days 非法
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OutcomeContract":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


__all__ = ["RecommendationContract", "OutcomeContract"]
