"""quantmind.line_a.harness.roles — 角色的输出 Schema 与系统提示（P2 垂直切片：A1 财务、A2 公告事件）.

角色只产出**声明**（事实、计算、推断、反证、未解问题），不写股票权重、不给买卖建议（第 5 节 F6）。
每条声明都要能被独立校验器核对：
- 事实要引用证据编号；
- 计算要写公式和输入声明；
- 推断要说明依据哪些声明。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceRef(BaseModel):
    evidence_id: str
    field: str | None = Field(None, description="引用的是记录里的哪个字段，例如 n_income_attr_p")
    quote: str | None = Field(None, description="文档证据：逐字引文")
    page: int | None = None
    para: int | None = None


class ClaimDraft(BaseModel):
    local_id: str = Field(description="本次输出内唯一，例如 c1、c2")
    claim_type: Literal["reported_fact", "computed", "inference", "counter_evidence", "open_question"]
    text: str = Field(description="一句中文陈述；数字要带单位和期间")
    period: str | None = Field(None, description="报告期 YYYYMMDD 或事件日期")
    metric: str | None = None
    value: float | None = None
    unit: str | None = Field(None, description="元 / 万元 / 亿元 / % / 股 等")
    formula: str | None = Field(None, description="computed 必填，用输入声明的 local_id 写，例如 (c1-c2)/abs(c2)*100")
    inputs: list[str] = Field(default_factory=list, description="computed / inference 依据的声明 local_id")
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class RoleOutput(BaseModel):
    ticker: str
    as_of: str
    claims: list[ClaimDraft]
    risk_flags: list[str] = Field(default_factory=list, description="待确认的风险事实，只描述事实，不打分")
    abstain: bool = False
    abstain_reason: str | None = None


COMMON_RULES = """规则：
1. 只用工具返回的数据。工具没给的，写成 open_question，不要猜。
2. reported_fact 必须在 evidence 里给出 evidence_id 和字段名，value 填记录里的原值（Tushare 财报金额单位是元）。
3. computed 必须给 formula（用输入声明的 local_id）和 inputs；不要在 text 里写没有出处的数字。
4. inference 要在 inputs 里列出依据的声明，并说明不确定性；需要反证时用 counter_evidence。
5. 区分"计划 / 预案"和"已实施 / 已完成"；区分上市公司本身与子公司、控股股东。
6. 不给买卖建议，不写目标价，不写组合权重。数据不足以下结论时设 abstain = true 并说明原因。
7. 所有日期都以 as_of 为上限；as_of 之后的信息不存在。"""

A1_SYSTEM = ("你是 A 股上市公司的财务研究员（角色 A1）。任务是根据按时点取得的财报、业绩预告和快报、披露日程、"
             "审计意见与股权质押数据，整理这家公司最近几期的经营与财务事实：收入与利润的同比变化、盈利质量"
             "（经营现金流与净利润的关系）、负债与流动性、预告与实际的差异、审计意见是否非标准、质押比例。\n"
             + COMMON_RULES)
A2_SYSTEM = ("你是 A 股公告与事件研究员（角色 A2）。任务是根据按时点取得的公告列表和事件链，整理这家公司当前进行中"
             "和最近结束的重要事项（回购、增减持、重组、诉讼、监管问询、立案、风险警示、限售解禁等）："
             "每个事项处于计划、进展、实施、完成还是终止，有没有更正公告，哪些需要进一步核实。\n" + COMMON_RULES)
