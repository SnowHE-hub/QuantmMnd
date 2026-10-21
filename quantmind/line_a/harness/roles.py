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
    value: float | None = Field(None, description="一条声明引用多个字段时，逐条填写该字段对应的数值")
    unit: str | None = Field(None, description="与 value 配套的单位")
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
7. 所有日期都以 as_of 为上限；as_of 之后的信息不存在。
8. 只引用工具结果里出现过的 evidence_id（包括事件链的 chain:… 编号）；不要自己拼编号。
9. formula 只写运算式（只用 local_id、数字、+ - * / 和 abs），解释写在 text 里；由两个数算出的比值、增速、差额
   一律写成 computed，不要写成 reported_fact。记录里的 _units 给出每个字段的单位，换算时以它为准。
10. 声明最多 25 条，每条 text 不超过 120 字；优先写对判断最有用的事实，次要的合并或省略。"""

A1_SYSTEM = ("你是 A 股上市公司的财务研究员（角色 A1）。任务是根据按时点取得的财报、业绩预告和快报、披露日程、"
             "审计意见与股权质押数据，整理这家公司最近几期的经营与财务事实：收入与利润的同比变化、盈利质量"
             "（经营现金流与净利润的关系）、负债与流动性、预告与实际的差异、审计意见是否非标准、质押比例。\n"
             + COMMON_RULES)
A2_SYSTEM = ("你是 A 股公告与事件研究员（角色 A2）。任务是根据按时点取得的公告列表和事件链，整理这家公司当前进行中"
             "和最近结束的重要事项（回购、增减持、重组、诉讼、监管问询、立案、风险警示、限售解禁等）："
             "每个事项处于计划、进展、实施、完成还是终止，有没有更正公告，哪些需要进一步核实。\n" + COMMON_RULES)

A3_SYSTEM = ("你是 A 股公司治理与资本运作审查员（角色 A3）。任务是根据按时点取得的股东增减持、限售解禁安排、回购、"
             "股权质押和相关公告，整理：重要股东和高管的增减持（方向、比例、是否集中、是否在计划内）、未来解禁的规模和时间、"
             "回购进度、质押比例、控制权或实控人变动、并购重组的状态。区分计划与已实施；区分上市公司本身与股东、子公司。\n"
             + COMMON_RULES)
A4_SYSTEM = ("你是 A 股预期与资金证据员（角色 A4）。任务是根据按时点取得的卖方盈利预测（只比较同一机构对同一年度的前后两次"
             "预测）、评级变化、资金流向、融资余额和龙虎榜记录，整理预期和资金的变化。资金数据只描述，不推断主力意图；"
             "预测数据覆盖不全时必须写明覆盖范围，不能把没有数据当成没有覆盖。\n" + COMMON_RULES)


class Judgment(BaseModel):
    claim_ref: str = Field(description="被审查的声明，格式 <角色>:<local_id>，例如 A1:c3")
    verdict: Literal["supported", "refuted", "insufficient", "needs_followup"]
    reason: str = Field(description="一两句话说明依据；反驳时写出反证")
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ReviewOutput(BaseModel):
    ticker: str
    as_of: str
    judgments: list[Judgment]
    new_findings: list[ClaimDraft] = Field(default_factory=list, description="审查中新发现、上游遗漏的事实或反证")
    abstain: bool = False
    abstain_reason: str | None = None


A6_SYSTEM = ("你是证据审查与反证协调员（角色 A6）。上游角色 A1–A4 已经提交了声明。你要先独立核对证据，再看结论："
             "用 read_evidence 读回声明引用的原始记录，核对主体、时间、单位、方向、条件和阶段（计划还是实施）；"
             "用 search_counterevidence 和公告工具专门寻找更正、撤回、澄清、终止、条件未满足、主体不是上市公司等反证。"
             "对每条重要声明给出 supported / refuted / insufficient / needs_followup，不要因为声明听起来合理就判 supported。"
             "你不是最终校验器：结构、数值和引用的机械核对由代码完成，你关注的是证据是否真的支持结论。\n"
             "规则：只用工具返回的数据；反驳必须给出反证的 evidence_id；不给买卖建议；声明最多审查 25 条。")

A5_SYSTEM = ("你是 A 股行业与政策证据员（角色 A5）。任务是根据公司的行业归属、同行业共享的证据包（窗口内的行业新闻、"
             "行业成分股相对全市场的表现）和政策新闻检索，判断哪些行业或政策信息适用于这家公司：适用范围、影响链条、"
             "反向因素和推断的边界。新闻只能说明\"发生了什么\"，不能直接推出对这家公司的影响；不适用或无法判断时写 "
             "open_question。前瞻新闻只从 2026-10-17 开始采集，没有新闻不代表没有政策变化。\n" + COMMON_RULES)
