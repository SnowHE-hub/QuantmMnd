"""quantmind.core.state — 全局 Pydantic Schemas.

定义 Agent 系统、模型层、回测层之间共享的数据结构。
所有结构都是 Pydantic v2 BaseModel，可序列化为 JSON / dict，
也可作为 LangGraph StateGraph 的 state_schema。

主要分组：
    1. 用户输入与任务      —— InvestmentQuery / TaskNode / TaskPlan
    2. 数据相关            —— DataSnapshot / TickerData
    3. 三大分析维度        —— FundamentalAnalysis / TechnicalAnalysis /
                               SentimentAnalysis
    4. Critic / 报告       —— CriticIssue / CriticFeedback / InvestmentReport
    5. Agent 全局 State   —— AgentState（LangGraph 用）

设计原则：
    - 严格 typing：所有字段标注类型
    - 时间字段统一用 ``datetime.date`` 或 ``datetime.datetime``，不用字符串
    - PIT 关键字段加 ``as_of: date``，明确"该结论是基于哪天可见的数据"
    - 所有金额/价格用 ``float``，所有比例用 ``float in [0, 1]`` 或 ``float`` 百分点
    - 默认值尽量保守，避免 ``None`` 漫天飞
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# 1. 用户输入与任务规划
# ============================================================================


class QueryIntent(StrEnum):
    """用户查询的意图分类（Planner 用于决定 DAG 模板）."""

    SINGLE_STOCK = "single_stock"  # 分析单只股票
    COMPARE_STOCKS = "compare_stocks"  # 对比多只股票
    INDUSTRY_RESEARCH = "industry_research"  # 行业研究
    RISK_ASSESSMENT = "risk_assessment"  # 风险评估
    PORTFOLIO_REVIEW = "portfolio_review"  # 组合审视
    GENERAL_QA = "general_qa"  # 一般问答


class InvestmentQuery(BaseModel):
    """用户原始投资查询."""

    raw_query: str = Field(..., description="用户原始问题文本")
    intent: QueryIntent = Field(default=QueryIntent.GENERAL_QA, description="意图")
    tickers: list[str] = Field(default_factory=list, description="涉及的股票代码")
    horizon_days: int = Field(default=60, ge=1, description="投资时间窗口（天）")
    as_of: date | None = Field(
        default=None,
        description="PIT 时点：Agent 只能看到 as_of 之前的数据；None=当前最新",
    )
    user_constraints: dict[str, Any] = Field(
        default_factory=dict, description="用户额外约束（行业偏好、风险偏好等）"
    )


class AgentType(StrEnum):
    """Agent 类型枚举."""

    PLANNER = "planner"
    DATA = "data"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    QUANT = "quant"
    CRITIC = "critic"
    REPORT = "report"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskNode(BaseModel):
    """Planner 生成的 DAG 节点."""

    task_id: str = Field(..., description="如 T1, T2, T3...")
    agent_type: AgentType
    action: str = Field(..., description="具体动作描述")
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list, description="依赖的 task_id")
    priority: int = Field(default=3, ge=1, le=5)
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TaskPlan(BaseModel):
    """Planner 输出：完整 DAG."""

    tasks: list[TaskNode] = Field(default_factory=list)
    rationale: str = Field(default="", description="Planner 的拆解思路")

    def pending(self) -> list[TaskNode]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def by_id(self, task_id: str) -> TaskNode | None:
        return next((t for t in self.tasks if t.task_id == task_id), None)

    def is_done(self) -> bool:
        return all(
            t.status in {TaskStatus.SUCCESS, TaskStatus.SKIPPED, TaskStatus.FAILED}
            for t in self.tasks
        )


# ============================================================================
# 2. 数据快照
# ============================================================================


class TickerData(BaseModel):
    """单只股票的原始数据集合（JSON 化版，用于 state 传递）.

    DataFrame 不直接放进 state（避免 JSON 序列化失败），而是存路径。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ticker: str
    name: str | None = None
    industry: str | None = None
    sector: str | None = None
    listing_date: date | None = None
    delisting_date: date | None = None
    is_tradable: bool = True
    # 存储路径（parquet）。state 流转时只传路径，需要时再 read
    price_path: str | None = None
    financials_path: str | None = None
    news_path: str | None = None
    # 关键统计量（避免每次都 read）
    latest_close: float | None = None
    market_cap: float | None = None
    pe_ttm: float | None = None
    pb: float | None = None


class DataSnapshot(BaseModel):
    """as_of 时点的全市场数据快照元信息."""

    as_of: date = Field(..., description="PIT 时点")
    universe: str = Field(..., description="股票池名，如 csi300")
    tickers: list[TickerData] = Field(default_factory=list)
    snapshot_dir: str | None = Field(default=None, description="data/snapshots/{as_of}/ 路径")
    fetched_at: datetime = Field(default_factory=datetime.now)
    data_sources: list[str] = Field(default_factory=list, description="如 [akshare, tushare]")
    coverage_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


# ============================================================================
# 3. 三大分析维度
# ============================================================================


class Recommendation(StrEnum):
    """评级."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class FundamentalAnalysis(BaseModel):
    """基本面分析结果."""

    ticker: str
    as_of: date
    # 计算后的关键指标（Python 严格计算，LLM 不算数）
    ratios: dict[str, float] = Field(default_factory=dict, description="财务比率")
    dcf_value: float | None = Field(default=None, description="DCF 内在价值（每股）")
    dcf_assumptions: dict[str, float] = Field(default_factory=dict)
    comparable_value: float | None = Field(default=None, description="可比公司估值（每股）")
    peer_comparison: dict[str, dict[str, float]] = Field(default_factory=dict)
    # LLM 解读
    profitability_analysis: str = ""
    solvency_analysis: str = ""
    growth_analysis: str = ""
    valuation_analysis: str = ""
    key_risks: list[str] = Field(default_factory=list)
    summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TechnicalAnalysis(BaseModel):
    """技术面分析结果."""

    ticker: str
    as_of: date
    indicators: dict[str, float] = Field(default_factory=dict)
    pattern: str | None = Field(default=None, description="如 突破/反转/整理")
    momentum_score: float = Field(default=0.0, description="动量综合分 [-1, 1]")
    volatility: float | None = None
    quant_score: float | None = Field(default=None, description="LightGBM 等量化模型给出的分数")
    quant_rank: int | None = None
    quant_explanation: dict[str, float] = Field(default_factory=dict, description="SHAP 因子贡献")
    summary: str = ""
    signal: Literal["bullish", "neutral", "bearish"] = "neutral"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class NewsItem(BaseModel):
    """单条新闻摘要."""

    title: str
    published_at: datetime
    source: str = ""
    url: str | None = None
    summary: str = ""
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    impact_magnitude: int = Field(default=1, ge=1, le=5)
    event_type: str = ""


class SentimentAnalysis(BaseModel):
    """情绪面分析结果."""

    ticker: str
    as_of: date
    news_items: list[NewsItem] = Field(default_factory=list)
    aggregated_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    analyst_rating_score: float | None = None
    analyst_estimate_revision_3m: float | None = None
    north_bound_flow_30d: float | None = None
    margin_balance_change: float | None = None
    summary: str = ""
    catalysts: list[str] = Field(default_factory=list, description="近期催化剂")
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ============================================================================
# 4. Critic 与最终报告
# ============================================================================


class IssueSeverity(StrEnum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class IssueType(StrEnum):
    DATA_MISSING = "data_missing"
    INCONSISTENCY = "inconsistency"
    CALCULATION_ERROR = "calculation_error"
    RISK_MISSING = "risk_missing"
    WEAK_ARGUMENT = "weak_argument"
    HALLUCINATION = "hallucination"
    PIT_VIOLATION = "pit_violation"


class FixAction(BaseModel):
    """Critic 给出的修复指令."""

    agent: AgentType
    instruction: str


class CriticIssue(BaseModel):
    severity: IssueSeverity
    type: IssueType
    description: str
    fix_action: FixAction | None = None


class CriticFeedback(BaseModel):
    """Critic Agent 输出."""

    passed: bool
    issues: list[CriticIssue] = Field(default_factory=list)
    overall_quality_score: float = Field(default=5.0, ge=1.0, le=10.0)
    approval_message: str = ""

    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    def major_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.MAJOR)


class InvestmentReport(BaseModel):
    """端到端 Agent 协作的最终交付物."""

    query: InvestmentQuery
    as_of: date
    generated_at: datetime = Field(default_factory=datetime.now)
    rating: Recommendation = Recommendation.HOLD
    target_price: float | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    horizon_days: int = 60
    thesis: str = Field(default="", description="核心投资逻辑")
    fundamental: FundamentalAnalysis | None = None
    technical: TechnicalAnalysis | None = None
    sentiment: SentimentAnalysis | None = None
    risk_warnings: list[str] = Field(default_factory=list)
    executive_summary: list[str] = Field(default_factory=list, description="3-5 要点")
    full_markdown: str = Field(default="", description="完整 Markdown 报告")
    data_sources: list[str] = Field(default_factory=list)
    iteration_count: int = 0
    token_usage_total: int = 0
    cost_cny_total: float = 0.0


# ============================================================================
# 5. LangGraph 全局 State
# ============================================================================


class HistoryEntry(BaseModel):
    """Agent 执行 trace 的单条记录."""

    agent: AgentType
    task_id: str | None = None
    started_at: datetime
    finished_at: datetime
    input_summary: str = ""
    output_summary: str = ""
    token_usage: int = 0
    cost_cny: float = 0.0
    success: bool = True
    error: str | None = None


class AgentState(BaseModel):
    """LangGraph StateGraph 全局状态.

    每个 Agent 是一个 node，输入输出都是 AgentState。
    LangGraph 会做不可变更新：返回 dict 时 LangGraph 自动 merge。

    用法（LangGraph 0.x）::

        from langgraph.graph import StateGraph
        graph = StateGraph(AgentState)
        graph.add_node("planner", planner_fn)
        ...
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 用户输入
    query: InvestmentQuery

    # 任务规划
    plan: TaskPlan | None = None

    # 数据快照
    snapshot: DataSnapshot | None = None

    # 三大分析（按 ticker 索引；多股票场景一个 ticker 一份）
    fundamentals: dict[str, FundamentalAnalysis] = Field(default_factory=dict)
    technicals: dict[str, TechnicalAnalysis] = Field(default_factory=dict)
    sentiments: dict[str, SentimentAnalysis] = Field(default_factory=dict)

    # Critic 与迭代
    critic_feedback: CriticFeedback | None = None
    iteration_count: int = 0
    max_iterations: int = 3

    # 最终报告
    report: InvestmentReport | None = None

    # 执行 trace
    history: list[HistoryEntry] = Field(default_factory=list)

    # 计费
    total_tokens: int = 0
    total_cost_cny: float = 0.0

    # 控制流
    should_terminate: bool = False
    terminate_reason: str = ""

    def add_history(self, entry: HistoryEntry) -> None:
        self.history.append(entry)
        self.total_tokens += entry.token_usage
        self.total_cost_cny += entry.cost_cny

    def critic_passed(self) -> bool:
        return self.critic_feedback is not None and self.critic_feedback.passed

    def reached_max_iterations(self) -> bool:
        return self.iteration_count >= self.max_iterations


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Enums
    "AgentType",
    "IssueSeverity",
    "IssueType",
    "QueryIntent",
    "Recommendation",
    "TaskStatus",
    # Query / Plan
    "InvestmentQuery",
    "TaskNode",
    "TaskPlan",
    # Data
    "DataSnapshot",
    "TickerData",
    # Analyses
    "FundamentalAnalysis",
    "NewsItem",
    "SentimentAnalysis",
    "TechnicalAnalysis",
    # Critic
    "CriticFeedback",
    "CriticIssue",
    "FixAction",
    # Report / State
    "AgentState",
    "HistoryEntry",
    "InvestmentReport",
]


# ----------------------------------------------------------------------------
# 自检：python -m quantmind.core.state
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    q = InvestmentQuery(
        raw_query="帮我分析一下宁德时代未来 60 天的投资价值",
        intent=QueryIntent.SINGLE_STOCK,
        tickers=["300750.SZ"],
        horizon_days=60,
        as_of=date(2024, 6, 30),
    )
    state = AgentState(query=q, max_iterations=3)
    print(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print()
    print(f"AgentState fields: {list(state.model_fields.keys())}")
    print(
        f"critic_passed = {state.critic_passed()}, "
        f"reached_max_iter = {state.reached_max_iterations()}"
    )
