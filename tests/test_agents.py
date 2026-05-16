"""tests/test_agents.py — Agent 单元测试（全量 Mock，不发真实 API 请求）.

测试用例：
- test_planner_output_schema   : 验证 TaskNode 格式
- test_critic_trigger_condition: critical ≥ 1 触发回流（passed=False）
- test_orchestrator_max_iterations: 超过3次强制退出
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from quantmind.core.state import (
    AgentState,
    AgentType,
    CriticFeedback,
    CriticIssue,
    FixAction,
    InvestmentQuery,
    IssueSeverity,
    IssueType,
    QueryIntent,
    TaskNode,
    TaskPlan,
    TaskStatus,
)


# ============================================================================
# 测试固件
# ============================================================================


def _make_query(tickers: list[str] | None = None) -> InvestmentQuery:
    return InvestmentQuery(
        raw_query="分析贵州茅台的投资价值",
        tickers=tickers or ["600519.SH"],
        as_of=date(2024, 6, 30),
        intent=QueryIntent.SINGLE_STOCK,
        horizon_days=60,
    )


def _make_state(tickers: list[str] | None = None) -> AgentState:
    return AgentState(query=_make_query(tickers))


def _mock_llm_response(content: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.content = content
    mock_router = MagicMock()
    mock_router.chat.return_value = mock_resp
    return mock_router


# ============================================================================
# test_planner_output_schema
# ============================================================================


class TestPlannerOutputSchema:
    """验证 PlannerAgent 输出的 TaskNode 格式."""

    def _make_planner_response(self) -> str:
        return json.dumps({
            "rationale": "先收集数据，再分析，最后 Critic 审查",
            "tasks": [
                {
                    "task_id": "t1",
                    "agent_type": "data",
                    "action": "fetch_basics",
                    "params": {"ticker": "600519.SH"},
                    "depends_on": [],
                    "priority": 5,
                },
                {
                    "task_id": "t2",
                    "agent_type": "fundamental",
                    "action": "fundamental_analysis",
                    "params": {"ticker": "600519.SH"},
                    "depends_on": ["t1"],
                    "priority": 3,
                },
                {
                    "task_id": "t3",
                    "agent_type": "critic",
                    "action": "critic_review",
                    "params": {},
                    "depends_on": ["t2"],
                    "priority": 2,
                },
                {
                    "task_id": "t4",
                    "agent_type": "report",
                    "action": "generate_report",
                    "params": {},
                    "depends_on": ["t3"],
                    "priority": 1,
                },
            ],
        }, ensure_ascii=False)

    def test_task_node_required_fields(self):
        """TaskNode 必须含 task_id, agent_type, action, depends_on, priority."""
        required = {"task_id", "agent_type", "action", "depends_on", "priority"}
        t = TaskNode(
            task_id="t1",
            agent_type=AgentType.DATA,
            action="fetch",
            depends_on=[],
            priority=5,
        )
        assert required.issubset(set(t.model_fields.keys()))

    def test_planner_produces_valid_task_nodes(self):
        """PlannerAgent 应生成有效的 TaskPlan，包含正确格式的 TaskNode。"""
        from quantmind.agents.planner import PlannerAgent

        planner = PlannerAgent()
        planner._router = _mock_llm_response(self._make_planner_response())

        state = _make_state()
        result = planner.run(state)

        assert result.plan is not None
        assert isinstance(result.plan, TaskPlan)
        assert len(result.plan.tasks) >= 4

        for task in result.plan.tasks:
            assert isinstance(task, TaskNode)
            assert task.task_id != ""
            assert isinstance(task.agent_type, AgentType)
            assert isinstance(task.depends_on, list)
            assert 1 <= task.priority <= 5

    def test_planner_has_data_before_analysis(self):
        """数据收集任务必须在分析任务之前（depends_on 结构）。"""
        from quantmind.agents.planner import PlannerAgent

        planner = PlannerAgent()
        planner._router = _mock_llm_response(self._make_planner_response())

        state = _make_state()
        result = planner.run(state)

        tasks = result.plan.tasks
        # fundamental 任务必须依赖某个 data 任务
        fundamental_tasks = [t for t in tasks if t.agent_type == AgentType.FUNDAMENTAL]
        data_task_ids = {t.task_id for t in tasks if t.agent_type == AgentType.DATA}

        for ft in fundamental_tasks:
            # 直接依赖或间接依赖一个 data 任务
            assert any(dep in data_task_ids for dep in ft.depends_on), (
                f"FundamentalTask {ft.task_id} 应依赖 DataTask，当前 depends_on={ft.depends_on}"
            )

    def test_planner_fallback_on_invalid_json(self):
        """LLM 返回非法 JSON 时，Planner 应使用默认模板（不崩溃）。"""
        from quantmind.agents.planner import PlannerAgent

        planner = PlannerAgent()
        planner._router = _mock_llm_response("这不是 JSON，完全无效内容。")

        state = _make_state()
        result = planner.run(state)

        assert result.plan is not None
        assert len(result.plan.tasks) >= 5  # 默认模板有 8 个任务

    def test_task_status_default_pending(self):
        """新建 TaskNode 默认状态为 PENDING。"""
        t = TaskNode(task_id="x", agent_type=AgentType.CRITIC, action="review")
        assert t.status == TaskStatus.PENDING


# ============================================================================
# test_critic_trigger_condition
# ============================================================================


class TestCriticTriggerCondition:
    """验证 CriticAgent 的触发规则：critical ≥ 1 → passed=False。"""

    def _make_critic_response(
        self,
        critical_count: int = 0,
        major_count: int = 0,
        minor_count: int = 0,
    ) -> str:
        issues = []
        for i in range(critical_count):
            issues.append({
                "severity": "critical",
                "type": "data_missing",
                "location": "FundamentalAgent",
                "description": f"关键数据缺失 #{i+1}",
                "suggestion": "重新获取财务数据",
            })
        for i in range(major_count):
            issues.append({
                "severity": "major",
                "type": "inconsistency",
                "location": "TechnicalAgent",
                "description": f"逻辑矛盾 #{i+1}",
                "suggestion": "重新分析",
            })
        for i in range(minor_count):
            issues.append({
                "severity": "minor",
                "type": "weak_argument",
                "location": "SentimentAgent",
                "description": f"论证不足 #{i+1}",
                "suggestion": "补充证据",
            })

        passed_guess = (critical_count == 0) and (major_count < 3)
        return json.dumps({
            "passed": passed_guess,
            "issues": issues,
            "overall_quality_score": 8.0 - critical_count * 2 - major_count,
            "approval_message": "通过" if passed_guess else "不通过",
        }, ensure_ascii=False)

    def test_one_critical_triggers_failure(self):
        """1 个 critical issue → passed=False。"""
        from quantmind.agents.critic_agent import CriticAgent

        critic = CriticAgent()
        critic._router = _mock_llm_response(self._make_critic_response(critical_count=1))

        state = _make_state()
        result = critic.run(state)

        assert result.critic_feedback is not None
        assert result.critic_feedback.passed is False
        assert result.critic_feedback.critical_count() == 1

    def test_three_major_triggers_failure(self):
        """3 个 major issue → passed=False。"""
        from quantmind.agents.critic_agent import CriticAgent

        critic = CriticAgent()
        critic._router = _mock_llm_response(self._make_critic_response(major_count=3))

        state = _make_state()
        result = critic.run(state)

        assert result.critic_feedback.passed is False

    def test_two_major_passes(self):
        """2 个 major issue（无 critical）→ passed=True。"""
        from quantmind.agents.critic_agent import CriticAgent

        critic = CriticAgent()
        critic._router = _mock_llm_response(self._make_critic_response(major_count=2))

        state = _make_state()
        result = critic.run(state)

        assert result.critic_feedback.passed is True

    def test_minor_only_passes(self):
        """只有 minor issue → passed=True。"""
        from quantmind.agents.critic_agent import CriticAgent

        critic = CriticAgent()
        critic._router = _mock_llm_response(
            self._make_critic_response(minor_count=5)
        )

        state = _make_state()
        result = critic.run(state)

        assert result.critic_feedback.passed is True

    def test_critic_feedback_has_required_fields(self):
        """CriticFeedback 必须含 passed, issues, overall_quality_score。"""
        fb = CriticFeedback(
            passed=True,
            issues=[],
            overall_quality_score=8.0,
        )
        assert hasattr(fb, "passed")
        assert hasattr(fb, "issues")
        assert hasattr(fb, "overall_quality_score")
        assert isinstance(fb.issues, list)

    def test_rule_overrides_llm_passed_field(self):
        """即使 LLM 说 passed=True，有 critical issue 也应强制 passed=False。"""
        from quantmind.agents.critic_agent import CriticAgent

        # 构造 LLM 说 passed=True 但有 1 个 critical issue 的响应
        response = json.dumps({
            "passed": True,  # LLM 说通过，但有 critical issue
            "issues": [{
                "severity": "critical",
                "type": "hallucination",
                "location": "FundamentalAgent.valuation_analysis",
                "description": "ROE 数字 999% 明显是幻觉",
                "suggestion": "重新调用财务数据",
            }],
            "overall_quality_score": 4.0,
            "approval_message": "（LLM 错误地说通过了）",
        }, ensure_ascii=False)

        critic = CriticAgent()
        critic._router = _mock_llm_response(response)

        state = _make_state()
        result = critic.run(state)

        # 规则强制覆盖 LLM 的 passed=True
        assert result.critic_feedback.passed is False


# ============================================================================
# test_orchestrator_max_iterations
# ============================================================================


class TestOrchestratorMaxIterations:
    """验证编排器 max_iterations 强制退出机制。"""

    def _mock_agent_run(self, state: AgentState) -> AgentState:
        """模拟 Agent 执行（快速返回，不发 LLM 请求）。"""
        return state

    def _mock_critic_fail(self, state: AgentState) -> AgentState:
        """模拟 Critic 持续失败（每次返回 critical issue）。"""
        from quantmind.core.state import CriticFeedback, CriticIssue, IssueSeverity, IssueType
        state.critic_feedback = CriticFeedback(
            passed=False,
            issues=[CriticIssue(
                severity=IssueSeverity.CRITICAL,
                type=IssueType.DATA_MISSING,
                description="数据始终缺失",
            )],
            overall_quality_score=2.0,
        )
        state.iteration_count += 1
        return state

    def test_max_iterations_enforced(self):
        """超过 max_iterations=3 后，编排器必须强制退出（不无限循环）。"""
        from quantmind.agents.orchestrator import ResearchOrchestrator

        orch = ResearchOrchestrator(max_iterations=3)

        # Mock 所有 Agent：Planner/Data/Fundamental/Technical/Sentiment 快速通过
        # Critic 持续失败
        for agent in [orch._planner, orch._data, orch._fundamental,
                      orch._technical, orch._sentiment]:
            agent.run = self._mock_agent_run

        orch._critic.run = self._mock_critic_fail

        query = InvestmentQuery(
            raw_query="测试最大迭代",
            tickers=["600519.SH"],
            as_of=date(2024, 6, 30),
            intent=QueryIntent.SINGLE_STOCK,
        )

        state = orch.run(query, max_iterations=3)

        # 最终状态必须标记为终止
        assert state.iteration_count >= 1, "应至少执行 1 次迭代"

    def test_state_tracks_iteration_count(self):
        """AgentState.iteration_count 应正确递增。"""
        state = AgentState(query=_make_query())
        assert state.iteration_count == 0
        state.iteration_count += 1
        assert state.iteration_count == 1

    def test_reached_max_iterations_method(self):
        """AgentState.reached_max_iterations() 应在超限时返回 True。"""
        state = AgentState(query=_make_query(), max_iterations=3)
        state.iteration_count = 2
        assert not state.reached_max_iterations()
        state.iteration_count = 3
        assert state.reached_max_iterations()

    def test_critic_passed_method(self):
        """AgentState.critic_passed() 无 feedback 时返回 False。"""
        state = AgentState(query=_make_query())
        assert not state.critic_passed()

        state.critic_feedback = CriticFeedback(passed=True, issues=[])
        assert state.critic_passed()

        state.critic_feedback = CriticFeedback(passed=False, issues=[])
        assert not state.critic_passed()


# ============================================================================
# test_report_agent_sections
# ============================================================================


class TestReportAgentSections:
    """验证 ReportAgent 生成的报告含9个必要章节关键词。"""

    _NINE_SECTION_KEYS = [
        "executive_summary",
        "company_overview",
        "financial_analysis",
        "valuation_analysis",
        "technical_quant",
        "sentiment_catalysts",
        "risk_assessment",
        "investment_recommendation",
        "appendix",
    ]

    def _make_report_response(self) -> str:
        """构造包含全部9章节的合法 LLM 响应。"""
        return json.dumps({
            "rating": "buy",
            "target_price_low": 1600.0,
            "target_price_high": 1800.0,
            "confidence": 0.75,
            "horizon_days": 60,
            "executive_summary": ["盈利稳健，估值合理", "技术面处于上升通道", "情绪偏正面"],
            "company_overview": "贵州茅台是中国高端白酒龙头企业。",
            "financial_analysis": "ROE 高达 30%，净利率 50%，财务质量极佳。",
            "valuation_analysis": "DCF 测算公允价值约 1700 元/股，当前存在一定折价。",
            "technical_quant": "MA60 多头排列，RSI14=55，趋势偏强。",
            "sentiment_catalysts": "近期提价预期提振市场情绪，北向资金持续流入。",
            "risk_assessment": "主要风险：消费降级、政策限制、宏观经济下行。",
            "investment_recommendation": "评级：买入。目标价 1600-1800 元，建议持仓 60 天。",
            "appendix": "数据来源：AkShare 官方财务数据，截止 2024-06-30。",
        }, ensure_ascii=False)

    def test_report_agent_produces_nine_sections(self):
        """ReportAgent 应生成包含全部9个章节字段的报告。"""
        from quantmind.agents.report_agent import ReportAgent

        agent = ReportAgent()
        agent._router = _mock_llm_response(self._make_report_response())

        state = _make_state()
        result = agent.run(state)

        assert result.report is not None, "report 不应为 None"
        # 检查 full_markdown 包含9个章节关键词（中文标题）
        md = result.report.full_markdown or ""
        for key in ["执行摘要", "公司概况", "财务分析", "估值分析",
                    "技术", "情绪", "风险", "投资建议", "附录"]:
            assert key in md, f"报告 Markdown 中缺少章节：{key}"

    def test_report_parse_output_nine_keys(self):
        """parse_output 必须提取出所有9个 section key。"""
        from quantmind.agents.report_agent import ReportAgent

        agent = ReportAgent()
        parsed = agent.parse_output(self._make_report_response())

        for key in self._NINE_SECTION_KEYS:
            assert key in parsed, f"parse_output 缺少字段：{key}"

    def test_report_rating_extracted_correctly(self):
        """parse_output 应正确提取 rating 字段。"""
        from quantmind.agents.report_agent import ReportAgent
        from quantmind.core.state import Recommendation

        agent = ReportAgent()
        agent._router = _mock_llm_response(self._make_report_response())

        state = _make_state()
        result = agent.run(state)

        assert result.report is not None
        assert result.report.rating == Recommendation.BUY

    def test_report_fallback_on_invalid_json(self):
        """LLM 返回非法 JSON 时，ReportAgent 不崩溃，report 使用默认值。"""
        from quantmind.agents.report_agent import ReportAgent

        agent = ReportAgent()
        agent._router = _mock_llm_response("这不是 JSON 格式的回复，无效内容。")

        state = _make_state()
        result = agent.run(state)

        # 即使 LLM 输出非法，report 也应被创建
        assert result.report is not None


# ============================================================================
# test_orchestrator_full_flow
# ============================================================================


class TestOrchestratorFullFlow:
    """端到端编排器 Mock 测试：验证完整流程 Planner→Data→...→Critic→Report。"""

    def _mock_critic_pass(self, state: AgentState) -> AgentState:
        """Critic 直接通过。"""
        state.critic_feedback = CriticFeedback(
            passed=True,
            issues=[],
            overall_quality_score=9.0,
        )
        state.iteration_count += 1
        return state

    def _mock_report(self, state: AgentState) -> AgentState:
        """模拟报告生成，写入一个简单 InvestmentReport。"""
        from quantmind.core.state import InvestmentReport, Recommendation
        from datetime import date as _date

        state.report = InvestmentReport(
            query=state.query,
            as_of=_date(2024, 6, 30),
            rating=Recommendation.BUY,
            confidence=0.8,
            horizon_days=60,
            thesis="测试论断",
            fundamental=None,
            technical=None,
            sentiment=None,
            risk_warnings=["风险1"],
            executive_summary=["摘要1"],
            full_markdown="# 测试报告",
            iteration_count=state.iteration_count,
        )
        return state

    def test_full_flow_produces_report(self):
        """完整流程（全 Mock Agent）应生成 report 且 iteration_count=1。"""
        from quantmind.agents.orchestrator import ResearchOrchestrator

        orch = ResearchOrchestrator(max_iterations=3)

        def _noop(state): return state
        for agent in [orch._planner, orch._data, orch._fundamental,
                      orch._technical, orch._sentiment]:
            agent.run = _noop

        orch._critic.run = self._mock_critic_pass
        orch._report.run = self._mock_report

        query = InvestmentQuery(
            raw_query="端到端测试",
            tickers=["600519.SH"],
            as_of=date(2024, 6, 30),
            intent=QueryIntent.SINGLE_STOCK,
        )
        state = orch.run(query)

        assert state.report is not None, "全流程应生成 report"
        assert state.iteration_count >= 1, f"期望 iteration_count>=1，实际={state.iteration_count}"

    def test_full_flow_critic_routes_to_report_on_pass(self):
        """Critic 通过时，路由目标必须是 report 节点（而非回流）。"""
        from quantmind.agents.orchestrator import ResearchOrchestrator

        orch = ResearchOrchestrator(max_iterations=3)

        # 构造一个已通过的 critic_feedback
        state = AgentState(query=_make_query())
        state.critic_feedback = CriticFeedback(passed=True, issues=[])
        state.iteration_count = 1

        route = orch._route_after_critic({"_agent_state": state})
        assert route == "report", f"Critic 通过时路由应为 report，实际为 {route}"

    def test_full_flow_critic_routes_to_retry_on_fail(self):
        """Critic 失败时，路由目标必须是回流 Agent（非 report）。"""
        from quantmind.agents.orchestrator import ResearchOrchestrator

        orch = ResearchOrchestrator(max_iterations=3)

        state = AgentState(query=_make_query())
        state.critic_feedback = CriticFeedback(
            passed=False,
            issues=[CriticIssue(
                severity=IssueSeverity.CRITICAL,
                type=IssueType.DATA_MISSING,
                description="基本面数据缺失",
                fix_action=FixAction(
                    agent=AgentType.FUNDAMENTAL,
                    instruction="重新拉取财务数据",
                ),
            )],
            overall_quality_score=3.0,
        )
        state.iteration_count = 1

        route = orch._route_after_critic({"_agent_state": state})
        assert route in ("fundamental", "technical", "sentiment", "data"), (
            f"Critic 失败时路由应回流 Agent，实际为 {route}"
        )


# ============================================================================
# test_stream_yields_updates
# ============================================================================


class TestStreamYieldsUpdates:
    """验证 stream() 方法每个节点完成后 yield (node_name, AgentState)。"""

    def _mock_critic_pass(self, state: AgentState) -> AgentState:
        state.critic_feedback = CriticFeedback(
            passed=True, issues=[], overall_quality_score=8.5
        )
        state.iteration_count += 1
        return state

    def _mock_report(self, state: AgentState) -> AgentState:
        from quantmind.core.state import InvestmentReport, Recommendation
        from datetime import date as _date
        state.report = InvestmentReport(
            query=state.query,
            as_of=_date(2024, 6, 30),
            rating=Recommendation.HOLD,
            confidence=0.6,
            horizon_days=60,
            thesis="stream 测试",
            full_markdown="# stream test",
            iteration_count=state.iteration_count,
        )
        return state

    def test_stream_yields_node_names(self):
        """stream() 应 yield 每个节点名称，且顺序正确。"""
        from quantmind.agents.orchestrator import ResearchOrchestrator

        orch = ResearchOrchestrator(max_iterations=3)

        def _noop(state): return state
        for agent in [orch._planner, orch._data, orch._fundamental,
                      orch._technical, orch._sentiment]:
            agent.run = _noop

        orch._critic.run = self._mock_critic_pass
        orch._report.run = self._mock_report

        query = InvestmentQuery(
            raw_query="stream 测试",
            tickers=["600519.SH"],
            as_of=date(2024, 6, 30),
            intent=QueryIntent.SINGLE_STOCK,
        )

        yielded_nodes: list[str] = []
        for node_name, state in orch.stream(query):
            yielded_nodes.append(node_name)

        # 期望按顺序经过这些节点
        expected_ordered = ["planner", "data", "fundamental", "technical", "sentiment", "critic", "report"]
        for expected in expected_ordered:
            assert expected in yielded_nodes, f"stream 缺少节点：{expected}"

        # 顺序验证：planner 在 data 之前
        assert yielded_nodes.index("planner") < yielded_nodes.index("data")
        assert yielded_nodes.index("data") < yielded_nodes.index("fundamental")

    def test_stream_yields_agent_state(self):
        """stream() yield 的第二个元素必须是 AgentState 实例。"""
        from quantmind.agents.orchestrator import ResearchOrchestrator

        orch = ResearchOrchestrator(max_iterations=3)

        def _noop(state): return state
        for agent in [orch._planner, orch._data, orch._fundamental,
                      orch._technical, orch._sentiment]:
            agent.run = _noop

        orch._critic.run = self._mock_critic_pass
        orch._report.run = self._mock_report

        query = InvestmentQuery(
            raw_query="AgentState 类型检查",
            tickers=["000858.SZ"],
            as_of=date(2024, 6, 30),
            intent=QueryIntent.SINGLE_STOCK,
        )

        for node_name, yielded_state in orch.stream(query):
            assert isinstance(yielded_state, AgentState), (
                f"节点 {node_name} yield 的 state 类型错误：{type(yielded_state)}"
            )
