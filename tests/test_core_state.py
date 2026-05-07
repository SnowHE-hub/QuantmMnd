"""测试 quantmind.core.state Pydantic schemas."""

from __future__ import annotations

from datetime import date

import pytest

from quantmind.core.state import (
    AgentState,
    AgentType,
    CriticFeedback,
    CriticIssue,
    FixAction,
    HistoryEntry,
    InvestmentQuery,
    InvestmentReport,
    IssueSeverity,
    IssueType,
    QueryIntent,
    Recommendation,
    TaskNode,
    TaskPlan,
    TaskStatus,
)


class TestInvestmentQuery:
    def test_minimal(self) -> None:
        q = InvestmentQuery(raw_query="hi")
        assert q.raw_query == "hi"
        assert q.intent == QueryIntent.GENERAL_QA
        assert q.tickers == []
        assert q.horizon_days == 60
        assert q.as_of is None

    def test_full(self) -> None:
        q = InvestmentQuery(
            raw_query="比较宁德和比亚迪",
            intent=QueryIntent.COMPARE_STOCKS,
            tickers=["300750.SZ", "002594.SZ"],
            as_of=date(2024, 6, 30),
        )
        assert len(q.tickers) == 2
        assert q.intent == QueryIntent.COMPARE_STOCKS

    def test_horizon_must_be_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            InvestmentQuery(raw_query="x", horizon_days=0)


class TestTaskPlan:
    def test_pending_filter(self) -> None:
        plan = TaskPlan(
            tasks=[
                TaskNode(task_id="T1", agent_type=AgentType.DATA, action="fetch"),
                TaskNode(
                    task_id="T2",
                    agent_type=AgentType.FUNDAMENTAL,
                    action="analyze",
                    depends_on=["T1"],
                    status=TaskStatus.SUCCESS,
                ),
            ]
        )
        pending = plan.pending()
        assert len(pending) == 1
        assert pending[0].task_id == "T1"

    def test_by_id(self) -> None:
        plan = TaskPlan(tasks=[TaskNode(task_id="T1", agent_type=AgentType.DATA, action="fetch")])
        assert plan.by_id("T1") is not None
        assert plan.by_id("T999") is None

    def test_is_done(self) -> None:
        plan = TaskPlan(
            tasks=[
                TaskNode(
                    task_id="T1",
                    agent_type=AgentType.DATA,
                    action="x",
                    status=TaskStatus.SUCCESS,
                ),
                TaskNode(
                    task_id="T2",
                    agent_type=AgentType.FUNDAMENTAL,
                    action="y",
                    status=TaskStatus.FAILED,
                ),
            ]
        )
        assert plan.is_done() is True


class TestCriticFeedback:
    def test_severity_count(self) -> None:
        fb = CriticFeedback(
            passed=False,
            issues=[
                CriticIssue(
                    severity=IssueSeverity.CRITICAL,
                    type=IssueType.PIT_VIOLATION,
                    description="leak detected",
                    fix_action=FixAction(agent=AgentType.DATA, instruction="re-fetch with as_of"),
                ),
                CriticIssue(
                    severity=IssueSeverity.MAJOR,
                    type=IssueType.WEAK_ARGUMENT,
                    description="needs more evidence",
                ),
                CriticIssue(
                    severity=IssueSeverity.MINOR,
                    type=IssueType.INCONSISTENCY,
                    description="typo",
                ),
            ],
        )
        assert fb.critical_count() == 1
        assert fb.major_count() == 1


class TestAgentState:
    def test_default_construction(self, sample_query: InvestmentQuery) -> None:
        s = AgentState(query=sample_query)
        assert s.iteration_count == 0
        assert s.max_iterations == 3
        assert not s.critic_passed()
        assert not s.reached_max_iterations()
        assert s.fundamentals == {}
        assert s.history == []

    def test_add_history_aggregates(self, sample_query: InvestmentQuery) -> None:
        from datetime import datetime

        s = AgentState(query=sample_query)
        s.add_history(
            HistoryEntry(
                agent=AgentType.DATA,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                token_usage=100,
                cost_cny=0.05,
            )
        )
        s.add_history(
            HistoryEntry(
                agent=AgentType.FUNDAMENTAL,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                token_usage=200,
                cost_cny=0.10,
            )
        )
        assert s.total_tokens == 300
        assert abs(s.total_cost_cny - 0.15) < 1e-9
        assert len(s.history) == 2

    def test_iteration_logic(self, sample_query: InvestmentQuery) -> None:
        s = AgentState(query=sample_query, max_iterations=2)
        s.iteration_count = 2
        assert s.reached_max_iterations()

    def test_json_round_trip(self, sample_query: InvestmentQuery) -> None:
        s = AgentState(query=sample_query)
        data = s.model_dump(mode="json")
        s2 = AgentState.model_validate(data)
        assert s2.query.raw_query == s.query.raw_query


class TestInvestmentReport:
    def test_minimal(self, sample_query: InvestmentQuery) -> None:
        r = InvestmentReport(query=sample_query, as_of=date(2024, 6, 30))
        assert r.rating == Recommendation.HOLD
        assert r.iteration_count == 0
        assert r.full_markdown == ""
