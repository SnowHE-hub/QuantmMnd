"""quantmind.agents.orchestrator — LangGraph 编排器.

DAG：planner → data → fundamental → technical → sentiment → critic → report
条件边：critic.passed=False 且 iteration < 3 → 回流对应 Agent 重做
Compaction 机制：每轮必须减少 issue 数，否则强制退出
流式：astream() 每个节点完成后立即 yield AgentState 快照
断点续传：run_with_checkpoint() 使用 SqliteSaver（.cache/checkpoints.db）
"""

from __future__ import annotations

import asyncio
import os
from datetime import date
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from loguru import logger

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise ImportError("pip install langgraph")

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False
    logger.warning("[Orchestrator] langgraph-checkpoint-sqlite 未安装，断点续传不可用")

from langgraph.checkpoint.memory import MemorySaver

from quantmind.agents.planner import PlannerAgent
from quantmind.agents.data_agent import DataAgent
from quantmind.agents.fundamental_agent import FundamentalAgent
from quantmind.agents.technical_agent import TechnicalAgent
from quantmind.agents.sentiment_agent import SentimentAgent
from quantmind.agents.critic_agent import CriticAgent
from quantmind.agents.report_agent import ReportAgent
from quantmind.core.state import (
    AgentState,
    AgentType,
    InvestmentQuery,
    QueryIntent,
)

__all__ = ["ResearchOrchestrator", "run_research"]

# SqliteSaver 数据库路径
_DEFAULT_DB_PATH = Path(".cache/checkpoints.db")


class ResearchOrchestrator:
    """基于 LangGraph 的多 Agent 研究编排器."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        max_iterations: int = 3,
        db_path: str | Path | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_iterations = max_iterations
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH

        # 初始化各 Agent
        kwargs = {"provider": provider, "model": model}
        self._planner     = PlannerAgent(**kwargs)
        self._data        = DataAgent(**kwargs)
        self._fundamental = FundamentalAgent(**kwargs)
        self._technical   = TechnicalAgent(**kwargs)
        self._sentiment   = SentimentAgent(**kwargs)
        self._critic      = CriticAgent(**kwargs)
        self._report      = ReportAgent(**kwargs)

        self._graph = self._build_graph()

    # ── 节点函数（dict ↔ AgentState 互转）────────────────────────────────────

    def _to_state(self, d: dict) -> AgentState:
        if "_agent_state" in d:
            return d["_agent_state"]
        return AgentState(**{k: v for k, v in d.items() if k != "_agent_state"})

    def _from_state(self, state: AgentState) -> dict:
        return {"_agent_state": state}

    def _node_planner(self, d: dict) -> dict:
        state = self._to_state(d)
        state = self._planner.run(state)
        return self._from_state(state)

    def _node_data(self, d: dict) -> dict:
        state = self._to_state(d)
        state = self._data.run(state)
        return self._from_state(state)

    def _node_fundamental(self, d: dict) -> dict:
        state = self._to_state(d)
        state = self._fundamental.run(state)
        return self._from_state(state)

    def _node_technical(self, d: dict) -> dict:
        state = self._to_state(d)
        state = self._technical.run(state)
        return self._from_state(state)

    def _node_sentiment(self, d: dict) -> dict:
        state = self._to_state(d)
        state = self._sentiment.run(state)
        return self._from_state(state)

    def _node_critic(self, d: dict) -> dict:
        state = self._to_state(d)
        state.iteration_count += 1
        state = self._critic.run(state)
        return self._from_state(state)

    def _node_report(self, d: dict) -> dict:
        state = self._to_state(d)
        state = self._report.run(state)
        return self._from_state(state)

    # ── 路由逻辑 ─────────────────────────────────────────────────────────────

    def _route_after_critic(self, d: dict) -> str:
        state = self._to_state(d)
        feedback = state.critic_feedback

        # 强制退出条件
        if state.iteration_count >= self.max_iterations:
            logger.warning(
                f"[Orchestrator] max_iterations={self.max_iterations} reached, "
                "forcing exit"
            )
            state.should_terminate = True
            state.terminate_reason = f"达到最大迭代次数 {self.max_iterations}"
            d["_agent_state"] = state
            return END

        if feedback is None or feedback.passed:
            return "report"

        # Compaction：本轮 issue 数必须少于上轮，否则强制退出
        prev_count = getattr(state, "_prev_issue_count", None)
        current_count = len(feedback.issues)
        if prev_count is not None and current_count >= prev_count:
            logger.warning(
                f"[Orchestrator] issue count not decreasing "
                f"({prev_count} → {current_count}), forcing exit"
            )
            state.should_terminate = True
            state.terminate_reason = "issue 数量未减少，强制退出"
            d["_agent_state"] = state
            return END
        state._prev_issue_count = current_count  # type: ignore[attr-defined]

        # 根据 issue 类型决定回流目标
        return self._pick_retry_target(feedback)

    def _pick_retry_target(self, feedback) -> str:
        """根据 issue 优先级选择回流 Agent."""
        # 优先修复 critical issue
        for issue in feedback.issues:
            if issue.severity.value == "critical" and issue.fix_action:
                agent = issue.fix_action.agent
                if agent == AgentType.DATA:
                    return "data"
                elif agent == AgentType.FUNDAMENTAL:
                    return "fundamental"
                elif agent == AgentType.TECHNICAL:
                    return "technical"
                elif agent == AgentType.SENTIMENT:
                    return "sentiment"

        # 没有 fix_action，回流到最常见问题的 Agent
        type_counts: dict[str, int] = {}
        for issue in feedback.issues:
            loc = ""
            if issue.fix_action:
                loc = issue.fix_action.agent.value
            type_counts[loc] = type_counts.get(loc, 0) + 1

        target = max(type_counts, key=lambda k: type_counts[k], default="fundamental")
        return target if target in ("fundamental", "technical", "sentiment", "data") else "fundamental"

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def run(
        self,
        query: InvestmentQuery,
        max_iterations: int | None = None,
    ) -> AgentState:
        """执行完整研究流程（同步，无断点续传）.

        Args:
            query: 投资查询
            max_iterations: 覆盖默认最大迭代次数

        Returns:
            最终 AgentState（含 report）
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations

        state = AgentState(
            query=query,
            max_iterations=self.max_iterations,
        )

        init_dict = {"_agent_state": state}
        result_dict = self._graph.invoke(init_dict)
        final_state = result_dict.get("_agent_state", state)

        return final_state

    def stream(
        self,
        query: InvestmentQuery,
        max_iterations: int | None = None,
    ) -> Iterator[tuple[str, AgentState]]:
        """同步流式执行，每个节点完成后立即 yield (node_name, AgentState).

        Args:
            query: 投资查询
            max_iterations: 覆盖默认最大迭代次数

        Yields:
            (node_name, AgentState) — 每个节点完成时 yield 一次
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations

        state = AgentState(query=query, max_iterations=self.max_iterations)
        init_dict = {"_agent_state": state}

        for chunk in self._graph.stream(init_dict, stream_mode="updates"):
            # chunk 格式：{node_name: {结果 dict}}
            for node_name, node_output in chunk.items():
                agent_state = node_output.get("_agent_state", state)
                logger.debug(f"[Orchestrator] ✓ {node_name}")
                yield node_name, agent_state
                state = agent_state  # 更新当前最新状态

    async def astream(
        self,
        query: InvestmentQuery,
        max_iterations: int | None = None,
    ) -> AsyncIterator[tuple[str, AgentState]]:
        """异步流式执行，每个节点完成后立即 yield (node_name, AgentState).

        用法::

            async for node_name, state in orch.astream(query):
                print(f"{node_name}: iteration={state.iteration_count}")

        Args:
            query: 投资查询
            max_iterations: 覆盖默认最大迭代次数

        Yields:
            (node_name, AgentState) — 每个节点完成时 yield 一次
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations

        state = AgentState(query=query, max_iterations=self.max_iterations)
        init_dict = {"_agent_state": state}

        async for chunk in self._graph.astream(init_dict, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                agent_state = node_output.get("_agent_state", state)
                logger.debug(f"[Orchestrator] ✓ {node_name} (async)")
                yield node_name, agent_state
                state = agent_state

    def run_with_checkpoint(
        self,
        query: InvestmentQuery,
        thread_id: str,
        max_iterations: int | None = None,
    ) -> AgentState:
        """带断点续传的研究流程（SqliteSaver）.

        相同 thread_id 的调用可从上次中断位置续跑。

        Args:
            query: 投资查询
            thread_id: 会话 ID（相同 ID 可断点续传）
            max_iterations: 覆盖默认最大迭代次数

        Returns:
            最终 AgentState（含 report）
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations

        # 确保数据库目录存在
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        if not _SQLITE_AVAILABLE:
            logger.warning("[Orchestrator] SqliteSaver 不可用，降级为 MemorySaver（无持久化）")
            checkpointer = MemorySaver()
            graph = self._build_graph(checkpointer=checkpointer)
            state = AgentState(query=query, max_iterations=self.max_iterations)
            init_dict = {"_agent_state": state}
            config = {"configurable": {"thread_id": thread_id}}
            result_dict = graph.invoke(init_dict, config=config)
            return result_dict.get("_agent_state", state)

        with SqliteSaver.from_conn_string(str(self._db_path)) as checkpointer:
            graph = self._build_graph(checkpointer=checkpointer)
            state = AgentState(query=query, max_iterations=self.max_iterations)
            init_dict = {"_agent_state": state}
            config = {"configurable": {"thread_id": thread_id}}

            logger.info(
                f"[Orchestrator] run_with_checkpoint: thread_id={thread_id}, "
                f"db={self._db_path}"
            )
            result_dict = graph.invoke(init_dict, config=config)
            final_state = result_dict.get("_agent_state", state)

            logger.info(
                f"[Orchestrator] checkpoint saved: thread_id={thread_id}, "
                f"iteration={final_state.iteration_count}"
            )
            return final_state

    def _build_graph(self, checkpointer=None):
        """构建（并可选编译带 checkpointer 的）图."""
        graph = StateGraph(dict)

        graph.add_node("planner",     self._node_planner)
        graph.add_node("data",        self._node_data)
        graph.add_node("fundamental", self._node_fundamental)
        graph.add_node("technical",   self._node_technical)
        graph.add_node("sentiment",   self._node_sentiment)
        graph.add_node("critic",      self._node_critic)
        graph.add_node("report",      self._node_report)

        graph.set_entry_point("planner")
        graph.add_edge("planner",     "data")
        graph.add_edge("data",        "fundamental")
        graph.add_edge("fundamental", "technical")
        graph.add_edge("technical",   "sentiment")
        graph.add_edge("sentiment",   "critic")

        graph.add_conditional_edges(
            "critic",
            self._route_after_critic,
            {
                "report":      "report",
                "fundamental": "fundamental",
                "technical":   "technical",
                "sentiment":   "sentiment",
                "data":        "data",
                END:           END,
            },
        )
        graph.add_edge("report", END)

        return graph.compile(checkpointer=checkpointer)


def run_research(
    ticker: str,
    as_of: date | None = None,
    query_text: str | None = None,
    provider: str = "ollama",
    model: str = "qwen2.5:7b",
    max_iterations: int = 3,
) -> AgentState:
    """便捷入口函数.

    Args:
        ticker:    股票代码
        as_of:     数据截止日期（None = 今天）
        query_text: 自定义查询文本（None = 自动生成）
        provider:  LLM 提供商
        model:     模型名称
        max_iterations: 最大迭代次数

    Returns:
        AgentState
    """
    as_of = as_of or date.today()
    query_text = query_text or f"请对 {ticker} 进行全面的投资分析"

    query = InvestmentQuery(
        raw_query=query_text,
        tickers=[ticker],
        as_of=as_of,
        intent=QueryIntent.SINGLE_STOCK,
        horizon_days=60,
    )

    orch = ResearchOrchestrator(
        provider=provider,
        model=model,
        max_iterations=max_iterations,
    )
    return orch.run(query)
