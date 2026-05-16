"""quantmind.agents.planner — PlannerAgent.

把用户 query 拆解为 DAG 任务列表，输出 TaskPlan。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from loguru import logger

from quantmind.agents.base import BaseAgent
from quantmind.core.state import (
    AgentState,
    AgentType,
    InvestmentQuery,
    TaskNode,
    TaskPlan,
    TaskStatus,
)

__all__ = ["PlannerAgent"]

_SYSTEM_PROMPT = """你是一位专业的投资研究任务规划师。
用户会给你一个投资分析需求，你的任务是将其拆解为一系列可执行的子任务（DAG）。

规则：
1. 任务数量：5-15 个（视复杂度而定）
2. 必须先有数据收集任务（agent_type="data"），才能有分析任务
3. 最后必须有 critic 任务审核，然后是 report 任务生成报告
4. 任务之间通过 depends_on 声明依赖关系
5. priority 范围 1-5（5最高），数据任务优先级高

输出严格 JSON 格式：
{
  "rationale": "任务规划思路（50字内）",
  "tasks": [
    {
      "task_id": "唯一ID（如 t1）",
      "agent_type": "data|fundamental|technical|sentiment|quant|critic|report",
      "action": "操作描述（如 fetch_financials）",
      "params": {"key": "value"},
      "depends_on": ["依赖的 task_id"],
      "priority": 1-5
    }
  ]
}"""


class PlannerAgent(BaseAgent):
    agent_type = AgentType.PLANNER
    description = "将用户投资查询拆解为 DAG 任务列表"
    system_prompt = _SYSTEM_PROMPT
    max_iterations = 1

    def format_input(self, state: AgentState) -> str:
        q = state.query
        tickers_str = "、".join(q.tickers) if q.tickers else "未指定"
        as_of_str = str(q.as_of) if q.as_of else "最新"
        return (
            f"投资分析需求：{q.raw_query}\n"
            f"目标股票：{tickers_str}\n"
            f"数据截止日期（as_of）：{as_of_str}\n"
            f"分析期限：{q.horizon_days}天\n"
            f"意图分类：{q.intent}\n\n"
            f"请生成任务 DAG（JSON 格式）。"
        )

    def parse_output(self, llm_response: str) -> dict[str, Any]:
        # 提取 JSON 块
        text = llm_response.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"rationale": "", "tasks": []}
        try:
            data = json.loads(m.group())
            return data
        except json.JSONDecodeError:
            logger.warning("[PlannerAgent] JSON parse failed")
            return {"rationale": "", "tasks": []}

    def _execute(self, state: AgentState) -> AgentState:
        prompt = self.format_input(state)
        resp = self.llm_chat(prompt, max_tokens=1024)
        parsed = self.parse_output(resp.content)

        tasks_raw = parsed.get("tasks", [])
        rationale = parsed.get("rationale", "")

        task_nodes: list[TaskNode] = []
        seen_ids: set[str] = set()

        for t in tasks_raw:
            task_id = str(t.get("task_id", f"t{len(task_nodes)+1}"))
            if task_id in seen_ids:
                task_id = f"{task_id}_{uuid.uuid4().hex[:4]}"
            seen_ids.add(task_id)

            agent_str = t.get("agent_type", "data").lower()
            try:
                agent_type = AgentType(agent_str)
            except ValueError:
                agent_type = AgentType.DATA

            task_nodes.append(TaskNode(
                task_id=task_id,
                agent_type=agent_type,
                action=t.get("action", ""),
                params=t.get("params", {}),
                depends_on=[str(d) for d in t.get("depends_on", [])],
                priority=int(t.get("priority", 3)),
                status=TaskStatus.PENDING,
            ))

        # 若 LLM 未生成有效 DAG，使用默认模板
        if not task_nodes:
            task_nodes = self._default_plan(state.query)
            rationale = "使用默认任务模板（LLM 解析失败）"

        state.plan = TaskPlan(tasks=task_nodes, rationale=rationale)
        logger.info(f"[PlannerAgent] plan: {len(task_nodes)} tasks, {rationale[:50]}")
        return state

    def _default_plan(self, query: InvestmentQuery) -> list[TaskNode]:
        tickers = query.tickers or ["unknown"]
        ticker = tickers[0]
        return [
            TaskNode(task_id="t1", agent_type=AgentType.DATA,
                     action="fetch_basics", params={"ticker": ticker}, priority=5),
            TaskNode(task_id="t2", agent_type=AgentType.DATA,
                     action="fetch_financials", params={"ticker": ticker},
                     depends_on=["t1"], priority=5),
            TaskNode(task_id="t3", agent_type=AgentType.DATA,
                     action="fetch_prices", params={"ticker": ticker},
                     depends_on=["t1"], priority=4),
            TaskNode(task_id="t4", agent_type=AgentType.FUNDAMENTAL,
                     action="fundamental_analysis", params={"ticker": ticker},
                     depends_on=["t2"], priority=3),
            TaskNode(task_id="t5", agent_type=AgentType.TECHNICAL,
                     action="technical_analysis", params={"ticker": ticker},
                     depends_on=["t3"], priority=3),
            TaskNode(task_id="t6", agent_type=AgentType.SENTIMENT,
                     action="sentiment_analysis", params={"ticker": ticker},
                     depends_on=["t1"], priority=2),
            TaskNode(task_id="t7", agent_type=AgentType.CRITIC,
                     action="critic_review", params={},
                     depends_on=["t4", "t5", "t6"], priority=2),
            TaskNode(task_id="t8", agent_type=AgentType.REPORT,
                     action="generate_report", params={},
                     depends_on=["t7"], priority=1),
        ]
