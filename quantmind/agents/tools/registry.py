"""quantmind.agents.tools.registry — 工具注册表.

统一管理所有工具，支持按 agent 类型过滤，生成 LangChain 兼容描述。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

from quantmind.core.state import AgentType

__all__ = [
    "ToolSpec",
    "ToolRegistry",
    "register_tool",
    "get_tool",
    "get_tools_for_agent",
    "list_tools",
    "REGISTRY",
]


@dataclass
class ToolSpec:
    """工具元信息."""

    name: str
    fn: Callable
    description: str
    agents: list[AgentType] = field(default_factory=list)  # 空列表 = 所有 agent 可用

    def to_langchain_dict(self) -> dict[str, Any]:
        """生成 LangChain 兼容的工具描述（用于 function calling）."""
        sig = inspect.signature(self.fn)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            ptype = "string"
            ann = param.annotation
            if ann is not inspect.Parameter.empty:
                if ann in (int,):
                    ptype = "integer"
                elif ann in (float,):
                    ptype = "number"
                elif ann in (bool,):
                    ptype = "boolean"
            properties[param_name] = {"type": ptype, "description": param_name}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """全局工具注册表."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        fn: Callable,
        name: str | None = None,
        description: str | None = None,
        agents: list[AgentType] | None = None,
    ) -> "ToolRegistry":
        tool_name = name or fn.__name__
        doc = description or (inspect.getdoc(fn) or "").split("\n")[0]
        spec = ToolSpec(
            name=tool_name,
            fn=fn,
            description=doc,
            agents=agents or [],
        )
        self._tools[tool_name] = spec
        logger.debug(f"[ToolRegistry] registered: {tool_name}")
        return self

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def for_agent(self, agent_type: AgentType) -> list[ToolSpec]:
        """返回 agent 可用的工具列表（空 agents 列表 = 通用工具）."""
        return [
            spec for spec in self._tools.values()
            if not spec.agents or agent_type in spec.agents
        ]

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def all_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())


# ── 全局单例 ─────────────────────────────────────────────────────────────────

REGISTRY = ToolRegistry()


def _register_defaults() -> None:
    from quantmind.agents.tools import data_tools, analysis_tools, kb_tools, quant_tools

    # data_tools — DataAgent + FundamentalAgent + TechnicalAgent + SentimentAgent
    data_agents = [AgentType.DATA, AgentType.FUNDAMENTAL, AgentType.TECHNICAL,
                   AgentType.SENTIMENT]
    REGISTRY.register(data_tools.fetch_stock_basics, agents=data_agents)
    REGISTRY.register(data_tools.fetch_financials_pit, agents=data_agents)
    REGISTRY.register(data_tools.fetch_price_history, agents=[AgentType.DATA, AgentType.TECHNICAL])
    REGISTRY.register(data_tools.fetch_industry_peers, agents=[AgentType.DATA, AgentType.FUNDAMENTAL])
    REGISTRY.register(data_tools.fetch_recent_news, agents=[AgentType.DATA, AgentType.SENTIMENT])

    # analysis_tools — FundamentalAgent + TechnicalAgent
    REGISTRY.register(analysis_tools.compute_financial_ratios, agents=[AgentType.FUNDAMENTAL])
    REGISTRY.register(analysis_tools.compute_dcf_valuation, agents=[AgentType.FUNDAMENTAL])
    REGISTRY.register(analysis_tools.compute_comparable_multiples, agents=[AgentType.FUNDAMENTAL])
    REGISTRY.register(analysis_tools.compute_technical_indicators, agents=[AgentType.TECHNICAL])
    REGISTRY.register(analysis_tools.run_factor_screening, agents=[])  # 通用

    # kb_tools — SentimentAgent
    REGISTRY.register(kb_tools.search_research_reports, agents=[AgentType.SENTIMENT])
    REGISTRY.register(kb_tools.search_news, agents=[AgentType.SENTIMENT])
    REGISTRY.register(kb_tools.search_company_filings, agents=[AgentType.SENTIMENT])

    # quant_tools — TechnicalAgent + FundamentalAgent
    REGISTRY.register(quant_tools.get_factor_signal, agents=[AgentType.TECHNICAL, AgentType.QUANT])
    REGISTRY.register(quant_tools.get_llm_rerank_thesis, agents=[AgentType.TECHNICAL, AgentType.QUANT])
    REGISTRY.register(quant_tools.get_backtest_performance, agents=[AgentType.QUANT])


_register_defaults()


# ── 便捷函数 ─────────────────────────────────────────────────────────────────

def register_tool(
    fn: Callable,
    name: str | None = None,
    description: str | None = None,
    agents: list[AgentType] | None = None,
) -> None:
    REGISTRY.register(fn, name=name, description=description, agents=agents)


def get_tool(name: str) -> ToolSpec | None:
    return REGISTRY.get(name)


def get_tools_for_agent(agent_type: AgentType) -> list[ToolSpec]:
    return REGISTRY.for_agent(agent_type)


def list_tools() -> list[str]:
    return REGISTRY.list_names()
