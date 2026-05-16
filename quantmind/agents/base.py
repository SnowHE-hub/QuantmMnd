"""quantmind.agents.base — BaseAgent 抽象基类.

所有专业 Agent 继承此类，统一执行流程、记录 trace、处理错误。
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from loguru import logger

from quantmind.core.llm_router import LLMRouter, LLMResponse, get_router
from quantmind.core.state import (
    AgentState,
    AgentType,
    HistoryEntry,
)

__all__ = ["BaseAgent", "ToolCall", "ToolResult"]


class ToolCall:
    """工具调用记录."""

    def __init__(self, name: str, params: dict[str, Any]) -> None:
        self.name = name
        self.params = params
        self.result: Any = None
        self.error: str | None = None
        self.elapsed_s: float = 0.0

    def succeeded(self) -> bool:
        return self.error is None


class ToolResult:
    def __init__(self, data: Any, error: str | None = None) -> None:
        self.data = data
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None


class BaseAgent(ABC):
    """所有专业 Agent 的抽象基类.

    子类必须实现：
        - agent_type: AgentType  (类属性)
        - format_input(state)    → str         prompt 文本
        - parse_output(llm_text) → dict        结构化结果
        - _execute(state)        → AgentState  核心逻辑

    run() 方法统一处理：执行 trace 记录、错误处理、state 更新。
    """

    # 子类设置
    agent_type: AgentType = AgentType.DATA
    description: str = ""
    system_prompt: str = ""
    max_iterations: int = 3

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        router: LLMRouter | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._router = router
        self._tool_calls: list[ToolCall] = []

    @property
    def router(self) -> LLMRouter:
        if self._router is None:
            self._router = get_router()
        return self._router

    # ── 子类必须实现 ─────────────────────────────────────────────────────────

    @abstractmethod
    def format_input(self, state: AgentState) -> str:
        """把 AgentState 格式化为 LLM prompt 文本."""

    @abstractmethod
    def parse_output(self, llm_response: str) -> dict[str, Any]:
        """把 LLM 文本解析为结构化 dict."""

    @abstractmethod
    def _execute(self, state: AgentState) -> AgentState:
        """Agent 核心逻辑：取数 + LLM 推理 + 写回 state."""

    # ── 可覆盖的钩子 ─────────────────────────────────────────────────────────

    def pre_check(self, state: AgentState) -> bool:
        """前置检查，返回 False 则跳过执行."""
        return True

    def post_validate(self, output: dict[str, Any]) -> bool:
        """验证 parse_output 的结果，返回 False 则视为失败."""
        return True

    # ── LLM 调用封装 ─────────────────────────────────────────────────────────

    def llm_chat(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """统一 LLM 调用入口，自动带 system_prompt."""
        sys = system or self.system_prompt
        messages: list[dict[str, str]] = []
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": prompt})
        return self.router.chat(
            messages=messages,
            provider=self.provider,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            fallback=None,
        )

    # ── 工具调用封装 ─────────────────────────────────────────────────────────

    def call_tool(
        self,
        tool_name: str,
        tool_fn,
        *args: Any,
        **kwargs: Any,
    ) -> ToolResult:
        """带错误处理和计时的工具调用."""
        tc = ToolCall(name=tool_name, params={"args": args, "kwargs": kwargs})
        self._tool_calls.append(tc)
        t0 = time.monotonic()
        try:
            result = tool_fn(*args, **kwargs)
            tc.result = result
            tc.elapsed_s = time.monotonic() - t0
            return ToolResult(data=result)
        except Exception as e:  # noqa: BLE001
            tc.error = str(e)
            tc.elapsed_s = time.monotonic() - t0
            logger.warning(f"[{self.agent_type}] tool {tool_name} failed: {e}")
            return ToolResult(data=None, error=str(e))

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def run(self, state: AgentState) -> AgentState:
        """执行 Agent 并记录完整 trace.

        流程：
        1. pre_check → 失败则跳过
        2. _execute（子类逻辑）
        3. 记录 HistoryEntry 到 state
        4. 更新 token 计数
        """
        self._tool_calls = []
        started_at = datetime.now()
        input_summary = self._summarize_input(state)

        if not self.pre_check(state):
            logger.info(f"[{self.agent_type}] pre_check failed, skipping")
            return state

        success = True
        error_msg: str | None = None
        output_summary = ""

        try:
            state = self._execute(state)
            output_summary = self._summarize_output(state)
        except Exception as e:  # noqa: BLE001
            success = False
            error_msg = str(e)
            logger.error(f"[{self.agent_type}] execution failed: {e}")
            warnings.warn(f"Agent {self.agent_type} failed: {e}", stacklevel=2)

        finished_at = datetime.now()

        # 统计本次 token 消耗（从最后一次 LLM 调用读取）
        token_usage = 0
        cost_cny = 0.0
        try:
            summary = self.router._tracker.summary()  # type: ignore[attr-defined]
            # 简单估算：用全局累计减去之前的——实际场景可用更精细的 per-call tracking
        except Exception:  # noqa: BLE001
            pass

        entry = HistoryEntry(
            agent=self.agent_type,
            started_at=started_at,
            finished_at=finished_at,
            input_summary=input_summary[:200],
            output_summary=output_summary[:200],
            token_usage=token_usage,
            cost_cny=cost_cny,
            success=success,
            error=error_msg,
        )
        state.add_history(entry)

        tool_names = [tc.name for tc in self._tool_calls]
        if tool_names:
            logger.debug(f"[{self.agent_type}] tools called: {tool_names}")

        return state

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _summarize_input(self, state: AgentState) -> str:
        tickers = state.query.tickers
        as_of = state.query.as_of
        return f"tickers={tickers}, as_of={as_of}, iter={state.iteration_count}"

    def _summarize_output(self, state: AgentState) -> str:
        parts = []
        if state.fundamentals:
            parts.append(f"fundamentals={list(state.fundamentals)}")
        if state.technicals:
            parts.append(f"technicals={list(state.technicals)}")
        if state.sentiments:
            parts.append(f"sentiments={list(state.sentiments)}")
        if state.critic_feedback:
            parts.append(f"critic_passed={state.critic_feedback.passed}")
        return "; ".join(parts) or "state updated"
