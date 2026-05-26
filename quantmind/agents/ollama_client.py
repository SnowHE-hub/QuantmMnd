"""quantmind/agents/ollama_client.py — Ollama 本地 LLM ReAct 客户端.

封装 Ollama qwen2.5:7b 的 ReAct（Reasoning + Acting）循环。
支持 OpenAI 格式的 tool_call，最多 max_steps 轮思考-行动。

设计原则
--------
1. 失败不崩溃：任何异常 → 返回 fallback_response（fallback=True）
2. 超时保护：单次 LLM 调用超时 timeout 秒，整个 Agent 超时 timeout×2 秒
3. 完整日志：每一步的思考/行动/结果都记录到 reasoning_trace
4. 温度 0.1：金融分析要确定性，不要创意
5. 工具格式：Ollama /api/chat 原生 tool_calls（qwen2.5:7b 支持）
             若模型未返回 tool_calls，则解析文本中的 ReAct 指令作为 fallback
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

import requests

log = logging.getLogger(__name__)

# ── 解析正则 ─────────────────────────────────────────────────────────────────
_SIGNAL_PATTERNS = [
    r"SIGNAL\s*[:：]\s*([+-]?\d+(?:\.\d+)?)",
    r"信号\s*[:：=]\s*([+-]?\d+(?:\.\d+)?)",
    r"signal\s*[=:]\s*([+-]?\d+(?:\.\d+)?)",
]
_CONFIDENCE_PATTERNS = [
    r"CONFIDENCE\s*[:：]\s*([+-]?\d+(?:\.\d+)?)",
    r"置信度\s*[:：=]\s*([+-]?\d+(?:\.\d+)?)",
    r"confidence\s*[=:]\s*([+-]?\d+(?:\.\d+)?)",
]
# ReAct 文本格式（当模型不支持原生工具调用时的 fallback）
_TEXT_ACTION_PATTERN = re.compile(
    r"Action\s*[:：]\s*(\w+)\s*\nAction\s*Input\s*[:：]\s*(\{.+?\})",
    re.DOTALL | re.IGNORECASE,
)


class OllamaReActClient:
    """Ollama ReAct 循环客户端（Reasoning + Acting）.

    Parameters
    ----------
    model : str
        Ollama 模型名，默认 'qwen2.5:7b'。
    timeout : int
        单次 HTTP 调用超时秒数，默认 60。
    max_steps : int
        最大思考-行动轮数，超过则强制返回当前最佳答案，默认 5。
    base_url : str
        Ollama 服务地址，默认 'http://localhost:11434'。

    Example
    -------
    >>> client = OllamaReActClient(model="qwen2.5:7b", timeout=60)
    >>> result = client.chat_with_tools(
    ...     system_prompt="你是 A 股分析师...",
    ...     user_message="分析 000858.SZ 的估值",
    ...     tools=[...],
    ...     tool_executors={"get_peers": lambda ticker, **kw: {...}},
    ... )
    >>> print(result["signal"], result["confidence"])
    """

    BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        timeout: int = 60,
        max_steps: int = 5,
        base_url: str = "",
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_steps = max_steps
        self.base_url = (base_url or self.BASE_URL).rstrip("/")

    # ─── 主入口 ────────────────────────────────────────────────────────────────

    def chat_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
        tool_executors: dict[str, Callable[..., Any]] | None = None,
        context: str = "",
    ) -> dict[str, Any]:
        """ReAct 循环：思考 → 行动 → 观察 → 最终回答.

        Parameters
        ----------
        system_prompt : str
            系统提示词（角色+任务+输出格式）。
        user_message : str
            用户消息（注入股票财务数据）。
        tools : list[dict]
            OpenAI/Ollama 格式的工具定义列表。
            支持两种格式：
              - {"name": ..., "description": ..., "parameters": ...}
              - {"type": "function", "function": {...}}
        tool_executors : dict | None
            {tool_name: callable} 映射，callable(**args) → Any。
        context : str
            额外上下文（RAG 检索到的研报内容），追加到用户消息末尾。

        Returns
        -------
        dict with keys:
            final_answer   : str    — LLM 最终回答文本
            signal         : float  — 提取的投资信号 [-1, +1]
            confidence     : float  — 提取的置信度 [0, 1]
            reasoning_trace: list   — 完整推理链（每步的思考+行动+结果）
            tools_called   : list   — 调用过的工具名列表
            fallback       : bool   — True 表示降级（LLM 失败，应使用规则）
        """
        executors = tool_executors or {}
        reasoning_trace: list[dict] = []
        tools_called: list[str] = []

        # 统一工具格式为 Ollama 格式
        ollama_tools = self._normalize_tools(tools)

        # 附加 RAG 上下文
        full_user_msg = user_message
        if context:
            full_user_msg += f"\n\n【参考研报摘要】\n{context[:2000]}"

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": full_user_msg},
        ]

        t0 = time.time()
        total_timeout = self.timeout * 2

        for step in range(self.max_steps):
            # 整体超时保护
            if time.time() - t0 > total_timeout:
                log.warning(
                    "[OllamaReAct] 超时 %.0fs（step=%d），强制返回最佳答案",
                    total_timeout, step,
                )
                break

            # ── LLM 调用 ─────────────────────────────────────────────────────
            try:
                resp = self._call_ollama(messages, ollama_tools)
            except Exception as exc:
                log.warning("[OllamaReAct] LLM 调用失败 step=%d: %s", step, exc)
                reasoning_trace.append({"step": step, "error": str(exc)})
                return self._fallback_response(reasoning_trace, tools_called)

            msg      = resp.get("message", {})
            content  = msg.get("content", "") or ""
            tc_list  = msg.get("tool_calls") or []

            step_trace: dict[str, Any] = {
                "step":        step,
                "content":     content[:500] if content else "(工具调用)",
                "tool_calls":  [_tc_name(tc) for tc in tc_list],
            }
            reasoning_trace.append(step_trace)

            # ── 原生工具调用处理 ──────────────────────────────────────────────
            if tc_list:
                messages.append({
                    "role":       "assistant",
                    "content":    content,
                    "tool_calls": tc_list,
                })
                for tc in tc_list:
                    tool_name, args = _parse_tool_call(tc)
                    tools_called.append(tool_name)
                    result_str = self._execute_tool(tool_name, args, executors)
                    step_trace["tool_result"] = result_str[:400]
                    log.debug("[OllamaReAct] %s(%s) → %s", tool_name, args, result_str[:100])
                    messages.append({"role": "tool", "content": result_str})
                continue  # 继续推理

            # ── 文本 ReAct 格式 fallback 解析 ─────────────────────────────────
            if content:
                text_tc = _TEXT_ACTION_PATTERN.search(content)
                if text_tc:
                    tool_name = text_tc.group(1).strip()
                    try:
                        args = json.loads(text_tc.group(2))
                    except Exception:
                        args = {}
                    tools_called.append(tool_name)
                    result_str = self._execute_tool(tool_name, args, executors)
                    step_trace["tool_result"] = result_str[:400]
                    step_trace["text_react"]  = True
                    messages.append({
                        "role":    "assistant",
                        "content": content,
                    })
                    messages.append({
                        "role":    "user",
                        "content": f"Observation: {result_str}",
                    })
                    continue  # 继续推理

                # ── 最终回答 ─────────────────────────────────────────────────
                return {
                    "final_answer":    content,
                    "signal":          self._parse_signal_from_text(content),
                    "confidence":      self._parse_confidence_from_text(content),
                    "reasoning_trace": reasoning_trace,
                    "tools_called":    tools_called,
                    "fallback":        False,
                }

        # ── 超出 max_steps：提取最后有内容的 assistant 消息 ──────────────────
        last_content = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                last_content = m["content"]
                break

        if last_content:
            return {
                "final_answer":    last_content,
                "signal":          self._parse_signal_from_text(last_content),
                "confidence":      self._parse_confidence_from_text(last_content),
                "reasoning_trace": reasoning_trace,
                "tools_called":    tools_called,
                "fallback":        False,
            }

        return self._fallback_response(reasoning_trace, tools_called)

    # ─── LLM HTTP 调用 ────────────────────────────────────────────────────────

    def _call_ollama(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """POST 到 Ollama /api/chat 端点."""
        payload: dict[str, Any] = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature":  0.1,
                "num_predict":  1024,
                "repeat_penalty": 1.1,
            },
        }
        if tools:
            payload["tools"] = tools

        r = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    # ─── 工具执行 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _execute_tool(
        tool_name: str,
        args: dict[str, Any],
        executors: dict[str, Callable],
    ) -> str:
        """执行工具函数，返回 JSON 字符串."""
        if tool_name not in executors:
            return json.dumps({"error": f"工具 {tool_name!r} 未注册"}, ensure_ascii=False)
        try:
            result = executors[tool_name](**args)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            log.warning("[OllamaReAct] 工具 %s 执行失败: %s", tool_name, exc)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # ─── 文本解析 ─────────────────────────────────────────────────────────────

    def _parse_signal_from_text(self, text: str) -> float:
        """从 LLM 输出文本中提取 signal 数值，支持多种格式.

        格式支持：
          SIGNAL: 0.75  |  SIGNAL: -0.3  |  信号: 0.5  |  signal=0.8
        """
        for pattern in _SIGNAL_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return max(-1.0, min(1.0, float(m.group(1))))
                except ValueError:
                    continue
        return 0.0

    def _parse_confidence_from_text(self, text: str) -> float:
        """从 LLM 输出文本中提取 confidence，有界到 [0, 1].

        自动处理百分比格式（0.8 或 80 → 0.8）。
        """
        for pattern in _CONFIDENCE_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    val = float(m.group(1))
                    if val >= 10.0:        # 百分比格式（如 80 → 0.80），1.5 之类的直接 clamp
                        val /= 100.0
                    return max(0.0, min(1.0, val))
                except ValueError:
                    continue
        return 0.5

    # ─── 可用性检查 ───────────────────────────────────────────────────────────

    @classmethod
    def is_available(cls, base_url: str = "", model: str = "qwen2.5:7b") -> bool:
        """快速检查 Ollama 服务 + 指定模型是否可用（超时 2s）."""
        url = (base_url or cls.BASE_URL).rstrip("/")
        try:
            r = requests.get(f"{url}/api/tags", timeout=2)
            if r.status_code != 200:
                return False
            models = [m.get("name", "") for m in r.json().get("models", [])]
            # 支持 "qwen2.5:7b" 和 "qwen2.5:7b-instruct" 等变体
            return any(model.split(":")[0] in m for m in models)
        except Exception:
            return False

    # ─── Fallback ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_response(
        reasoning_trace: list[dict],
        tools_called: list[str],
    ) -> dict[str, Any]:
        """LLM 完全失败时的降级响应."""
        return {
            "final_answer":    "LLM 分析失败，建议降级到规则模式",
            "signal":          0.0,
            "confidence":      0.3,
            "reasoning_trace": reasoning_trace,
            "tools_called":    tools_called,
            "fallback":        True,
        }

    @staticmethod
    def _normalize_tools(tools: list[dict]) -> list[dict]:
        """统一工具格式为 Ollama 原生格式（{type, function: {...}}）."""
        normalized = []
        for t in tools:
            if "type" in t and "function" in t:
                normalized.append(t)                     # 已是标准格式
            elif "name" in t:
                normalized.append({"type": "function", "function": t})
            else:
                log.debug("[OllamaReAct] 跳过无效工具定义: %s", t)
        return normalized


# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def _tc_name(tc: dict) -> str:
    """从 tool_call 字典中提取工具名."""
    return tc.get("function", {}).get("name", "unknown")


def _parse_tool_call(tc: dict) -> tuple[str, dict[str, Any]]:
    """解析 tool_call 字典，返回 (name, args)."""
    fn   = tc.get("function", {})
    name = fn.get("name", "")
    raw  = fn.get("arguments", {})
    if isinstance(raw, str):
        try:
            args = json.loads(raw)
        except Exception:
            args = {}
    else:
        args = raw or {}
    return name, args


__all__ = ["OllamaReActClient"]
