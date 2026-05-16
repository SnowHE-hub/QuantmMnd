"""quantmind.agents.llm_client — 轻量级 LLM 客户端适配层.

支持 provider:
  - none       : 不调用外部 LLM，返回 None（调用方用模板生成）
  - dashscope  : 阿里云 DashScope（通义千问）
  - openai     : OpenAI 兼容接口
  - anthropic  : Anthropic Claude
  - deepseek   : DeepSeek（openai 兼容接口）

环境变量（不打印）：
  DASHSCOPE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY

API key 不存在时自动 fallback 到 provider=none，不崩溃。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

__all__ = ["LLMClient", "LLMClientResponse", "build_client"]

_ENV_KEYS: dict[str, str] = {
    "dashscope": "DASHSCOPE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class LLMClientResponse:
    content: str
    provider: str
    model: str
    fallback_used: bool = False


class LLMClient:
    """极简 LLM 客户端，支持多 provider，key 缺失时自动 fallback none.

    Args:
        provider: 'none' | 'dashscope' | 'openai' | 'anthropic' | 'deepseek'
        model:    模型名称（provider=none 时忽略）
    """

    def __init__(
        self,
        provider: str = "none",
        model: str | None = None,
    ) -> None:
        self.requested_provider = provider
        self.model = model or self._default_model(provider)
        self.provider, self.fallback_used = self._resolve_provider(provider)

    # ── 公开 API ─────────────────────────────────────────────────────────────

    def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMClientResponse | None:
        """发送 chat 请求。provider=none 时返回 None（调用方使用模板）.

        不打印 API key。

        Returns:
            LLMClientResponse or None（provider=none）
        """
        if self.provider == "none":
            return None

        try:
            if self.provider == "anthropic":
                return self._call_anthropic(system, user, max_tokens, temperature)
            else:
                return self._call_openai_compat(system, user, max_tokens, temperature)
        except Exception as e:
            logger.warning(f"[LLMClient] {self.provider} 调用失败，fallback none: {e}")
            return None

    @property
    def is_none_provider(self) -> bool:
        return self.provider == "none"

    # ── 内部实现 ─────────────────────────────────────────────────────────────

    def _resolve_provider(self, requested: str) -> tuple[str, bool]:
        """检查 API key，不存在则 fallback none."""
        if requested == "none":
            return "none", False

        env_key = _ENV_KEYS.get(requested)
        if not env_key:
            logger.warning(f"[LLMClient] 未知 provider={requested!r}，fallback none")
            return "none", True

        key_val = os.environ.get(env_key, "")
        if not key_val.strip():
            logger.warning(
                f"[LLMClient] {env_key} 未设置，provider={requested} fallback → none"
            )
            return "none", True

        logger.info(f"[LLMClient] provider={requested}  model={self.model}")
        return requested, False

    def _call_openai_compat(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMClientResponse | None:
        """调用 OpenAI 兼容接口（openai / deepseek / dashscope）."""
        try:
            import openai
        except ImportError:
            logger.warning("[LLMClient] openai 包未安装，fallback none")
            return None

        api_key = os.environ.get(_ENV_KEYS[self.provider], "")
        base_url: str | None = None
        if self.provider == "deepseek":
            base_url = _DEEPSEEK_BASE_URL
        elif self.provider == "dashscope":
            base_url = _DASHSCOPE_BASE_URL

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.choices[0].message.content or ""
        return LLMClientResponse(
            content=content,
            provider=self.provider,
            model=self.model,
            fallback_used=self.fallback_used,
        )

    def _call_anthropic(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMClientResponse | None:
        """调用 Anthropic Claude API."""
        try:
            import anthropic as ant
        except ImportError:
            logger.warning("[LLMClient] anthropic 包未安装，fallback none")
            return None

        api_key = os.environ.get(_ENV_KEYS["anthropic"], "")
        client = ant.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=self.model,
            system=system or "",
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = resp.content[0].text if resp.content else ""
        return LLMClientResponse(
            content=content,
            provider=self.provider,
            model=self.model,
            fallback_used=self.fallback_used,
        )

    @staticmethod
    def _default_model(provider: str) -> str:
        defaults: dict[str, str] = {
            "none": "none",
            "dashscope": "qwen-plus",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "deepseek": "deepseek-chat",
        }
        return defaults.get(provider, "unknown")


def build_client(
    provider: str = "none",
    model: str | None = None,
) -> LLMClient:
    """工厂函数：构建 LLMClient，key 缺失自动 fallback none."""
    return LLMClient(provider=provider, model=model)
