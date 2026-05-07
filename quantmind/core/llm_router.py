"""quantmind.core.llm_router — 多 LLM Provider 统一路由层.

提供 5 个 Provider 的统一调用接口：

    - OllamaProvider   ：本地推理（开发期推荐，零成本）
    - DeepSeekProvider ：用 openai SDK + base_url 切换（生产推荐，最便宜）
    - QwenProvider     ：用 dashscope SDK
    - OpenAIProvider   ：标准 OpenAI
    - AnthropicProvider：Claude

核心特性：
    1. 统一 ``chat(messages) -> LLMResponse`` 接口
    2. ``LLMRouter`` 自动按 ``configs/llm_providers.yaml`` 选 provider 与 fallback
    3. tenacity 指数回退重试（默认 3 次）
    4. ``TokenUsageTracker`` 全局聚合 token / 估算成本
    5. ``chat_for_task(task_name, ...)`` 按任务路由（如 critic 用 deepseek、
       sentiment 用 ollama）

用法::

    from quantmind.core.llm_router import LLMRouter

    router = LLMRouter()
    resp = router.prompt("用一句话解释什么是 RAG。")
    print(resp.content)
    print(router.tracker.summary())
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from quantmind.core.config import PROJECT_ROOT, Settings, get_settings
from quantmind.core.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# Schemas
# ============================================================================

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: Role
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_cny: float = 0.0


class LLMResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw: Any = None
    finish_reason: str | None = None
    elapsed_s: float = 0.0


def _normalize_messages(
    messages: str | Iterable[Message] | Iterable[dict[str, str]],
) -> list[Message]:
    """接受多种输入格式，统一转 list[Message]."""
    if isinstance(messages, str):
        return [Message(role="user", content=messages)]
    out: list[Message] = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
        elif isinstance(m, dict):
            out.append(Message(role=m["role"], content=m["content"]))  # type: ignore[arg-type]
        else:
            raise TypeError(f"Unsupported message type: {type(m)}")
    return out


# ============================================================================
# Token usage tracking
# ============================================================================


@dataclass
class _TrackerEntry:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_cny: float = 0.0
    failures: int = 0


@dataclass
class TokenUsageTracker:
    """聚合 token 与成本，按 provider/model 分类."""

    by_provider_model: dict[tuple[str, str], _TrackerEntry] = field(default_factory=dict)
    daily_cost_limit_cny: float = float("inf")

    def record(self, provider: str, model: str, usage: TokenUsage) -> None:
        key = (provider, model)
        e = self.by_provider_model.setdefault(key, _TrackerEntry())
        e.calls += 1
        e.prompt_tokens += usage.prompt_tokens
        e.completion_tokens += usage.completion_tokens
        e.total_tokens += usage.total_tokens
        e.cost_cny += usage.cost_cny
        if self.total_cost() > self.daily_cost_limit_cny:
            log.warning(
                f"⚠ daily LLM cost {self.total_cost():.4f} CNY exceeded limit "
                f"{self.daily_cost_limit_cny:.2f}"
            )

    def record_failure(self, provider: str, model: str) -> None:
        key = (provider, model)
        e = self.by_provider_model.setdefault(key, _TrackerEntry())
        e.failures += 1

    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.by_provider_model.values())

    def total_cost(self) -> float:
        return sum(e.cost_cny for e in self.by_provider_model.values())

    def summary(self) -> dict[str, Any]:
        rows = [
            {
                "provider": p,
                "model": m,
                "calls": e.calls,
                "failures": e.failures,
                "prompt_tokens": e.prompt_tokens,
                "completion_tokens": e.completion_tokens,
                "total_tokens": e.total_tokens,
                "cost_cny": round(e.cost_cny, 4),
            }
            for (p, m), e in self.by_provider_model.items()
        ]
        return {
            "rows": rows,
            "total_tokens": self.total_tokens(),
            "total_cost_cny": round(self.total_cost(), 4),
        }

    def reset(self) -> None:
        self.by_provider_model.clear()


# ============================================================================
# Provider 配置（从 llm_providers.yaml 加载）
# ============================================================================


def _resolve_env_in_obj(obj: Any) -> Any:
    """递归替换字符串中的 ``${ENV_VAR}``."""
    import re

    pat = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
    if isinstance(obj, str):
        return pat.sub(lambda m: os.getenv(m.group(1), ""), obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_in_obj(x) for x in obj]
    return obj


def _load_providers_yaml(path: Path | None = None) -> dict[str, Any]:
    p = path or (PROJECT_ROOT / "configs" / "llm_providers.yaml")
    if not p.exists():
        return {"providers": {}, "fallback_chain": {}, "task_routing": {}}
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _resolve_env_in_obj(data)


def _model_cost(provider_cfg: dict[str, Any], model_name: str) -> tuple[float, float]:
    """返回 (cost_per_1k_input, cost_per_1k_output)；找不到时返回 0."""
    for m in provider_cfg.get("models", []):
        if m.get("name") == model_name:
            return float(m.get("cost_per_1k_input", 0.0)), float(m.get("cost_per_1k_output", 0.0))
    return 0.0, 0.0


# ============================================================================
# Provider 抽象基类
# ============================================================================


class ProviderUnavailableError(Exception):
    """Provider 缺少 key / 网络不通，可被 fallback 链兜住."""


# 历史别名（向后兼容；新代码请用 *Error 版本）
ProviderUnavailable = ProviderUnavailableError


class ProviderTransientError(Exception):
    """临时错误（429、5xx、超时），值得重试."""


class ProviderFatalError(Exception):
    """永久错误（认证失败、模型不存在），不重试."""


class BaseLLMProvider(ABC):
    name: str = "base"

    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.cfg = cfg
        self.api_key: str = str(cfg.get("api_key", "")).strip()
        self.base_url: str = cfg.get("base_url", "")
        self.default_model: str = (cfg.get("models") or [{}])[0].get("name", "")
        self._client: Any = None  # lazy

    def list_models(self) -> list[str]:
        return [m["name"] for m in self.cfg.get("models", [])]

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        in_rate, out_rate = _model_cost(self.cfg, model)
        return prompt_tokens / 1000 * in_rate + completion_tokens / 1000 * out_rate

    def _build_response(
        self,
        content: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        raw: Any = None,
        finish_reason: str | None = None,
        elapsed_s: float = 0.0,
    ) -> LLMResponse:
        cost = self.estimate_cost(model, prompt_tokens, completion_tokens)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_cny=round(cost, 6),
        )
        return LLMResponse(
            content=content,
            model=model,
            provider=self.name,
            usage=usage,
            raw=raw,
            finish_reason=finish_reason,
            elapsed_s=elapsed_s,
        )

    @abstractmethod
    def _chat_impl(self, messages: list[Message], model: str, **kwargs: Any) -> LLMResponse: ...

    def chat(self, messages: list[Message], model: str | None = None, **kwargs: Any) -> LLMResponse:
        m = model or self.default_model
        if not m:
            raise ProviderFatalError(f"No model specified for provider {self.name}")
        log.debug(f"[{self.name}] chat call: model={m}, msgs={len(messages)}")
        return self._chat_impl(messages, m, **kwargs)


# ============================================================================
# 各 Provider 实现
# ============================================================================


class OllamaProvider(BaseLLMProvider):
    """本地 Ollama 推理（http://localhost:11434）.

    依赖 ``ollama`` python 包；如未安装，回退到 httpx 直连。
    """

    def _client_lazy(self) -> Any:
        if self._client is None:
            try:
                from ollama import Client

                self._client = Client(host=self.base_url or "http://localhost:11434")
            except ImportError as e:
                raise ProviderUnavailable(
                    "ollama package not installed; pip install ollama"
                ) from e
        return self._client

    def _chat_impl(self, messages: list[Message], model: str, **kwargs: Any) -> LLMResponse:
        c = self._client_lazy()
        t0 = time.monotonic()
        try:
            resp = c.chat(
                model=model,
                messages=[m.to_openai() for m in messages],
                options={
                    "temperature": kwargs.get("temperature", 0.1),
                    "num_predict": kwargs.get("max_tokens", -1),
                },
            )
        except Exception as e:
            # ollama 客户端的 ConnectError / ResponseError 都视为 transient
            raise ProviderTransientError(f"ollama error: {e}") from e
        elapsed = time.monotonic() - t0

        # ollama 0.3+ 返回 dict 或 ChatResponse 对象
        msg = resp["message"] if isinstance(resp, dict) else resp.message
        content = msg["content"] if isinstance(msg, dict) else msg.content
        eval_count = (
            resp.get("eval_count", 0)
            if isinstance(resp, dict)
            else getattr(resp, "eval_count", 0) or 0
        )
        prompt_count = (
            resp.get("prompt_eval_count", 0)
            if isinstance(resp, dict)
            else getattr(resp, "prompt_eval_count", 0) or 0
        )
        return self._build_response(
            content=content,
            model=model,
            prompt_tokens=int(prompt_count),
            completion_tokens=int(eval_count),
            raw=resp,
            elapsed_s=elapsed,
        )


class _OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI 兼容 SDK 的基类（DeepSeek / OpenAI / 自建 vLLM 等）."""

    def _client_lazy(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ProviderUnavailable("openai package not installed") from e
            if not self.api_key:
                raise ProviderUnavailable(f"{self.name}: api_key 未配置")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        return self._client

    def _chat_impl(self, messages: list[Message], model: str, **kwargs: Any) -> LLMResponse:
        client = self._client_lazy()
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[m.to_openai() for m in messages],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 4096),
                timeout=kwargs.get("timeout_s", 120),
            )
        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ("rate limit", "timeout", "503", "502", "429")):
                raise ProviderTransientError(f"{self.name}: {e}") from e
            if any(s in msg for s in ("auth", "unauthorized", "invalid api key", "401")):
                raise ProviderFatalError(f"{self.name} auth: {e}") from e
            raise ProviderTransientError(f"{self.name}: {e}") from e
        elapsed = time.monotonic() - t0

        choice = resp.choices[0]
        usage = resp.usage
        return self._build_response(
            content=choice.message.content or "",
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=resp,
            finish_reason=getattr(choice, "finish_reason", None),
            elapsed_s=elapsed,
        )


class DeepSeekProvider(_OpenAICompatibleProvider):
    """DeepSeek 走 OpenAI 兼容 API；base_url=https://api.deepseek.com/v1."""


class OpenAIProvider(_OpenAICompatibleProvider):
    """标准 OpenAI；base_url 默认走官方."""


class QwenProvider(BaseLLMProvider):
    """阿里通义千问，走 dashscope SDK."""

    def _client_lazy(self) -> Any:
        if self._client is None:
            try:
                import dashscope  # type: ignore[import-untyped]
            except ImportError as e:
                raise ProviderUnavailable("dashscope package not installed") from e
            if not self.api_key:
                raise ProviderUnavailable("qwen: DASHSCOPE_API_KEY 未配置")
            dashscope.api_key = self.api_key
            self._client = dashscope
        return self._client

    def _chat_impl(self, messages: list[Message], model: str, **kwargs: Any) -> LLMResponse:
        ds = self._client_lazy()
        t0 = time.monotonic()
        try:
            resp = ds.Generation.call(
                model=model,
                messages=[m.to_openai() for m in messages],
                result_format="message",
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 4096),
            )
        except Exception as e:
            raise ProviderTransientError(f"qwen: {e}") from e
        elapsed = time.monotonic() - t0

        if getattr(resp, "status_code", 200) != 200:
            msg = f"qwen non-200: code={resp.code}, msg={resp.message}"
            if resp.status_code in {401, 403}:
                raise ProviderFatalError(msg)
            raise ProviderTransientError(msg)

        out = resp.output
        choice = out["choices"][0] if isinstance(out, dict) else out.choices[0]
        msg = choice["message"] if isinstance(choice, dict) else choice.message
        content = msg["content"] if isinstance(msg, dict) else msg.content
        usage_obj = resp.usage
        prompt_tokens = (
            usage_obj["input_tokens"] if isinstance(usage_obj, dict) else usage_obj.input_tokens
        )
        completion_tokens = (
            usage_obj["output_tokens"] if isinstance(usage_obj, dict) else usage_obj.output_tokens
        )

        return self._build_response(
            content=content or "",
            model=model,
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            raw=resp,
            elapsed_s=elapsed,
        )


class AnthropicProvider(BaseLLMProvider):
    """Claude (anthropic SDK)."""

    def _client_lazy(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise ProviderUnavailable("anthropic package not installed") from e
            if not self.api_key:
                raise ProviderUnavailable("anthropic: ANTHROPIC_API_KEY 未配置")
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def _chat_impl(self, messages: list[Message], model: str, **kwargs: Any) -> LLMResponse:
        client = self._client_lazy()
        sys_msgs = [m.content for m in messages if m.role == "system"]
        user_msgs = [m for m in messages if m.role != "system"]
        t0 = time.monotonic()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.1),
                system="\n\n".join(sys_msgs) if sys_msgs else None,
                messages=[m.to_openai() for m in user_msgs],
            )
        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ("auth", "401", "403", "invalid")):
                raise ProviderFatalError(f"anthropic: {e}") from e
            raise ProviderTransientError(f"anthropic: {e}") from e
        elapsed = time.monotonic() - t0

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return self._build_response(
            content=text,
            model=model,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            raw=resp,
            finish_reason=resp.stop_reason,
            elapsed_s=elapsed,
        )


PROVIDER_CLASS: dict[str, type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "deepseek": DeepSeekProvider,
    "openai": OpenAIProvider,
    "qwen": QwenProvider,
    "anthropic": AnthropicProvider,
}


# ============================================================================
# Router
# ============================================================================


class LLMRouter:
    """根据 settings + llm_providers.yaml 自动选 provider，含 fallback 与重试.

    一次性初始化，全局共享：

        router = LLMRouter()
        router.prompt("hello")              # 用默认 provider
        router.chat(msgs, provider="qwen")  # 强制用 qwen
        router.chat_for_task("critic_agent", msgs)  # 按 task_routing 选
    """

    def __init__(
        self,
        settings: Settings | None = None,
        providers_yaml_path: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.providers_yaml = _load_providers_yaml(providers_yaml_path)
        self.fallback_chains: dict[str, list[str]] = self.providers_yaml.get("fallback_chain", {})
        self.task_routing: dict[str, str] = self.providers_yaml.get("task_routing", {})
        self._providers: dict[str, BaseLLMProvider] = {}
        self.tracker = TokenUsageTracker(daily_cost_limit_cny=self.settings.max_daily_api_cost_cny)
        log.info(
            f"LLMRouter ready: providers={list(self.providers_yaml.get('providers', {}).keys())}, "
            f"default={self.settings.default_llm_provider}"
        )

    # ----- Provider 管理 -----
    def get_provider(self, name: str) -> BaseLLMProvider:
        if name in self._providers:
            return self._providers[name]
        cfg = self.providers_yaml.get("providers", {}).get(name)
        if not cfg:
            raise ValueError(f"Unknown provider: {name}")
        cls = PROVIDER_CLASS.get(name)
        if cls is None:
            raise ValueError(f"No implementation for provider: {name}")
        prov = cls(name, cfg)
        self._providers[name] = prov
        return prov

    def available_providers(self) -> list[str]:
        names = []
        for name in self.providers_yaml.get("providers", {}):
            try:
                p = self.get_provider(name)
                if p.api_key or name == "ollama":
                    names.append(name)
            except Exception:  # noqa: BLE001
                continue
        return names

    # ----- 调用入口 -----
    def chat(
        self,
        messages: str | Iterable[Message] | Iterable[dict[str, str]],
        provider: str | None = None,
        model: str | None = None,
        fallback: str | list[str] | None = "default",
        retry_attempts: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """统一入口."""
        msgs = _normalize_messages(messages)
        primary = provider or self.settings.default_llm_provider

        # 构造尝试链
        chain: list[str] = [primary]
        if fallback:
            extras: list[str]
            if isinstance(fallback, str):
                extras = list(self.fallback_chains.get(fallback, []))
            else:
                extras = list(fallback)
            for p in extras:
                if p not in chain:
                    chain.append(p)

        attempts = retry_attempts or self.settings.llm.retry_attempts
        last_err: Exception | None = None
        for prov_name in chain:
            try:
                prov = self.get_provider(prov_name)
            except Exception as e:  # noqa: BLE001
                log.warning(f"provider {prov_name} init failed: {e}; skipping")
                last_err = e
                continue

            actual_model = (
                model
                if (provider == prov_name and model)
                else (self.settings.llm.model if prov_name == primary and not model else None)
                or prov.default_model
            )

            try:
                resp = self._chat_with_retry(prov, msgs, actual_model, attempts=attempts, **kwargs)
                self.tracker.record(prov_name, resp.model, resp.usage)
                log.info(
                    f"[{prov_name}/{actual_model}] tokens={resp.usage.total_tokens} "
                    f"cost={resp.usage.cost_cny:.4f} elapsed={resp.elapsed_s:.2f}s"
                )
                return resp
            except ProviderFatalError as e:
                self.tracker.record_failure(prov_name, actual_model)
                log.error(f"[{prov_name}] fatal: {e}; trying next in chain")
                last_err = e
                continue
            except Exception as e:  # noqa: BLE001
                self.tracker.record_failure(prov_name, actual_model)
                log.warning(f"[{prov_name}] failed after retries: {e}; trying next")
                last_err = e
                continue

        raise RuntimeError(f"All providers in chain {chain} failed. Last error: {last_err}")

    def chat_for_task(
        self,
        task_name: str,
        messages: str | Iterable[Message] | Iterable[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """按 task_routing 选 provider（任务-provider 映射在 yaml 里）."""
        prov = self.task_routing.get(task_name, self.settings.default_llm_provider)
        return self.chat(messages, provider=prov, **kwargs)

    def prompt(
        self,
        text: str,
        system: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """便利方法：单 user prompt（可选 system）."""
        msgs: list[Message] = []
        if system:
            msgs.append(Message(role="system", content=system))
        msgs.append(Message(role="user", content=text))
        return self.chat(msgs, provider=provider, **kwargs)

    # ----- 重试包装 -----
    def _chat_with_retry(
        self,
        prov: BaseLLMProvider,
        messages: list[Message],
        model: str,
        attempts: int,
        **kwargs: Any,
    ) -> LLMResponse:
        # 动态构造 retry 装饰器，避免在 self 上挂死参数
        @retry(
            reraise=True,
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=self.settings.llm.retry_min_wait,
                max=self.settings.llm.retry_max_wait,
            ),
            retry=retry_if_exception_type(ProviderTransientError),
        )
        def _call() -> LLMResponse:
            return prov.chat(messages, model=model, **kwargs)

        return _call()


# ============================================================================
# 便捷全局实例
# ============================================================================

_router_singleton: LLMRouter | None = None


def get_router(reload: bool = False) -> LLMRouter:
    global _router_singleton
    if reload or _router_singleton is None:
        _router_singleton = LLMRouter()
    return _router_singleton


__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "DeepSeekProvider",
    "LLMResponse",
    "LLMRouter",
    "Message",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderFatalError",
    "ProviderTransientError",
    "ProviderUnavailable",
    "QwenProvider",
    "TokenUsage",
    "TokenUsageTracker",
    "get_router",
]


if __name__ == "__main__":
    router = get_router()
    print(f"available providers: {router.available_providers()}")
    print(f"task routing: {router.task_routing}")

    print("\n--- Test Ollama (local qwen2.5:7b) ---")
    try:
        resp = router.prompt(
            "用一句话解释什么是 RAG。",
            provider="ollama",
            model="qwen2.5:7b",
        )
        print(f"  content   : {resp.content[:200]}")
        print(f"  model     : {resp.model}")
        print(
            f"  tokens    : {resp.usage.total_tokens} (in={resp.usage.prompt_tokens} "
            f"out={resp.usage.completion_tokens})"
        )
        print(f"  elapsed   : {resp.elapsed_s:.2f}s")
        print(f"  cost      : ¥{resp.usage.cost_cny:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"  Ollama call failed: {e}")

    print("\n--- Tracker summary ---")
    import json as _json

    print(_json.dumps(router.tracker.summary(), indent=2, ensure_ascii=False))
