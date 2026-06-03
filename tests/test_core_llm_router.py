"""测试 quantmind.core.llm_router."""

from __future__ import annotations

from typing import Any

import pytest

from quantmind.core.llm_router import (
    BaseLLMProvider,
    LLMRouter,
    Message,
    ProviderFatalError,
    ProviderTransientError,
    ProviderUnavailable,
    TokenUsage,
    TokenUsageTracker,
    _normalize_messages,
)

# ============================================================================
# Fake Provider 用于隔离测试
# ============================================================================


class _FakeProvider(BaseLLMProvider):
    """可控行为的 fake provider，用于测试 router 逻辑."""

    def __init__(
        self,
        name: str,
        cfg: dict[str, Any],
        *,
        behavior: str = "ok",
        fail_n: int = 0,
    ) -> None:
        super().__init__(name, cfg)
        self.behavior = behavior
        self.calls = 0
        self.fail_n = fail_n

    def _chat_impl(self, messages: list[Message], model: str, **kwargs: Any) -> Any:
        self.calls += 1
        if self.behavior == "transient_then_ok" and self.calls <= self.fail_n:
            raise ProviderTransientError("temporary glitch")
        if self.behavior == "fatal":
            raise ProviderFatalError("auth fail")
        if self.behavior == "always_transient":
            raise ProviderTransientError("always fails")
        return self._build_response(
            content=f"echo[{self.name}/{model}]: {messages[-1].content}",
            model=model,
            prompt_tokens=10,
            completion_tokens=20,
        )


def _make_cfg(name: str = "fake", model: str = "fake-model") -> dict[str, Any]:
    return {
        "type": "fake",
        "base_url": "http://fake",
        "api_key": "fake-key",
        "models": [
            {
                "name": model,
                "context_length": 8192,
                "cost_per_1k_input": 0.001,
                "cost_per_1k_output": 0.002,
            }
        ],
    }


# ============================================================================
# Tests
# ============================================================================


class TestMessageNormalization:
    def test_string_input(self) -> None:
        msgs = _normalize_messages("hello")
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "hello"

    def test_dict_list_input(self) -> None:
        msgs = _normalize_messages(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        )
        assert len(msgs) == 2
        assert msgs[0].role == "system"

    def test_message_objects(self) -> None:
        msgs = _normalize_messages([Message(role="user", content="x")])
        assert msgs[0].content == "x"

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _normalize_messages([123])  # type: ignore[arg-type]


class TestTokenUsageTracker:
    def test_record_aggregates(self) -> None:
        t = TokenUsageTracker()
        t.record(
            "a",
            "m",
            TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_cny=0.5),
        )
        t.record(
            "a",
            "m",
            TokenUsage(prompt_tokens=5, completion_tokens=15, total_tokens=20, cost_cny=0.3),
        )
        assert t.total_tokens() == 50
        assert abs(t.total_cost() - 0.8) < 1e-9

    def test_summary_format(self) -> None:
        t = TokenUsageTracker()
        t.record("a", "m1", TokenUsage(total_tokens=100, cost_cny=0.1))
        t.record("b", "m2", TokenUsage(total_tokens=200, cost_cny=0.2))
        s = t.summary()
        assert s["total_tokens"] == 300
        assert len(s["rows"]) == 2

    def test_failures_recorded(self) -> None:
        t = TokenUsageTracker()
        t.record_failure("x", "y")
        t.record_failure("x", "y")
        s = t.summary()
        assert s["rows"][0]["failures"] == 2


class TestProviderCostEstimation:
    def test_cost_calculation(self) -> None:
        prov = _FakeProvider("fake", _make_cfg(), behavior="ok")
        # 1000 input @ 0.001 + 2000 output @ 0.002 = 0.001 + 0.004 = 0.005
        cost = prov.estimate_cost("fake-model", 1000, 2000)
        assert abs(cost - 0.005) < 1e-9


class TestLLMRouter:
    @pytest.fixture
    def router(self, monkeypatch: pytest.MonkeyPatch) -> LLMRouter:
        """构造一个使用 _FakeProvider 的 router."""
        from quantmind.core import llm_router as mod

        # patch provider class map
        monkeypatch.setitem(mod.PROVIDER_CLASS, "fakeA", _FakeProvider)  # type: ignore[arg-type]
        monkeypatch.setitem(mod.PROVIDER_CLASS, "fakeB", _FakeProvider)  # type: ignore[arg-type]

        # patch yaml loader
        fake_yaml = {
            "providers": {
                "fakeA": _make_cfg("fakeA"),
                "fakeB": _make_cfg("fakeB"),
            },
            "fallback_chain": {"default": ["fakeA", "fakeB"], "test_chain": ["fakeB"]},
            "task_routing": {"my_task": "fakeB"},
        }
        monkeypatch.setattr(mod, "_load_providers_yaml", lambda *a, **k: fake_yaml)

        # 让 settings.default_llm_provider = fakeA
        from quantmind.core.config import get_settings

        settings = get_settings(reload=True)
        settings.default_llm_provider = "fakeA"  # type: ignore[misc]
        settings.llm.retry_attempts = 2  # 加快测试
        settings.llm.retry_min_wait = 0.01
        settings.llm.retry_max_wait = 0.05

        return LLMRouter(settings=settings)

    def test_basic_call(self, router: LLMRouter) -> None:
        resp = router.prompt("hi")
        assert "echo[fakeA" in resp.content
        assert resp.usage.total_tokens == 30
        assert resp.provider == "fakeA"

    def test_fallback_on_fatal(self, router: LLMRouter, monkeypatch: pytest.MonkeyPatch) -> None:
        """fakeA fatal → fallback to fakeB."""
        prov_a = router.get_provider("fakeA")
        prov_a.behavior = "fatal"  # type: ignore[attr-defined]
        resp = router.prompt("hi")
        assert "echo[fakeB" in resp.content

    def test_transient_retried_then_succeeds(self, router: LLMRouter) -> None:
        """transient 错误应被 tenacity 重试."""
        prov = router.get_provider("fakeA")
        prov.behavior = "transient_then_ok"  # type: ignore[attr-defined]
        prov.fail_n = 1  # type: ignore[attr-defined]
        resp = router.prompt("hi", retry_attempts=3)
        assert "echo[fakeA" in resp.content
        assert prov.calls == 2  # type: ignore[attr-defined]

    def test_chain_exhausted_raises(self, router: LLMRouter) -> None:
        for name in ("fakeA", "fakeB"):
            p = router.get_provider(name)
            p.behavior = "always_transient"  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="All providers"):
            router.prompt("hi", retry_attempts=2)

    def test_chat_for_task_routing(self, router: LLMRouter) -> None:
        resp = router.chat_for_task("my_task", "hi")
        assert "echo[fakeB" in resp.content

    def test_tracker_records(self, router: LLMRouter) -> None:
        router.prompt("a")
        router.prompt("b")
        s = router.tracker.summary()
        assert s["rows"][0]["calls"] == 2

    def test_explicit_provider_overrides(self, router: LLMRouter) -> None:
        resp = router.prompt("x", provider="fakeB")
        assert "echo[fakeB" in resp.content


# ============================================================================
# Live integration: Ollama（仅本机有 ollama 时跑）
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_ollama_live(ollama_available: bool) -> None:
    # integration：真实调用本地 Ollama 服务，可能 hang（服务在但模型加载/繁忙时
    # router.prompt 会阻塞），默认/CI 单元跑中跳过。
    if not ollama_available:
        pytest.skip("ollama not running")

    from quantmind.core.llm_router import LLMRouter

    router = LLMRouter()
    resp = router.prompt(
        "请用一个词回答：太阳从哪个方向升起？",
        provider="ollama",
        model="qwen2.5:7b",
    )
    assert resp.content
    assert resp.usage.total_tokens > 0
    assert resp.provider == "ollama"
