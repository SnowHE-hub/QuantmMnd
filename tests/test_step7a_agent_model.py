"""T-024 第三轮验收：日更 Step 7a 的预热 / Agent 调用 / 统计记录用同一个模型（环境变量 > 配置 > 默认）。

不连 ollama：探测与预热都注入假函数；Agent 客户端只构造、不调用。
"""
from __future__ import annotations

import json

import pytest

import quantmind.agents.ollama_client as oc
from scripts.daily_update import _resolve_agent_runtime


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    def write(model: str | None):
        p = tmp_path / "strategy_config_v2.json"
        p.write_text(json.dumps({"agent_mode": "full", **({"ollama_model": model} if model else {})}), encoding="utf-8")
        monkeypatch.setattr(oc, "_STRATEGY_CONFIG", p)
        return json.loads(p.read_text(encoding="utf-8"))
    return write


def fake_warm(available: set[str], calls: list[str]):
    def warm_up(model, timeout):
        calls.append(model)
        return (True, 1.0, "ok") if model in available else (False, 0.1, f"HTTP 404 model '{model}' not found")
    return warm_up


def ok_probe():
    return None


def test_env_model_is_warmed_used_and_recorded_even_if_config_model_is_missing(cfg_file, monkeypatch):
    config = cfg_file("qwen2.5:7b")
    monkeypatch.setenv("QM_AGENT_MODEL", "qwen3.5:9b")
    calls: list[str] = []
    rt = _resolve_agent_runtime(config, probe=ok_probe, warm_up=fake_warm({"qwen3.5:9b"}, calls))
    assert calls == ["qwen3.5:9b"]                                    # 预热的是环境指定的模型
    assert rt["agent_mode"] == "full"                                 # 配置里的模型 404 不能触发降级
    assert rt["ollama_model"] == "qwen3.5:9b" and rt["model_source"] == "env"
    assert rt["warmup"]["model"] == "qwen3.5:9b"
    assert oc.OllamaReActClient().model == rt["ollama_model"]         # Agent 客户端用同一个模型


def test_env_model_missing_degrades_to_fast(cfg_file, monkeypatch):
    config = cfg_file("qwen2.5:7b")
    monkeypatch.setenv("QM_AGENT_MODEL", "not-installed:1b")
    calls: list[str] = []
    rt = _resolve_agent_runtime(config, probe=ok_probe, warm_up=fake_warm({"qwen2.5:7b"}, calls))
    assert calls == ["not-installed:1b"] and rt["agent_mode"] == "fast"


def test_config_model_without_env(cfg_file, monkeypatch):
    config = cfg_file("qwen2.5:7b")
    monkeypatch.delenv("QM_AGENT_MODEL", raising=False)
    calls: list[str] = []
    rt = _resolve_agent_runtime(config, probe=ok_probe, warm_up=fake_warm({"qwen2.5:7b"}, calls))
    assert calls == ["qwen2.5:7b"] and rt["agent_mode"] == "full" and rt["model_source"] == "config"
    assert oc.OllamaReActClient().model == "qwen2.5:7b"


def test_default_model_without_env_or_config_entry(cfg_file, monkeypatch):
    config = cfg_file(None)
    monkeypatch.delenv("QM_AGENT_MODEL", raising=False)
    calls: list[str] = []
    rt = _resolve_agent_runtime(config, probe=ok_probe, warm_up=fake_warm({"qwen2.5:7b"}, calls))
    assert rt["ollama_model"] == "qwen2.5:7b" and rt["model_source"] == "default" and calls == ["qwen2.5:7b"]


def test_unreachable_ollama_degrades_without_warmup(cfg_file, monkeypatch):
    config = cfg_file("qwen2.5:7b")
    monkeypatch.delenv("QM_AGENT_MODEL", raising=False)
    calls: list[str] = []

    def down():
        raise ConnectionError("refused")
    rt = _resolve_agent_runtime(config, probe=down, warm_up=fake_warm({"qwen2.5:7b"}, calls))
    assert rt["agent_mode"] == "fast" and calls == [] and rt["warmup"] == {}
