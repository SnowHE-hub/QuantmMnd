"""统一 LLM 客户端 — 支持 DashScope(百炼) / Ollama 本地模型."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

# ── 读取 API Key ──────────────────────────────────────────────────────────────
_KEY_FILE = Path(__file__).resolve().parents[3] / "api_key.txt"

def _read_key(name: str) -> str:
    """从 api_key.txt 读取指定 key."""
    if _KEY_FILE.exists():
        for line in _KEY_FILE.read_text(encoding="utf-8").splitlines():
            if name in line and "：" in line:
                return line.split("：", 1)[1].strip()
            if name in line and ":" in line:
                return line.split(":", 1)[1].strip()
    return os.getenv(name, "")

DASHSCOPE_API_KEY = _read_key("DASHSCOPE_API_KEY")
DEEPSEEK_API_KEY  = _read_key("DEEPSEEK_API_KEY")

DASHSCOPE_BASE    = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE     = "https://api.deepseek.com"
OLLAMA_HOST       = "http://localhost:11434"

# 默认模型
DEFAULT_OLLAMA_MODEL    = "qwen2.5:7b"
DEFAULT_DASHSCOPE_MODEL = "qwen-turbo"
DEFAULT_DEEPSEEK_MODEL  = "deepseek-chat"


# ── DashScope ─────────────────────────────────────────────────────────────────
def dashscope_chat(messages: list[dict], model: str = DEFAULT_DASHSCOPE_MODEL,
                   temperature: float = 0.3) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


def dashscope_stream(messages: list[dict], model: str = DEFAULT_DASHSCOPE_MODEL,
                     temperature: float = 0.3) -> Generator[str, None, None]:
    from openai import OpenAI
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE)
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── DeepSeek ──────────────────────────────────────────────────────────────────
def deepseek_chat(messages: list[dict], model: str = DEFAULT_DEEPSEEK_MODEL,
                  temperature: float = 0.3) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE)
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


# ── Ollama ────────────────────────────────────────────────────────────────────
def ollama_chat(messages: list[dict], model: str = DEFAULT_OLLAMA_MODEL,
                temperature: float = 0.3) -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return resp.message.content or ""


def ollama_stream(messages: list[dict], model: str = DEFAULT_OLLAMA_MODEL,
                  temperature: float = 0.3) -> Generator[str, None, None]:
    import ollama
    for chunk in ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
        stream=True,
    ):
        yield chunk.message.content or ""


def list_ollama_models() -> list[str]:
    try:
        import ollama
        client = ollama.Client()
        return [m.model for m in client.list().models]
    except Exception:
        return []


# ── 统一调用入口 ──────────────────────────────────────────────────────────────
def chat(messages: list[dict],
         provider: str = "dashscope",
         model: str | None = None,
         temperature: float = 0.3) -> str:
    """provider: 'dashscope' | 'ollama' | 'deepseek'"""
    try:
        if provider == "ollama":
            return ollama_chat(messages, model or DEFAULT_OLLAMA_MODEL, temperature)
        elif provider == "deepseek":
            return deepseek_chat(messages, model or DEFAULT_DEEPSEEK_MODEL, temperature)
        else:  # dashscope (default)
            return dashscope_chat(messages, model or DEFAULT_DASHSCOPE_MODEL, temperature)
    except Exception as e:
        # 降级：dashscope → ollama → deepseek
        if provider == "dashscope":
            try:
                return ollama_chat(messages, DEFAULT_OLLAMA_MODEL, temperature)
            except Exception:
                pass
        return f"[LLM错误] {e}"


def stream(messages: list[dict],
           provider: str = "dashscope",
           model: str | None = None,
           temperature: float = 0.3) -> Generator[str, None, None]:
    """流式输出."""
    try:
        if provider == "ollama":
            yield from ollama_stream(messages, model or DEFAULT_OLLAMA_MODEL, temperature)
        elif provider == "deepseek":
            # deepseek 暂不流式，整体返回
            yield deepseek_chat(messages, model or DEFAULT_DEEPSEEK_MODEL, temperature)
        else:
            yield from dashscope_stream(messages, model or DEFAULT_DASHSCOPE_MODEL, temperature)
    except Exception as e:
        try:
            yield from ollama_stream(messages, DEFAULT_OLLAMA_MODEL, temperature)
        except Exception:
            yield f"[LLM错误] {e}"
