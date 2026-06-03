"""共享 pytest fixtures."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

# 测试启动时即加载 .env 让 TUSHARE_TOKEN / DEEPSEEK_API_KEY 等可用
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv()
except ImportError:
    pass

from quantmind.core.state import (
    AgentState,
    InvestmentQuery,
    QueryIntent,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """每个 test 用独立 cache/log 目录，避免互相干扰."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / ".cache"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))


@pytest.fixture(autouse=True)
def block_external_network(request: pytest.FixtureRequest):
    """CI 闸门：NO_NETWORK=1 时，非 integration 测试禁止访问外部网络。

    任何漏标 @pytest.mark.integration 的联网测试会**立即 RuntimeError 响亮失败**，
    而不是 hang 住整套（曾因一个未隔离的 akshare/tushare 调用 hang 8 小时）。
    放行 localhost/回环（本地 PG/Mongo、pytest-xdist worker 通信不受影响）。
    本地默认不开启（不设 NO_NETWORK）；CI 在 ci.yml 里设 NO_NETWORK=1。
    """
    if os.environ.get("NO_NETWORK") != "1" or "integration" in request.keywords:
        yield
        return

    import socket

    _real_getaddrinfo = socket.getaddrinfo
    _real_create_connection = socket.create_connection
    _LOCAL = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}

    def _is_local(host: object) -> bool:
        h = str(host)
        return h in _LOCAL or h.startswith("127.")

    def _guard_getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            raise RuntimeError(
                f"Network blocked in unit test (NO_NETWORK=1): DNS '{host}'. "
                f"给该测试加 @pytest.mark.integration，或 mock 掉网络调用。"
            )
        return _real_getaddrinfo(host, *args, **kwargs)

    def _guard_create_connection(address, *args, **kwargs):
        host = address[0] if isinstance(address, (tuple, list)) else address
        if not _is_local(host):
            raise RuntimeError(
                f"Network blocked in unit test (NO_NETWORK=1): connect '{host}'. "
                f"给该测试加 @pytest.mark.integration，或 mock 掉网络调用。"
            )
        return _real_create_connection(address, *args, **kwargs)

    socket.getaddrinfo = _guard_getaddrinfo  # type: ignore[assignment]
    socket.create_connection = _guard_create_connection  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.getaddrinfo = _real_getaddrinfo
        socket.create_connection = _real_create_connection


@pytest.fixture
def sample_query() -> InvestmentQuery:
    return InvestmentQuery(
        raw_query="分析宁德时代",
        intent=QueryIntent.SINGLE_STOCK,
        tickers=["300750.SZ"],
        horizon_days=60,
        as_of=date(2024, 6, 30),
    )


@pytest.fixture
def sample_state(sample_query: InvestmentQuery) -> AgentState:
    return AgentState(query=sample_query, max_iterations=3)


@pytest.fixture
def have_internet() -> bool:
    """简易联网检测."""
    import socket

    try:
        socket.create_connection(("api.deepseek.com", 443), timeout=2)
        return True
    except OSError:
        return False


@pytest.fixture
def ollama_available() -> bool:
    try:
        import httpx

        return httpx.get("http://localhost:11434/api/tags", timeout=1.5).status_code == 200
    except Exception:
        return False
