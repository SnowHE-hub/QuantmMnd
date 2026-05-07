"""共享 pytest fixtures."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

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
