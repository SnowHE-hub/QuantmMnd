"""tests/test_agent_registry.py — AgentModelRegistry 单元测试."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from quantmind.agents.investment_agents.agent_registry import (
    AgentModelRecord,
    AgentModelRegistry,
    initialize_default_registry,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def registry(tmp_path):
    """创建使用临时路径的注册表."""
    return AgentModelRegistry(registry_path=tmp_path / "registry.json")


@pytest.fixture
def base_record():
    return AgentModelRecord(
        agent_name="TestAgent",
        model_version="rules_v1",
        model_type="rules",
        model_path=None,
        created_at="2024-01-01T00:00:00",
        performance={"ic_mean": 0.030, "accuracy": 0.55},
        is_active=True,
        upgrade_notes="基线规则版本",
    )


@pytest.fixture
def v2_record():
    return AgentModelRecord(
        agent_name="TestAgent",
        model_version="lgbm_v2",
        model_type="ml",
        model_path="models/agents/test_lgbm_v2.pkl",
        created_at="2024-06-01T00:00:00",
        performance={"ic_mean": 0.055, "accuracy": 0.62},
        is_active=False,
        upgrade_notes="LGBM升级版本",
    )


# ── 测试：注册新模型版本 ──────────────────────────────────────────────────────

def test_register_new_version_increases_history(registry, base_record, v2_record):
    """注册新模型版本后，history 增加."""
    registry.register(base_record)
    assert len(registry.get_history("TestAgent")) == 1

    registry.register(v2_record)
    assert len(registry.get_history("TestAgent")) == 2


def test_register_duplicate_version_updates_not_appends(registry, base_record):
    """注册已有版本时更新，不重复追加."""
    registry.register(base_record)
    updated = AgentModelRecord(
        agent_name="TestAgent",
        model_version="rules_v1",
        model_type="rules",
        model_path=None,
        created_at="2024-01-01T00:00:00",
        performance={"ic_mean": 0.040},  # 更新性能
        is_active=True,
        upgrade_notes="更新后",
    )
    registry.register(updated)
    history = registry.get_history("TestAgent")
    assert len(history) == 1  # 不追加
    assert history[0].performance["ic_mean"] == 0.040


# ── 测试：set_active 切换激活版本 ─────────────────────────────────────────────

def test_set_active_switches_correctly(registry, base_record, v2_record):
    """set_active 切换后 get_active 返回新版本."""
    registry.register(base_record)   # rules_v1, is_active=True
    registry.register(v2_record)     # lgbm_v2, is_active=False

    # 初始：rules_v1 是激活的
    active = registry.get_active("TestAgent")
    assert active.model_version == "rules_v1"

    # 切换到 lgbm_v2
    success = registry.set_active("TestAgent", "lgbm_v2")
    assert success is True

    active = registry.get_active("TestAgent")
    assert active.model_version == "lgbm_v2"
    assert active.is_active is True


def test_set_active_deactivates_others(registry, base_record, v2_record):
    """set_active 后其他版本 is_active 变为 False."""
    registry.register(base_record)
    registry.register(v2_record)
    registry.set_active("TestAgent", "lgbm_v2")

    history = registry.get_history("TestAgent")
    for rec in history:
        if rec.model_version == "lgbm_v2":
            assert rec.is_active is True
        else:
            assert rec.is_active is False


def test_set_active_nonexistent_version_returns_false(registry, base_record):
    """设置不存在的版本返回 False."""
    registry.register(base_record)
    success = registry.set_active("TestAgent", "nonexistent_v99")
    assert success is False


# ── 测试：compare_versions 输出正确 ───────────────────────────────────────────

def test_compare_versions_returns_correct_dataframe(registry, base_record, v2_record):
    """compare_versions 输出正确的对比表."""
    registry.register(base_record)
    registry.register(v2_record)

    df = registry.compare_versions("TestAgent")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "version" in df.columns
    assert "ic_mean" in df.columns

    versions = df["version"].tolist()
    assert "rules_v1" in versions
    assert "lgbm_v2" in versions


def test_compare_versions_empty_for_unknown_agent(registry):
    """未注册的 Agent 返回空 DataFrame."""
    df = registry.compare_versions("UnknownAgent")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ── 测试：模型加载失败降级到 rules ────────────────────────────────────────────

def test_model_load_failure_falls_back_to_rules(tmp_path, registry):
    """模型文件不存在时降级到规则模式，不崩溃."""
    from quantmind.agents.investment_agents.base_agent import BaseInvestmentAgent, AgentSignal

    # 注册一个 ml 版本，但模型文件不存在
    ml_record = AgentModelRecord(
        agent_name="MomentumAgent",
        model_version="lgbm_v2",
        model_type="ml",
        model_path=str(tmp_path / "nonexistent_model.pkl"),
        created_at="2024-06-01T00:00:00",
        performance={"ic_mean": 0.055},
        is_active=True,
        upgrade_notes="测试不存在的模型",
    )
    registry.register(ml_record)
    registry.set_active("MomentumAgent", "lgbm_v2")

    # 使用这个注册表的 Agent 应该不崩溃，_ml_model 为 None（降级）
    class MockMomentumAgent(BaseInvestmentAgent):
        REGISTRY = registry

        def analyze(self) -> AgentSignal:
            return AgentSignal("MockMomentumAgent", "000001.SZ", 0.0, 0.0, "测试")

    agent = MockMomentumAgent("000001.SZ", "2024-12-31", {})
    assert agent._ml_model is None  # 文件不存在，降级为 None


# ── 测试：initialize_default_registry ────────────────────────────────────────

def test_initialize_default_registry_creates_six_agents(tmp_path):
    """initialize_default_registry 创建 6 个 Agent 的基线记录."""
    registry = AgentModelRegistry(registry_path=tmp_path / "registry.json")

    # 手动调用初始化
    from quantmind.agents.investment_agents.agent_registry import _BASELINE_RECORDS
    for rec in _BASELINE_RECORDS:
        registry.register(rec)

    agents = registry.list_agents()
    expected = {"ValuationAgent", "MomentumAgent", "QualityAgent",
                "SentimentAgent", "RiskAgent", "StrategyAgent"}
    assert expected.issubset(set(agents))


def test_registry_persists_to_disk(tmp_path):
    """注册表写入磁盘，重新加载后数据仍然存在."""
    path = tmp_path / "registry.json"
    r1 = AgentModelRegistry(registry_path=path)
    r1.register(AgentModelRecord(
        agent_name="MomentumAgent",
        model_version="rules_v1",
        model_type="rules",
        model_path=None,
        created_at="2024-01-01T00:00:00",
        performance={"ic_mean": 0.045},
        is_active=True,
    ))

    # 重新加载
    r2 = AgentModelRegistry(registry_path=path)
    history = r2.get_history("MomentumAgent")
    assert len(history) == 1
    assert history[0].performance["ic_mean"] == 0.045


def test_update_performance_merges_metrics(registry, base_record):
    """update_performance 合并而非覆盖 metrics."""
    registry.register(base_record)
    registry.update_performance("TestAgent", "rules_v1", {
        "last_actual_return": 0.05,
        "last_evaluated_at": "2024-12-31",
    })

    active = registry.get_active("TestAgent")
    assert active.performance.get("ic_mean") == 0.030     # 原有字段保留
    assert active.performance.get("last_actual_return") == 0.05  # 新增字段
