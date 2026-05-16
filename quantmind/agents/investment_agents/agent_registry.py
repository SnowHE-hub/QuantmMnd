"""quantmind.agents.investment_agents.agent_registry — Agent 专属模型注册表.

每个 Agent 有自己的：
  - 当前激活模型（rules / ml / dl / llm）
  - 历史版本记录
  - 性能指标（IC / 准确率 / 校准误差）
  - 升级路径定义

持久化到 data/agent_models/registry.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REGISTRY_PATH = _ROOT / "data" / "agent_models" / "registry.json"


@dataclass
class AgentModelRecord:
    """Agent 的单个模型版本记录."""

    agent_name: str
    model_version: str       # "rules_v1" / "lgbm_v1" / "lstm_v1" / "llm_gpt4"
    model_type: str          # "rules" / "ml" / "dl" / "llm"
    model_path: str | None   # pkl/pt/None
    created_at: str
    performance: dict        # {ic_mean, accuracy, calibration_error, ...}
    is_active: bool
    upgrade_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentModelRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AgentModelRegistry:
    """Agent 模型注册表.

    记录每个 Agent 历史上用过的所有模型版本，
    支持切换激活版本、记录性能、触发升级流程。
    持久化到 data/agent_models/registry.json
    """

    _lock: Lock = Lock()

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self._path = Path(registry_path or _REGISTRY_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, list[dict]] = {}  # agent_name → list of record dicts
        self._load()

    # ── 持久化 ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"[AgentRegistry] 加载失败: {e}，使用空注册表")
                self._data = {}

    def _save(self) -> None:
        with self._lock:
            try:
                self._path.write_text(
                    json.dumps(self._data, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.warning(f"[AgentRegistry] 保存失败: {e}")

    # ── 注册与查询 ───────────────────────────────────────────────────────────

    def register(self, record: AgentModelRecord) -> None:
        """注册新模型版本（若已存在同版本，更新它）."""
        agent = record.agent_name
        records = self._data.setdefault(agent, [])
        # 若已有同版本，替换
        for i, r in enumerate(records):
            if r.get("model_version") == record.model_version:
                records[i] = record.to_dict()
                self._save()
                logger.info(f"[AgentRegistry] 更新 {agent}/{record.model_version}")
                return
        records.append(record.to_dict())
        self._save()
        logger.info(f"[AgentRegistry] 注册 {agent}/{record.model_version}")

    def get_active(self, agent_name: str) -> AgentModelRecord | None:
        """获取当前激活的模型版本；若无，返回 None."""
        records = self._data.get(agent_name, [])
        for r in records:
            if r.get("is_active"):
                return AgentModelRecord.from_dict(r)
        # 若无 active，返回最后一条
        if records:
            return AgentModelRecord.from_dict(records[-1])
        return None

    def set_active(self, agent_name: str, version: str) -> bool:
        """切换激活版本，返回是否成功."""
        records = self._data.get(agent_name, [])
        found = False
        for r in records:
            if r.get("model_version") == version:
                r["is_active"] = True
                found = True
            else:
                r["is_active"] = False
        if found:
            self._save()
            logger.info(f"[AgentRegistry] 切换 {agent_name} → {version}")
        else:
            logger.warning(f"[AgentRegistry] 未找到版本 {agent_name}/{version}")
        return found

    def get_history(self, agent_name: str) -> list[AgentModelRecord]:
        """获取 Agent 的所有历史版本，按创建时间排序."""
        records = self._data.get(agent_name, [])
        parsed = [AgentModelRecord.from_dict(r) for r in records]
        return sorted(parsed, key=lambda r: r.created_at)

    def compare_versions(self, agent_name: str) -> pd.DataFrame:
        """对比同一 Agent 所有版本的性能指标，返回 DataFrame."""
        history = self.get_history(agent_name)
        if not history:
            return pd.DataFrame()
        rows = []
        for rec in history:
            row = {
                "agent": rec.agent_name,
                "version": rec.model_version,
                "model_type": rec.model_type,
                "is_active": rec.is_active,
                "created_at": rec.created_at,
                **rec.performance,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def update_performance(
        self, agent_name: str, version: str, metrics: dict
    ) -> None:
        """更新指定版本的性能指标."""
        records = self._data.get(agent_name, [])
        for r in records:
            if r.get("model_version") == version:
                r.setdefault("performance", {}).update(metrics)
                self._save()
                return
        logger.warning(f"[AgentRegistry] 未找到 {agent_name}/{version}")

    def list_agents(self) -> list[str]:
        """列出注册表中所有 Agent 名称."""
        return list(self._data.keys())

    def to_summary_dict(self) -> dict[str, Any]:
        """返回整个注册表的摘要（各 Agent 激活版本）."""
        summary = {}
        for agent_name in self._data:
            active = self.get_active(agent_name)
            summary[agent_name] = {
                "active_version": active.model_version if active else None,
                "total_versions": len(self._data[agent_name]),
            }
        return summary


# ── 默认注册表（单例）────────────────────────────────────────────────────────

_default_registry: AgentModelRegistry | None = None


def get_default_registry() -> AgentModelRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = AgentModelRegistry()
    return _default_registry


# ── 初始化各 Agent 的 rules_v1 基线记录 ──────────────────────────────────────

_BASELINE_RECORDS = [
    AgentModelRecord(
        agent_name="ValuationAgent",
        model_version="rules_v1",
        model_type="rules",
        model_path=None,
        created_at="2024-01-01T00:00:00",
        performance={"ic_mean": 0.030, "accuracy": 0.55},
        is_active=True,
        upgrade_notes="规则基线：PE/PB阈值判断",
    ),
    AgentModelRecord(
        agent_name="MomentumAgent",
        model_version="rules_v1",
        model_type="rules",
        model_path=None,
        created_at="2024-01-01T00:00:00",
        performance={"ic_mean": 0.045, "accuracy": 0.58},
        is_active=True,
        upgrade_notes="规则基线：MA趋势+RSI",
    ),
    AgentModelRecord(
        agent_name="QualityAgent",
        model_version="rules_v1",
        model_type="rules",
        model_path=None,
        created_at="2024-01-01T00:00:00",
        performance={"ic_mean": 0.038, "accuracy": 0.56},
        is_active=True,
        upgrade_notes="规则基线：ROE/毛利率阈值",
    ),
    AgentModelRecord(
        agent_name="SentimentAgent",
        model_version="rules_v1",
        model_type="rules",
        model_path=None,
        created_at="2024-01-01T00:00:00",
        performance={"ic_mean": 0.020, "accuracy": 0.52},
        is_active=True,
        upgrade_notes="规则基线：关键词匹配+可选LLM打分",
    ),
    AgentModelRecord(
        agent_name="RiskAgent",
        model_version="rules_v1",
        model_type="rules",
        model_path=None,
        created_at="2024-01-01T00:00:00",
        performance={"ic_mean": 0.025, "accuracy": 0.53},
        is_active=True,
        upgrade_notes="规则基线：波动率/杠杆阈值",
    ),
    AgentModelRecord(
        agent_name="StrategyAgent",
        model_version="rules_v1",
        model_type="rules",
        model_path=None,
        created_at="2024-01-01T00:00:00",
        performance={"ic_mean": 0.035, "accuracy": 0.55},
        is_active=True,
        upgrade_notes="规则基线：规则综合+LLM生成策略文本",
    ),
]


def initialize_default_registry(force: bool = False) -> AgentModelRegistry:
    """初始化注册表，写入各 Agent 的 rules_v1 基线记录（若未存在）."""
    registry = AgentModelRegistry()
    for rec in _BASELINE_RECORDS:
        existing = registry.get_history(rec.agent_name)
        if not existing or force:
            registry.register(rec)
    return registry
