"""quantmind.agents.investment_agents.base_agent — 投资分析 Agent 抽象基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from quantmind.agents.investment_agents.agent_registry import (
        AgentModelRecord,
        AgentModelRegistry,
    )

_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class AgentSignal:
    agent_name: str
    ticker: str
    signal: float          # -1.0 到 +1.0（-1=看空，0=中性，+1=看多）
    confidence: float      # 0.0 到 1.0
    summary: str           # 一句话摘要（中文，≤50字）
    evidence: dict = field(default_factory=dict)    # 支撑数据
    warnings: list[str] = field(default_factory=list)  # 风险提示


class BaseInvestmentAgent(ABC):
    """投资分析 Agent 抽象基类.

    与 quantmind.agents.base.BaseAgent 解耦，专门用于投资分析流水线。
    支持 AgentModelRegistry 集成：自动加载当前激活的模型版本。
    """

    # 类级别的注册表（懒加载，所有子类共享）
    _registry: "AgentModelRegistry | None" = None

    def __init__(
        self,
        ticker: str,
        as_of: str,
        context: dict,
        model_version: str = "active",
    ) -> None:
        self.ticker = ticker
        self.as_of = as_of
        self.context = context  # retrieve_stock_context 返回的 JSON

        # 加载注册表中的模型
        self._model_record: AgentModelRecord | None = None
        self._ml_model: Any = None
        self._init_model(model_version)

    # ── 注册表集成 ────────────────────────────────────────────────────────────

    @classmethod
    def _get_registry(cls) -> "AgentModelRegistry":
        if cls._registry is None:
            try:
                from quantmind.agents.investment_agents.agent_registry import (
                    get_default_registry,
                )
                cls._registry = get_default_registry()
            except Exception as e:
                logger.debug(f"[BaseAgent] 注册表加载失败: {e}")
                cls._registry = None  # type: ignore
        return cls._registry  # type: ignore

    def _init_model(self, model_version: str = "active") -> None:
        """从注册表加载指定版本的模型."""
        try:
            registry = self._get_registry()
            if registry is None:
                return
            agent_name = self.__class__.__name__
            if model_version == "active":
                record = registry.get_active(agent_name)
            else:
                history = registry.get_history(agent_name)
                record = next((r for r in history if r.model_version == model_version), None)
            if record is None:
                return
            self._model_record = record
            self._ml_model = self._load_model(record)
        except Exception as e:
            logger.debug(f"[BaseAgent] 模型初始化失败: {e}")

    def _load_model(self, record: "AgentModelRecord") -> Any:
        """根据 model_type 加载不同类型的模型."""
        if record is None or record.model_type == "rules":
            return None

        model_path = record.model_path
        if not model_path:
            return None

        path = Path(model_path)
        if not path.is_absolute():
            path = _ROOT / path
        if not path.exists():
            logger.warning(f"[BaseAgent] 模型文件不存在: {path}，降级到规则模式")
            return None

        try:
            if record.model_type == "ml":
                import pickle
                with open(path, "rb") as f:
                    return pickle.load(f)
            elif record.model_type == "dl":
                import torch  # type: ignore
                return torch.load(path, map_location="cpu")
            elif record.model_type == "llm":
                from quantmind.agents.llm_client import build_client
                return build_client(provider="dashscope", model="qwen-plus")
        except Exception as e:
            logger.warning(f"[BaseAgent] 加载模型失败 ({path}): {e}，降级到规则模式")
            return None

    def record_performance(self, actual_return: float, predicted_signal: float | None = None) -> None:
        """事后记录预测效果（用于持续优化）.

        每季度末调用，将本季度预测和实际收益写回注册表。
        """
        try:
            registry = self._get_registry()
            if registry is None or self._model_record is None:
                return
            metrics = {
                "last_actual_return": actual_return,
                "last_predicted_signal": predicted_signal,
                "last_evaluated_at": self.as_of,
            }
            registry.update_performance(
                self.__class__.__name__,
                self._model_record.model_version,
                metrics,
            )
        except Exception as e:
            logger.debug(f"[BaseAgent] 记录性能失败: {e}")

    # ── 抽象接口 ─────────────────────────────────────────────────────────────

    @abstractmethod
    def analyze(self) -> AgentSignal:
        """执行分析，返回 AgentSignal."""

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    def _safe_float(self, val, default=None):
        """安全转换为 float，失败返回 default."""
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    def _pct_to_float(self, val, default=None) -> float | None:
        """将百分比值规范化为小数（自动识别是否已是小数）."""
        v = self._safe_float(val, default)
        if v is None:
            return default
        if abs(v) > 1.5:
            return v / 100.0
        return v

    def _clamp(self, val: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, val))

    def _extract_snapshot_field(
        self,
        snapshot_key: str,
        field_name: str,
        default=None,
    ):
        """从 context[snapshot_key] 列表中提取最新条目的指定字段."""
        items = self.context.get(snapshot_key, [])
        if not items:
            return default
        first = items[0]
        text = first.get("text", "") if isinstance(first, dict) else str(first)
        meta = first.get("metadata", {}) if isinstance(first, dict) else {}
        if field_name in meta:
            return meta[field_name]
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                k, _, v = line.partition(":")
                if k.strip().lower() == field_name.lower():
                    return v.strip()
        return default

    def _get_snapshot_text(self, snapshot_key: str) -> str:
        """拼接 snapshot 列表中所有条目的 text."""
        items = self.context.get(snapshot_key, [])
        texts = []
        for item in items:
            if isinstance(item, dict):
                texts.append(item.get("text", ""))
            else:
                texts.append(str(item))
        return "\n".join(texts)
