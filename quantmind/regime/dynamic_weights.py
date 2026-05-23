"""quantmind/regime/dynamic_weights.py

Regime 感知的动态权重管理器。

背景
----
HMM 将市场划分为 bull / neutral / bear 三种状态。
各状态下因子 IC 差异显著（基于 alpha_panel_v4 实证，28季）：

  bull   : quality IC=+0.034  momentum IC=-0.033  value IC=+0.019
  neutral: quality IC=+0.011  momentum IC=-0.031  value IC=+0.008
  bear   : quality IC=-0.018  momentum IC=-0.015  value IC=+0.023

结论：
  - Bull regime：质量因子最有效 → CNN（擅长质量特征）权重上调
  - Bear regime：价值因子相对最有效 → 防御模式，LGBM 主导
  - 动量因子全 regime 为负（A股均值回归效应）→ 保守配置

管理三类权重
------------
1. System2 四维权重（value / momentum / quality / technical）
2. Model ensemble 权重（LGBM vs FactorCNN）
3. Factor 补充权重（EM 因子、文本因子叠加强度）

热更新
------
可通过 ``update_from_config()`` 从 strategy_config_v2.json 读取最新权重，
无需重启流水线。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ─── 默认权重 Schema ───────────────────────────────────────────────────────────

_DEFAULT_WEIGHT_SCHEMA: Dict[str, Dict] = {
    "bull": {
        # System2 四维权重（IC 校准 + quality 偏高）
        "system2": {
            "value": 0.242,
            "momentum": 0.223,
            "quality": 0.333,
            "technical": 0.202,
        },
        # Ensemble 权重：bull 下 CNN 权重最高（质量模式识别能力强）
        "ensemble": {
            "lgbm": 0.65,
            "cnn": 0.35,
        },
        # 文本反转因子权重（bull 情绪高涨时反转信号更弱）
        "text_factor": 0.04,
        # EM 基本面质量因子权重（bull 下质量溢价最高）
        "em_factor": 0.25,
    },
    "neutral": {
        "system2": {
            "value": 0.280,
            "momentum": 0.220,
            "quality": 0.300,
            "technical": 0.200,
        },
        "ensemble": {
            "lgbm": 0.70,
            "cnn": 0.30,
        },
        "text_factor": 0.05,
        "em_factor": 0.20,
    },
    "bear": {
        # Bear 下价值因子 IC 最高 → 提高 value 权重，降低 momentum
        "system2": {
            "value": 0.350,
            "momentum": 0.150,
            "quality": 0.280,
            "technical": 0.220,
        },
        # 防御模式：LGBM 更稳健，CNN 权重最低
        "ensemble": {
            "lgbm": 0.80,
            "cnn": 0.20,
        },
        # Bear 下情绪反转信号弱（恐慌持续），文本因子权重低
        "text_factor": 0.03,
        # EM 质量因子在 bear 下 IC 为负，降至最低
        "em_factor": 0.15,
    },
}

# 有效 regime 标签
VALID_REGIMES = frozenset({"bull", "neutral", "bear"})


class DynamicWeightManager:
    """根据 HMM regime 动态调整所有 ensemble 权重。

    Parameters
    ----------
    config_path : str | Path, optional
        strategy_config_v2.json 路径。若提供且文件存在，
        则在实例化时从文件读取权重覆盖默认值。
    auto_reload : bool
        是否在每次 get_weights() 时检查配置文件是否更新（热更新）。
        默认 False（高频调用时设为 True 会增加 IO 开销）。

    Examples
    --------
    >>> mgr = DynamicWeightManager()
    >>> w = mgr.get_weights("bull")
    >>> w["ensemble"]["cnn"]
    0.35
    >>> mgr.log_current_weights("bull")
    """

    # 完整 schema 作为类属性（允许子类覆盖）
    WEIGHT_SCHEMA: Dict[str, Dict] = _DEFAULT_WEIGHT_SCHEMA

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        auto_reload: bool = False,
    ) -> None:
        import copy
        self._schema: Dict[str, Dict] = copy.deepcopy(self.WEIGHT_SCHEMA)
        self._config_path: Optional[Path] = (
            Path(config_path) if config_path is not None else None
        )
        self._auto_reload = auto_reload
        self._last_mtime: float = 0.0

        if self._config_path is not None and self._config_path.exists():
            self.update_from_config()

    # ── 主要接口 ────────────────────────────────────────────────────────────────

    def get_weights(self, regime: str) -> Dict:
        """返回指定 regime 的完整权重字典（深拷贝，修改不影响内部状态）。

        Parameters
        ----------
        regime : str
            'bull' | 'neutral' | 'bear'（大小写不敏感）

        Returns
        -------
        dict with keys: 'system2', 'ensemble', 'text_factor', 'em_factor'
        """
        import copy
        if self._auto_reload:
            self._maybe_reload()

        regime_lower = regime.lower()
        if regime_lower not in VALID_REGIMES:
            log.warning(
                "get_weights: 未知 regime '%s'，降级为 'neutral'", regime
            )
            regime_lower = "neutral"

        return copy.deepcopy(self._schema[regime_lower])

    def get_system2_weights(self, regime: str) -> Dict[str, float]:
        """便捷接口：仅返回 System2 四维权重。"""
        return self.get_weights(regime)["system2"]

    def get_ensemble_weights(self, regime: str) -> Dict[str, float]:
        """便捷接口：仅返回 ensemble 权重 {'lgbm': ..., 'cnn': ...}。"""
        return self.get_weights(regime)["ensemble"]

    def get_em_weight(self, regime: str) -> float:
        """便捷接口：返回 EM 基本面因子权重。"""
        return float(self.get_weights(regime)["em_factor"])

    def get_text_weight(self, regime: str) -> float:
        """便捷接口：返回文本情绪因子权重。"""
        return float(self.get_weights(regime)["text_factor"])

    # ── 热更新 ──────────────────────────────────────────────────────────────────

    def update_from_config(self) -> bool:
        """从 strategy_config_v2.json 读取权重并覆盖当前 schema。

        JSON 结构示例（仅覆盖存在的字段）::

            {
              "regime_weights": {
                "bull": {
                  "ensemble": {"lgbm": 0.60, "cnn": 0.40},
                  "em_factor": 0.30
                }
              }
            }

        Returns
        -------
        bool: True if config was loaded and schema updated; False otherwise.
        """
        if self._config_path is None or not self._config_path.exists():
            return False

        try:
            with open(self._config_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("update_from_config: 读取配置失败 — %s", e)
            return False

        self._last_mtime = self._config_path.stat().st_mtime

        regime_cfg = cfg.get("regime_weights", {})
        if not regime_cfg:
            log.debug("update_from_config: 配置文件中无 'regime_weights' 字段")
            return False

        for regime, overrides in regime_cfg.items():
            if regime not in VALID_REGIMES:
                continue
            for key, val in overrides.items():
                if key in self._schema[regime]:
                    if isinstance(val, dict):
                        self._schema[regime][key].update(val)
                    else:
                        self._schema[regime][key] = val

        self._validate_schema()
        log.info("update_from_config: 权重已从 %s 更新", self._config_path)
        return True

    def _maybe_reload(self) -> None:
        """如文件修改时间有变化，重新加载配置。"""
        if self._config_path is None or not self._config_path.exists():
            return
        mtime = self._config_path.stat().st_mtime
        if mtime > self._last_mtime:
            self.update_from_config()

    # ── 验证 ─────────────────────────────────────────────────────────────────────

    def _validate_schema(self) -> None:
        """校验权重合法性（每组和为 1，值在 [0,1]）。"""
        for regime in VALID_REGIMES:
            s2 = self._schema[regime]["system2"]
            total = sum(s2.values())
            if abs(total - 1.0) > 1e-6:
                log.warning(
                    "_validate_schema: %s system2 权重和=%.4f，非 1.0，已归一化",
                    regime, total,
                )
                for k in s2:
                    s2[k] /= total

            ens = self._schema[regime]["ensemble"]
            ens_total = sum(ens.values())
            if abs(ens_total - 1.0) > 1e-6:
                for k in ens:
                    ens[k] /= ens_total

    # ── 日志 ────────────────────────────────────────────────────────────────────

    def log_current_weights(self, regime: Optional[str] = None) -> None:
        """打印当前权重到 logger（INFO 级别）。"""
        regimes = [regime] if regime else list(VALID_REGIMES)
        for r in regimes:
            w = self.get_weights(r)
            s2 = w["system2"]
            ens = w["ensemble"]
            log.info(
                "[DynamicWeightManager] %s | "
                "system2=[val=%.3f mom=%.3f qual=%.3f tech=%.3f] | "
                "ensemble=[lgbm=%.2f cnn=%.2f] | "
                "em=%.2f text=%.2f",
                r.upper(),
                s2["value"], s2["momentum"], s2["quality"], s2["technical"],
                ens["lgbm"], ens["cnn"],
                w["em_factor"], w["text_factor"],
            )

    def to_dict(self) -> Dict:
        """导出当前完整 schema 为 dict（用于序列化 / 前端展示）。"""
        import copy
        return copy.deepcopy(self._schema)

    def summary_dataframe(self):
        """返回 pd.DataFrame 格式的权重汇总，便于 Streamlit 展示。"""
        import pandas as pd

        rows = []
        for regime in ["bull", "neutral", "bear"]:
            w = self._schema[regime]
            s2 = w["system2"]
            ens = w["ensemble"]
            rows.append({
                "Regime": regime.upper(),
                "Value": round(s2["value"], 3),
                "Momentum": round(s2["momentum"], 3),
                "Quality": round(s2["quality"], 3),
                "Technical": round(s2["technical"], 3),
                "LGBM": round(ens["lgbm"], 3),
                "CNN": round(ens["cnn"], 3),
                "EM Factor": round(w["em_factor"], 3),
                "Text Factor": round(w["text_factor"], 3),
            })
        return pd.DataFrame(rows).set_index("Regime")

    def __repr__(self) -> str:
        lines = [f"DynamicWeightManager(config={self._config_path})"]
        for regime in ["bull", "neutral", "bear"]:
            ens = self._schema[regime]["ensemble"]
            lines.append(
                f"  {regime}: lgbm={ens['lgbm']:.2f} cnn={ens['cnn']:.2f} "
                f"em={self._schema[regime]['em_factor']:.2f}"
            )
        return "\n".join(lines)


__all__ = [
    "DynamicWeightManager",
    "VALID_REGIMES",
]
