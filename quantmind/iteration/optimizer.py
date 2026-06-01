"""quantmind/iteration/optimizer.py — 基于诊断结果生成/应用参数调整建议.

ParameterOptimizer
  .generate_suggestions(diag) → list[ParameterSuggestion]
  .apply_suggestions(suggestions, config_path) → 更新后的配置 dict

建议范围
--------
- system2_updates.weights_calibrated    — 因子权重调整
- holding_period.recommended            — 推荐持仓周期
- system1_updates.top_n                 — 初筛候选数
- layer6_overweight.overweight_industries — 行业超配列表
- barra_constrained                     — 是否启用 Barra 约束
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = _ROOT / "data" / "paper_trading" / "strategy_config_v2.json"
WEIGHT_SUM_TOL = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# ParameterSuggestion
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParameterSuggestion:
    """单条参数调整建议."""

    param_path:  str          # JSON 路径，eg. "system2_updates.weights_calibrated.value"
    old_value:   Any          = None
    new_value:   Any          = None
    reason:      str          = ""
    confidence:  float        = 0.5   # [0, 1]
    category:    str          = "weight"  # weight / horizon / universe / risk

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_path": self.param_path,
            "old_value":  self.old_value,
            "new_value":  self.new_value,
            "reason":     self.reason,
            "confidence": self.confidence,
            "category":   self.category,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParameterSuggestion":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────────
# ParameterOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class ParameterOptimizer:
    """根据 SimDiagnosis 生成参数调整建议，并可一键写回配置文件.

    Parameters
    ----------
    config_path : Path | str | None
        strategy_config_v2.json 路径，默认使用项目标准路径。
    weight_step : float
        单次权重调整步长（占 [0,1] 区间），默认 0.05。
    max_weight_change : float
        单次权重变动上限，默认 0.15（防止调参过激）。
    value_floor : float
        value 权重下限，默认 0.15。
        即使 IC 为负，value 因子在 bear 市场仍有防御价值，
        不应完全压至 0。超出下限的部分由其他因子按比例承担。
    """

    WEIGHT_KEYS = ["value", "momentum", "quality", "technical"]

    # 各因子权重下限（key = WEIGHT_KEYS 中的名称）
    WEIGHT_FLOORS: dict[str, float] = {"value": 0.15}

    def __init__(
        self,
        config_path: Path | str | None = None,
        weight_step: float = 0.05,
        max_weight_change: float = 0.15,
        value_floor: float = 0.15,
    ) -> None:
        self._config_path    = Path(config_path) if config_path else CONFIG_PATH
        self._weight_step    = weight_step
        self._max_wt_change  = max_weight_change
        self._value_floor    = float(value_floor)
        # 运行时下限 dict（允许子类/测试覆盖）
        self._floors: dict[str, float] = {"value": self._value_floor}

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def generate_suggestions(
        self,
        diag: "SimDiagnosis",  # noqa: F821
    ) -> list[ParameterSuggestion]:
        """基于 SimDiagnosis 生成建议列表."""
        suggestions: list[ParameterSuggestion] = []

        config   = self._load_config()
        cur_weights = (
            config.get("system2_updates", {})
            .get("weights_calibrated", {})
        )
        if not cur_weights:
            cur_weights = {k: 0.25 for k in self.WEIGHT_KEYS}

        # 1. 因子权重调整（基于 IC_3m）
        ic3 = diag.ic_summary.get("ic_3m", {})
        weight_suggestions = self._suggest_weight_adjustments(
            ic3, cur_weights, diag
        )
        suggestions.extend(weight_suggestions)

        # 2. 持仓周期建议
        horizon_sugg = self._suggest_holding_period(diag, config)
        if horizon_sugg:
            suggestions.append(horizon_sugg)

        # 3. 行业超配建议
        if diag.top_industries:
            industry_sugg = self._suggest_industry_overweight(
                diag.top_industries, config
            )
            if industry_sugg:
                suggestions.append(industry_sugg)

        # 4. 风险管控
        if diag.loss_pct_gt5 > 0.3:
            suggestions.append(ParameterSuggestion(
                param_path = "barra_constrained",
                old_value  = config.get("barra_constrained", False),
                new_value  = True,
                reason     = (
                    f"3m 亏损>5% 占比 {diag.loss_pct_gt5:.1%}，"
                    f"建议启用 Barra 风险约束以控制下行风险"
                ),
                confidence = 0.65,
                category   = "risk",
            ))

        return suggestions

    def apply_suggestions(
        self,
        suggestions: list[ParameterSuggestion],
        config_path: Path | str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """将建议应用到 strategy_config_v2.json.

        Parameters
        ----------
        dry_run : 若 True，只返回更新后的配置 dict，不写磁盘。

        Returns
        -------
        更新后的完整配置 dict。
        """
        path   = Path(config_path) if config_path else self._config_path
        config = self._load_config(path)

        applied: list[str] = []
        for sugg in suggestions:
            try:
                self._set_nested(config, sugg.param_path, sugg.new_value)
                applied.append(sugg.param_path)
                log.info(
                    "[Optimizer] 应用建议: %s  %s → %s",
                    sugg.param_path, sugg.old_value, sugg.new_value,
                )
            except Exception as e:
                log.warning("[Optimizer] 应用失败(%s): %s", sugg.param_path, e)

        # 记录本次迭代元信息
        config.setdefault("iteration_history", []).append({
            "applied_at":    datetime.now().isoformat(timespec="seconds"),
            "applied_params": applied,
            "n_suggestions": len(suggestions),
        })
        config["updated_at"] = datetime.now().isoformat()

        if not dry_run:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(config, fh, ensure_ascii=False, indent=2)
            log.info("[Optimizer] 配置已写回 %s", path)

        return config

    # ── 私有：建议生成逻辑 ────────────────────────────────────────────────────

    def _suggest_weight_adjustments(
        self,
        ic3:         dict[str, float],
        cur_weights: dict[str, float],
        diag:        "SimDiagnosis",
    ) -> list[ParameterSuggestion]:
        """基于 IC_3m 调整四因子权重，保持权重和 = 1."""
        # 只处理四个核心权重维度
        factor_to_weight = {
            "value_score":     "value",
            "momentum_score":  "momentum",
            "quality_score":   "quality",
            "technical_score": "technical",
        }

        # 计算目标权重（归一化 max(IC, 0) 作为软目标）
        ic_vals: dict[str, float] = {}
        for fc, wk in factor_to_weight.items():
            v = ic3.get(fc, float("nan"))
            ic_vals[wk] = v if not np.isnan(v) else 0.0

        # 若所有 IC 均 ≤ 0，用均等权重
        pos_sum = sum(max(v, 0) for v in ic_vals.values())
        if pos_sum < 1e-9:
            target_weights = {k: 0.25 for k in self.WEIGHT_KEYS}
        else:
            raw_target = {k: max(ic_vals.get(k, 0), 0) for k in self.WEIGHT_KEYS}
            total = sum(raw_target.values())
            target_weights = {k: v / total for k, v in raw_target.items()}

        # 对每个权重做步长限制 clamp
        new_weights: dict[str, float] = {}
        for k in self.WEIGHT_KEYS:
            cur = cur_weights.get(k, 0.25)
            tgt = target_weights.get(k, 0.25)
            delta = np.clip(tgt - cur, -self._max_wt_change, self._max_wt_change)
            new_weights[k] = max(0.0, cur + delta)

        # 第一次归一化（确保和 = 1）
        total = sum(new_weights.values())
        if total > WEIGHT_SUM_TOL:
            new_weights = {k: v / total for k, v in new_weights.items()}

        # ── 下限保护 ─────────────────────────────────────────────────────────
        # value 权重不低于 _value_floor（默认 0.15）。
        # 即使本轮 IC 为负，价值因子在 bear 市场有防御价值，不应完全清零。
        # 超出下限占用的权重空间，由其他因子按原比例均摊（proportional reduce）。
        floored_keys = {
            k for k, fv in self._floors.items()
            if new_weights.get(k, 0.0) < fv
        }
        if floored_keys:
            for k in floored_keys:
                new_weights[k] = self._floors[k]
            floored_total   = sum(new_weights[k] for k in floored_keys)
            remaining_budget = max(0.0, 1.0 - floored_total)
            other_keys       = [k for k in self.WEIGHT_KEYS if k not in floored_keys]
            other_raw_total  = sum(new_weights[k] for k in other_keys)
            if other_raw_total > WEIGHT_SUM_TOL:
                scale = remaining_budget / other_raw_total
                for k in other_keys:
                    new_weights[k] = new_weights[k] * scale
            elif other_keys:
                per_w = remaining_budget / len(other_keys)
                for k in other_keys:
                    new_weights[k] = per_w

        suggestions: list[ParameterSuggestion] = []
        for k in self.WEIGHT_KEYS:
            old = float(round(float(cur_weights.get(k, 0.25)), 4))
            new = float(round(float(new_weights[k]), 4))
            if abs(new - old) < 0.005:
                continue   # 变化太小，跳过
            ic_val = ic_vals.get(k, float("nan"))
            ic_str = f"{ic_val:+.4f}" if not np.isnan(ic_val) else "N/A"
            suggestions.append(ParameterSuggestion(
                param_path = f"system2_updates.weights_calibrated.{k}",
                old_value  = old,
                new_value  = new,
                reason     = (
                    f"{k}_score IC_3m={ic_str}，"
                    f"权重 {old:.3f} → {new:.3f}"
                ),
                confidence = min(0.9, 0.5 + abs(ic_val) * 3) if not np.isnan(ic_val) else 0.5,
                category   = "weight",
            ))

        return suggestions

    def _suggest_holding_period(
        self,
        diag:   "SimDiagnosis",
        config: dict[str, Any],
    ) -> ParameterSuggestion | None:
        """若短期 IR 均负而 3m IR 正，建议持仓 3m."""
        cur_period = config.get("holding_period", {}).get("recommended", "")
        if diag.ir_1w < 0 and diag.ir_2w < 0 and diag.ir_3m > 0:
            if cur_period != "3m":
                return ParameterSuggestion(
                    param_path = "holding_period.recommended",
                    old_value  = cur_period,
                    new_value  = "3m",
                    reason     = (
                        f"1w IR={diag.ir_1w:+.3f}，2w IR={diag.ir_2w:+.3f}，"
                        f"3m IR={diag.ir_3m:+.3f}，推荐持仓调整为 3m"
                    ),
                    confidence = 0.8,
                    category   = "horizon",
                )
        return None

    def _suggest_industry_overweight(
        self,
        top_industries: list[dict],
        config:         dict[str, Any],
    ) -> ParameterSuggestion | None:
        """将 top3 行业加入 layer6_overweight."""
        new_industries = [
            ind["industry"] for ind in top_industries[:3]
            if ind.get("ret_3m_mean", 0) > 0.1  # 只有 3m 均值 > 10% 才超配
        ]
        if not new_industries:
            return None
        cur = config.get("layer6_overweight", {}).get("overweight_industries", [])
        if set(new_industries) == set(cur[:3]):
            return None
        return ParameterSuggestion(
            param_path = "layer6_overweight.overweight_industries",
            old_value  = cur,
            new_value  = new_industries,
            reason     = (
                f"当前轮 Top 行业: {new_industries}，3m 超额收益显著，"
                f"建议 Layer6 超配"
            ),
            confidence = 0.7,
            category   = "universe",
        )

    # ── 私有：配置 IO ─────────────────────────────────────────────────────────

    def _load_config(self, path: Path | None = None) -> dict[str, Any]:
        p = path or self._config_path
        if not p.exists():
            log.warning("[Optimizer] 配置文件不存在: %s，使用空配置", p)
            return {}
        try:
            with p.open(encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            log.warning("[Optimizer] 配置读取失败: %s", e)
            return {}

    @staticmethod
    def _set_nested(obj: dict, path: str, value: Any) -> None:
        """按 "a.b.c" 路径设置嵌套 dict 中的值，路径不存在时自动创建."""
        keys = path.split(".")
        for key in keys[:-1]:
            obj = obj.setdefault(key, {})
        obj[keys[-1]] = value
