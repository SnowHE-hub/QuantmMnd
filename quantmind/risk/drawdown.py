"""quantmind.risk.drawdown — 动态回撤风控与仓位调整.

参考文献：
  [1] Grossman, S.J. & Zhou, Z. (1993).
      "Optimal Investment Strategies for Controlling Drawdowns."
      Mathematical Finance. （最优回撤控制）
  [2] Perold, A.F. & Sharpe, W.F. (1988).
      "Dynamic Strategies for Asset Allocation." FAJ.
      （CPPI 恒定比例组合保险）
  [3] Meucci, A. (2009).
      "Risk and Asset Allocation." Springer.
      （目标波动率框架）

规则层级（DrawdownController）：
  回撤阈值  │ 目标仓位比例
  ──────────┼───────────────
  < 10%     │ 100%（满仓）
  10%~20%   │ 70%（降仓）
  20%~30%   │ 40%（大幅降仓）
  > 30%     │ 0%（清仓）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

__all__ = ["DrawdownController", "DrawdownRule"]


@dataclass
class DrawdownRule:
    """单条回撤控制规则."""
    threshold: float   # 触发阈值（如 0.10 = -10%）
    target_pct: float  # 目标仓位比例（如 0.70 = 70%）

    def __post_init__(self) -> None:
        assert 0 <= self.threshold <= 1, "阈值应在 [0, 1]"
        assert 0 <= self.target_pct <= 1, "目标仓位应在 [0, 1]"


class DrawdownController:
    """动态回撤风控器.

    功能：
    1. check_and_adjust()   — 根据当前回撤动态调整目标仓位
    2. volatility_targeting() — 目标波动率对应的杠杆/仓位系数
    3. cppi()               — CPPI 安全垫计算

    Args:
        rules:   回撤控制规则列表（按阈值升序，将自动排序）
        verbose: 是否打印风控日志
    """

    # 默认规则（与 Spec 对齐）
    _DEFAULT_RULES = [
        DrawdownRule(threshold=0.10, target_pct=0.70),  # 回撤 >10% → 70%
        DrawdownRule(threshold=0.20, target_pct=0.40),  # 回撤 >20% → 40%
        DrawdownRule(threshold=0.30, target_pct=0.00),  # 回撤 >30% → 清仓
    ]

    def __init__(
        self,
        rules: list[DrawdownRule] | None = None,
        verbose: bool = True,
    ) -> None:
        self.rules: list[DrawdownRule] = sorted(
            rules or self._DEFAULT_RULES,
            key=lambda r: r.threshold,
            reverse=True,   # 从大到小，先匹配最严格规则
        )
        self.verbose = verbose
        self._last_position_pct: float = 1.0

    # ── 1. 回撤检查与仓位调整 ─────────────────────────────────────────────────

    def check_and_adjust(
        self,
        current_drawdown: float,
        current_nav: float | None = None,
        peak_nav: float | None = None,
    ) -> float:
        """根据当前回撤返回目标仓位比例.

        若同时传入 current_nav 和 peak_nav，则自动计算回撤：
            drawdown = (peak_nav - current_nav) / peak_nav

        Args:
            current_drawdown: 当前回撤（正数表示亏损，如 0.15 = 回撤 15%）
                               注意：通常回撤为负（-0.15），此处统一取绝对值
            current_nav:      当前净值（可选，与 peak_nav 配合使用）
            peak_nav:         历史最高净值（可选）

        Returns:
            目标仓位比例 [0, 1]（1.0 = 满仓，0.0 = 空仓）
        """
        # 若传入 nav，则重新计算回撤
        if current_nav is not None and peak_nav is not None and peak_nav > 0:
            current_drawdown = (peak_nav - current_nav) / peak_nav

        # 统一为正数
        dd = abs(float(current_drawdown))

        # 匹配规则（规则已按阈值降序排列）
        target_pct = 1.0  # 默认满仓
        triggered_rule: DrawdownRule | None = None

        for rule in self.rules:
            if dd > rule.threshold:
                target_pct = rule.target_pct
                triggered_rule = rule
                break  # 匹配最严格（阈值最高）的触发规则

        if self.verbose and triggered_rule is not None:
            status = "清仓" if target_pct == 0 else f"降仓至 {target_pct:.0%}"
            logger.warning(
                f"[DrawdownController] 回撤 {dd:.2%} > {triggered_rule.threshold:.0%}，"
                f"{status}"
            )
        elif self.verbose and abs(target_pct - self._last_position_pct) > 0.01:
            logger.info(
                f"[DrawdownController] 回撤 {dd:.2%} 恢复，仓位恢复至 {target_pct:.0%}"
            )

        self._last_position_pct = target_pct
        return target_pct

    # ── 2. 目标波动率仓位系数 ─────────────────────────────────────────────────

    @staticmethod
    def volatility_targeting(
        returns: pd.Series | np.ndarray,
        target_vol: float = 0.10,
        lookback: int = 63,
        min_leverage: float = 0.0,
        max_leverage: float = 2.0,
    ) -> float:
        """目标波动率对应的杠杆系数.

        参考文献: Meucci (2009).

        公式：
            σ_realized = std(r) × √252  （年化实现波动率）
            leverage = target_vol / σ_realized
            leverage = clip(leverage, min_leverage, max_leverage)

        例如：实现波动率 20%，目标 10%，则杠杆 = 0.5（半仓）
              实现波动率 5%，目标 10%，则杠杆 = 2.0（上限）

        Args:
            returns:      日收益率序列
            target_vol:   目标年化波动率（默认 10%）
            lookback:     计算波动率的回看窗口（交易日）
            min_leverage: 最小杠杆（默认 0，不允许做空）
            max_leverage: 最大杠杆（默认 2x）

        Returns:
            仓位系数（杠杆）[min_leverage, max_leverage]
        """
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]

        if len(r) < 5:
            logger.warning("[VolTargeting] 数据不足，返回 1.0（不调整）")
            return 1.0

        # 使用最近 lookback 期
        r = r[-lookback:]
        realized_vol = float(np.std(r, ddof=1) * np.sqrt(252))

        if realized_vol <= 0:
            return max_leverage

        leverage = target_vol / realized_vol
        leverage = float(np.clip(leverage, min_leverage, max_leverage))

        logger.debug(
            f"[VolTargeting] 实现波动率={realized_vol:.2%}，"
            f"目标={target_vol:.2%}，杠杆={leverage:.3f}"
        )
        return leverage

    # ── 3. CPPI（恒定比例组合保险）─────────────────────────────────────────────

    @staticmethod
    def cppi(
        nav: float,
        floor: float,
        multiplier: float = 3.0,
        risky_pct: float | None = None,
    ) -> float:
        """CPPI（Constant Proportion Portfolio Insurance）安全垫计算.

        参考文献: Perold & Sharpe (1988).

        公式：
            Cushion  = NAV - Floor    （安全垫：资产净值超过保底的部分）
            Exposure = m × Cushion    （风险资产敞口 = 乘数 × 安全垫）
            Risky%   = Exposure / NAV （风险资产占组合比例）

        当 NAV <= Floor 时，全部转为安全资产（Risky% = 0）。

        例子：NAV=100，Floor=80，m=3
              Cushion=20，Exposure=60，Risky%=60%
              安全资产=40%（保证即便风险资产归零，NAV不低于 Floor）

        Args:
            nav:        当前净值
            floor:      保底净值（通常为初始净值的 80%-90%）
            multiplier: 乘数 m（越大越激进，通常 2~5）
            risky_pct:  已有的风险资产比例（若提供则返回调整后的目标比例）

        Returns:
            风险资产目标占比 [0, 1]
        """
        if nav <= 0:
            return 0.0

        if nav <= floor:
            logger.warning(
                f"[CPPI] NAV={nav:.2f} <= Floor={floor:.2f}，触发保底，清空风险资产"
            )
            return 0.0

        cushion = nav - floor
        exposure = multiplier * cushion
        risky_weight = float(np.clip(exposure / nav, 0.0, 1.0))

        logger.debug(
            f"[CPPI] NAV={nav:.2f}，Floor={floor:.2f}，"
            f"Cushion={cushion:.2f}，m={multiplier}，"
            f"风险资产={risky_weight:.2%}"
        )
        return risky_weight

    # ── 4. 组合回撤计算工具 ────────────────────────────────────────────────────

    @staticmethod
    def compute_max_drawdown(nav_series: pd.Series) -> float:
        """计算最大回撤（负数）.

        Args:
            nav_series: 净值序列

        Returns:
            最大回撤（如 -0.25 = -25%）
        """
        if nav_series.empty or len(nav_series) < 2:
            return 0.0
        peak = nav_series.expanding().max()
        drawdown = (nav_series - peak) / peak
        return float(drawdown.min())

    @staticmethod
    def compute_current_drawdown(nav_series: pd.Series) -> float:
        """计算当前（最新）回撤（负数）.

        Args:
            nav_series: 净值序列

        Returns:
            当前回撤（如 -0.10 = -10%）
        """
        if nav_series.empty:
            return 0.0
        peak = float(nav_series.max())
        current = float(nav_series.iloc[-1])
        if peak <= 0:
            return 0.0
        return (current - peak) / peak

    @staticmethod
    def compute_drawdown_series(nav_series: pd.Series) -> pd.Series:
        """计算逐期回撤序列（负数）."""
        if nav_series.empty:
            return pd.Series(dtype=float)
        peak = nav_series.expanding().max()
        return (nav_series - peak) / peak

    # ── 5. 组合风险汇总报告 ────────────────────────────────────────────────────

    def risk_report(
        self,
        nav_series: pd.Series,
        target_vol: float = 0.10,
        floor_pct: float = 0.85,
        cppi_multiplier: float = 3.0,
    ) -> dict[str, Any]:
        """生成当前时点的风险控制建议报告.

        Args:
            nav_series:       历史净值序列
            target_vol:       目标波动率
            floor_pct:        CPPI 保底比例（相对于初始净值）
            cppi_multiplier:  CPPI 乘数

        Returns:
            dict: 包含各风控指标和建议仓位
        """
        if nav_series.empty:
            return {"error": "净值序列为空"}

        # 计算收益序列
        returns = nav_series.pct_change().dropna()
        current_nav = float(nav_series.iloc[-1])
        peak_nav = float(nav_series.max())
        initial_nav = float(nav_series.iloc[0])

        # 当前回撤
        current_dd = self.compute_current_drawdown(nav_series)
        max_dd = self.compute_max_drawdown(nav_series)

        # 各风控方法的仓位建议
        dd_position = self.check_and_adjust(abs(current_dd))
        vol_leverage = self.volatility_targeting(returns, target_vol=target_vol)
        floor = initial_nav * floor_pct
        cppi_position = self.cppi(current_nav, floor=floor, multiplier=cppi_multiplier)

        # 综合建议：取最保守（最小）的仓位
        conservative_position = float(min(dd_position, vol_leverage, cppi_position))

        return {
            "current_nav": current_nav,
            "peak_nav": peak_nav,
            "current_drawdown": current_dd,
            "max_drawdown": max_dd,
            "dd_position_pct": dd_position,
            "vol_leverage": vol_leverage,
            "cppi_position_pct": cppi_position,
            "recommended_position": conservative_position,
            "interpretation": (
                f"当前回撤={current_dd:.2%}（历史最大={max_dd:.2%}），"
                f"回撤规则建议仓位={dd_position:.0%}，"
                f"波动率目标杠杆={vol_leverage:.2f}，"
                f"CPPI建议={cppi_position:.2%}，"
                f"综合建议（最保守）={conservative_position:.0%}"
            ),
        }
