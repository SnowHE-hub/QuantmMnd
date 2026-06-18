"""quantmind.backtest.wf_costs — Walk-forward 闸门成本扩展（A股, T+1）.

依据 ``docs/plans/walkforward_design_plan.md`` §3。**新增配置/逻辑，不改
``execution.py`` 默认行为**。三处扩展：

1. 🔴 **滑点流动性分层（硬条件 H-B，本版必做）**：按 amihud 流动性分位三档
   （大盘 5bp / 中盘 15bp / 小盘 30bp，可配）。幸存者池里中小盘用平 10bp 会
   **低估成本→高估 alpha**（假阳性方向），必须根除。
2. **时变印花税**：2023-08-28 起卖出印花税由 0.1% 减半至 0.05%（按 fill_date 取）。
3. 🔴 **板块×时变涨跌停（硬条件 H-E）**：科创板 688* 20%、创业板 30* **2020-08-24 起** 20%
   （之前 10%）、主板 60*/00* 10%。错阈值会误判可成交 → NAV 失真高估 alpha。

时点（§3.2）：信号 as_of 收盘出；**T+1** 次日开盘入场；持有 12 交易日；as_of+13 开盘出场。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

# ============================================================================
# 时变成本生效日（S0 已核官方公告；如需改动须在此单点修改）
# ============================================================================

#: 印花税减半生效日：2023-08-28 起 0.1% → 0.05%（财政部/税务总局 2023-08-27 公告，28 日起施行）
STAMP_DUTY_HALVING_DATE = pd.Timestamp("2023-08-28")
STAMP_DUTY_BEFORE = 0.001    # 0.1%
STAMP_DUTY_AFTER = 0.0005    # 0.05%

#: 创业板（30*）涨跌停由 10% 改 20% 生效日：2020-08-24（注册制首日）
CHINEXT_20PCT_DATE = pd.Timestamp("2020-08-24")


def stamp_duty_rate(fill_date: date | str | pd.Timestamp) -> float:
    """卖出印花税率（按成交日时变）。2023-08-28 起减半。"""
    d = pd.Timestamp(fill_date).normalize()
    return STAMP_DUTY_AFTER if d >= STAMP_DUTY_HALVING_DATE else STAMP_DUTY_BEFORE


# ============================================================================
# 板块 × 时变涨跌停
# ============================================================================


def board_of(ticker: str) -> str:
    """按代码前缀判板块。"""
    t = str(ticker).split(".")[0]
    if t.startswith("688"):
        return "STAR"          # 科创板
    if t.startswith("8") or t.startswith("4"):
        return "BSE"           # 北交所（30%；本池通常无）
    if t.startswith("30"):
        return "CHINEXT"       # 创业板
    if t.startswith(("60", "00")):
        return "MAIN"          # 沪/深主板
    return "MAIN"


def price_limit_pct(
    ticker: str, fill_date: date | str | pd.Timestamp, *, is_st: bool = False
) -> float:
    """该票该日的涨跌停幅度（小数）。

    - 科创板 688*：20%
    - 创业板 30*：2020-08-24 起 20%，之前 10%
    - 北交所：30%
    - 主板 60*/00*：10%
    - ST：5%（本池无 ST，按需）
    """
    if is_st:
        return 0.05
    d = pd.Timestamp(fill_date).normalize()
    board = board_of(ticker)
    if board == "STAR":
        return 0.20
    if board == "BSE":
        return 0.30
    if board == "CHINEXT":
        return 0.20 if d >= CHINEXT_20PCT_DATE else 0.10
    return 0.10  # MAIN


# ============================================================================
# 滑点流动性分层（硬条件 H-B）
# ============================================================================


@dataclass(frozen=True)
class SlippageTiers:
    """按流动性分位的滑点档位（bp）。事前定，事后不调。"""

    large_bp: float = 5.0     # 流动性最好（amihud 分位低）
    mid_bp: float = 15.0
    small_bp: float = 30.0    # 流动性最差（amihud 分位高 = 越不流动）
    # amihud 分位阈值：< q_large → 大盘档；>= q_small → 小盘档；中间 → 中盘档
    q_large: float = 1.0 / 3.0
    q_small: float = 2.0 / 3.0

    def bp_for_quantile(self, amihud_quantile: float) -> float:
        """amihud 分位（0=最流动, 1=最不流动）→ 滑点 bp。"""
        q = float(amihud_quantile)
        if q != q:  # NaN
            return self.small_bp  # 缺流动性信息按最保守（最贵）处理，绝不低估成本
        if q < self.q_large:
            return self.large_bp
        if q >= self.q_small:
            return self.small_bp
        return self.mid_bp


def amihud_to_quantile(amihud: pd.Series) -> pd.Series:
    """把一个截面的 amihud_illiquidity 值映射到 [0,1] 横截面分位（rank/百分位）。

    amihud 越大 = 越不流动 → 分位越接近 1（小盘档）。NaN 保留 NaN。
    """
    return amihud.rank(pct=True, na_option="keep")


def slippage_bp_series(
    amihud_cross_section: pd.Series, tiers: SlippageTiers | None = None
) -> pd.Series:
    """一个截面内每只票的滑点 bp（按 amihud 分位三档）。"""
    tiers = tiers or SlippageTiers()
    q = amihud_to_quantile(amihud_cross_section)
    return q.apply(tiers.bp_for_quantile).rename("slippage_bp")


# ============================================================================
# T+1 入场/出场时点
# ============================================================================


def entry_fill_index(as_of_idx: int) -> int:
    """信号 as_of 收盘出 → **T+1** 次日开盘入场。返回成交日的交易日索引。"""
    return as_of_idx + 1


def exit_fill_index(as_of_idx: int, holding_td: int = 12) -> int:
    """持有 ``holding_td`` 个交易日 → as_of+1+12 的开盘出场（与入场口径一致）。"""
    return as_of_idx + 1 + holding_td


@dataclass
class WFCostModel:
    """打包闸门成本：滑点分层 + 时变印花 + 板块涨跌停。供 gate/engine 桥接调用。"""

    slippage_tiers: SlippageTiers = field(default_factory=SlippageTiers)
    commission_bp: float = 3.0          # 万3 双边（复用既有口径）
    transfer_fee_bp_sh: float = 0.2     # 沪市过户费万0.2 双边
    holding_td: int = 12

    def stamp_duty(self, fill_date) -> float:
        return stamp_duty_rate(fill_date)

    def limit_pct(self, ticker, fill_date, *, is_st: bool = False) -> float:
        return price_limit_pct(ticker, fill_date, is_st=is_st)

    def slippage_bp(self, amihud_cross_section: pd.Series) -> pd.Series:
        return slippage_bp_series(amihud_cross_section, self.slippage_tiers)


__all__ = [
    "STAMP_DUTY_HALVING_DATE", "STAMP_DUTY_BEFORE", "STAMP_DUTY_AFTER",
    "CHINEXT_20PCT_DATE",
    "stamp_duty_rate", "board_of", "price_limit_pct",
    "SlippageTiers", "amihud_to_quantile", "slippage_bp_series",
    "entry_fill_index", "exit_fill_index", "WFCostModel",
]
