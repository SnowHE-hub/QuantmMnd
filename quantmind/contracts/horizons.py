"""HorizonRegistry —— horizon 口径单一来源（消除散落各处的硬编码 horizon）.

短(short)=12d 客户短线 / 稳健(robust)=21d 只做稳健性验证 / 长(long)=63d 客户长线。
各处（评估/成本/再平衡/产品线判断）一律从这里引用，不再各自硬编码。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Horizon:
    name: str            # short | robust | long
    days: int            # 持有交易日 H
    label_col: str       # 对应 panel 标签列
    is_product_line: bool  # 是否客户产品线（robust=否，仅稳健性验证）
    note: str = ""

    @property
    def holding_td(self) -> int:
        return self.days

    @property
    def rebalance_step(self) -> int:
        """非重叠持有期的 as_of 步长（周频=5td/格）：ceil(H/5)。与 evaluate_bakeoff 一致。"""
        return max(1, math.ceil(self.days / 5.0))

    @property
    def periods_per_year(self) -> float:
        return 250.0 / self.days


_HORIZONS: dict[str, Horizon] = {
    "short": Horizon("short", 12, "forward_return_12d", True, "客户短线主信号"),
    "robust": Horizon("robust", 21, "forward_return_21d", False, "仅稳健性验证，非产品线"),
    "long": Horizon("long", 63, "forward_return_63d", True, "客户长线（63d 基本面版）"),
}
# 反查：days -> name
_BY_DAYS: dict[int, str] = {h.days: n for n, h in _HORIZONS.items()}


class HorizonRegistry:
    """horizon 单一来源。按 name 或 days 取 Horizon。"""

    @staticmethod
    def get(name: str) -> Horizon:
        if name not in _HORIZONS:
            raise KeyError(f"unknown horizon {name!r}; valid={list(_HORIZONS)}")
        return _HORIZONS[name]

    @staticmethod
    def by_days(days: int) -> Horizon:
        if days not in _BY_DAYS:
            raise KeyError(f"no horizon with days={days}; valid={sorted(_BY_DAYS)}")
        return _HORIZONS[_BY_DAYS[days]]

    @staticmethod
    def all() -> list[Horizon]:
        return list(_HORIZONS.values())

    @staticmethod
    def product_lines() -> list[Horizon]:
        return [h for h in _HORIZONS.values() if h.is_product_line]

    @staticmethod
    def names() -> list[str]:
        return list(_HORIZONS)


__all__ = ["Horizon", "HorizonRegistry"]
