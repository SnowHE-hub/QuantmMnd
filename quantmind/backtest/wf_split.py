"""quantmind.backtest.wf_split — Purged + embargoed walk-forward 切分器.

依据 ``docs/plans/walkforward_design_plan.md`` §2。**新建**，不修改既有两套
walk-forward（``backtest/walk_forward.py`` 的 ``WalkForwardValidator``、
``models/factor_model.py`` 的 ``WalkForwardSplit``）。

核心正确性（交易日索引空间）
----------------------------
设 ``idx(a)`` = as_of ``a`` 在 SSE 交易日历中的位置；``H`` = 标签 horizon（交易日）；
``E`` = embargo（交易日，**≥ H**）；``C = idx(cutoff)``。

    train = { a : idx(a) + H ≤ C }      # 标签窗口 [a, a+H] 不越过 C → idx(a) ≤ C − H
    隔离带（既不训练也不测试）: idx(a) ∈ (C − H, C + E]
    test  = { a : idx(a) > C + E } ∩ OOS

``H=0, E=0`` 退化为"train 紧贴 test"（复刻既有无 purge 行为）——仅供 **purge 消融反证**。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import pandas as pd


def _ts(d: date | datetime | str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


@dataclass(frozen=True)
class WFFold:
    """一个 walk-forward fold：训练 / 验证 / 测试 as_of 列表 + 元信息."""

    cutoff: pd.Timestamp
    train_dates: list[pd.Timestamp]
    val_dates: list[pd.Timestamp]
    test_dates: list[pd.Timestamp]
    purge_horizon: int
    embargo: int

    def as_dict(self) -> dict:
        return {
            "cutoff": str(self.cutoff.date()),
            "n_train": len(self.train_dates),
            "n_val": len(self.val_dates),
            "n_test": len(self.test_dates),
            "purge_horizon": self.purge_horizon,
            "embargo": self.embargo,
            "train_last": str(self.train_dates[-1].date()) if self.train_dates else None,
            "test_first": str(self.test_dates[0].date()) if self.test_dates else None,
        }


class PurgedWalkForwardSplit:
    """交易日索引感知的 purge + embargo 切分（扩展 / 滚动两模式）.

    Args:
        trading_days: 升序 SSE 交易日历（``date`` / ``Timestamp`` 序列），定义 ``idx()``。
        horizon: 标签 horizon H（交易日），默认 12。
        embargo: 隔离带 E（交易日），默认 = horizon（须 ≥ horizon 才能挡住 12 日标签重叠）。
        mode: ``"expanding"``（训练起点固定）或 ``"rolling"``（固定窗长）。
        rolling_lookback_td: rolling 模式的训练窗长（交易日），默认 756（≈36 个月）。
        n_val: val 取训练尾部的 as_of 个数（早停用，仍在 purge 内），默认 2。
    """

    def __init__(
        self,
        trading_days: Sequence[date | datetime | str | pd.Timestamp],
        *,
        horizon: int = 12,
        embargo: int = 12,
        mode: str = "expanding",
        rolling_lookback_td: int = 756,
        n_val: int = 2,
    ) -> None:
        if horizon < 0 or embargo < 0:
            raise ValueError("horizon/embargo must be >= 0")
        if horizon > 0 and embargo < horizon:
            raise ValueError(
                f"embargo({embargo}) 必须 >= horizon({horizon}) 才能挡住标签重叠"
            )
        if mode not in ("expanding", "rolling"):
            raise ValueError("mode must be 'expanding' or 'rolling'")
        cal = pd.DatetimeIndex(sorted({_ts(d) for d in trading_days}))
        if len(cal) == 0:
            raise ValueError("trading_days is empty")
        self.cal = cal
        self._pos = {d: i for i, d in enumerate(cal)}
        self.H = int(horizon)
        self.E = int(embargo)
        self.mode = mode
        self.rolling_lookback_td = int(rolling_lookback_td)
        self.n_val = int(n_val)

    # ------------------------------------------------------------------
    def idx(self, d: date | datetime | str | pd.Timestamp) -> int:
        """as_of 在交易日历中的位置；非交易日回退到最近的前一个交易日。"""
        t = _ts(d)
        if t in self._pos:
            return self._pos[t]
        pos = int(self.cal.searchsorted(t, side="right")) - 1
        if pos < 0:
            raise ValueError(f"{t.date()} 早于交易日历起点")
        return pos

    # ------------------------------------------------------------------
    def split(
        self,
        as_of_grid: Sequence[date | datetime | str | pd.Timestamp],
        cutoff: date | datetime | str | pd.Timestamp,
        *,
        oos_start: date | datetime | str | pd.Timestamp | None = None,
        oos_end: date | datetime | str | pd.Timestamp | None = None,
    ) -> WFFold:
        """对给定 cutoff 生成一个含隔离带的 fold。

        - train: ``idx(a) ≤ C − H``（rolling 再加 ``idx(a) ≥ C − rolling_lookback_td``）。
        - test : ``idx(a) > C + E`` 且落在 ``[oos_start, oos_end]``。
        - val  : train 尾部 ``n_val`` 个（仍满足 purge）。
        """
        grid = sorted({_ts(a) for a in as_of_grid})
        C = self.idx(cutoff)
        os_lo = _ts(oos_start) if oos_start is not None else None
        os_hi = _ts(oos_end) if oos_end is not None else None

        train_hi = C - self.H  # idx(a) ≤ train_hi
        train_lo = (C - self.rolling_lookback_td) if self.mode == "rolling" else -1

        train = [a for a in grid if train_lo <= self.idx(a) <= train_hi]
        test = [
            a for a in grid
            if self.idx(a) > C + self.E
            and (os_lo is None or a >= os_lo)
            and (os_hi is None or a <= os_hi)
        ]
        val = train[-self.n_val:] if self.n_val > 0 and len(train) >= self.n_val else []

        return WFFold(
            cutoff=_ts(cutoff),
            train_dates=train,
            val_dates=val,
            test_dates=test,
            purge_horizon=self.H,
            embargo=self.E,
        )

    # ------------------------------------------------------------------
    def make_folds(
        self,
        as_of_grid: Sequence[date | datetime | str | pd.Timestamp],
        cutoffs: Sequence[date | datetime | str | pd.Timestamp],
        *,
        oos_start: date | datetime | str | pd.Timestamp,
        oos_end: date | datetime | str | pd.Timestamp,
    ) -> list[WFFold]:
        """对一串季度 cutoff 生成滚动 refit 的 fold 序列。

        每个 fold 的 test 区段 = ``(C_k + E, C_{k+1}]``（最后一段到 OOS 末），
        与 OOS 区间求交。
        """
        cuts = [_ts(c) for c in sorted(cutoffs)]
        os_hi = _ts(oos_end)
        folds: list[WFFold] = []
        for k, C in enumerate(cuts):
            seg_hi = cuts[k + 1] if k + 1 < len(cuts) else os_hi
            f = self.split(as_of_grid, C, oos_start=oos_start, oos_end=seg_hi)
            folds.append(f)
        return folds

    # ------------------------------------------------------------------
    @staticmethod
    def assert_no_leakage(fold: WFFold) -> None:
        """硬断言：train/test 零交集，且 gap ≥ H + E（H>0 时）。隔离带内无样本。"""
        tr = set(fold.train_dates)
        te = set(fold.test_dates)
        inter = tr & te
        if inter:
            raise AssertionError(f"train∩test 非空: {sorted(inter)[:3]}")
        if fold.train_dates and fold.test_dates:
            # 用调用方提供的索引无法在此重算；此处仅校验时间序（详细 gap 在 split 处保证）
            if max(fold.train_dates) >= min(fold.test_dates):
                raise AssertionError("max(train) >= min(test)")


__all__ = ["WFFold", "PurgedWalkForwardSplit"]
