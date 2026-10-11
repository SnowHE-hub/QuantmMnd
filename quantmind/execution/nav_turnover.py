"""带换手控制的 executable NAV 引擎（T-020，研究用；不修改 nav_engine.py）.

在 ExecutableNavEngine 的决策步骤上加两种常见的换手控制，其余（成交、成本、涨跌停、停牌、现金、退市处理、汇总）完全沿用父类：

- ``exit_rank``（排名缓冲带）：已持有的票只要仍排在候选的前 ``exit_rank`` 名就继续持有；
  空出的名额按排名从高到低补入新票，组合仍为 ``top_n`` 只。``None`` = 不设缓冲（与父类相同）。
- ``drift_band``（权重容忍带）：目标内已持有的票，若当前市值与等权目标的偏离不超过
  ``drift_band × 目标市值``，本次不调仓；新进的票仍按等权目标买入。``None`` = 每次都调回等权（与父类相同）。

两个参数都为 ``None`` 时，行为与 ExecutableNavEngine 逐笔相同（测试覆盖）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantmind.execution.nav_engine import (
    R_CANCELLED_REBAL,
    R_FILTER_NO_RECENT,
    R_FILTER_NOT_UNIVERSE,
    ExecutableNavEngine,
    NavConfig,
)


class TurnoverControlledNavEngine(ExecutableNavEngine):
    def __init__(self, preds: pd.DataFrame, prices: pd.DataFrame, config: NavConfig,
                 amihud: pd.DataFrame | None = None, *,
                 exit_rank: int | None = None, drift_band: float | None = None) -> None:
        if exit_rank is not None and exit_rank < config.top_n:
            raise ValueError("exit_rank must be >= top_n")
        if drift_band is not None and drift_band < 0:
            raise ValueError("drift_band must be >= 0")
        super().__init__(preds, prices, config, amihud=amihud)
        self.exit_rank = exit_rank
        self.drift_band = drift_band
        self.kept_by_buffer: list[dict] = []

    def _select(self, ranked: list[str], as_of: pd.Timestamp) -> list[str]:
        n = self.cfg.top_n
        if self.exit_rank is None:
            return ranked[:n]
        within = set(ranked[: self.exit_rank])
        keep = [t for t in ranked if t in within and t in self.positions][:n]
        keep_set = set(keep)
        fill = [t for t in ranked if t not in keep_set][: n - len(keep)]
        chosen = keep_set | set(fill)
        top = set(ranked[:n])
        self.kept_by_buffer.append({"as_of": str(as_of.date()), "n_kept_outside_top_n": len(keep_set - top)})
        return [t for t in ranked if t in chosen]

    # 与 ExecutableNavEngine._decide 相同，只在"选目标"和"是否调仓"两处插入换手控制
    def _decide(self, as_of: pd.Timestamp) -> None:
        cfg = self.cfg
        universe = self.px.universe_topn(as_of, cfg.universe_size)
        cross = self.preds[self.preds["as_of"] == as_of]
        scores = cross.set_index("ticker")["score"]

        candidates, filtered = [], []
        recent_floor_idx = self.px.calendar.searchsorted(as_of, side="right") - cfg.min_recent_bar_td
        recent_floor_i8 = self.px.calendar[max(recent_floor_idx, 0)].value
        try:
            lb_row = self.px.last_bar_upto.loc[:as_of].iloc[-1]
        except IndexError:
            lb_row = pd.Series(dtype=float)
        for t in scores.index:
            if t not in universe:
                filtered.append((t, R_FILTER_NOT_UNIVERSE))
                continue
            lb = lb_row.get(t, np.nan)
            if pd.isna(lb) or lb < recent_floor_i8:
                filtered.append((t, R_FILTER_NO_RECENT))
                continue
            candidates.append(t)

        ranked = list(scores.loc[candidates].sort_values(ascending=False, kind="mergesort").index)
        target_names = self._select(ranked, as_of)                      # ← 换手控制 1
        nav_now = self._nav(as_of)
        tgt_value = nav_now / max(len(target_names), 1)

        still = []
        for o in self.pending:
            conflict = ((o["side"] == "buy" and o["ticker"] not in target_names)
                        or (o["side"] == "sell" and o["ticker"] in target_names))
            if conflict:
                self.rejected.append({**self._order_row(o), "fill_date": None,
                                      "reason_code": R_CANCELLED_REBAL, "reject_date": str(as_of.date())})
            else:
                still.append(o)
        self.pending = still

        slip = self._slippage_bp_map(as_of, sorted(set(target_names) | set(self.positions)))

        expiry = None
        later = [r for r in self.rebalance_dates if r > as_of]
        if later:
            expiry = later[0]

        for o in self.pending:
            if o["side"] == "buy" and o["ticker"] in target_names:
                o["expiry"] = expiry
                o["target_value"] = tgt_value
                o["decision_asof"] = as_of
                o["rank"] = target_names.index(o["ticker"])

        pend_keys = {(o["side"], o["ticker"]) for o in self.pending}
        for t in sorted(set(target_names) | set(self.positions)):
            cur_px = self.px.last_price_upto(as_of, t)
            cur_val = (self.positions[t].shares * cur_px
                       if t in self.positions and cur_px is not None else 0.0)
            want = tgt_value if t in target_names else 0.0
            delta = want - cur_val
            if abs(delta) < nav_now * 1e-6:
                continue
            if (self.drift_band is not None and t in target_names and cur_val > 0
                    and abs(delta) <= self.drift_band * tgt_value):        # ← 换手控制 2
                continue
            side = "buy" if delta > 0 else "sell"
            if (side, t) in pend_keys:
                continue
            order = {
                "side": side, "ticker": t, "decision_asof": as_of,
                "target_value": abs(delta) if side == "buy" else None,
                "sell_value": abs(delta) if side == "sell" else None,
                "sell_all": (side == "sell" and want == 0.0),
                "slippage_bp": float(slip.get(t, self.cfg.tiers.small_bp)),
                "expiry": expiry if side == "buy" else None,
                "rank": (target_names.index(t) if t in target_names else -1),
            }
            self.pending.append(order)

        self.targets_log.append({
            "as_of": str(as_of.date()), "n_scored": int(len(scores)),
            "n_universe": int(len(universe)), "n_candidates": int(len(candidates)),
            "n_target": int(len(target_names)), "nav_at_decision": float(nav_now),
            "target_weight_each": float(1.0 / max(len(target_names), 1)),
            "targets": target_names,
        })
        for t, reason in filtered:
            self.filtered_log.append({"as_of": str(as_of.date()), "ticker": t, "reason_code": reason})

        self._bench_members = universe


__all__ = ["TurnoverControlledNavEngine"]
