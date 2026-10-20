"""quantmind.line_a.f8_books — F8 参考账本与容量情景（03_final_plan_v2 第 5 节 F8）.

- 参考账本：直接用冻结的 ExecutableNavEngine 和参考契约（configs/line_a_v2/reference_strategies.json）里的参数，
  不改任何一项；账户权限线 = 先剔除这条线不能新买的板块的分数，再交给引擎（引擎照常在 PIT adv20 前 1500 里选前 N）。
- 容量情景：对每次调仓的目标持仓，按账户规模、最小申报单位、最低佣金、参与率上限计算可执行性与偏离，
  只报告，不改变参考账本（集中持仓等改动是单独登记的挑战臂）。
注意：Tushare 日线 amount 的单位是千元。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import SlippageTiers, board_of
from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig

LOT = {"MAIN": (100, 100), "CHINEXT": (100, 100), "STAR": (200, 1), "BSE": (100, 1)}   # trading_rules.yaml（待核实）


def nav_config(contract: dict) -> NavConfig:
    c, p = contract["costs"], contract["portfolio"]
    t = c["slippage_tiers_bp"]
    return NavConfig(horizon_td=contract["holding_td"], rebalance_step=contract["rebalance_step_weekly_periods"],
                     top_n=p["top_n"], universe_size=1500, commission_bp=c["commission_bp_each_side"],
                     transfer_bp=c["transfer_bp_each_side"],
                     tiers=SlippageTiers(large_bp=t["large_bp"], mid_bp=t["mid_bp"], small_bp=t["small_bp"],
                                         q_large=t["q_large"], q_small=t["q_small"]),
                     delist_writeoff_td=contract["execution"]["delist_writeoff_td"])


def restrict_to_profile(preds: pd.DataFrame, boards: dict[str, bool]) -> pd.DataFrame:
    ok = preds["ticker"].map(lambda t: bool(boards.get(board_of(t), False)))
    return preds[ok.to_numpy()].reset_index(drop=True)


def run_reference_book(preds: pd.DataFrame, prices: pd.DataFrame, contract: dict,
                       amihud: pd.DataFrame | None) -> dict:
    return ExecutableNavEngine(preds, prices, nav_config(contract), amihud=amihud).run()


def _shares(value: float, price: float, board: str) -> int:
    lo, step = LOT.get(board, (100, 100))
    if not (price > 0) or value < lo * price:
        return 0
    n = int(value // price)
    return lo + ((n - lo) // step) * step


def capacity_report(targets: pd.DataFrame, prices: pd.DataFrame, aum: float, *, min_commission: float = 5.0,
                    commission_bp: float = 3.0, max_participation: float = 0.01) -> pd.DataFrame:
    """targets：引擎的 targets_log（as_of、targets 列表、target_weight_each）。返回每次调仓的可执行性统计。"""
    px = prices[["ts_code", "trade_date", "close", "amount"]].copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["ts_code", "trade_date"])
    px["adv20_rmb"] = px.groupby("ts_code")["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean()) * 1000
    px = px.set_index(["trade_date", "ts_code"])
    rows = []
    for _, r in targets.iterrows():
        d, names = pd.Timestamp(r["as_of"]), list(r["targets"])
        if not names:
            continue
        w = float(r["target_weight_each"])
        k = pd.MultiIndex.from_arrays([[d] * len(names), names])
        snap = px.reindex(k)
        value = aum * w
        sh = np.array([_shares(value, p, board_of(t)) for t, p in zip(names, snap["close"].to_numpy(), strict=True)])
        spent = sh * snap["close"].to_numpy()
        part = value / snap["adv20_rmb"].to_numpy()
        comm = np.maximum(min_commission, spent * commission_bp / 1e4)
        rows.append({"as_of": d.date(), "aum": aum, "n_targets": len(names),
                     "n_unbuyable_one_lot": int((sh == 0).sum()),
                     "cash_residual_share": float(1 - np.nansum(spent) / aum),
                     "max_participation": float(np.nanmax(part)) if np.isfinite(part).any() else None,
                     "n_over_participation": int((part > max_participation).sum()),
                     "min_commission_drag_bp": float(np.nansum(comm[sh > 0]) / aum * 1e4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 前瞻决策网格与账本
FREEZE = pd.Timestamp("2026-10-09")         # 研究期数据冻结日（T-030 / T-034）


def extended_grid(full_cal, frozen_grid, step: int = 5) -> list[pd.Timestamp]:
    """冻结的 v6_1 周频网格按同一相位向后延伸：从冻结网格最后一个日期起，每 step 个交易日取一点。
    （build_asof_grid 的锚点随日历末日移动，不能直接用新日历重算，否则相位漂移。）"""
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(list(full_cal))))
    i = cal.get_loc(pd.Timestamp(frozen_grid[-1]))
    return [pd.Timestamp(x) for x in frozen_grid] + [pd.Timestamp(cal[j]) for j in range(i + step, len(cal), step)]


def forward_grid(full_cal, frozen_grid, step: int = 5) -> list[pd.Timestamp]:
    return [d for d in extended_grid(full_cal, frozen_grid, step) if d > FREEZE]


def run_forward_books(scores: dict[str, pd.DataFrame], prices: pd.DataFrame, contracts: dict, profiles: dict,
                      amihud: pd.DataFrame | None, aums=(1e5, 1e6, 1e7), **cap_kw) -> dict:
    """scores：{参考臂名: (as_of, ticker, score)，只含前瞻网格日}。每个参考臂 × 账户权限线跑一次引擎，
    再对目标持仓做容量情景。返回 {臂: {线: {"nav", "targets", "capacity"}}}。"""
    out: dict = {}
    for arm, preds in scores.items():
        out[arm] = {}
        for prof, boards in profiles.items():
            p = restrict_to_profile(preds, boards)
            if p.empty:
                continue
            r = run_reference_book(p, prices, contracts[arm], amihud)
            cap = pd.concat([capacity_report(r["targets"], prices, a, **cap_kw) for a in aums], ignore_index=True) \
                if len(r["targets"]) else pd.DataFrame()
            out[arm][prof] = {"nav": r["nav_daily"], "targets": r["targets"], "capacity": cap,
                              "holdings": r["holdings_daily"], "rejected": r["rejected_trades"],
                              "summary": r["summary"]}
    return out
