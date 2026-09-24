"""阶段 1.5 审计回归测试（合成数据，锁定已审计的引擎语义）.

这些测试固化审计结论，防止回归：
- NAV 恒等式（nav = cash + 持仓市值；gross = net + 累计成本）；
- 再平衡日历 = as_of[::step] 的位置步进语义（频率依赖网格间距，非时间）；
- 拒单计数的两个口径：n_rejection_events（每日重试膨胀）vs n_distinct_rejected_orders（按单去重）；
- 绝对回撤 与 超额回撤 是两个不同量（gate 语义问题的根源）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantmind.execution.nav_engine import ExecutableNavEngine, NavConfig

DATES = pd.bdate_range("2024-01-01", periods=140)


def _prices(spec, dates=DATES, missing=None, drift=None):
    missing = missing or {}
    drift = drift or {}
    rows = []
    for t, (base, amount) in spec.items():
        prev = base
        for i, d in enumerate(dates):
            close = base * (1 + drift.get(t, 0.0)) ** i
            if d in missing.get(t, set()):
                prev = close
                continue
            rows.append({"ts_code": t, "trade_date": d, "open": close,
                         "high": close * 1.02, "low": close * 0.98, "close": close,
                         "pre_close": prev, "adj_factor": 1.0, "amount": amount})
            prev = close
    return pd.DataFrame(rows)


def _preds(as_ofs, scores):
    return pd.DataFrame([{"as_of": a, "ticker": t, "score": s}
                         for a in as_ofs for t, s in scores.items()])


def _cfg(**kw):
    d = {"horizon_td": 12, "rebalance_step": 3, "top_n": 2, "universe_size": 10,
         "min_recent_bar_td": 5}
    d.update(kw)
    return NavConfig(**d)


# ── 再平衡日历 = as_of[::step]（频率依赖网格间距，非时间）───────────────────

def test_rebalance_is_positional_stepping_not_time():
    """审计任务2：再平衡按 as_of 位置步进；网格有大间隙时实际持有期会偏离设计意图。"""
    # 构造带大间隙的 as_of 网格（模拟 walk-forward fold 之间的缺口）
    dense = list(DATES[10:23])         # 13 个连续（周频代理）
    gap = list(DATES[70:75])           # 隔一大段后的另一簇
    as_ofs = dense + gap
    prices = _prices({"AAA.SZ": (10, 100), "BBB.SZ": (20, 90), "CCC.SZ": (30, 80)})
    preds = _preds(as_ofs, {"AAA.SZ": 0.9, "BBB.SZ": 0.5, "CCC.SZ": 0.1})
    eng = ExecutableNavEngine(preds, prices, _cfg(rebalance_step=13, top_n=1))
    # 18 个 as_of，step=13 → 位置 0,13 → 2 次再平衡，且第二次跨越大间隙
    assert len(eng.rebalance_dates) == 2
    assert eng.rebalance_dates[0] == pd.Timestamp(dense[0])
    assert eng.rebalance_dates[1] == pd.Timestamp(gap[0])


# ── NAV 恒等式 ───────────────────────────────────────────────────────────────

def test_nav_equals_cash_plus_holdings_and_gross_identity():
    prices = _prices({"AAA.SZ": (10, 100), "BBB.SZ": (20, 90), "CCC.SZ": (5, 80)},
                     drift={"AAA.SZ": 0.002, "BBB.SZ": -0.001})
    preds = pd.concat([
        _preds([DATES[30]], {"AAA.SZ": 0.9, "BBB.SZ": 0.8, "CCC.SZ": 0.1}),
        _preds([DATES[45]], {"AAA.SZ": 0.1, "BBB.SZ": 0.8, "CCC.SZ": 0.9}),
    ])
    res = ExecutableNavEngine(preds, prices, _cfg(top_n=2, rebalance_step=1)).run()
    nav = res["nav_daily"].set_index("date")
    hold = res["holdings_daily"]
    hv = hold.groupby("date")["value"].sum().reindex(nav.index).fillna(0.0)
    assert np.allclose(nav["nav_net"], nav["cash"] + hv, atol=1e-9)
    assert np.allclose(nav["nav_gross"], nav["nav_net"] + nav["cum_cost"], atol=1e-12)


# ── 拒单计数：事件口径 vs 去重按单口径 ───────────────────────────────────────

def test_rejection_event_count_inflated_by_daily_retries():
    """审计任务3：一个买单因停牌连续 K 天不可成交 → K 个 rejection 事件但仅 1 个底层订单。

    锁定 summary 同时报告 n_rejection_events（膨胀）与 n_distinct_rejected_orders（真实）。
    """
    susp = set(DATES[31:37])   # as_of=DATES[30]，随后 6 天停牌
    prices = _prices({"AAA.SZ": (10, 100), "BBB.SZ": (20, 90)}, missing={"AAA.SZ": susp})
    preds = _preds([DATES[30]], {"AAA.SZ": 0.9, "BBB.SZ": 0.1})
    res = ExecutableNavEngine(preds, prices, _cfg(top_n=1)).run()
    execu = res["summary"]["execution"]
    rej = res["rejected_trades"]
    n_susp_events = int((rej["reason_code"] == "suspended_no_bar").sum())
    assert n_susp_events == 6                              # 每个停牌日一个事件（膨胀口径）
    # 去重口径：同一 (side,ticker,decision_asof,reason) 只算一单
    assert execu["n_distinct_rejected_orders"] == 1
    assert execu["rejection_distinct_order_counts"]["suspended_no_bar"] == 1
    assert execu["n_rejection_events"] >= execu["n_distinct_rejected_orders"]


# ── 绝对回撤 vs 超额回撤 是两个不同量 ────────────────────────────────────────

def test_absolute_vs_excess_drawdown_are_distinct():
    """审计任务5：满仓多头的绝对回撤含市场 β；超额回撤(net/bench)剔除 β。两者不同。"""
    # 让全池齐跌（β 回撤），策略相对基准仅小幅波动 → 绝对DD大、超额DD小
    dates = DATES
    rows = []
    for t, base in [("AAA.SZ", 10.0), ("BBB.SZ", 10.0), ("CCC.SZ", 10.0)]:
        for i, d in enumerate(dates):
            # 市场在中段深跌 30% 再回升
            f = 1.0 - 0.30 * np.sin(min(i, 70) / 70 * np.pi)
            close = base * f
            rows.append({"ts_code": t, "trade_date": d, "open": close,
                         "high": close * 1.02, "low": close * 0.98, "close": close,
                         "pre_close": close, "adj_factor": 1.0, "amount": 100})
    prices = pd.DataFrame(rows)
    preds = _preds(list(dates[20:120:5]), {"AAA.SZ": 0.9, "BBB.SZ": 0.5, "CCC.SZ": 0.1})
    res = ExecutableNavEngine(preds, prices, _cfg(top_n=2, universe_size=3, rebalance_step=1)).run()
    nav = res["nav_daily"]

    def mdd(s):
        s = np.asarray(s, dtype=float)
        rm = np.maximum.accumulate(s)
        return float(((s - rm) / rm).min())

    dd_abs = mdd(nav["nav_net"])
    dd_excess = mdd(nav["nav_net"].values / nav["nav_bench"].values)
    assert dd_abs < -0.03                        # 绝对回撤含市场 β，显著为负
    assert dd_excess > dd_abs + 0.02             # 超额回撤明显更浅（剔除 β）
    assert res["summary"]["nav"]["max_drawdown_net"] == dd_abs   # 引擎报告的是绝对口径
