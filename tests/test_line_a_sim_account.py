"""T-052 模拟账户：整数手、最低佣金、涨跌停（含 ST 规则切换）、T+1、参与率、分红与红利税、现金不透支。"""
from __future__ import annotations

import pandas as pd
import pytest

from quantmind.line_a.sim_account import AccountSimulator, SimConfig, dividend_tax_rate, limit_pct, lot_shares

DAYS = pd.bdate_range("2026-06-01", periods=30)


def _px(rows: dict[str, dict]) -> pd.DataFrame:
    """rows：{ticker: {"close": 价格或列表, 其他列覆盖}}；默认平价、成交量充足。"""
    out = []
    for t, spec in rows.items():
        closes = spec.get("close", 10.0)
        closes = closes if isinstance(closes, list) else [closes] * len(DAYS)
        for i, d in enumerate(DAYS):
            c = closes[i]
            pre = closes[i - 1] if i else c
            r = {"ts_code": t, "trade_date": d, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                 "pre_close": pre, "vol": spec.get("vol", 1e5), "amount": c * spec.get("vol", 1e5) * 100 / 1000,
                 "adj_factor": 1.0}
            r.update(spec.get("override", {}).get(i, {}))
            out.append(r)
    return pd.DataFrame(out)


def _scores(names, day_idx=(0,)):
    return pd.DataFrame([{"as_of": DAYS[i], "ticker": t, "score": float(len(names) - k)}
                         for i in day_idx for k, t in enumerate(names)])


def _cfg(**kw):
    base = dict(aum=1e5, top_n=2, rebalance_step=1, universe_size=None, buffer_mult=None, participation=None,
                min_recent_bar_td=1)
    return SimConfig(**(base | kw))


def test_lot_rules():
    assert lot_shares(10000, 12.3, "MAIN") == 800
    assert lot_shares(10000, 30, "STAR") == 333
    assert lot_shares(5000, 30, "STAR") == 0
    assert lot_shares(2000, 15, "BSE") == 133
    assert lot_shares(1000, 15, "MAIN") == 0
    assert lot_shares(1000, 15, "MAIN", integer=False) == pytest.approx(1000 / 15)


def test_limit_rules():
    assert limit_pct("600001.SH", pd.Timestamp("2026-07-03"), True) == 0.05
    assert limit_pct("600001.SH", pd.Timestamp("2026-07-06"), True) == 0.10
    assert limit_pct("300001.SZ", pd.Timestamp("2026-07-03"), True) == 0.20
    assert limit_pct("688001.SH", pd.Timestamp("2026-01-05"), False) == 0.20
    assert dividend_tax_rate(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-30")) == 0.20
    assert dividend_tax_rate(pd.Timestamp("2026-06-01"), pd.Timestamp("2026-08-30")) == 0.10


def test_buy_lots_min_commission_and_cash():
    px = _px({"600001.SH": {"close": 12.3}, "600002.SH": {"close": 47.0}})
    sim = AccountSimulator(_scores(["600001.SH", "600002.SH"]), px, _cfg(aum=2e4), start=DAYS[0], end=DAYS[5])
    r = sim.run()
    tr = r["trades"]
    assert set(tr["date"]) == {DAYS[1]}                              # 决策日收盘出单，次日成交
    sh = dict(zip(tr["ticker"], tr["shares"], strict=True))
    assert sh["600001.SH"] == 800 and sh["600002.SH"] == 200        # 1 万元 / 12.3 → 813 → 800；1 万 / 47 → 212 → 200
    assert r["costs"]["commission"] == pytest.approx(10.0)          # 两笔都不足 5 元，按最低 5 元
    assert r["nav"]["cash"].min() >= 0


def test_t_plus_1_and_limit_up_block():
    ov = {1: {"high": 11.0, "pre_close": 10.0}}                      # 第 1 天最高价触及涨停 → 买不进
    px = _px({"600001.SH": {"close": 10.0, "override": ov}, "600002.SH": {"close": 10.0}})
    sim = AccountSimulator(_scores(["600001.SH", "600002.SH"]), px, _cfg(), start=DAYS[0], end=DAYS[5])
    r = sim.run()
    first = r["trades"].groupby("ticker")["date"].min()
    assert first["600002.SH"] == DAYS[1] and first["600001.SH"] == DAYS[2]
    assert (r["rejects"]["reason"] == "limit_up_locked").any()
    # T+1：当天买入的持仓，当天的卖单不能成交（决策都在收盘，正常流程里碰不到，直接构造状态检查）
    from quantmind.line_a.sim_account import _Pos
    sim2 = AccountSimulator(_scores(["600002.SH"]), px, _cfg(), start=DAYS[0], end=DAYS[5])
    sim2.pos["600002.SH"] = _Pos(100.0, DAYS[2], DAYS[2])
    sim2.pending = [{"side": "sell", "ticker": "600002.SH", "decided": DAYS[1], "value": 1000.0, "sell_all": True,
                     "slip_bp": 5.0, "expiry": None, "rank": -1}]
    sim2._execute(DAYS[2])
    assert sim2.pos["600002.SH"].shares == 100 and sim2.rejects[-1]["reason"] == "t_plus_1_blocked"
    sim2._execute(DAYS[3])
    assert "600002.SH" not in sim2.pos


def test_st_limit_switch():
    d_before = DAYS.get_indexer([pd.Timestamp("2026-07-03")])[0]
    ov = {d_before: {"high": 10.5, "pre_close": 10.0}}               # +5%：ST 在 7-06 之前是涨停
    px = _px({"600001.SH": {"close": 10.0, "override": ov}})
    st = pd.DataFrame({"ts_code": ["600001.SH"], "s": [DAYS[0]], "e": [DAYS[-1]]})
    sim = AccountSimulator(_scores(["600001.SH"], (d_before - 1,)), px, _cfg(top_n=1), start=DAYS[0], end=DAYS[-1],
                           st_intervals=st)
    r = sim.run()
    assert r["trades"]["date"].min() > DAYS[d_before]


def test_participation_cap_splits_order():
    px = _px({"600001.SH": {"close": 10.0, "vol": 50.0}})            # 每天 5,000 股；5% = 250 股
    sim = AccountSimulator(_scores(["600001.SH"]), px, _cfg(top_n=1, aum=1e4, participation=0.05),
                           start=DAYS[0], end=DAYS[8])
    r = sim.run()
    buys = r["trades"][r["trades"]["side"] == "buy"]
    assert len(buys) >= 3 and buys["shares"].max() <= 250 and (buys["shares"] % 100 == 0).all()


def test_dividend_and_tax():
    px = _px({"600001.SH": {"close": 10.0}})
    dv = pd.DataFrame({"ts_code": ["600001.SH"], "div_proc": ["实施"], "ex_date": [DAYS[5].strftime("%Y%m%d")],
                       "cash_div_tax": [0.5], "stk_div": [0.0]})
    sc = pd.concat([_scores(["600001.SH"], (0,)),
                    pd.DataFrame({"as_of": [DAYS[10]], "ticker": ["600002.SH"], "score": [1.0]})])
    px = pd.concat([px, _px({"600002.SH": {"close": 10.0}})])
    sim = AccountSimulator(sc, px, _cfg(top_n=1, aum=1e4), start=DAYS[0], end=DAYS[15], dividends=dv)
    r = sim.run()
    held = r["trades"][(r["trades"]["ticker"] == "600001.SH") & (r["trades"]["side"] == "buy")]["shares"].sum()
    assert r["costs"]["dividend_tax"] == pytest.approx(held * 0.5 * 0.20)   # 持有不到 1 个月：20%


def test_review_t053_fixes():
    from quantmind.line_a.sim_account import _Pos, board_of, limit_price
    assert limit_price(10.25, 0.10, False) == 9.23 and limit_price(9.45, 0.10, True) == 10.40   # 十进制四舍五入
    assert board_of("689009.SH") == "STAR" and board_of("920090.BJ") == "BSE" and board_of("600000.SH") == "MAIN"
    px = _px({"600001.SH": {"close": 10.0}, "600002.SH": {"close": 100.0}})
    sim = AccountSimulator(_scores(["600001.SH"]), px, _cfg(), start=DAYS[0], end=DAYS[5])
    sim.pos["600001.SH"] = _Pos(150.0, DAYS[0], DAYS[0])          # 部分卖出 50 股：不足一手，订单终止，不无限重试
    sim.pending = [{"side": "sell", "ticker": "600001.SH", "decided": DAYS[1], "value": 500.0, "sell_all": False,
                    "slip_bp": 5.0, "expiry": None, "rank": -1}]
    sim._execute(DAYS[2])
    assert sim.pending == [] and sim.rejects[-1]["reason"] == "below_one_lot" and sim.pos["600001.SH"].shares == 150
    sim.cash = 5000.0                                              # 现金不够一手：委托被拒，不留到第二天
    sim.pending = [{"side": "buy", "ticker": "600002.SH", "decided": DAYS[1], "value": 11000.0, "sell_all": False,
                    "slip_bp": 5.0, "expiry": DAYS[20], "rank": 0}]
    sim._execute(DAYS[3])
    assert sim.pending == [] and sim.rejects[-1]["reason"] == "insufficient_cash"
