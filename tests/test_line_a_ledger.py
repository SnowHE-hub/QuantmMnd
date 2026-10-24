"""模拟账户账本（ledger.py）的会计核对：用已生成的账本文件逐项验证；账本不在就跳过。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

L = Path("/home/lenovo/projects/quantmind/data/line_a_v2/ledger")
T052 = sorted(p.name for p in L.glob("t052_*")) if L.exists() else []
pytestmark = pytest.mark.skipif(not T052, reason="no ledger built")


def load(acc):
    d = L / acc
    rd = lambda n: pd.read_parquet(d / f"{n}.parquet") if (d / f"{n}.parquet").exists() else pd.DataFrame()  # noqa: E731
    return json.loads((d / "meta.json").read_text(encoding="utf-8")), rd("daily"), rd("positions"), rd("trades"), rd("actions"), rd("round_trips")


@pytest.mark.parametrize("acc", T052)
def test_accounting_identities(acc):
    meta, daily, pos, tr, ac, rt = load(acc)
    aum = meta["aum"]
    mv = (pos["shares"] * pos["close"]).groupby(pos["date"]).sum().reindex(daily["date"]).fillna(0).to_numpy()
    assert np.abs(daily["cash"].to_numpy() + mv - daily["equity"].to_numpy()).max() < 0.01                 # 总资产 = 现金 + 市值
    assert np.abs(daily["equity"] - aum - daily["realized_cum"] - daily["unrealized"] - daily["dividends_cum"]).max() < 0.01
    assert abs(aum + tr["net_cash"].sum() + (ac["cash_dividend"].sum() if len(ac) else 0) - daily["cash"].iloc[-1]) < 0.01
    assert meta["checks"]["passed"] and meta["checks"]["t052_nav_reproduced_max_abs"] < 0.01


@pytest.mark.parametrize("acc", T052)
def test_fees_follow_rules(acc):
    _, _, _, tr, _, _ = load(acc)
    assert (tr["commission"] >= 5.0 - 1e-9).all()                                                      # 最低 5 元
    assert np.allclose(tr["commission"], np.maximum(tr["gross"] * 2.5e-4, 5.0))
    assert np.allclose(tr["transfer"], tr["gross"] * 1e-5)
    buys, sells = tr[tr["side"] == "buy"], tr[tr["side"] != "buy"]
    assert (buys["stamp"] == 0).all() and np.allclose(sells["stamp"], sells["gross"] * 5e-4)           # 印花税只在卖出
    assert np.allclose(tr["fee_total"], tr[["commission", "transfer", "stamp", "slippage", "dividend_tax"]].sum(axis=1))
    assert np.allclose(buys["net_cash"], -(buys["gross"] + buys["fee_total"]))


@pytest.mark.parametrize("acc", T052)
def test_lots_and_t_plus_1(acc):
    _, _, pos, tr, _, _ = load(acc)
    from quantmind.line_a.sim_account import LOT, board_of
    for r in tr[tr["side"] == "buy"].itertuples():
        lo, step = LOT[board_of(r.ticker)]
        assert r.shares >= lo and (r.shares - lo) % step == 0
    assert (pos["available"] <= pos["shares"] + 1e-9).all()
    new = pos[pos["bought_today"]]
    assert ((new["shares"] - new["available"]) > 0).all()                                                # 当天买入的部分不可卖
    # 卖出永远不发生在买入当天
    b = set(zip(tr.loc[tr["side"] == "buy", "date"], tr.loc[tr["side"] == "buy", "ticker"]))
    full_new = {(d, t) for d, t in b if not ((pos["date"] == d) & (pos["ticker"] == t) & (pos["available"] > 0)).any()}
    assert not any((d, t) in full_new for d, t in zip(tr.loc[tr["side"] == "sell", "date"], tr.loc[tr["side"] == "sell", "ticker"]))


@pytest.mark.parametrize("acc", T052)
def test_cash_after_is_sequential(acc):
    _, _, _, tr, ac, _ = load(acc)
    prev = None
    div = ac.groupby("date")["cash_dividend"].sum() if len(ac) else pd.Series(dtype=float)
    for d, g in tr.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            assert abs(g.loc[i - 1, "cash_after"] + g.loc[i, "net_cash"] - g.loc[i, "cash_after"]) < 1e-6
        prev = g.loc[len(g) - 1, "cash_after"]
    assert prev is not None


@pytest.mark.parametrize("acc", T052)
def test_round_trips_sum(acc):
    meta, daily, pos, tr, ac, rt = load(acc)
    if not len(rt):
        return
    # 已清仓回合的净盈亏 = 这些回合里卖出的已实现盈亏 + 回合期间的分红
    closed = set(zip(rt["ticker"], rt["close_date"]))
    assert len(closed) == len(rt)
    assert (rt["days_held"] >= 1).all()
    assert abs(rt["net_pnl"].sum() - (rt["sell_net"] + rt["dividends"] - rt["buy_cost"]).sum()) < 1e-6
