"""T-025 第三轮验收：买入可行性按 max(最低佣金, 比例佣金) 只算一次；现金恰好够 / 差一点 / 不透支（合成数据）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_P = Path(__file__).resolve().parents[1] / "docs/work/evidence/T-025/simulate_2026.py"
_spec = importlib.util.spec_from_file_location("t025_sim", _P)
sim = importlib.util.module_from_spec(_spec)
sys.modules["t025_sim"] = sim
_spec.loader.exec_module(sim)

CB, TB = 3.0, 0.2           # 佣金万 3、过户费万 0.2（与引擎默认一致）


def test_cash_exactly_one_lot_is_affordable():
    # Codex 反例：50 元、100 股、现金 5005.10 元 = 成交额 5000 + 过户费 0.10 + 最低佣金 5 → 应能买，v1 误判买不起
    total = sim.buy_total(100, 50.0, 0.0, CB, TB)
    assert total == pytest.approx(5005.10)
    assert sim.max_affordable(total, 50.0, 100, 100, 0.0, CB, TB) == 100
    assert sim.max_affordable(total - 0.01, 50.0, 100, 100, 0.0, CB, TB) == 0         # 差一分钱：买不起


def test_large_order_uses_percentage_commission_once():
    # 成交额足够大时比例佣金 > 5 元：只按比例收一次
    n = sim.max_affordable(100_000.0, 10.0, 100, 100, 15.0, CB, TB)
    assert sim.buy_total(n, 10.0, 15.0, CB, TB) <= 100_000.0
    assert sim.buy_total(n + 100, 10.0, 15.0, CB, TB) > 100_000.0                      # 再多一手就超
    assert n == 9_900            # 99,000 + 滑点 148.5 + 过户费 1.98 + 佣金 29.7 = 99,180.18；10,000 股为 100,182


def test_star_market_min_200_then_step_1():
    total_200 = sim.buy_total(200, 30.0, 0.0, CB, TB)
    assert sim.max_affordable(total_200, 30.0, 200, 1, 0.0, CB, TB) == 200
    assert sim.max_affordable(total_200 - 0.01, 30.0, 200, 1, 0.0, CB, TB) == 0
    n = sim.max_affordable(total_200 + 31.0, 30.0, 200, 1, 0.0, CB, TB)
    assert n == 201 and sim.buy_total(n, 30.0, 0.0, CB, TB) <= total_200 + 31.0


DATES = pd.bdate_range("2026-01-05", periods=45)
D1, D2, JUMP = 21, 35, 28      # 两个决策日（需要约 20 个交易日的成交额历史才有选股池）；复权跳变在两者之间


def prices(adj_jump_day: int | None = None, ratio: float = 1.1) -> pd.DataFrame:
    rows = []
    for t, px in (("000001.SZ", 50.0), ("000002.SZ", 20.0)):
        adj = 1.0
        prev = px
        for k, d in enumerate(DATES):
            close = px
            if adj_jump_day is not None and k >= adj_jump_day and t == "000001.SZ":
                adj, close = ratio, px / ratio          # 送转 / 除息：原始价下跌，复权因子上升，复权价不变
            rows.append({"ts_code": t, "trade_date": d, "open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "pre_close": prev, "adj_factor": adj, "amount": 1e9})
            prev = close
    return pd.DataFrame(rows)


def run(px: pd.DataFrame, capital: float, affordable: bool = False):
    # 第一次决策选 000001，第二次换成 000002 → 必然卖出 000001
    preds = pd.DataFrame([{"as_of": DATES[D1], "ticker": "000001.SZ", "score": 2.0},
                          {"as_of": DATES[D1], "ticker": "000002.SZ", "score": 1.0},
                          {"as_of": DATES[D2], "ticker": "000001.SZ", "score": 1.0},
                          {"as_of": DATES[D2], "ticker": "000002.SZ", "score": 2.0}])
    cfg = sim.NavConfig(horizon_td=10, rebalance_step=1, top_n=1, universe_size=2, min_recent_bar_td=1, initial_nav=capital)
    eng = sim.RetailNavEngine(preds, px, cfg, drift_band=0.5)
    eng.affordable = affordable
    return eng, eng.run()


@pytest.mark.parametrize("affordable", [False, True])
def test_engine_buys_exact_lot_and_never_overdraws(affordable):
    # 现金恰好够一手（无滑点信息 → 引擎用最保守档；按该档算出的恰好够一手的资金）
    eng0 = sim.RetailNavEngine(pd.DataFrame({"as_of": [DATES[D1]], "ticker": ["000001.SZ"], "score": [1.0]}), prices(),
                               sim.NavConfig(horizon_td=10, rebalance_step=1, top_n=1, universe_size=2, min_recent_bar_td=1))
    slip = eng0.cfg.tiers.small_bp
    cap = sim.buy_total(100, 50.0, slip, CB, TB)
    eng, r = run(prices(), cap, affordable)
    buys = r["trades"][r["trades"]["side"] == "buy"]
    assert len(buys) >= 1 and buys["raw_shares"].iloc[0] == 100
    assert (r["nav_daily"]["cash"] >= -1e-9).all()


def test_adjustment_change_sells_total_return_position():
    # 总收益近似：持有期间复权因子 ×1.1，清仓卖出的原始股数 = 100 × 1.1（不是整数手也允许），现金不透支
    eng, r = run(prices(adj_jump_day=JUMP, ratio=1.1), 20_000.0)
    tr = r["trades"]
    sells = tr[(tr["side"] == "sell") & (tr["ticker"] == "000001.SZ")]
    bought = tr[(tr["side"] == "buy") & (tr["ticker"] == "000001.SZ")]["raw_shares"].iloc[0]
    assert len(sells) == 1 and sells["raw_shares"].iloc[0] == pytest.approx(bought * 1.1)
    assert (r["nav_daily"]["cash"] >= -1e-9).all()
