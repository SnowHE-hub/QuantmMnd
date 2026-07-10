"""Executable NAV 引擎 + gate 测试（合成数据，无网络/无真实数据依赖）.

覆盖任务要求的七类：无未来数据 / 成本计算 / 持仓现金守恒 / 无法成交处理 /
重复运行确定性 / registry 只在 gate 通过时更新 / 12d & 63d 最小集成。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantmind.backtest.wf_costs import stamp_duty_rate
from quantmind.execution.nav_engine import (
    R_SUSPENDED,
    ExecutableNavEngine,
    NavConfig,
)
from quantmind.execution.nav_gate import (
    GateThresholds,
    apply_gate_to_registry,
    evaluate_gate,
)

# ── 合成数据工具 ─────────────────────────────────────────────────────────────

DATES = pd.bdate_range("2024-01-01", periods=120)


def synth_prices(
    spec: dict[str, tuple[float, float]],       # ticker -> (close, amount)
    dates: pd.DatetimeIndex = DATES,
    missing: dict[str, set] | None = None,       # ticker -> 缺 bar 的日期集合（停牌）
    drift: dict[str, float] | None = None,       # ticker -> 每日漂移（如 0.01）
) -> pd.DataFrame:
    missing = missing or {}
    drift = drift or {}
    rows = []
    for t, (base, amount) in spec.items():
        close_prev = base
        for i, d in enumerate(dates):
            close = base * (1 + drift.get(t, 0.0)) ** i
            if d in missing.get(t, set()):
                close_prev = close
                continue
            rows.append({
                "ts_code": t, "trade_date": d,
                "open": close, "high": close * 1.015, "low": close * 0.985,
                "close": close, "pre_close": close_prev,
                "adj_factor": 1.0, "amount": amount,
            })
            close_prev = close
    return pd.DataFrame(rows)


def synth_preds(as_of_dates, scores: dict[str, float]) -> pd.DataFrame:
    rows = [{"as_of": a, "ticker": t, "score": s}
            for a in as_of_dates for t, s in scores.items()]
    return pd.DataFrame(rows)


def base_cfg(**kw) -> NavConfig:
    d = {"horizon_td": 12, "rebalance_step": 3, "top_n": 2, "universe_size": 10,
         "min_recent_bar_td": 5}
    d.update(kw)
    return NavConfig(**d)


# ── 1. 无未来数据 ────────────────────────────────────────────────────────────

def test_no_lookahead_fills_and_universe():
    """成交严格在决策日之后；PIT universe 不受 as_of 之后的成交额影响。"""
    # B 的 amount 在 as_of 之后暴涨：若引擎偷看未来，B 会进 universe
    as_of = DATES[40]
    rows = []
    for t, amt_before, amt_after in [("AAA.SZ", 100.0, 100.0), ("BBB.SZ", 1.0, 99999.0)]:
        for d in DATES:
            amt = amt_before if d <= as_of else amt_after
            close = 10.0
            rows.append({"ts_code": t, "trade_date": d, "open": close,
                         "high": close * 1.01, "low": close * 0.99, "close": close,
                         "pre_close": close, "adj_factor": 1.0, "amount": amt})
    prices = pd.DataFrame(rows)
    preds = synth_preds([as_of], {"AAA.SZ": 0.1, "BBB.SZ": 0.9})  # B 分数更高
    eng = ExecutableNavEngine(preds, prices, base_cfg(top_n=1, universe_size=1))
    res = eng.run()

    assert res["targets"].iloc[0]["targets"] == ["AAA.SZ"]        # PIT：B 未进池
    assert (res["filtered"]["ticker"] == "BBB.SZ").any()
    trades = res["trades"]
    fills = pd.to_datetime(trades["fill_date"])
    decisions = pd.to_datetime(trades["decision_asof"])
    assert (fills > decisions).all()                              # T+1：成交严格晚于决策


def test_t_plus_1_no_same_day_buy_sell():
    """同一票不存在 buy 与 sell 同日成交（T+1 结构性保证）。"""
    prices = synth_prices({"AAA.SZ": (10, 100), "BBB.SZ": (20, 90), "CCC.SZ": (30, 80)})
    # 每个 as_of 目标轮换 → 产生频繁买卖
    preds = pd.concat([
        synth_preds([DATES[30]], {"AAA.SZ": 0.9, "BBB.SZ": 0.5, "CCC.SZ": 0.1}),
        synth_preds([DATES[35]], {"AAA.SZ": 0.1, "BBB.SZ": 0.9, "CCC.SZ": 0.5}),
        synth_preds([DATES[40]], {"AAA.SZ": 0.5, "BBB.SZ": 0.1, "CCC.SZ": 0.9}),
    ])
    eng = ExecutableNavEngine(preds, prices, base_cfg(top_n=1, rebalance_step=1))
    res = eng.run()
    tr = res["trades"]
    dup = tr.groupby(["ticker", "fill_date"])["side"].nunique()
    assert (dup <= 1).all()


# ── 2. 交易成本 ──────────────────────────────────────────────────────────────

def test_cost_calculation_exact():
    """买卖成本逐笔核对：滑点档（无 amihud→最保守 30bp）+佣金+过户+时变印花（卖）。"""
    prices = synth_prices({"AAA.SZ": (10, 100), "BBB.SZ": (10, 90)})
    preds = pd.concat([
        synth_preds([DATES[30]], {"AAA.SZ": 0.9, "BBB.SZ": 0.1}),
        synth_preds([DATES[40]], {"AAA.SZ": 0.1, "BBB.SZ": 0.9}),   # A 出 B 进
    ])
    cfg = base_cfg(top_n=1, rebalance_step=1)
    res = ExecutableNavEngine(preds, prices, cfg).run()
    tr = res["trades"]

    buy = tr[(tr["side"] == "buy") & (tr["ticker"] == "AAA.SZ")].iloc[0]
    exp_buy_rate = (30.0 + 3.0 + 0.2) / 1e4
    assert buy["cost_rate"] == pytest.approx(exp_buy_rate)
    assert buy["cost"] == pytest.approx(buy["gross_value"] * exp_buy_rate)

    sell = tr[(tr["side"] == "sell") & (tr["ticker"] == "AAA.SZ")].iloc[0]
    exp_sell_rate = exp_buy_rate + stamp_duty_rate(sell["fill_date"])  # 2024 → 0.0005
    assert stamp_duty_rate(sell["fill_date"]) == pytest.approx(0.0005)
    assert sell["cost_rate"] == pytest.approx(exp_sell_rate)
    assert sell["cost"] == pytest.approx(sell["gross_value"] * exp_sell_rate)


# ── 3. 持仓与现金守恒 ────────────────────────────────────────────────────────

def test_cash_and_value_conservation():
    prices = synth_prices({"AAA.SZ": (10, 100), "BBB.SZ": (20, 90), "CCC.SZ": (5, 80)},
                          drift={"AAA.SZ": 0.002, "BBB.SZ": -0.001})
    preds = pd.concat([
        synth_preds([DATES[30]], {"AAA.SZ": 0.9, "BBB.SZ": 0.8, "CCC.SZ": 0.1}),
        synth_preds([DATES[45]], {"AAA.SZ": 0.1, "BBB.SZ": 0.8, "CCC.SZ": 0.9}),
    ])
    res = ExecutableNavEngine(preds, prices, base_cfg(top_n=2, rebalance_step=1)).run()
    nav = res["nav_daily"].set_index("date")
    hold = res["holdings_daily"]

    assert (nav["cash"] >= -1e-12).all()                          # 现金永不透支
    hv = hold.groupby("date")["value"].sum().reindex(nav.index).fillna(0.0)
    assert np.allclose(nav["nav_net"], nav["cash"] + hv, atol=1e-9)   # NAV=现金+持仓
    # 守恒：NAV 变化 = 持仓 mtm 盈亏 − 成本（无凭空创造/消失）
    assert nav["nav_gross"].iloc[-1] == pytest.approx(
        nav["nav_net"].iloc[-1] + nav["cum_cost"].iloc[-1])


# ── 4. 无法成交处理 ──────────────────────────────────────────────────────────

def test_suspended_buy_retries_then_fills():
    """成交日停牌 → 记拒单 reason code，复牌日按当日开盘补建（设计 2.2）。"""
    susp = set(DATES[31:34])   # as_of=DATES[30]，随后 3 天停牌
    prices = synth_prices({"AAA.SZ": (10, 100), "BBB.SZ": (20, 90)},
                          missing={"AAA.SZ": susp})
    preds = synth_preds([DATES[30]], {"AAA.SZ": 0.9, "BBB.SZ": 0.1})
    res = ExecutableNavEngine(preds, prices, base_cfg(top_n=1)).run()

    rej = res["rejected_trades"]
    assert (rej["reason_code"] == R_SUSPENDED).sum() == 3          # 每次尝试都有记录
    tr = res["trades"]
    fill = tr[(tr["side"] == "buy") & (tr["ticker"] == "AAA.SZ")].iloc[0]
    assert pd.Timestamp(fill["fill_date"]) == DATES[34]            # 复牌日成交


def test_held_suspended_forced_hold_and_writeoff():
    """持仓票长期无 bar 且无未来数据 → 按末价强制清仓并记 reason。"""
    # AAA 在 DATES[40] 之后彻底消失（退市）
    gone = set(DATES[41:])
    prices = synth_prices({"AAA.SZ": (10, 100), "BBB.SZ": (20, 90)},
                          missing={"AAA.SZ": gone})
    preds = synth_preds([DATES[30]], {"AAA.SZ": 0.9, "BBB.SZ": 0.1})
    res = ExecutableNavEngine(preds, prices,
                              base_cfg(top_n=1, delist_writeoff_td=10)).run()
    tr = res["trades"]
    wo = tr[tr["status"] == "delisted_writeoff"]
    assert len(wo) == 1 and wo.iloc[0]["ticker"] == "AAA.SZ"
    nav = res["nav_daily"]
    assert nav["n_positions"].iloc[-1] == 0                        # 已清仓
    assert nav["cash"].iloc[-1] > 0.9                              # 资金按末价回笼（扣成本）


# ── 5. 重复运行确定性 ────────────────────────────────────────────────────────

def test_deterministic_reruns():
    prices = synth_prices({"AAA.SZ": (10, 100), "BBB.SZ": (20, 90), "CCC.SZ": (5, 80)},
                          drift={"AAA.SZ": 0.003, "CCC.SZ": -0.002})
    preds = pd.concat([
        synth_preds([DATES[30]], {"AAA.SZ": 0.9, "BBB.SZ": 0.5, "CCC.SZ": 0.1}),
        synth_preds([DATES[45]], {"AAA.SZ": 0.1, "BBB.SZ": 0.5, "CCC.SZ": 0.9}),
    ])
    r1 = ExecutableNavEngine(preds, prices, base_cfg(rebalance_step=1)).run()
    r2 = ExecutableNavEngine(preds, prices, base_cfg(rebalance_step=1)).run()
    pd.testing.assert_frame_equal(r1["nav_daily"], r2["nav_daily"])
    pd.testing.assert_frame_equal(r1["trades"], r2["trades"])
    pd.testing.assert_frame_equal(r1["rejected_trades"], r2["rejected_trades"])
    assert json.dumps(r1["summary"], default=str) == json.dumps(r2["summary"], default=str)


# ── 6. registry 只在 gate 通过时更新 ─────────────────────────────────────────

def _fake_summary(ann_excess, maxdd, ir, yearly) -> dict:
    return {"nav": {"ann_net_excess": ann_excess, "max_drawdown_net": maxdd,
                    "information_ratio": ir, "ann_return_net": 0.1,
                    "yearly_net_excess": yearly},
            "turnover": {"annualized_twoway": 6.0},
            "costs": {"total_cost_nav_units": 0.01},
            "period": {"start": "2022-01-01", "end": "2026-01-01",
                       "n_trading_days": 1000, "n_rebalances": 40}}


def _register_fake(monkeypatch, tmp_path):
    from quantmind.contracts import model_registry as MR  # noqa: N812
    monkeypatch.setattr(MR, "REGISTRY_PATH", tmp_path / "reg.json")
    MR.register(MR.ModelRecord(
        model_id="fake_seed", label="forward_return_12d", horizon="short",
        feature_set="full_35_16_158", data_version="v6",
        gate_status="research_candidate_pending_nav",
        metrics={"gate_pass": False}))
    return MR


def test_registry_untouched_on_gate_fail(tmp_path, monkeypatch):
    MR = _register_fake(monkeypatch, tmp_path)
    before = (tmp_path / "reg.json").read_bytes()
    summary = _fake_summary(0.02, -0.10, 1.5, {"2023": 0.05})   # 净超额不达标
    gate = evaluate_gate(summary, GateThresholds())
    assert not gate["gate_pass"]
    out = apply_gate_to_registry("fake_seed", gate, summary)
    assert not out["registry_updated"]
    assert (tmp_path / "reg.json").read_bytes() == before        # 一个字节都没变
    assert MR.get("fake_seed").gate_status == "research_candidate_pending_nav"


def test_registry_updated_only_on_pass(tmp_path, monkeypatch):
    MR = _register_fake(monkeypatch, tmp_path)
    summary = _fake_summary(0.08, -0.10, 1.5, {"2023": 0.05, "2024": 0.03})
    gate = evaluate_gate(summary, GateThresholds())
    assert gate["gate_pass"]
    out = apply_gate_to_registry("fake_seed", gate, summary)
    assert out["registry_updated"]
    rec = MR.get("fake_seed")
    assert rec.metrics["gate_pass"] is True
    assert rec.metrics["executable_nav"]["ann_net_excess"] == pytest.approx(0.08)
    assert rec.gate_status == "research_candidate_pending_nav"    # 不自动升 production


def test_gate_fails_on_negative_year():
    summary = _fake_summary(0.08, -0.10, 1.5, {"2023": 0.05, "2024": -0.01})
    gate = evaluate_gate(summary, GateThresholds())
    assert not gate["gate_pass"]
    bad = next(c for c in gate["checks"]
               if c["criterion"] == "yearly_net_excess_all_positive")
    assert not bad["passed"]


def test_gate_dry_run_never_writes(tmp_path, monkeypatch):
    _register_fake(monkeypatch, tmp_path)
    before = (tmp_path / "reg.json").read_bytes()
    summary = _fake_summary(0.08, -0.10, 1.5, {"2023": 0.05})
    gate = evaluate_gate(summary, GateThresholds())
    assert gate["gate_pass"]
    out = apply_gate_to_registry("fake_seed", gate, summary, dry_run=True)
    assert not out["registry_updated"]
    assert (tmp_path / "reg.json").read_bytes() == before


# ── 7. 12d / 63d 最小集成 ────────────────────────────────────────────────────

@pytest.mark.parametrize("step,horizon", [(3, 12), (13, 63)])
def test_mini_integration(step, horizon):
    """12d(step3) / 63d(step13) 配置在合成面板上端到端跑通并产出全套结果。"""
    tickers = {f"T{i:03d}.SZ": (10.0 + i, 100.0 - i) for i in range(6)}
    prices = synth_prices(tickers, drift={"T000.SZ": 0.002, "T001.SZ": 0.001})
    as_of_list = list(DATES[30:110:5])   # 16 个 as_of（周频网格）
    rng_scores = {t: (0.9 - 0.1 * i) for i, t in enumerate(tickers)}
    preds = synth_preds(as_of_list, rng_scores)

    cfg = NavConfig(horizon_td=horizon, rebalance_step=step, top_n=2,
                    universe_size=6, min_recent_bar_td=5)
    res = ExecutableNavEngine(preds, prices, cfg).run()
    s = res["summary"]

    assert s["period"]["n_rebalances"] == len(as_of_list[::step])
    assert len(res["nav_daily"]) > 0
    for key in ("nav", "costs", "turnover", "execution", "coverage"):
        assert key in s
    assert np.isfinite(s["nav"]["ann_net_excess"])
    assert np.isfinite(s["nav"]["max_drawdown_net"])
    assert s["execution"]["n_trades_filled"] > 0
    # gate 判定端到端（不落盘）
    gate = evaluate_gate(s, GateThresholds())
    assert isinstance(gate["gate_pass"], bool)
    assert len(gate["checks"]) == 5
