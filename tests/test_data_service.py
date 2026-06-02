"""tests/test_data_service.py — 统一数据服务层单元测试 (≥15).

全部基于临时目录的合成数据，不依赖真实 data/。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 合成数据根目录
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synth_root(tmp_path: Path) -> Path:
    """构造最小化的 QuantMind 数据目录树。"""
    root = tmp_path
    (root / "data" / "recommendations").mkdir(parents=True)
    (root / "data" / "feedback").mkdir(parents=True)
    (root / "data" / "paper_trading").mkdir(parents=True)
    (root / "data" / "snapshots" / "2026-01-02").mkdir(parents=True)
    (root / "data" / "regime").mkdir(parents=True)
    (root / "data" / "loss_signals_v4").mkdir(parents=True)
    (root / "data" / "meta_learner").mkdir(parents=True)
    (root / "data" / "alpha_universe").mkdir(parents=True)
    (root / "reports" / "investment_pipeline" / "2026-01-02").mkdir(parents=True)

    # 推荐（已 enrich）
    rec = {
        "as_of": "2026-01-02",
        "generated_at": "2026-01-02T16:00:00",
        "market_summary": "今日推荐 2 只股票，平均 PE 20.0x",
        "top10": [
            {"rank": 1, "ticker": "600000.SH", "lgbm_rank": 1, "lgbm_score": 0.95,
             "reason": "理由A", "name": "浦发银行", "industry": "银行",
             "entry_price": 10.0, "raw_pe_ttm": 8.0, "raw_roe": 12.0},
            {"rank": 2, "ticker": "000001.SZ", "lgbm_rank": 2, "lgbm_score": 0.90,
             "reason": "理由B", "name": "平安银行", "industry": "银行",
             "entry_price": 20.0, "raw_pe_ttm": 32.0, "raw_roe": 9.0},
        ],
    }
    (root / "data" / "recommendations" / "2026-01-02.json").write_text(
        json.dumps(rec), encoding="utf-8")

    # alpha_universe（名称/行业映射）
    pd.DataFrame({
        "ts_code": ["600000.SH", "000001.SZ"],
        "name": ["浦发银行", "平安银行"],
        "industry": ["银行", "银行"],
    }).to_parquet(root / "data" / "alpha_universe" / "alpha_universe.parquet", index=False)

    # snapshot daily_basic（当前价）
    pd.DataFrame({
        "ticker": ["600000.SH", "000001.SZ"],
        "trade_date": ["2026-01-02", "2026-01-02"],
        "close": [11.0, 18.0],   # 600000 +10%, 000001 -10%
    }).to_parquet(root / "data" / "snapshots" / "2026-01-02" / "daily_basic.parquet", index=False)

    # realized_pnl
    pd.DataFrame({
        "as_of_date": ["2025-09-30", "2025-09-30"],
        "ticker": ["600000.SH", "000002.SZ"],
        "entry_date": ["2025-09-30", "2025-09-30"],
        "exit_date": ["2025-12-30", "2025-12-30"],
        "actual_return_63d": [0.15, -0.05],
        "hit": [True, False],
        "entry_price": [9.0, 8.0],
        "exit_price": [10.35, 7.6],
    }).to_parquet(root / "data" / "feedback" / "realized_pnl.parquet", index=False)

    # forward_positions（1 OPEN）
    (root / "data" / "paper_trading" / "forward_positions.json").write_text(json.dumps({
        "generated_at": "2026-01-02T17:00:00",
        "tracking_horizon": "3m",
        "n_open_positions": 1,
        "positions": [{
            "as_of": "2026-01-02", "ticker": "600000.SH", "name": "浦发银行",
            "predicted_rank": 1, "holding_period": "3m",
            "estimated_exit_date": "2026-04-02",
            "entry_price": 10.0, "exit_price": None, "actual_return": None,
            "status": "OPEN",
        }],
    }), encoding="utf-8")

    # strategies.json — 同时含 rich + flat 两种 agent_signals 格式
    strategies = [
        {  # rich
            "ticker": "600000.SH", "rating": "买入", "composite_signal": 0.42,
            "confidence": 0.7, "target_price_1m": 11.0, "target_price_3m": 12.0,
            "stop_loss_price": 9.0, "investment_thesis": "综合论证文字",
            "key_risks": ["风险1"], "key_catalysts": ["催化1"],
            "agent_signals": {
                "ValuationAgent": {"signal": 0.8, "confidence": 0.75, "summary": "估值低"},
                "MomentumAgent":  {"signal": 0.1, "confidence": 0.6, "summary": "动量弱"},
                "QualityAgent":   {"signal": 0.5, "confidence": 0.9, "summary": "质量好"},
                "SentimentAgent": {"signal": 0.0, "confidence": 0.4, "summary": "中性"},
                "RiskAgent":      {"signal": -0.2, "confidence": 0.5, "summary": "尾部风险"},
            },
        },
        {  # flat（近期格式：值为 float）
            "ticker": "000001.SZ", "rating": "谨慎关注", "composite_signal": 0.05,
            "confidence": 0.56,
            "agent_signals": {
                "ValuationAgent": 0.6, "MomentumAgent": 0.0, "QualityAgent": -0.4,
                "SentimentAgent": 0.0, "RiskAgent": 0.0,
            },
        },
    ]
    (root / "reports" / "investment_pipeline" / "2026-01-02" / "strategies.json").write_text(
        json.dumps(strategies), encoding="utf-8")

    # regime_history
    pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "regime": ["neutral", "bull"],
        "bull_prob": [0.3, 0.8], "neutral_prob": [0.6, 0.15], "bear_prob": [0.1, 0.05],
    }).to_parquet(root / "data" / "regime" / "regime_history.parquet", index=False)

    # loss_signals_v4
    (root / "data" / "loss_signals_v4" / "latest.json").write_text(
        json.dumps({"run_ts": "2026-01-02", "overall_health": "OK"}), encoding="utf-8")
    (root / "data" / "loss_signals_v4" / "factor_health.json").write_text(
        json.dumps({"volatility_3m": 0.2}), encoding="utf-8")
    (root / "data" / "loss_signals_v4" / "action_plan.json").write_text(
        json.dumps({"actions": []}), encoding="utf-8")

    # meta_learner
    (root / "data" / "meta_learner" / "meta_learner_v3.meta.json").write_text(json.dumps({
        "version": "v3", "cv_auc": 0.59, "train_auc": 0.75, "n_samples": 140,
        "trained_at": "2026-01-01T00:00:00",
    }), encoding="utf-8")

    return root


@pytest.fixture
def svc(synth_root):
    from app.services.data_service import DataService
    return DataService(root=synth_root)


# ─────────────────────────────────────────────────────────────────────────────
# 推荐
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendations:

    def test_recommendation_dates(self, svc):
        assert svc.get_recommendation_dates() == ["2026-01-02"]

    def test_get_recommendations_has_required_columns(self, svc):
        df = svc.get_recommendations()
        for col in ("ticker", "name", "industry", "entry_price",
                    "current_price", "pnl_pct", "status"):
            assert col in df.columns

    def test_get_recommendations_values(self, svc):
        df = svc.get_recommendations()
        assert len(df) == 2
        row = df[df["ticker"] == "600000.SH"].iloc[0]
        assert row["name"] == "浦发银行"
        assert row["industry"] == "银行"
        assert row["entry_price"] == pytest.approx(10.0)

    def test_get_recommendations_current_price_and_pnl(self, svc):
        df = svc.get_recommendations()
        row = df[df["ticker"] == "600000.SH"].iloc[0]
        assert row["current_price"] == pytest.approx(11.0)
        assert row["pnl_pct"] == pytest.approx(0.10, abs=1e-6)  # 11/10-1

    def test_get_recommendations_missing_date_empty(self, svc):
        df = svc.get_recommendations(date="1999-01-01")
        assert df.empty

    def test_get_recommendations_status_open(self, svc):
        df = svc.get_recommendations()
        row = df[df["ticker"] == "600000.SH"].iloc[0]
        assert row["status"] == "持仓中"  # 在 forward_positions OPEN


# ─────────────────────────────────────────────────────────────────────────────
# 6-Agent 六维分析
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentAnalysis:

    def test_rich_format(self, svc):
        a = svc.get_agent_analysis("600000.SH")
        assert a is not None
        assert a["rating"] == "买入"
        assert set(a["agents"].keys()) == {
            "Valuation", "Momentum", "Quality", "Sentiment", "Risk"}
        val = a["agents"]["Valuation"]
        assert val["signal"] == pytest.approx(0.8)
        assert val["confidence"] == pytest.approx(0.75)
        assert val["summary"] == "估值低"

    def test_flat_format(self, svc):
        a = svc.get_agent_analysis("000001.SZ")
        assert a is not None
        assert len(a["agents"]) == 5
        val = a["agents"]["Valuation"]
        assert val["signal"] == pytest.approx(0.6)
        assert val["confidence"] is None     # flat 格式无置信度

    def test_unknown_ticker_returns_none(self, svc):
        assert svc.get_agent_analysis("999999.SH") is None

    def test_get_all_agent_analysis(self, svc):
        allp = svc.get_all_agent_analysis()
        assert set(allp.keys()) == {"600000.SH", "000001.SZ"}
        assert "investment_thesis" in allp["600000.SH"]


# ─────────────────────────────────────────────────────────────────────────────
# PnL / 持仓
# ─────────────────────────────────────────────────────────────────────────────

class TestPnLAndPositions:

    def test_realized_pnl_shape(self, svc):
        df = svc.get_realized_pnl()
        assert len(df) == 2
        assert "actual_return_63d" in df.columns

    def test_realized_pnl_missing_returns_empty(self, tmp_path):
        from app.services.data_service import DataService
        s = DataService(root=tmp_path)   # 空目录
        df = s.get_realized_pnl()
        assert df.empty

    def test_forward_positions_columns_and_pnl(self, svc):
        df = svc.get_forward_positions()
        assert len(df) == 1
        for col in ("entry_price", "current_price", "pnl_pct"):
            assert col in df.columns
        row = df.iloc[0]
        assert row["entry_price"] == pytest.approx(10.0)
        assert row["current_price"] == pytest.approx(11.0)
        assert row["pnl_pct"] == pytest.approx(0.10, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 模型状态 / Regime / 损失信号
# ─────────────────────────────────────────────────────────────────────────────

class TestModelStatus:

    def test_returns_all_model_keys(self, svc):
        ms = svc.get_model_status()
        for key in ("lgbm_main", "lgbm_gem", "lgbm_star", "lgbm_alpha",
                    "factor_cnn", "board_router", "hmm_regime", "meta_learner"):
            assert key in ms

    def test_missing_lgbm_graceful(self, svc):
        # 合成 root 无 models/ → exists False，不崩溃
        ms = svc.get_model_status()
        assert ms["lgbm_main"].get("exists") is False

    def test_hmm_regime_from_history(self, svc):
        ms = svc.get_model_status()
        assert ms["hmm_regime"]["current_regime"] == "bull"

    def test_meta_learner_status(self, svc):
        ms = svc.get_model_status()
        assert ms["meta_learner"]["version"] == "v3"


class TestRegimeAndLossSignals:

    def test_get_regime(self, svc):
        rg = svc.get_regime()
        assert rg["current"]["current_regime"] == "bull"
        assert not rg["history"].empty

    def test_get_loss_signals(self, svc):
        ls = svc.get_loss_signals()
        assert set(ls.keys()) == {"latest", "factor_health", "action_plan"}
        assert ls["latest"]["overall_health"] == "OK"


# ─────────────────────────────────────────────────────────────────────────────
# 缓存 + 健壮性
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheAndRobustness:

    def test_cache_avoids_second_disk_read(self, svc, monkeypatch):
        import app.services.data_service as mod
        calls = {"n": 0}
        orig = mod.pd.read_parquet

        def counted(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(mod.pd, "read_parquet", counted)
        svc.get_realized_pnl()
        n1 = calls["n"]
        svc.get_realized_pnl()      # 第二次应命中缓存
        assert calls["n"] == n1

    def test_clear_cache_forces_reload(self, svc, monkeypatch):
        import app.services.data_service as mod
        calls = {"n": 0}
        orig = mod.pd.read_parquet

        def counted(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(mod.pd, "read_parquet", counted)
        svc.get_realized_pnl()
        n1 = calls["n"]
        svc.clear_cache()
        svc.get_realized_pnl()
        assert calls["n"] > n1

    def test_empty_root_no_crash(self, tmp_path):
        from app.services.data_service import DataService
        s = DataService(root=tmp_path)
        assert s.get_recommendation_dates() == []
        assert s.get_recommendations().empty
        assert s.get_agent_analysis("X") is None
        assert s.get_realized_pnl().empty
        assert s.get_forward_positions().empty
        assert isinstance(s.get_loss_signals(), dict)
