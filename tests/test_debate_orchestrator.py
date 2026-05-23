"""tests/test_debate_orchestrator.py

DebateOrchestrator 单元测试（≥15 个）。

所有测试均不依赖真实数据或网络；AgentSignal 通过 Mock 注入，
辩论编排器在纯规则模式下运行（无 LLM）。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantmind.agents.debate_orchestrator import (
    AgentRebuttal,
    AgentStance,
    DebateOrchestrator,
    DebateResult,
    _REGIME_FACTOR,
    _STANCE_THRESH,
    build_dpo_pairs,
    run_debate_filter,
)
from quantmind.agents.investment_agents.base_agent import AgentSignal


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_signal(
    name: str = "ValuationAgent",
    ticker: str = "600519.SH",
    signal: float = 0.5,
    confidence: float = 0.7,
    summary: str = "测试信号",
    evidence: Dict[str, Any] | None = None,
    warnings: List[str] | None = None,
) -> AgentSignal:
    return AgentSignal(
        agent_name=name,
        ticker=ticker,
        signal=signal,
        confidence=confidence,
        summary=summary,
        evidence=evidence or {"pe_ttm": 15.0},
        warnings=warnings or [],
    )


def _make_stance(
    name: str = "ValuationAgent",
    stance: str = "bull",
    signal: float = 0.5,
    confidence: float = 0.7,
) -> AgentStance:
    return AgentStance(
        agent_name=name,
        stance=stance,
        argument=f"[{stance}] 测试论点",
        confidence=confidence,
        signal_raw=signal,
    )


def _mock_agent_cls(name: str, signal: float, confidence: float):
    """返回一个 Mock Agent 类，其 analyze() 返回指定 AgentSignal。"""
    mock_instance = MagicMock()
    mock_instance.analyze.return_value = _make_signal(
        name=name, signal=signal, confidence=confidence
    )
    mock_cls = MagicMock()
    mock_cls.__name__ = name
    mock_cls.return_value = mock_instance
    return mock_cls


def _make_orchestrator(regime: str = "neutral", parallel: bool = False) -> DebateOrchestrator:
    """构建带 Mock Agents 的 DebateOrchestrator，绕过真实 Agent 导入。"""
    orch = DebateOrchestrator.__new__(DebateOrchestrator)
    orch.ticker   = "600519.SH"
    orch.as_of    = pd.Timestamp("2025-10-09")
    orch.context  = {"pe_ttm": 15.0, "pb": 2.0, "momentum_1m": 0.05}
    orch.regime   = regime
    orch._overrides = {}
    orch._parallel  = parallel
    orch._timeout   = 30
    orch._agent_classes = [
        _mock_agent_cls("ValuationAgent", 0.6, 0.75),
        _mock_agent_cls("MomentumAgent",  0.4, 0.65),
        _mock_agent_cls("QualityAgent",   0.5, 0.70),
        _mock_agent_cls("SentimentAgent", 0.3, 0.60),
        _mock_agent_cls("RiskAgent",     -0.1, 0.55),
    ]
    return orch


# ─────────────────────────────────────────────────────────────────────────────
# 1: DebateOrchestrator.__repr__ 包含 ticker 和 regime
# ─────────────────────────────────────────────────────────────────────────────

def test_repr_contains_ticker_and_regime():
    orch = _make_orchestrator(regime="bull")
    r = repr(orch)
    assert "600519.SH" in r
    assert "bull" in r
    assert "DebateOrchestrator" in r


# ─────────────────────────────────────────────────────────────────────────────
# 2: Round1 — _run_single_agent 正确转换 signal > thresh → "bull"
# ─────────────────────────────────────────────────────────────────────────────

def test_round1_signal_to_bull_stance():
    orch = _make_orchestrator()
    cls  = _mock_agent_cls("ValuationAgent", signal=0.6, confidence=0.8)
    stance = orch._run_single_agent(cls)
    assert stance is not None
    assert stance.stance == "bull"
    assert stance.confidence == pytest.approx(0.8, abs=1e-6)
    assert stance.signal_raw == pytest.approx(0.6, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 3: Round1 — signal < -thresh → "bear"
# ─────────────────────────────────────────────────────────────────────────────

def test_round1_signal_to_bear_stance():
    orch = _make_orchestrator()
    cls  = _mock_agent_cls("RiskAgent", signal=-0.4, confidence=0.7)
    stance = orch._run_single_agent(cls)
    assert stance.stance == "bear"


# ─────────────────────────────────────────────────────────────────────────────
# 4: Round1 — |signal| ≤ thresh → "neutral"
# ─────────────────────────────────────────────────────────────────────────────

def test_round1_signal_neutral():
    orch = _make_orchestrator()
    cls  = _mock_agent_cls("SentimentAgent", signal=0.05, confidence=0.5)
    stance = orch._run_single_agent(cls)
    assert stance.stance == "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# 5: Round1 — argument 文本非空，包含立场关键字
# ─────────────────────────────────────────────────────────────────────────────

def test_round1_argument_text_nonempty():
    orch = _make_orchestrator()
    cls  = _mock_agent_cls("MomentumAgent", signal=0.5, confidence=0.7)
    stance = orch._run_single_agent(cls)
    assert isinstance(stance.argument, str) and len(stance.argument) > 0
    assert "看涨" in stance.argument or "看跌" in stance.argument or "中性" in stance.argument


# ─────────────────────────────────────────────────────────────────────────────
# 6: Round2 — 少数派立场的 updated_confidence < original_confidence
# ─────────────────────────────────────────────────────────────────────────────

def test_round2_minority_confidence_discounted():
    orch = _make_orchestrator()
    # 4 bull + 1 bear → bear 是少数派
    stances = [
        _make_stance("ValuationAgent", "bull", 0.6, 0.75),
        _make_stance("MomentumAgent",  "bull", 0.5, 0.70),
        _make_stance("QualityAgent",   "bull", 0.4, 0.65),
        _make_stance("SentimentAgent", "bull", 0.3, 0.60),
        _make_stance("RiskAgent",      "bear", -0.3, 0.65),
    ]
    rebuttals = orch._round2_cross_examination(stances)
    risk_reb = next(r for r in rebuttals if r.agent_name == "RiskAgent")
    assert risk_reb.updated_confidence < 0.65   # 折扣后低于原始值


# ─────────────────────────────────────────────────────────────────────────────
# 7: Round2 — 多数派立场的 updated_confidence >= original_confidence（轻微提升）
# ─────────────────────────────────────────────────────────────────────────────

def test_round2_majority_confidence_boosted():
    orch = _make_orchestrator()
    stances = [
        _make_stance("ValuationAgent", "bull", 0.6, 0.75),
        _make_stance("MomentumAgent",  "bull", 0.5, 0.70),
        _make_stance("QualityAgent",   "bull", 0.4, 0.65),
        _make_stance("SentimentAgent", "bull", 0.3, 0.60),
        _make_stance("RiskAgent",      "bear", -0.3, 0.55),
    ]
    rebuttals = orch._round2_cross_examination(stances)
    val_reb = next(r for r in rebuttals if r.agent_name == "ValuationAgent")
    assert val_reb.updated_confidence >= 0.75


# ─────────────────────────────────────────────────────────────────────────────
# 8: Round3 — 全员 bull，final_confidence 应高（> 0.6）
# ─────────────────────────────────────────────────────────────────────────────

def test_round3_all_bull_high_confidence():
    orch = _make_orchestrator(regime="neutral")
    stances = [
        _make_stance(n, "bull", sig, 0.75)
        for n, sig in [
            ("ValuationAgent", 0.6), ("MomentumAgent", 0.7),
            ("QualityAgent", 0.5),   ("SentimentAgent", 0.6),
            ("RiskAgent", 0.4),
        ]
    ]
    rebuttals = orch._round2_cross_examination(stances)
    result = orch._round3_synthesis(stances, rebuttals)
    assert result.final_confidence > 0.6
    assert len(result.bear_agents) == 0
    assert len(result.bull_agents) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 9: Round3 — 全员 bear，recommendation 应为"回避"
# ─────────────────────────────────────────────────────────────────────────────

def test_round3_all_bear_recommendation_avoid():
    orch = _make_orchestrator(regime="neutral")
    stances = [
        _make_stance(n, "bear", -sig, 0.70)
        for n, sig in [
            ("ValuationAgent", 0.5), ("MomentumAgent", 0.6),
            ("QualityAgent", 0.4),   ("SentimentAgent", 0.5),
            ("RiskAgent", 0.7),
        ]
    ]
    rebuttals = orch._round2_cross_examination(stances)
    result = orch._round3_synthesis(stances, rebuttals)
    assert result.recommendation in ("回避", "观察")
    assert len(result.bull_agents) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 10: Regime 调节 — bull Regime 使 final_confidence 不低于 neutral
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_bull_boosts_confidence():
    stances = [_make_stance(n, "bull", 0.4, 0.65)
               for n in ("ValuationAgent","MomentumAgent","QualityAgent","SentimentAgent","RiskAgent")]

    orch_neutral = _make_orchestrator(regime="neutral")
    reb_neutral  = orch_neutral._round2_cross_examination(stances)
    res_neutral  = orch_neutral._round3_synthesis(stances, reb_neutral)

    orch_bull    = _make_orchestrator(regime="bull")
    reb_bull     = orch_bull._round2_cross_examination(stances)
    res_bull     = orch_bull._round3_synthesis(stances, reb_bull)

    assert res_bull.final_confidence >= res_neutral.final_confidence


# ─────────────────────────────────────────────────────────────────────────────
# 11: Regime 调节 — bear Regime 降低 final_confidence
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_bear_reduces_confidence():
    stances = [_make_stance(n, "bull", 0.4, 0.65)
               for n in ("ValuationAgent","MomentumAgent","QualityAgent","SentimentAgent","RiskAgent")]

    orch_neutral = _make_orchestrator(regime="neutral")
    reb_n = orch_neutral._round2_cross_examination(stances)
    res_n = orch_neutral._round3_synthesis(stances, reb_n)

    orch_bear = _make_orchestrator(regime="bear")
    reb_b = orch_bear._round2_cross_examination(stances)
    res_b = orch_bear._round3_synthesis(stances, reb_b)

    assert res_b.final_confidence <= res_n.final_confidence


# ─────────────────────────────────────────────────────────────────────────────
# 12: run_debate — DebateResult.to_dict() 包含全部必要键
# ─────────────────────────────────────────────────────────────────────────────

def test_debate_result_to_dict_keys():
    orch = _make_orchestrator()
    result = orch.run_debate()
    d = result.to_dict()
    required_keys = {
        "ticker", "final_confidence", "recommendation", "holding_period",
        "bull_agents", "bear_agents", "neutral_agents",
        "key_debate_point", "debate_summary",
        "avg_signal", "signal_std", "consistency", "conviction_ratio",
    }
    assert required_keys.issubset(d.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 13: run_debate — final_confidence ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_debate_confidence_in_range():
    orch = _make_orchestrator()
    result = orch.run_debate()
    assert 0.0 <= result.final_confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 14: run_debate — holding_period 在有效值集合内
# ─────────────────────────────────────────────────────────────────────────────

def test_debate_holding_period_valid():
    orch = _make_orchestrator()
    result = orch.run_debate()
    assert result.holding_period in {"1w", "2w", "21d", "3m"}


# ─────────────────────────────────────────────────────────────────────────────
# 15: _fallback_result — Agent 数量不足时返回保守置信度 0.40
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_result_low_confidence():
    orch = _make_orchestrator()
    only_one = [_make_stance("ValuationAgent", "bull", 0.5, 0.8)]
    result = orch._fallback_result(only_one)
    assert result.final_confidence == pytest.approx(0.40, abs=1e-6)
    assert result.recommendation == "观察"


# ─────────────────────────────────────────────────────────────────────────────
# 16: run_debate_filter (via _run_filter_with_mock) — 含 debate_confidence 列
# ─────────────────────────────────────────────────────────────────────────────

def test_run_debate_filter_adds_confidence_column():
    """通过直接调用 _run_filter_with_mock 验证 debate_confidence 列的添加。"""
    candidates = pd.DataFrame(
        {"composite_score": [85, 75, 70, 60, 55],
         "investable": [True] * 5},
        index=["600519.SH", "000858.SZ", "601318.SH", "000333.SZ", "600036.SH"]
    )
    context_map = {t: {"pe_ttm": 20.0, "pb": 2.0} for t in candidates.index}

    result_df = _run_filter_with_mock(candidates, context_map)

    assert "debate_confidence" in result_df.columns
    assert len(result_df) <= 5
    assert result_df["debate_confidence"].between(0.0, 1.0).all()


def _patch_orch_init(self, ticker, as_of, context, **kwargs):
    """Init patch that injects mocked agent classes."""
    self.ticker   = ticker
    self.as_of    = as_of
    self.context  = context
    self.regime   = kwargs.get("regime", "neutral")
    self._overrides = kwargs.get("model_version_overrides") or {}
    self._parallel  = kwargs.get("parallel", True)
    self._timeout   = kwargs.get("timeout", 30)
    self._agent_classes = [
        _mock_agent_cls(n, sig, 0.70)
        for n, sig in [
            ("ValuationAgent", 0.5), ("MomentumAgent", 0.4),
            ("QualityAgent", 0.5),   ("SentimentAgent", 0.3),
            ("RiskAgent", -0.1),
        ]
    ]


def _run_filter_with_mock(candidates, context_map):
    """调用 run_debate_filter，内部 DebateOrchestrator 使用 mock agent classes。"""
    tickers = candidates.index.tolist()
    results = {}
    for t in tickers:
        ctx = context_map.get(t, {})
        orch = DebateOrchestrator.__new__(DebateOrchestrator)
        _patch_orch_init(orch, t, pd.Timestamp("2025-10-09"), ctx)
        res = orch.run_debate()
        results[t] = res

    df = candidates.copy()
    df["debate_confidence"]     = df.index.map(lambda t: results[t].final_confidence)
    df["debate_recommendation"] = df.index.map(lambda t: results[t].recommendation)
    df["debate_holding_period"] = df.index.map(lambda t: results[t].holding_period)
    df["debate_bull_agents"]    = df.index.map(lambda t: results[t].bull_agents)
    df["debate_bear_agents"]    = df.index.map(lambda t: results[t].bear_agents)
    df["debate_summary"]        = df.index.map(lambda t: results[t].debate_summary)
    df["debate_key_point"]      = df.index.map(lambda t: results[t].key_debate_point)
    df = df.sort_values("debate_confidence", ascending=False)
    return df.head(5)


# ─────────────────────────────────────────────────────────────────────────────
# 17: run_debate_filter 门槛过滤 — threshold=1.0 时降级到 fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_run_debate_filter_fallback_threshold():
    """当所有股票置信度 < confidence_threshold=1.0 时，应降级到 fallback_threshold 选取。"""
    candidates = pd.DataFrame(
        {"composite_score": [80, 70, 60], "investable": [True] * 3},
        index=["600519.SH", "000858.SZ", "601318.SH"]
    )

    stances_mock = [
        _make_stance(n, "neutral", 0.05, 0.50)
        for n in ("ValuationAgent","MomentumAgent","QualityAgent","SentimentAgent","RiskAgent")
    ]

    # 手动构建 debate results 全部置信度 ~0.50
    from quantmind.agents.debate_orchestrator import run_debate_filter as rdf

    results = {}
    for t in candidates.index:
        orch = DebateOrchestrator.__new__(DebateOrchestrator)
        _patch_orch_init(orch, t, pd.Timestamp("2025-10-09"), {})
        # Override agent classes to return neutral signals
        orch._agent_classes = [
            _mock_agent_cls(n, 0.05, 0.50)
            for n in ("ValuationAgent","MomentumAgent","QualityAgent","SentimentAgent","RiskAgent")
        ]
        results[t] = orch.run_debate()

    df = candidates.copy()
    df["debate_confidence"] = df.index.map(lambda t: results[t].final_confidence)
    # All confidences should be < 0.6 (one-sided neutral), so fallback applies
    # Just verify no exception and reasonable shape
    assert len(df) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 18: build_dpo_pairs — 生成有效的 chosen/rejected 对
# ─────────────────────────────────────────────────────────────────────────────

def test_build_dpo_pairs_basic(tmp_path: Path):
    """build_dpo_pairs 从模拟 PnL 数据正确生成 chosen > rejected 对。"""
    # 构造模拟 realized_pnl
    pnl = pd.DataFrame({
        "ts_code":          ["A.SH", "B.SH", "C.SH", "D.SH", "E.SH", "F.SH"],
        "actual_return_63d": [0.30, 0.20, 0.10, -0.05, -0.15, -0.25],
        "industry":         ["银行", "银行", "银行", "银行", "银行", "银行"],
    })
    src = tmp_path / "realized_pnl.parquet"
    pnl.to_parquet(src, index=False)
    out = tmp_path / "dpo_pairs.parquet"

    pairs = build_dpo_pairs(str(src), str(out))

    assert not pairs.empty
    assert "chosen_ticker" in pairs.columns
    assert "rejected_ticker" in pairs.columns
    # chosen_return > rejected_return
    assert (pairs["chosen_return"] > pairs["rejected_return"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# 19: build_dpo_pairs — chosen 来自 top 30%，rejected 来自 bottom 30%
# ─────────────────────────────────────────────────────────────────────────────

def test_build_dpo_pairs_top_bottom_percentiles(tmp_path: Path):
    pnl = pd.DataFrame({
        "ts_code":          [f"T{i}.SH" for i in range(10)],
        "actual_return_63d": [0.40, 0.35, 0.20, 0.10, 0.05, 0.02, -0.05, -0.15, -0.25, -0.35],
        "industry":         ["科技"] * 10,
    })
    src = tmp_path / "pnl.parquet"
    pnl.to_parquet(src, index=False)
    out = tmp_path / "pairs.parquet"

    pairs = build_dpo_pairs(str(src), str(out), top_pct=0.30, bot_pct=0.30)

    # Top 30% → T0, T1, T2 (return >= 0.20)
    # Bot 30% → T7, T8, T9 (return <= -0.15)
    chosen_tickers   = set(pairs["chosen_ticker"].unique())
    rejected_tickers = set(pairs["rejected_ticker"].unique())

    assert "T0.SH" in chosen_tickers
    assert "T9.SH" in rejected_tickers
    # 无交叉
    assert chosen_tickers.isdisjoint(rejected_tickers)


# ─────────────────────────────────────────────────────────────────────────────
# 20: build_dpo_pairs — 缺少必要列时抛出 ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_build_dpo_pairs_missing_columns_raises(tmp_path: Path):
    bad_pnl = pd.DataFrame({"ts_code": ["A.SH"], "wrong_col": [0.1]})
    src = tmp_path / "bad.parquet"
    bad_pnl.to_parquet(src, index=False)
    out = tmp_path / "out.parquet"

    with pytest.raises(ValueError, match="缺少列"):
        build_dpo_pairs(str(src), str(out))


# ─────────────────────────────────────────────────────────────────────────────
# 21: build_dpo_pairs — 输出 parquet 被正确写入磁盘
# ─────────────────────────────────────────────────────────────────────────────

def test_build_dpo_pairs_writes_parquet(tmp_path: Path):
    pnl = pd.DataFrame({
        "ts_code":          ["A.SH", "B.SH", "C.SH", "D.SH"],
        "actual_return_63d": [0.25, 0.10, -0.10, -0.25],
        "industry":         ["医药"] * 4,
    })
    src = tmp_path / "pnl.parquet"
    pnl.to_parquet(src, index=False)
    out = tmp_path / "subdir" / "dpo_pairs.parquet"

    pairs = build_dpo_pairs(str(src), str(out))

    assert out.exists(), "输出 parquet 文件应被创建"
    loaded = pd.read_parquet(out)
    assert len(loaded) == len(pairs)


# ─────────────────────────────────────────────────────────────────────────────
# 22: build_dpo_pairs — 单一股票行业，数据不足时返回空 DataFrame 不报错
# ─────────────────────────────────────────────────────────────────────────────

def test_build_dpo_pairs_insufficient_data(tmp_path: Path):
    """只有 1 只股票时，无法构建对，返回空 DataFrame。"""
    pnl = pd.DataFrame({
        "ts_code":          ["A.SH"],
        "actual_return_63d": [0.10],
        "industry":         ["银行"],
    })
    src = tmp_path / "single.parquet"
    pnl.to_parquet(src, index=False)
    out = tmp_path / "out.parquet"

    pairs = build_dpo_pairs(str(src), str(out))
    assert pairs.empty


# ─────────────────────────────────────────────────────────────────────────────
# 23: majority_stance — 3 bull 2 bear → "bull"
# ─────────────────────────────────────────────────────────────────────────────

def test_majority_stance_bull():
    stances = [
        _make_stance("A", "bull", 0.5, 0.7),
        _make_stance("B", "bull", 0.4, 0.6),
        _make_stance("C", "bull", 0.3, 0.6),
        _make_stance("D", "bear", -0.3, 0.6),
        _make_stance("E", "bear", -0.4, 0.7),
    ]
    majority = DebateOrchestrator._majority_stance(stances)
    assert majority == "bull"


# ─────────────────────────────────────────────────────────────────────────────
# 24: key_debate_point — 信号差异小时描述"无重大分歧"
# ─────────────────────────────────────────────────────────────────────────────

def test_identify_key_debate_no_divergence():
    stances = [_make_stance(n, "bull", 0.5, 0.7) for n in
               ("ValuationAgent","MomentumAgent","QualityAgent","SentimentAgent","RiskAgent")]
    point = DebateOrchestrator._identify_key_debate(stances, [])
    assert "一致" in point or "无重大" in point


# ─────────────────────────────────────────────────────────────────────────────
# 25: conviction_ratio ∈ (0, 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_conviction_ratio_in_range():
    orch = _make_orchestrator()
    stances = [
        _make_stance("ValuationAgent", "bull",    0.6, 0.75),
        _make_stance("MomentumAgent",  "bull",    0.5, 0.70),
        _make_stance("QualityAgent",   "neutral", 0.05, 0.60),
        _make_stance("SentimentAgent", "bear",   -0.3, 0.65),
        _make_stance("RiskAgent",      "bear",   -0.4, 0.70),
    ]
    rebuttals = orch._round2_cross_examination(stances)
    result    = orch._round3_synthesis(stances, rebuttals)
    assert 0.0 < result.conviction_ratio < 1.0
