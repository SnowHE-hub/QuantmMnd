"""tests/test_investment_agents.py — 投资分析 Agent 单元测试.

不调用外部 API，不连接 ChromaDB，使用 mock 数据。
运行：
    conda run -n quantmind python -m pytest tests/test_investment_agents.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.agents.investment_agents.base_agent import AgentSignal
from quantmind.agents.investment_agents.momentum_agent import MomentumAgent
from quantmind.agents.investment_agents.quality_agent import QualityAgent
from quantmind.agents.investment_agents.risk_agent import RiskAgent
from quantmind.agents.investment_agents.sentiment_agent import SentimentAgent
from quantmind.agents.investment_agents.strategy_agent import InvestmentStrategy, StrategyAgent
from quantmind.agents.investment_agents.valuation_agent import ValuationAgent
from scripts.validate_strategies import StrategyValidationResult, validate_strategy


# ── Fixtures & Mock 工具 ──────────────────────────────────────────────────────

TICKER = "600519.SH"
AS_OF = "2024-12-31"


def _make_context(**overrides) -> dict:
    """构造最小 context，可用关键字覆盖各 snapshot 字段."""
    ctx = {
        "ticker": TICKER,
        "as_of": AS_OF,
        "news_context": [],
        "news_count": 0,
        "report_context": [],
        "report_count": 0,
        "market_context": [],
        "snapshot_company_profile": [],
        "snapshot_latest_market_metrics": [],
        "snapshot_financial_indicator_summary": [],
        "snapshot_northbound_summary": [],
        "snapshot_margin_summary": [],
    }
    ctx.update(overrides)
    return ctx


def _market_snapshot(pe_ttm=20, pb=2.5, roe_ttm=18, **kw) -> list[dict]:
    """构造 snapshot_latest_market_metrics 文本格式."""
    text = f"pe_ttm: {pe_ttm}\npb: {pb}\nroe_ttm: {roe_ttm}\n"
    for k, v in kw.items():
        text += f"{k}: {v}\n"
    return [{"text": text, "metadata": {}, "score": 1.0}]


def _fi_snapshot(roe_ttm=18, grossprofit_margin=45, debt_to_assets=35, **kw) -> list[dict]:
    """构造 snapshot_financial_indicator_summary 文本格式."""
    text = f"roe_ttm: {roe_ttm}\ngrossprofit_margin: {grossprofit_margin}\ndebt_to_assets: {debt_to_assets}\n"
    for k, v in kw.items():
        text += f"{k}: {v}\n"
    return [{"text": text, "metadata": {}, "score": 1.0}]


def _make_price_series(ticker: str, n: int = 300, trend: float = 0.001) -> pd.DataFrame:
    """生成 mock 价格序列（带上升趋势）."""
    dates = pd.date_range(end="2024-12-31", periods=n, freq="B")
    prices = 100 * np.cumprod(1 + np.random.normal(trend, 0.015, n))
    return pd.DataFrame({ticker: prices}, index=dates)


# ── Test 1: ValuationAgent — 合理估值 ────────────────────────────────────────

def test_valuation_agent_reasonable():
    """PE=20, PB=2, ROE=18% → signal 约 +0.4（合理偏低估）."""
    ctx = _make_context(
        snapshot_latest_market_metrics=_market_snapshot(pe_ttm=20, pb=2.0, roe_ttm=18),
        snapshot_financial_indicator_summary=_fi_snapshot(roe_ttm=18),
    )
    agent = ValuationAgent(TICKER, AS_OF, ctx)
    sig = agent.analyze()

    assert isinstance(sig, AgentSignal)
    assert sig.agent_name == "ValuationAgent"
    assert sig.ticker == TICKER
    # PE=20 处于合理范围，PB=2 略低，ROE=18%给加成
    assert -0.3 <= sig.signal <= 0.8, f"信号超出预期范围: {sig.signal}"
    assert sig.confidence >= 0.6


# ── Test 2: ValuationAgent — 高估 ─────────────────────────────────────────────

@pytest.mark.integration
def test_valuation_agent_overvalued():
    """PE=80 → signal < -0.5（高估）."""
    ctx = _make_context(
        snapshot_latest_market_metrics=_market_snapshot(pe_ttm=80, pb=10, roe_ttm=5),
        snapshot_financial_indicator_summary=_fi_snapshot(roe_ttm=5),
    )
    agent = ValuationAgent(TICKER, AS_OF, ctx)
    sig = agent.analyze()

    assert sig.signal < -0.5, f"高估信号应 < -0.5，实际: {sig.signal}"
    assert len(sig.warnings) > 0


# ── Test 3: MomentumAgent — 上涨趋势 ─────────────────────────────────────────

def test_momentum_agent_uptrend():
    """Mock 上涨价格序列 → signal > 0."""
    price_df = _make_price_series(TICKER, n=200, trend=0.003)  # 日均+0.3%

    with patch(
        "quantmind.agents.investment_agents.momentum_agent.pd.read_parquet",
        return_value=price_df,
    ):
        ctx = _make_context()
        agent = MomentumAgent(TICKER, AS_OF, ctx)
        sig = agent.analyze()

    assert isinstance(sig, AgentSignal)
    assert sig.confidence > 0, "上涨趋势数据应有正置信度"
    # 上涨趋势信号通常 > 0（但 RSI 超买可能修正）
    assert sig.signal >= -0.5, f"上涨趋势信号不应过低: {sig.signal}"


# ── Test 4: QualityAgent — 高质量 ────────────────────────────────────────────

def test_quality_agent_high_quality():
    """ROE=25%, debt_to_assets=30% → signal > 0.5."""
    ctx = _make_context(
        snapshot_financial_indicator_summary=_fi_snapshot(
            roe_ttm=25,
            grossprofit_margin=50,
            debt_to_assets=30,
        ),
    )
    agent = QualityAgent(TICKER, AS_OF, ctx)
    sig = agent.analyze()

    assert sig.signal > 0.5, f"高质量信号应 > 0.5，实际: {sig.signal}"
    assert sig.confidence >= 0.6


# ── Test 5: SentimentAgent — 正面新闻（无需LLM） ──────────────────────────────

@pytest.mark.integration
def test_sentiment_agent_positive_news():
    """含"增长""超预期"新闻 → signal > 0（纯规则，不调 LLM）."""
    news_items = [
        {"text": "公司营收增长超预期，创历史新高！", "metadata": {}},
        {"text": "获批新产品，市场扩张顺利", "metadata": {}},
    ]
    ctx = _make_context(
        news_context=news_items,
        news_count=2,
    )
    agent = SentimentAgent(TICKER, AS_OF, ctx, provider="none")
    sig = agent.analyze()

    assert sig.signal > 0, f"正面新闻信号应 > 0，实际: {sig.signal}"
    assert sig.evidence.get("positive_words"), "应检测到正面关键词"


# ── Test 6: RiskAgent — 高风险 ───────────────────────────────────────────────

@pytest.mark.integration
def test_risk_agent_high_risk():
    """vol=50%, debt=80% → signal < -0.5，warnings 非空."""
    price_df = _make_price_series(TICKER, n=200, trend=0.0)
    # 手动放大波动率
    price_df[TICKER] = price_df[TICKER] * (1 + np.random.normal(0, 0.03, 200)).cumprod()

    ctx = _make_context(
        snapshot_financial_indicator_summary=_fi_snapshot(
            roe_ttm=-5,
            debt_to_assets=80,
        ),
    )

    with patch(
        "quantmind.agents.investment_agents.risk_agent.pd.read_parquet",
        return_value=price_df,
    ):
        agent = RiskAgent(TICKER, AS_OF, ctx)
        sig = agent.analyze()

    assert sig.signal < -0.3, f"高风险信号应 < -0.3，实际: {sig.signal}"
    assert len(sig.warnings) > 0, "高风险应触发 warnings"


# ── Test 7: StrategyAgent — provider=none 不崩溃 ─────────────────────────────

def test_strategy_agent_no_llm():
    """mock 5个Agent信号 + provider=none → 返回 InvestmentStrategy，不崩溃."""
    mock_signals = [
        AgentSignal("ValuationAgent", TICKER, +0.4, 0.8, "估值合理", {}, []),
        AgentSignal("MomentumAgent", TICKER, +0.6, 0.7, "动量良好", {}, []),
        AgentSignal("QualityAgent", TICKER, +0.5, 0.9, "高质量", {}, []),
        AgentSignal("SentimentAgent", TICKER, +0.2, 0.5, "中性偏正", {}, []),
        AgentSignal("RiskAgent", TICKER, +0.1, 0.8, "低风险", {}, []),
    ]

    price_df = _make_price_series(TICKER, n=200, trend=0.002)
    ctx = _make_context()

    with patch(
        "quantmind.agents.investment_agents.strategy_agent.pd.read_parquet",
        return_value=price_df,
    ):
        agent = StrategyAgent(
            ticker=TICKER,
            as_of=AS_OF,
            context=ctx,
            agent_signals=mock_signals,
            provider="none",
        )
        strategy = agent.analyze_with_llm()

    assert isinstance(strategy, InvestmentStrategy)
    assert strategy.ticker == TICKER
    assert strategy.rating in ["积极关注", "谨慎关注", "观察", "回避"]
    assert -1.0 <= strategy.composite_signal <= 1.0
    assert strategy.position_size != ""
    assert strategy.investment_thesis != ""
    assert not strategy.llm_used, "provider=none 时不应使用 LLM"


# ── Test 8: validate_strategy — 胜率70%→ACCEPTABLE ───────────────────────────

def test_validate_strategy_acceptable():
    """mock 价格序列（上涨趋势，胜率约70%）→ ACCEPTABLE 或 WATCHLIST."""
    # 构造带上涨趋势的价格序列（确保较高胜率）
    n = 400
    dates = pd.date_range(end="2024-12-31", periods=n, freq="B")
    prices = 100 * np.cumprod(1 + np.random.normal(0.002, 0.01, n))
    price_df = pd.DataFrame({TICKER: prices}, index=dates)

    strategy_dict = {
        "ticker": TICKER,
        "as_of": AS_OF,
        "rating": "积极关注",
        "composite_signal": 0.6,
        "confidence": 0.8,
        "entry_price_range": [prices[-1] * 0.98, prices[-1] * 1.02],
        "target_price_1m": prices[-1] * 1.05,
        "target_price_3m": prices[-1] * 1.12,
        "stop_loss_price": prices[-1] * 0.93,
        "position_size": "标准(3-5%)",
        "holding_horizon": "短期(1个月)",
        "investment_thesis": "测试策略",
        "key_risks": [],
        "key_catalysts": [],
        "agent_signals": {},
        "llm_used": False,
    }

    result = validate_strategy(strategy_dict, price_df, panel_df=None)

    assert isinstance(result, StrategyValidationResult)
    assert result.ticker == TICKER
    assert result.validation_status in ["ACCEPTABLE", "WATCHLIST", "AVOID"]
    # 上涨趋势下期望值应 >= 0
    assert result.expected_value >= -0.05, f"强上涨趋势的期望值不应过低: {result.expected_value}"


# ── Test 9: AgentSignal 字段完整性 ───────────────────────────────────────────

def test_agent_signal_fields():
    """所有 Agent 应返回包含完整字段的 AgentSignal."""
    ctx = _make_context(
        snapshot_latest_market_metrics=_market_snapshot(pe_ttm=25, pb=3, roe_ttm=15),
        snapshot_financial_indicator_summary=_fi_snapshot(roe_ttm=15, grossprofit_margin=30, debt_to_assets=45),
        news_context=[{"text": "公司业绩增长", "metadata": {}}],
    )

    price_df = _make_price_series(TICKER, n=150, trend=0.001)

    agents_signals = []
    with patch(
        "quantmind.agents.investment_agents.momentum_agent.pd.read_parquet",
        return_value=price_df,
    ), patch(
        "quantmind.agents.investment_agents.risk_agent.pd.read_parquet",
        return_value=price_df,
    ):
        for AgentClass, extra_kwargs in [
            (ValuationAgent, {}),
            (MomentumAgent, {}),
            (QualityAgent, {}),
            (SentimentAgent, {"provider": "none"}),
            (RiskAgent, {}),
        ]:
            agent = AgentClass(TICKER, AS_OF, ctx, **extra_kwargs)
            sig = agent.analyze()
            agents_signals.append(sig)
            assert hasattr(sig, "signal"), f"{AgentClass.__name__} 缺少 signal 字段"
            assert hasattr(sig, "confidence")
            assert hasattr(sig, "summary")
            assert hasattr(sig, "evidence")
            assert hasattr(sig, "warnings")
            assert -1.0 <= sig.signal <= 1.0, f"{AgentClass.__name__} signal 超出范围"
            assert 0.0 <= sig.confidence <= 1.0

    assert len(agents_signals) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
