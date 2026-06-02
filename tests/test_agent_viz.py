"""tests/test_agent_viz.py — 6-Agent 可视化组件单元测试."""
from __future__ import annotations

import plotly.graph_objects as go


def _sample_analysis() -> dict:
    return {
        "ticker": "600000.SH", "rating": "买入",
        "composite_signal": 0.42, "confidence": 0.7,
        "investment_thesis": "综合论证文字",
        "key_risks": ["风险1"], "key_catalysts": ["催化1"],
        "agents": {
            "Valuation": {"signal": 0.8, "confidence": 0.75, "summary": "估值低"},
            "Momentum":  {"signal": 0.1, "confidence": 0.6, "summary": "动量弱"},
            "Quality":   {"signal": 0.5, "confidence": 0.9, "summary": "质量好"},
            "Sentiment": {"signal": 0.0, "confidence": 0.4, "summary": "中性"},
            "Risk":      {"signal": -0.2, "confidence": 0.5, "summary": "尾部风险"},
        },
    }


class TestRenderAgentRadar:

    def test_returns_figure(self):
        from app.utils.agent_viz import render_agent_radar
        fig = render_agent_radar(_sample_analysis())
        assert isinstance(fig, go.Figure)

    def test_has_signal_and_baseline_traces(self):
        from app.utils.agent_viz import render_agent_radar
        fig = render_agent_radar(_sample_analysis())
        # 至少 2 条 trace：零基准 + 信号
        assert len(fig.data) >= 2

    def test_empty_agents_placeholder(self):
        from app.utils.agent_viz import render_agent_radar
        fig = render_agent_radar({"agents": {}})
        assert isinstance(fig, go.Figure)

    def test_flat_format_signal_only(self):
        """flat 格式（confidence=None）也能渲染。"""
        from app.utils.agent_viz import render_agent_radar
        a = {"agents": {
            "Valuation": {"signal": 0.6, "confidence": None, "summary": ""},
            "Risk": {"signal": 0.0, "confidence": None, "summary": ""},
        }}
        fig = render_agent_radar(a)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2


class TestSignalLabel:

    def test_bullish_bearish_neutral(self):
        from app.utils.agent_viz import _signal_label
        assert _signal_label(0.5)[0] == "看多"
        assert _signal_label(-0.5)[0] == "看空"
        assert _signal_label(0.0)[0] == "中性"
        assert _signal_label(None)[0] == "无数据"
