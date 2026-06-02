"""app/utils/agent_viz.py — 6-Agent 六维分析可复用展示组件.

输入 analysis 字典（来自 DataService.get_agent_analysis 或 compute_agent_analysis_live）：
    {
      "ticker", "rating", "composite_signal", "confidence",
      "target_price_1m", "target_price_3m", "stop_loss_price",
      "position_size", "holding_horizon",
      "investment_thesis", "key_risks", "key_catalysts",
      "agents": {Valuation/Momentum/Quality/Sentiment/Risk: {signal, confidence, summary}},
      "live": bool (可选，实时计算标记),
    }

页面 3（单股分析）与页面 12（历史推荐）共用，避免重复代码。
"""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

# 维度顺序 + 中文 + 图标
DIM_ORDER = ["Valuation", "Momentum", "Quality", "Sentiment", "Risk"]
DIM_CN = {
    "Valuation": "估值", "Momentum": "动量", "Quality": "质量",
    "Sentiment": "情绪", "Risk": "风险",
}
DIM_ICON = {
    "Valuation": "💰", "Momentum": "📈", "Quality": "⭐",
    "Sentiment": "💬", "Risk": "⚠️",
}

_STANCE_THRESH = 0.10


def _signal_label(sig: float | None) -> tuple[str, str]:
    """signal → (标签, 颜色)。"""
    if not isinstance(sig, (int, float)):
        return "无数据", "#95a5a6"
    if sig > _STANCE_THRESH:
        return "看多", "#27ae60"
    if sig < -_STANCE_THRESH:
        return "看空", "#e74c3c"
    return "中性", "#95a5a6"


# ─────────────────────────────────────────────────────────────────────────────
# 区域1：六维雷达图
# ─────────────────────────────────────────────────────────────────────────────

def render_agent_radar(analysis: dict[str, Any], height: int = 340) -> go.Figure:
    """五维信号雷达图（signal ∈ [-1, +1]），marker 大小映射 confidence。"""
    agents = analysis.get("agents", {}) or {}
    dims = [d for d in DIM_ORDER if d in agents]
    if not dims:
        # 空图占位
        fig = go.Figure()
        fig.update_layout(height=height,
                          annotations=[dict(text="无六维数据", showarrow=False)])
        return fig

    theta = [DIM_CN[d] for d in dims]
    signals = [float(agents[d].get("signal") or 0.0) for d in dims]
    confs = [agents[d].get("confidence") for d in dims]
    marker_sizes = [
        8 + (float(c) * 14 if isinstance(c, (int, float)) else 4) for c in confs
    ]

    fig = go.Figure()
    # 零基准圈
    fig.add_trace(go.Scatterpolar(
        r=[0] * (len(dims) + 1), theta=theta + [theta[0]],
        line=dict(color="rgba(178,190,195,0.5)", width=1, dash="dot"),
        name="中性(0)", hoverinfo="skip",
    ))
    # 信号
    fig.add_trace(go.Scatterpolar(
        r=signals + [signals[0]], theta=theta + [theta[0]],
        fill="toself", fillcolor="rgba(108,92,231,0.18)",
        line=dict(color="#6C5CE7", width=2.5),
        marker=dict(size=marker_sizes + [marker_sizes[0]], color="#6C5CE7"),
        name="Agent 信号",
        customdata=[f"{c:.0%}" if isinstance(c, (int, float)) else "—" for c in confs] + [
            f"{confs[0]:.0%}" if isinstance(confs[0], (int, float)) else "—"],
        hovertemplate="%{theta}: signal=%{r:+.2f}<br>置信度=%{customdata}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1], tickfont_size=10),
                   angularaxis=dict(tickfont_size=13)),
        showlegend=True, height=height, margin=dict(t=30, b=20),
        legend=dict(orientation="h", y=-0.08, font_size=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 区域2：逐 Agent 论证卡片
# ─────────────────────────────────────────────────────────────────────────────

def render_agent_cards(analysis: dict[str, Any]) -> None:
    """渲染 5 张 Agent 卡片（signal + 看多/看空 + confidence 进度条 + summary）。"""
    agents = analysis.get("agents", {}) or {}
    dims = [d for d in DIM_ORDER if d in agents]
    if not dims:
        st.info("无逐 Agent 分析数据。")
        return

    cols = st.columns(len(dims))
    for col, dim in zip(cols, dims):
        info = agents[dim]
        sig = info.get("signal")
        conf = info.get("confidence")
        label, color = _signal_label(sig)
        sig_str = f"{sig:+.2f}" if isinstance(sig, (int, float)) else "—"
        with col:
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:1.4rem'>{DIM_ICON[dim]}</div>"
                f"<div style='font-weight:600'>{DIM_CN[dim]}</div>"
                f"<div style='font-size:1.3rem;color:{color};font-weight:700'>{sig_str}</div>"
                f"<span style='background:{color};color:white;padding:1px 8px;"
                f"border-radius:8px;font-size:.75rem'>{label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if isinstance(conf, (int, float)):
                st.progress(min(max(float(conf), 0.0), 1.0))
                st.caption(f"置信度 {conf:.0%}")
            else:
                st.caption("置信度 —")
            summ = info.get("summary", "") or "（无结论）"
            st.caption(summ[:120])


# ─────────────────────────────────────────────────────────────────────────────
# 区域3：综合策略
# ─────────────────────────────────────────────────────────────────────────────

_RATING_COLOR = {
    "强烈买入": "#27ae60", "买入": "#2ecc71", "谨慎关注": "#f39c12",
    "观察": "#f39c12", "持有": "#95a5a6", "回避": "#e74c3c", "卖出": "#e74c3c",
}


def render_strategy_summary(analysis: dict[str, Any]) -> None:
    """渲染综合策略：评级 + 论证 + 目标价/止损/仓位 + 风险/催化剂。"""
    rating = analysis.get("rating", "—")
    cs = analysis.get("composite_signal")
    conf = analysis.get("confidence")
    rcolor = _RATING_COLOR.get(rating, "#636e72")

    # 顶部指标
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f"<div style='text-align:center'><div style='font-size:.8rem;color:#636e72'>综合评级</div>"
        f"<div style='font-size:1.3rem;font-weight:700;color:{rcolor}'>{rating}</div></div>",
        unsafe_allow_html=True,
    )
    m2.metric("综合信号", f"{cs:+.3f}" if isinstance(cs, (int, float)) else "—")
    m3.metric("置信度", f"{conf:.0%}" if isinstance(conf, (int, float)) else "—")
    m4.metric("建议持仓", analysis.get("holding_horizon", "—") or "—")

    # 目标价 / 止损 / 仓位
    p1m = analysis.get("target_price_1m")
    p3m = analysis.get("target_price_3m")
    sl = analysis.get("stop_loss_price")
    pos = analysis.get("position_size")

    def _price(v):
        return f"¥{v:.2f}" if isinstance(v, (int, float)) and v else "—"

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("1月目标价", _price(p1m))
    pc2.metric("3月目标价", _price(p3m))
    pc3.metric("止损价", _price(sl))
    pc4.metric("建议仓位", pos if pos else "—")

    # 投资论证
    thesis = analysis.get("investment_thesis", "")
    if thesis:
        st.markdown("**📝 投资论证**")
        st.markdown(
            f"<div style='background:#f8f9fa;border-left:4px solid {rcolor};"
            f"padding:10px 14px;border-radius:4px;white-space:pre-wrap;"
            f"font-size:.9rem'>{thesis}</div>",
            unsafe_allow_html=True,
        )

    # 风险 + 催化剂
    risks = analysis.get("key_risks", []) or []
    catalysts = analysis.get("key_catalysts", []) or []
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**⚠️ 关键风险**")
        if risks:
            for r in risks:
                st.markdown(f"- {r}")
        else:
            st.caption("无")
    with rc2:
        st.markdown("**🚀 关键催化剂**")
        if catalysts:
            for c in catalysts:
                st.markdown(f"- {c}")
        else:
            st.caption("无")


# ─────────────────────────────────────────────────────────────────────────────
# 一站式渲染（雷达 + 卡片 + 策略）
# ─────────────────────────────────────────────────────────────────────────────

def render_full_analysis(analysis: dict[str, Any]) -> None:
    """完整渲染：雷达图 + 卡片 + 综合策略。供页面直接调用。"""
    if not analysis:
        st.info("无 6-Agent 分析数据。")
        return
    if analysis.get("live"):
        st.success("⚡ 实时计算结果（fast 模式）")

    st.markdown("##### 🎯 六维信号雷达")
    st.plotly_chart(render_agent_radar(analysis), use_container_width=True)

    st.markdown("##### 🤖 逐 Agent 论证")
    render_agent_cards(analysis)

    st.divider()
    st.markdown("##### 📋 综合策略")
    render_strategy_summary(analysis)
