"""01 Overview — 项目首页：架构图 + 进度表 + 最新回测摘要."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent


def render() -> None:
    st.title("📈 QuantMind — AI Agent 量化研究系统")
    st.markdown(
        "> Multi-Agent 协作 × 生成式量化选股 × 严格 PIT 回测",
    )

    # ── 关键指标 ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">9</div>'
            '<div class="metric-label">完成 Phase</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">91</div>'
            '<div class="metric-label">测试通过</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">41</div>'
            '<div class="metric-label">因子数量</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">7</div>'
            '<div class="metric-label">Agent 数量</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 两列布局：架构图 + 进度表 ─────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🏗️ 系统架构")
        _render_architecture_chart()

    with col_right:
        st.subheader("✅ 开发进度")
        _render_progress_table()

    st.divider()

    # ── 最新回测结果 ──────────────────────────────────────────────────────────
    st.subheader("📊 最新回测结果")
    _render_latest_backtest()


def _render_architecture_chart() -> None:
    """用 Plotly 绘制系统架构图（无图片依赖）."""
    fig = go.Figure()

    # 定义节点（x, y, label, color, layer）
    nodes = [
        # 数据层
        (1, 4.5, "Data Layer\n(PIT)", "#1f6feb"),
        # 特征层
        (2, 4.5, "Features\n41 Factors", "#1f6feb"),
        # Agent 层
        (3.5, 6, "Planner", "#388bfd"),
        (3.5, 5, "Data Agent", "#388bfd"),
        (3.5, 4, "Fundamental", "#388bfd"),
        (3.5, 3, "Technical", "#388bfd"),
        (3.5, 2, "Sentiment", "#388bfd"),
        (3.5, 1, "Critic", "#f85149"),
        (3.5, 0, "Report", "#3fb950"),
        # 回测层
        (5, 5, "BacktestEngine\nT+1/涨跌停/PIT", "#1f6feb"),
        (5, 3.5, "Statistical\nTests", "#1f6feb"),
        (5, 2, "WalkForward\nValidator", "#1f6feb"),
        # 风险层
        (6.5, 4, "Risk Manager\nHRP/CPPI", "#a371f7"),
        # UI 层
        (8, 3, "Streamlit\nUI", "#d29922"),
    ]

    # 绘制节点
    for x, y, label, color, *_ in nodes:
        fig.add_shape(
            type="rect",
            x0=x - 0.45, y0=y - 0.38, x1=x + 0.45, y1=y + 0.38,
            fillcolor=color, opacity=0.2, line_color=color, line_width=1.5,
        )
        fig.add_annotation(
            x=x, y=y, text=label, showarrow=False,
            font=dict(size=8.5, color="#e6edf3"),
            align="center",
        )

    # 连线（数据流向）
    edges = [
        (1, 4.5, 2, 4.5),        # Data → Features
        (2, 4.5, 3.5, 4.5),      # Features → Agents
        (3.5, 0.5, 5, 5),        # Report → Backtest
        (5, 3.5, 5, 2),          # Tests → WalkForward
        (5, 4, 6.5, 4),          # Backtest → Risk
        (6.5, 4, 8, 3),          # Risk → UI
    ]
    for x0, y0, x1, y1 in edges:
        fig.add_annotation(
            ax=x0, ay=y0, x=x1, y=y1,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=1.2,
            arrowcolor="#30363d", showarrow=True, text="",
        )

    # 层标签
    for x, label in [(1, "数据层"), (2, "特征层"), (3.5, "Agent层"),
                     (5, "回测层"), (6.5, "风险层"), (8, "UI层")]:
        fig.add_annotation(
            x=x, y=6.7, text=f"<b>{label}</b>",
            showarrow=False, font=dict(size=9, color="#8b949e"),
        )

    fig.update_layout(
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.3, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.6, 7.3]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_progress_table() -> None:
    """渲染 Phase 0-8 完成状态表格."""
    phases = [
        (0, "环境与基础设施", "✅"),
        (1, "数据层（PIT 严格）", "✅"),
        (2, "特征工程（41 因子）", "✅"),
        (3, "模型训练（LGBM+DPO）", "✅"),
        (4, "Multi-Agent 研究", "✅"),
        (5, "知识库（BGE-M3+BM25）", "✅"),
        (6, "回测引擎（A股规则）", "✅"),
        (7, "风险与组合管理", "✅"),
        (8, "Streamlit UI", "✅"),
    ]
    import pandas as pd
    df = pd.DataFrame(phases, columns=["Phase", "任务", "状态"])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Phase": st.column_config.NumberColumn("Phase", width=70),
            "任务": st.column_config.TextColumn("任务", width=180),
            "状态": st.column_config.TextColumn("状态", width=60),
        },
    )


def _render_latest_backtest() -> None:
    """显示最新回测结果或引导用户运行回测."""
    reports_dir = ROOT / "reports"
    html_files = sorted(reports_dir.glob("*.html")) if reports_dir.exists() else []

    if html_files:
        latest = html_files[-1]
        st.success(f"找到最新回测报告：`{latest.name}`")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**报告路径：** `{latest}`")
        with col2:
            with open(latest, "rb") as f:
                st.download_button(
                    "📥 下载 HTML 报告",
                    data=f.read(),
                    file_name=latest.name,
                    mime="text/html",
                )
    else:
        st.info(
            "暂无回测报告。请前往 **策略回测** 页面运行回测，"
            "或执行：`python scripts/run_backtest.py --strategy equal_weight`"
        )
        # 展示 demo 指标
        import pandas as pd
        demo_metrics = {
            "指标": ["年化收益", "夏普比率", "最大回撤", "波动率", "Beta"],
            "等权基准（Demo）": ["12.3%", "0.89", "-18.5%", "14.2%", "0.82"],
        }
        st.dataframe(pd.DataFrame(demo_metrics), use_container_width=True, hide_index=True)
