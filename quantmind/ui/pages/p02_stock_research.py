"""02 个股研究 — Multi-Agent 深度分析（调用 Orchestrator + Ollama）."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent


def render() -> None:
    st.title("🔍 个股深度研究")
    st.markdown("调用 Multi-Agent（Planner→Data→Fundamental→Technical→Sentiment→Critic→Report）进行深度分析。")

    # ── 输入控件 ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 2])
    with col1:
        ticker = st.text_input(
            "股票代码",
            value="000001.SZ",
            placeholder="如 600519.SH / 000001.SZ",
            help="A股标准代码格式",
        )
    with col2:
        as_of = st.date_input(
            "分析日期（as_of）",
            value=date.today(),
            max_value=date.today(),
        )

    dimensions = st.multiselect(
        "分析维度",
        ["财务分析", "技术分析", "情绪分析", "估值分析", "风险评估"],
        default=["财务分析", "技术分析", "情绪分析"],
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
    with col_btn2:
        demo_btn = st.button("🎯 运行 Demo", use_container_width=True)

    st.divider()

    # ── 执行分析 ──────────────────────────────────────────────────────────────
    if run_btn or demo_btn:
        if demo_btn:
            ticker = "600519.SH"
            as_of = date(2024, 6, 30)
        _run_analysis(ticker, str(as_of), dimensions, demo_mode=demo_btn)


def _run_analysis(ticker: str, as_of: str, dimensions: list[str], demo_mode: bool = False) -> None:
    """调用 Orchestrator 或展示 Demo 结果."""
    if demo_mode:
        _show_demo_result(ticker)
        return

    status_container = st.status(f"正在分析 {ticker}（{as_of}）...", expanded=True)

    try:
        with status_container:
            st.write("🔄 初始化 Agent 系统...")

            # 延迟导入（避免启动时加载全部依赖）
            try:
                from quantmind.agents.orchestrator import Orchestrator
                from quantmind.core.state import AgentState
            except ImportError as e:
                st.error(f"导入失败：{e}。请确认环境安装正确。")
                status_container.update(label="❌ 导入失败", state="error")
                return

            st.write("📋 Planner 规划分析任务...")
            st.write("📊 Data Agent 获取数据...")
            st.write("💰 Fundamental Agent 分析财务...")
            if "技术分析" in dimensions:
                st.write("📈 Technical Agent 分析技术...")
            if "情绪分析" in dimensions:
                st.write("🗞️ Sentiment Agent 分析情绪...")
            st.write("🔍 Critic Agent 质量审核...")
            st.write("📝 Report Agent 生成报告...")

            # 运行 Orchestrator
            orchestrator = Orchestrator()
            state = orchestrator.run(ticker=ticker, as_of=as_of)

            status_container.update(label="✅ 分析完成", state="complete", expanded=False)

        # 显示报告
        _render_report(state, ticker)

    except Exception as e:
        status_container.update(label=f"❌ 分析失败: {e}", state="error")
        st.error(f"分析过程出错：{e}")
        st.info("提示：请确认 Ollama 已启动（`ollama run qwen2.5:7b`）")


def _render_report(state: dict, ticker: str) -> None:
    """渲染分析报告."""
    report = state.get("report") or state.get("_agent_state", {}).get("report", {})

    if not report:
        st.warning("未获取到报告内容，请检查 Agent 运行日志。")
        return

    # 评级标签
    rating = report.get("rating", "N/A")
    rating_color = {"买入": "🟢", "增持": "🔵", "中性": "🟡", "减持": "🔴", "卖出": "🔴"}.get(rating, "⚪")
    st.markdown(f"## {rating_color} {ticker} — 投资评级：**{rating}**")

    confidence = report.get("confidence", 0)
    st.progress(confidence / 100, text=f"置信度：{confidence}%")

    # 各章节展示
    tab_names = ["执行摘要", "财务分析", "技术分析", "情绪分析", "估值", "风险", "催化剂", "逻辑链", "行动建议"]
    section_keys = [
        "executive_summary", "fundamental_analysis", "technical_analysis",
        "sentiment_analysis", "valuation", "risk_factors",
        "catalysts", "investment_thesis", "action_recommendation",
    ]

    tabs = st.tabs(tab_names)
    for tab, key in zip(tabs, section_keys):
        with tab:
            content = report.get(key, "暂无内容")
            if isinstance(content, dict):
                st.json(content)
            else:
                st.markdown(str(content))

    # 下载报告
    report_md = json.dumps(report, ensure_ascii=False, indent=2)
    st.download_button(
        "📥 下载 JSON 报告",
        data=report_md,
        file_name=f"report_{ticker}_{state.get('as_of', 'unknown')}.json",
        mime="application/json",
    )

    # HTML 报告
    reports_dir = ROOT / "reports"
    html_files = list(reports_dir.glob(f"*{ticker.replace('.', '_')}*.html")) if reports_dir.exists() else []
    if html_files:
        with open(html_files[-1], "rb") as f:
            st.download_button(
                "📥 下载 HTML 报告",
                data=f.read(),
                file_name=html_files[-1].name,
                mime="text/html",
            )


def _show_demo_result(ticker: str) -> None:
    """展示 Demo 报告（无需真实 API）."""
    st.success(f"✅ {ticker} Demo 分析完成（使用模拟数据）")

    st.markdown(f"## 🟢 {ticker} — 投资评级：**增持**")
    st.progress(78, text="置信度：78%")

    tab_names = ["执行摘要", "财务分析", "技术分析", "情绪分析"]
    contents = [
        "**执行摘要**：该股基本面稳健，技术面呈上升趋势，情绪整体偏正面。"
        "建议在当前估值水平增持，目标价较当前上涨约 15%。",
        "**财务分析**：ROE 18.5%，净利润增速 12.3%，资产负债率 35%，现金流充裕。",
        "**技术分析**：价格站上 MA20/MA60，MACD 金叉，RSI(14)=58，成交量温和放大。",
        "**情绪分析**：北向资金近 30 日净流入 12 亿元，机构持仓比例提升，分析师评级中性偏多。",
    ]
    tabs = st.tabs(tab_names)
    for tab, content in zip(tabs, contents):
        with tab:
            st.markdown(content)
