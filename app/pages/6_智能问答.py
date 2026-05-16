"""交互式投资问答（读取本地流水线结果 + LLM 总结）。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.data_loader import list_top10_dates
from app.utils.qa_engine import QAEngine

st.set_page_config(page_title="智能问答", layout="wide")
st.title("💬 智能投资问答")
st.markdown(
    """
> 你可以直接问：
> - 「帮我分析一下 600519.SH 茅台」
> - 「000651.SZ 格力电器现在能买吗？」
> - 「给我找几只当前候选池里的低估值标的」
> - 「对比一下宁德时代和比亚迪」

回答基于仓库内已生成的 **top10 / strategies / validations**，并调用 LLM 做摘要（无密钥且无本地模型时自动降级为模板）。
"""
)

dates = list_top10_dates()
pick = st.selectbox("对齐流水线日期（与 recommendations 目录一致）", dates, index=len(dates) - 1) if dates else None

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("charts"):
            for fig in msg["charts"]:
                st.plotly_chart(fig, use_container_width=True)
        if msg["role"] == "assistant" and msg.get("full_report"):
            with st.expander("📄 查看完整报告"):
                st.markdown(msg["full_report"])

if prompt := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 正在分析，请稍候..."):
            engine = QAEngine(pipeline_date=pick)
            result = engine.answer(prompt)
        st.markdown(result.text_response)
        chart_objs = []
        for fig in result.charts:
            st.plotly_chart(fig, use_container_width=True)
            chart_objs.append(fig)
        if result.full_report:
            with st.expander("📄 查看完整报告"):
                st.markdown(result.full_report)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.text_response,
            "charts": chart_objs,
            "full_report": result.full_report or "",
        }
    )
