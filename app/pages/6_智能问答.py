"""6_智能问答.py — AI驱动问答（DashScope/Ollama + 命令执行 + 图表分析）."""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.sim_data import (
    load_sim30d_days, load_sim30d_stock_returns,
    load_ic_analysis, load_realized_pnl,
    horizon_portfolio_ts,
)

st.set_page_config(page_title="智能问答 · QuantMind", page_icon="💬", layout="wide")

API_BASE = "http://localhost:8000"

# ── 检查 API 是否可用 ─────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def check_api() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return {}

# ── Session State 初始化 ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider" not in st.session_state:
    st.session_state.provider = "dashscope"
if "auto_execute" not in st.session_state:
    st.session_state.auto_execute = True

# ── 标题栏 ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,#2D3436,#636E72);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:16px'>
  <h2 style='margin:0'>💬 智能问答 · AI 助手</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    DashScope 百炼 / Ollama 本地模型 · 自动执行系统命令 · 图表解读
  </p>
</div>
""", unsafe_allow_html=True)

# ── 侧栏：模型设置 + 快捷问题 ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ AI 模型设置")

    api_status = check_api()
    if api_status:
        st.success("✅ API 服务正常")
        ollama_models = api_status.get("ollama_models", [])
    else:
        st.warning("⚠️ API 未启动，请先运行 `start_api.sh`")
        ollama_models = []

    provider = st.radio(
        "选择模型",
        options=["dashscope", "ollama", "deepseek"],
        format_func=lambda x: {
            "dashscope": "🌐 百炼 qwen-turbo（推荐）",
            "ollama": f"💻 Ollama 本地",
            "deepseek": "🔮 DeepSeek",
        }[x],
        index=["dashscope", "ollama", "deepseek"].index(st.session_state.provider),
    )
    st.session_state.provider = provider

    if provider == "ollama" and ollama_models:
        st.caption(f"可用模型：{', '.join(ollama_models)}")

    st.session_state.auto_execute = st.toggle(
        "🤖 自动执行命令",
        value=st.session_state.auto_execute,
        help="开启后，AI识别到系统命令请求时会自动执行并返回结果"
    )

    st.divider()
    st.markdown("### 💡 快捷问题")

    quick_qs = [
        "📊 最近30天整体表现如何？",
        "🏆 哪些股票表现最好？",
        "🏭 哪个行业收益最高？",
        "🔬 哪个因子预测能力最强？",
        "⚖️ System2权重为什么这样设置？",
        "🚀 帮我运行优化分析",
        "📈 实盘PnL情况怎么样？",
        "🎯 下一步应该做什么？",
    ]

    for q in quick_qs:
        if st.button(q, use_container_width=True, key=f"q_{q[:8]}"):
            st.session_state._pending_question = q

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── 生成图表（根据问题类型）─────────────────────────────────────────────────
def make_perf_chart(days) -> go.Figure | None:
    try:
        HZ_COLORS = {"1w": "#D63031", "2w": "#E17055", "21d": "#FDCB6E", "3m": "#00B894"}
        fig = go.Figure()
        for hz, color in HZ_COLORS.items():
            ts = horizon_portfolio_ts(days, hz)
            if ts.empty:
                continue
            fig.add_trace(go.Scatter(
                x=ts["date"].dt.strftime("%m-%d"),
                y=ts["cum_return"] * 100,
                name={"1w":"1周","2w":"2周","21d":"21天","3m":"3月"}[hz],
                line=dict(color=color, width=2),
                mode="lines+markers", marker_size=3,
            ))
        fig.update_layout(
            title="四持仓期累计收益曲线（30日模拟）",
            height=300, margin=dict(t=40, b=40, l=50, r=20),
            yaxis_title="累计收益 (%)",
            legend=dict(orientation="h", y=1.1),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    except Exception:
        return None


def make_ic_chart(ic_data: dict) -> go.Figure | None:
    try:
        ic_final = ic_data.get("ic_final_picks", {})
        if not ic_final:
            return None
        items = [(k, v.get("ic_3m", 0) or 0) for k, v in ic_final.items() if v.get("ic_3m") is not None]
        items.sort(key=lambda x: x[1], reverse=True)
        factors = [i[0] for i in items]
        vals    = [i[1] for i in items]
        colors  = ["#00B894" if v > 0 else "#D63031" for v in vals]
        fig = go.Figure(go.Bar(
            x=factors, y=vals,
            marker_color=colors,
            text=[f"{v:+.4f}" for v in vals],
            textposition="outside",
        ))
        fig.update_layout(
            title="因子 3月 IC（预测能力）",
            height=300, margin=dict(t=40, b=80, l=10, r=10),
            yaxis_title="IC",
            xaxis=dict(tickangle=-30),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    except Exception:
        return None


def make_industry_chart(sr: pd.DataFrame) -> go.Figure | None:
    try:
        if sr.empty or "industry" not in sr.columns or "return_3m" not in sr.columns:
            return None
        final_all = sr[sr["in_final"]].copy()
        ind_agg = final_all.groupby("industry")["return_3m"].mean().sort_values(ascending=False).head(10)
        colors = ["#00B894" if v > 0 else "#D63031" for v in ind_agg.values]
        fig = go.Figure(go.Bar(
            x=ind_agg.index, y=ind_agg.values * 100,
            marker_color=colors,
            text=[f"{v*100:+.1f}%" for v in ind_agg.values],
            textposition="outside",
        ))
        fig.update_layout(
            title="行业 3月平均收益（最终推荐）",
            height=300, margin=dict(t=40, b=80, l=10, r=10),
            yaxis_title="3月均值(%)",
            xaxis=dict(tickangle=-30),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    except Exception:
        return None


def make_top_stocks_chart(sr: pd.DataFrame) -> go.Figure | None:
    try:
        if sr.empty:
            return None
        final_all = sr[sr["in_final"]].copy()
        if "return_3m" not in final_all.columns:
            return None
        top = (final_all.groupby("ticker")["return_3m"].mean()
               .sort_values(ascending=False).head(15))
        colors = ["#00B894" if v > 0 else "#D63031" for v in top.values]
        fig = go.Figure(go.Bar(
            x=top.index, y=top.values * 100,
            marker_color=colors,
            text=[f"{v*100:+.1f}%" for v in top.values],
            textposition="outside",
        ))
        fig.update_layout(
            title="股票 3月均值收益 Top15",
            height=300, margin=dict(t=40, b=60, l=10, r=10),
            yaxis_title="3月收益(%)",
            xaxis=dict(tickfont_size=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    except Exception:
        return None


def make_pnl_chart(rpnl: pd.DataFrame) -> go.Figure | None:
    try:
        if rpnl.empty or "pnl_pct" not in rpnl.columns:
            return None
        fig = go.Figure(go.Histogram(
            x=rpnl["pnl_pct"] * 100,
            nbinsx=30,
            marker_color="#0984E3",
            opacity=0.75,
        ))
        fig.update_layout(
            title=f"realized PnL 分布（{len(rpnl)}条记录）",
            height=280, margin=dict(t=40, b=40),
            xaxis_title="收益率(%)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    except Exception:
        return None


def get_charts_for_message(message: str, days, sr, ic_data, rpnl) -> list[go.Figure]:
    """根据消息关键词返回相关图表."""
    msg = message.lower()
    charts = []
    if any(k in msg for k in ["表现", "收益", "绩效", "总体", "整体", "怎么样"]):
        fig = make_perf_chart(days)
        if fig:
            charts.append(("累计收益曲线", fig))
    if any(k in msg for k in ["股票", "最好", "top", "推荐", "排名"]):
        fig = make_top_stocks_chart(sr)
        if fig:
            charts.append(("股票排行", fig))
    if any(k in msg for k in ["行业", "sector", "归因"]):
        fig = make_industry_chart(sr)
        if fig:
            charts.append(("行业收益", fig))
    if any(k in msg for k in ["因子", "ic", "预测", "权重", "system2"]):
        fig = make_ic_chart(ic_data)
        if fig:
            charts.append(("因子IC", fig))
    if any(k in msg for k in ["pnl", "实盘", "实际", "盈亏"]):
        fig = make_pnl_chart(rpnl)
        if fig:
            charts.append(("PnL分布", fig))
    return charts


# ── 主对话区 ──────────────────────────────────────────────────────────────────
# 加载数据
with st.spinner("加载系统数据..."):
    days    = load_sim30d_days()
    sr      = load_sim30d_stock_returns()
    ic_data = load_ic_analysis()
    rpnl    = load_realized_pnl()

# 欢迎消息（无历史时）
if not st.session_state.messages:
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
**🤖 AI 能力**
- 回答系统绩效、因子、选股问题
- 自动识别并执行系统命令
- 分析图表，生成专业解读
- 对话记忆，上下文连贯
        """)
    with col2:
        st.info("""
**📊 当前数据**
- 30日模拟：5535只全A股
- 3月期均收益：+20.22%（胜率96.7%）
- realized_pnl：379条记录
- LGBM v6 · 38特征 · IC驱动权重
        """)

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"],
                         avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        # 渲染附带图表
        for chart_title, fig_dict in msg.get("charts", []):
            st.plotly_chart(
                go.Figure(fig_dict) if isinstance(fig_dict, dict) else fig_dict,
                use_container_width=True,
                key=f"chart_{msg.get('id','')}_{chart_title}"
            )
        # 命令执行日志
        if msg.get("exec_log"):
            with st.expander("📋 执行日志", expanded=False):
                st.code(msg["exec_log"], language="text")

# ── 处理输入（快捷按钮或文本框）─────────────────────────────────────────────
user_input = st.chat_input("请输入您的问题，或让 AI 帮您执行系统命令...")

# 处理快捷问题按钮
if hasattr(st.session_state, "_pending_question"):
    user_input = st.session_state._pending_question
    del st.session_state._pending_question

if user_input:
    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "id": str(time.time()),
    })

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # AI 处理
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        status_placeholder   = st.empty()
        charts_placeholder   = st.container()
        log_placeholder      = st.empty()

        exec_log = ""
        charts   = []
        reply    = ""

        if api_status:
            # ── 调用 API ──────────────────────────────────────────────────
            with st.spinner("AI 思考中..."):
                try:
                    # 构建历史
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[-8:]
                        if m["role"] in ("user", "assistant")
                    ]

                    resp = requests.post(
                        f"{API_BASE}/api/chat",
                        json={
                            "message": user_input,
                            "history": history,
                            "provider": st.session_state.provider,
                            "auto_execute": st.session_state.auto_execute,
                        },
                        timeout=120,
                    )
                    data = resp.json()
                    reply   = data.get("reply", "")
                    action  = data.get("action", "none")
                    exec_res = data.get("exec_result")

                    # 显示执行状态
                    if action != "none" and action in (exec_svc_commands := {}):
                        pass

                    # 处理命令执行结果
                    if exec_res:
                        exec_log = exec_res.get("output", "")
                        ai_analysis = exec_res.get("ai_analysis", "")
                        if ai_analysis:
                            reply = f"{reply}\n\n**📊 执行分析：**\n{ai_analysis}"

                        status_icon = "✅" if exec_res.get("success") else "❌"
                        elapsed = exec_res.get("elapsed", 0)
                        status_placeholder.success(
                            f"{status_icon} 命令执行完成 | 耗时 {elapsed:.1f}s"
                        )

                except requests.exceptions.ConnectionError:
                    reply = "⚠️ API 服务未启动。请运行：`bash start_api.sh`"
                except Exception as e:
                    reply = f"❌ 请求失败：{e}"
        else:
            # ── 降级：本地直接调用 LLM ───────────────────────────────────
            try:
                from app.api.services.analyzer import (
                    build_data_context, generate_full_response
                )
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[-8:]
                ]
                data_ctx = build_data_context()
                reply = generate_full_response(
                    user_input, history,
                    data_context=data_ctx,
                    provider=st.session_state.provider,
                )
            except Exception as e:
                reply = f"❌ LLM 调用失败：{e}"

        # 显示回复
        response_placeholder.markdown(reply)

        # 生成相关图表
        with charts_placeholder:
            chart_list = get_charts_for_message(user_input, days, sr, ic_data, rpnl)
            for chart_title, fig in chart_list:
                st.plotly_chart(fig, use_container_width=True)
                # 请求 AI 图表解读
                if api_status:
                    try:
                        commentary_resp = requests.post(
                            f"{API_BASE}/api/analyze/chart",
                            json={
                                "title": chart_title,
                                "data": {"chart": chart_title, "message": user_input},
                                "provider": st.session_state.provider,
                            },
                            timeout=30,
                        )
                        commentary = commentary_resp.json().get("commentary", "")
                        if commentary:
                            st.caption(f"🤖 **AI解读**：{commentary}")
                    except Exception:
                        pass

        # 显示执行日志
        if exec_log:
            with log_placeholder:
                with st.expander("📋 执行日志（点击展开）", expanded=False):
                    st.code(exec_log[-3000:], language="text")

    # 保存助手消息
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "id": str(time.time()),
        "exec_log": exec_log,
        "charts": [(t, None) for t, _ in charts],  # 图表在渲染时已显示
    })
