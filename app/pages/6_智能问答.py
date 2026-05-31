"""6_智能问答.py — AI对话中心（DashScope / Ollama · 命令识别执行 · 图表生成 · 上下文记忆）."""
from __future__ import annotations

import sys
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

st.set_page_config(
    page_title="智能问答 · QuantMind",
    page_icon="💬",
    layout="wide",
)

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "messages"   not in st.session_state:
    st.session_state.messages   = []
if "provider"   not in st.session_state:
    st.session_state.provider   = "dashscope"
if "auto_exec"  not in st.session_state:
    st.session_state.auto_exec  = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=20)
def get_health() -> dict:
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).json()
    except Exception:
        return {}


def api_chat(message: str, history: list[dict], provider: str, auto_exec: bool) -> dict:
    resp = requests.post(
        f"{API_BASE}/api/chat",
        json={"message": message, "history": history,
              "provider": provider, "auto_execute": auto_exec},
        timeout=180,
    )
    return resp.json()


def local_llm_chat(message: str, history: list[dict], provider: str) -> str:
    """API不可用时，直接调用本地LLM."""
    from app.api.services.analyzer import build_data_context, generate_full_response
    ctx = build_data_context()
    return generate_full_response(message, history, data_context=ctx, provider=provider)


# ─────────────────────────────────────────────────────────────────────────────
# 图表生成（按关键词匹配）
# ─────────────────────────────────────────────────────────────────────────────
def charts_for_message(msg: str, days, sr, ic_data, rpnl) -> list[tuple[str, go.Figure]]:
    msg_l  = msg.lower()
    charts = []

    # 收益 / 表现
    if any(k in msg_l for k in ["表现","收益","绩效","总体","整体","怎么样","成绩","回报"]):
        try:
            HZ = {"1w":("#D63031","1周"), "2w":("#E17055","2周"),
                  "21d":("#FDCB6E","21天"), "3m":("#00B894","3月")}
            fig = go.Figure()
            for hz, (color, label) in HZ.items():
                ts = horizon_portfolio_ts(days, hz)
                if ts.empty: continue
                fig.add_trace(go.Scatter(
                    x=ts["date"].dt.strftime("%Y-%m-%d"), y=ts["cum_return"]*100,
                    name=label, line=dict(color=color, width=2.5),
                    mode="lines+markers", marker_size=3,
                ))
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(178,190,195,0.4)")
            fig.update_layout(title="累计收益曲线（30日模拟）", height=300,
                              margin=dict(t=35,b=35,l=45,r=15),
                              yaxis_title="累计收益(%)",
                              legend=dict(orientation="h", y=1.1),
                              hovermode="x unified",
                              plot_bgcolor="rgba(0,0,0,0)")
            charts.append(("累计收益曲线", fig))
        except Exception:
            pass

    # 股票排行
    if any(k in msg_l for k in ["股票","最好","top","推荐","排名","涨幅","上涨"]):
        try:
            final = sr[sr["in_final"]].copy()
            if "return_3m" in final.columns and not final.empty:
                top = (final.groupby("ticker")["return_3m"].mean()
                       .sort_values(ascending=False).head(15))
                colors = ["#00B894" if v>0 else "#D63031" for v in top.values]
                fig = go.Figure(go.Bar(
                    x=top.index, y=top.values*100, marker_color=colors,
                    text=[f"{v*100:+.1f}%" for v in top.values],
                    textposition="outside",
                ))
                fig.update_layout(title="股票3月均值收益 Top15", height=280,
                                  margin=dict(t=35,b=55,l=10,r=10),
                                  yaxis_title="3月收益(%)",
                                  xaxis=dict(tickfont_size=9),
                                  plot_bgcolor="rgba(0,0,0,0)")
                charts.append(("股票收益排行", fig))
        except Exception:
            pass

    # 行业分析
    if any(k in msg_l for k in ["行业","sector","板块","归因","领域"]):
        try:
            final = sr[sr["in_final"]].copy()
            if "return_3m" in final.columns and "industry" in final.columns:
                ind = (final.groupby("industry")["return_3m"].mean()
                       .sort_values(ascending=False).head(10))
                colors = ["#00B894" if v>0 else "#D63031" for v in ind.values]
                fig = go.Figure(go.Bar(
                    x=ind.index, y=ind.values*100, marker_color=colors,
                    text=[f"{v*100:+.1f}%" for v in ind.values],
                    textposition="outside",
                ))
                fig.update_layout(title="行业3月均值收益 Top10", height=280,
                                  margin=dict(t=35,b=75,l=10,r=10),
                                  yaxis_title="3月均值(%)",
                                  xaxis=dict(tickangle=-30, tickfont_size=10),
                                  plot_bgcolor="rgba(0,0,0,0)")
                charts.append(("行业收益归因", fig))
        except Exception:
            pass

    # 因子 / IC / 权重
    if any(k in msg_l for k in ["因子","ic","预测","权重","system2","维度","质量","价值","动量"]):
        try:
            ic_final = ic_data.get("ic_final_picks", {})
            items = [(k, v.get("ic_3m") or 0) for k,v in ic_final.items()
                     if v.get("ic_3m") is not None]
            if items:
                items.sort(key=lambda x: x[1], reverse=True)
                factors, vals = zip(*items)
                colors = ["#00B894" if v>0 else "#D63031" for v in vals]
                fig = go.Figure(go.Bar(
                    x=list(factors), y=list(vals), marker_color=colors,
                    text=[f"{v:+.4f}" for v in vals],
                    textposition="outside",
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="rgba(178,190,195,0.4)")
                fig.update_layout(title="因子3月IC（预测能力）", height=280,
                                  margin=dict(t=35,b=80,l=10,r=10),
                                  yaxis_title="Spearman IC",
                                  xaxis=dict(tickangle=-30),
                                  plot_bgcolor="rgba(0,0,0,0)")
                charts.append(("因子IC分析", fig))
        except Exception:
            pass

    # PnL / 实盘
    if any(k in msg_l for k in ["pnl","实盘","实际","盈亏","损益","收益分布"]):
        try:
            if not rpnl.empty and "actual_return_63d" in rpnl.columns:
                fig = go.Figure(go.Histogram(
                    x=rpnl["actual_return_63d"]*100, nbinsx=30,
                    marker_color="#0984E3", opacity=0.8,
                ))
                mean_v = float(rpnl["actual_return_63d"].mean())*100
                fig.add_vline(x=mean_v, line_dash="dash", line_color="#D63031",
                              annotation_text=f"均值 {mean_v:+.2f}%")
                fig.update_layout(title=f"realized PnL分布（{len(rpnl)}条）",
                                  height=260, margin=dict(t=35,b=35),
                                  xaxis_title="收益率(%)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                charts.append(("PnL收益分布", fig))
        except Exception:
            pass

    return charts


# ─────────────────────────────────────────────────────────────────────────────
# 页面渲染
# ─────────────────────────────────────────────────────────────────────────────
health   = get_health()
api_ok   = bool(health)
provider = st.session_state.provider

# 标题
st.markdown("""
<div style='background:linear-gradient(135deg,#2D3436,#0984E3 80%,#00B894);
            padding:18px 28px;border-radius:14px;color:white;margin-bottom:18px'>
  <h2 style='margin:0;font-size:1.5rem'>💬 智能问答</h2>
  <p style='margin:5px 0 0;opacity:.85;font-size:.9rem'>
    AI理解你的问题 · 自动生成图表 · 执行系统命令 · 上下文记忆对话
  </p>
</div>
""", unsafe_allow_html=True)

# ── 侧栏 ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 模型设置")

    if api_ok:
        ollama_models = health.get("ollama_models", [])
        st.success("✅ API 在线")
        if ollama_models:
            st.caption(f"Ollama: {', '.join(ollama_models)}")
    else:
        st.warning("⚠️ API未启动，将直接调用LLM")

    new_provider = st.radio(
        "选择模型",
        options=["dashscope", "ollama", "deepseek"],
        format_func=lambda x: {
            "dashscope": "🌐 百炼 qwen-turbo",
            "ollama":    "💻 Ollama 本地",
            "deepseek":  "🔮 DeepSeek",
        }[x],
        index=["dashscope","ollama","deepseek"].index(st.session_state.provider),
    )
    if new_provider != st.session_state.provider:
        st.session_state.provider = new_provider
        st.rerun()

    st.session_state.auto_exec = st.toggle(
        "🤖 自动执行命令",
        value=st.session_state.auto_exec,
        help="识别到系统操作指令时自动执行并返回结果",
    )

    st.divider()
    st.markdown("### 💡 试试问这些")

    QUICK_QS = [
        ("📊", "最近30天整体表现？"),
        ("🏆", "哪些股票收益最高？"),
        ("🏭", "哪个行业表现最好？"),
        ("🔬", "哪个因子预测最准？"),
        ("⚖️", "System2权重是怎么校准的？"),
        ("📈", "帮我运行一次绩效评估"),
        ("🚀", "帮我运行IC优化分析"),
        ("🎯", "现在系统最建议做什么？"),
    ]
    for icon, q in QUICK_QS:
        if st.button(f"{icon} {q}", use_container_width=True, key=f"sq_{q[:6]}"):
            st.session_state._pending = q

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(f"""
<div style='font-size:.75rem;color:#636E72;margin-top:8px'>
模型：{st.session_state.provider}<br>
消息数：{len(st.session_state.messages)}<br>
API：{'在线' if api_ok else '离线（降级）'}
</div>
""", unsafe_allow_html=True)

# ── 数据加载 ──────────────────────────────────────────────────────────────────
with st.spinner("加载数据..."):
    try:
        days    = load_sim30d_days()
        sr      = load_sim30d_stock_returns()
        ic_data = load_ic_analysis()
        rpnl    = load_realized_pnl()
    except Exception:
        days, sr, ic_data, rpnl = [], pd.DataFrame(), {}, pd.DataFrame()

# ── 欢迎屏（无消息时）────────────────────────────────────────────────────────
if not st.session_state.messages:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("""
**🤖 AI 能力**
- 回答绩效、因子、选股问题
- 自动识别并执行系统命令
- 按问题内容生成相关图表
- 连续对话，上下文记忆
        """)
    with c2:
        p3m_text = "—"
        if days:
            ts = horizon_portfolio_ts(days, "3m")
            if not ts.empty:
                p3m_text = f"+{ts['mean'].mean()*100:.2f}%"
        st.info(f"""
**📊 当前系统状态**
- 模拟天数：{len(days)} 天
- 3月期均收益：{p3m_text}
- PnL记录：{len(rpnl)} 条
- 主模型：LGBM v6 · 38特征
        """)
    with c3:
        st.info("""
**🎯 快捷指令示例**
- 「最近表现怎么样」→ 生成收益图
- 「哪个行业最好」→ 行业归因图
- 「帮我运行绩效评估」→ 执行命令
- 「质量因子IC多少」→ 因子分析图
        """)

# ── 历史消息渲染 ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        for chart_spec in msg.get("charts", []):
            try:
                fig_data = chart_spec.get("fig")
                if isinstance(fig_data, dict):
                    st.plotly_chart(go.Figure(fig_data),
                                    use_container_width=True,
                                    key=f"hist_chart_{i}_{chart_spec['title']}")
            except Exception:
                pass
            ai_c = chart_spec.get("ai_comment", "")
            if ai_c:
                st.caption(f"🤖 {ai_c}")
        if msg.get("exec_log"):
            with st.expander("📋 执行日志", expanded=False):
                st.code(msg["exec_log"], language="text")
        if msg.get("exec_status"):
            color = "#00B894" if msg.get("exec_ok") else "#D63031"
            st.markdown(f"<span style='color:{color};font-size:.8rem'>{msg['exec_status']}</span>",
                        unsafe_allow_html=True)

# ── 输入处理 ──────────────────────────────────────────────────────────────────
user_input = st.chat_input("问我任何关于系统的问题，或说「帮我运行XX」…")

# 快捷按钮触发
if hasattr(st.session_state, "_pending"):
    user_input = st.session_state._pending
    del st.session_state._pending

if user_input:
    # 1. 显示用户消息
    st.session_state.messages.append({
        "role": "user", "content": user_input,
    })
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # 2. AI 处理
    with st.chat_message("assistant", avatar="🤖"):
        reply_ph  = st.empty()
        status_ph = st.empty()
        charts_ct = st.container()
        log_ph    = st.empty()

        reply      = ""
        exec_log   = ""
        exec_ok    = False
        exec_status = ""
        charts_rendered: list[dict] = []

        with st.spinner("AI思考中…"):
            if api_ok:
                # ── 通过 API 调用 ─────────────────────────────────────────
                try:
                    history_for_api = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[-10:]
                        if m["role"] in ("user", "assistant")
                    ]
                    data = api_chat(
                        user_input, history_for_api,
                        st.session_state.provider,
                        st.session_state.auto_exec,
                    )
                    reply   = data.get("reply", "")
                    action  = data.get("action", "none")
                    exec_res = data.get("exec_result")

                    # 处理命令执行结果
                    if exec_res:
                        exec_log    = exec_res.get("output", "")
                        exec_ok     = exec_res.get("success", False)
                        elapsed     = exec_res.get("elapsed", 0)
                        ai_analysis = exec_res.get("ai_analysis", "")
                        exec_status = f"{'✅' if exec_ok else '❌'} 命令执行{'成功' if exec_ok else '失败'} · 耗时 {elapsed:.1f}s"
                        if ai_analysis and ai_analysis not in reply:
                            reply = f"{reply}\n\n**📊 执行结果分析：**\n{ai_analysis}"

                except requests.exceptions.ConnectionError:
                    reply = "⚠️ 无法连接 API，已切换到本地模式。"
                    try:
                        history_local = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[-6:]
                        ]
                        reply = local_llm_chat(user_input, history_local,
                                               st.session_state.provider)
                    except Exception as e:
                        reply = f"❌ LLM调用失败：{e}"
                except Exception as e:
                    reply = f"❌ 请求失败：{e}"
            else:
                # ── 直接本地 LLM ─────────────────────────────────────────
                try:
                    history_local = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[-6:]
                    ]
                    reply = local_llm_chat(user_input, history_local,
                                           st.session_state.provider)
                except Exception as e:
                    reply = f"❌ LLM调用失败：{e}"

        # 3. 显示回复
        reply_ph.markdown(reply)
        if exec_status:
            color = "#00B894" if exec_ok else "#D63031"
            status_ph.markdown(
                f"<span style='color:{color};font-size:.8rem'>{exec_status}</span>",
                unsafe_allow_html=True,
            )

        # 4. 生成图表
        with charts_ct:
            raw_charts = charts_for_message(user_input, days, sr, ic_data, rpnl)
            for title, fig in raw_charts:
                st.plotly_chart(fig, use_container_width=True,
                                key=f"new_chart_{title}_{time.time()}")
                # AI 图表点评
                ai_comment = ""
                if api_ok:
                    try:
                        r = requests.post(
                            f"{API_BASE}/api/analyze/chart",
                            json={"title": title,
                                  "data": {"chart": title, "query": user_input},
                                  "provider": st.session_state.provider},
                            timeout=20,
                        )
                        ai_comment = r.json().get("commentary", "")
                        if ai_comment:
                            st.caption(f"🤖 {ai_comment}")
                    except Exception:
                        pass
                charts_rendered.append({
                    "title":      title,
                    "fig":        fig.to_dict(),
                    "ai_comment": ai_comment,
                })

        # 5. 执行日志
        if exec_log:
            with log_ph:
                with st.expander("📋 执行日志（点击展开）", expanded=False):
                    st.code(exec_log[-4000:] if len(exec_log) > 4000 else exec_log,
                            language="text")

    # 6. 保存助手消息
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     reply,
        "charts":      charts_rendered,
        "exec_log":    exec_log,
        "exec_ok":     exec_ok,
        "exec_status": exec_status,
    })
