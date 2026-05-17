"""7_系统控制台.py — 交互式系统控制：执行命令 + 实时日志 + AI分析 + 可视化."""
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
    horizon_portfolio_ts, load_nav_curves,
    load_strategy_config,
)

st.set_page_config(page_title="系统控制台 · QuantMind", page_icon="🖥️", layout="wide")

API_BASE = "http://localhost:8000"

# ── 检查 API ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def check_api() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return {}

# ── Session State ─────────────────────────────────────────────────────────────
if "exec_result" not in st.session_state:
    st.session_state.exec_result = None
if "exec_cmd" not in st.session_state:
    st.session_state.exec_cmd = None
if "provider" not in st.session_state:
    st.session_state.provider = "dashscope"

api_status = check_api()

# ── 标题 ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#2D3436 0%,#0984E3 100%);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:16px'>
  <h2 style='margin:0'>🖥️ 系统控制台</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    执行系统命令 · 实时日志 · AI 分析解读 · 结果可视化
  </p>
</div>
""", unsafe_allow_html=True)

# ── API 状态栏 ─────────────────────────────────────────────────────────────────
if api_status:
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.success("✅ FastAPI 后端在线")
    ollama_models = api_status.get("ollama_models", [])
    col_s2.info(f"💻 Ollama: {', '.join(ollama_models) or '无'}")
    col_s3.info(f"🌐 百炼: {'已配置' if api_status.get('dashscope_key') else '未配置'}")
else:
    st.error("""
⚠️ **FastAPI 后端未启动**

请运行：
```bash
bash /home/lenovo/projects/quantmind/start_api.sh
```
或：
```bash
cd /home/lenovo/projects/quantmind
/home/lenovo/miniforge3/envs/quantmind/bin/uvicorn app.api.server:app --port 8000 --reload
```
    """)

st.divider()

# ── 布局：左栏=命令面板，右栏=结果 ──────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 🎛️ 命令面板")

    # AI 模型选择
    provider = st.selectbox(
        "AI 分析模型",
        options=["dashscope", "ollama", "deepseek"],
        format_func=lambda x: {
            "dashscope": "🌐 百炼 qwen-turbo",
            "ollama":    "💻 Ollama qwen2.5:7b",
            "deepseek":  "🔮 DeepSeek",
        }[x],
        index=["dashscope", "ollama", "deepseek"].index(st.session_state.provider),
    )
    st.session_state.provider = provider

    analyze_result = st.toggle("🤖 执行后自动AI分析", value=True)
    st.markdown("---")

    # ── 命令按钮组 ────────────────────────────────────────────────────────────
    COMMANDS_META = {
        "simulate":          ("🚀", "30日全A股模拟",   "red"),
        "simulate_evaluate": ("📈", "绩效评估",        "blue"),
        "optimize":          ("📊", "IC优化分析",      "green"),
        "backtest":          ("📉", "NAV回测",         "orange"),
        "train_meta":        ("🧠", "重训 meta-learner", "violet"),
        "build_features":    ("🔧", "重建特征面板",    "gray"),
    }

    st.markdown("**📋 可用操作**")
    for cmd_key, (icon, label, color) in COMMANDS_META.items():
        btn_disabled = not bool(api_status)
        if st.button(f"{icon} {label}", use_container_width=True,
                     key=f"btn_{cmd_key}", disabled=btn_disabled,
                     type="primary" if cmd_key in ("simulate_evaluate", "optimize") else "secondary"):
            st.session_state.exec_cmd = cmd_key

    st.markdown("---")

    # ── 数据快速查看 ──────────────────────────────────────────────────────────
    st.markdown("**📂 数据快速查看**")
    if st.button("🔄 刷新数据摘要", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if api_status:
        try:
            summary_resp = requests.get(f"{API_BASE}/api/data/summary", timeout=10)
            summary = summary_resp.json()
            perf = summary.get("performance", {})
            p3m = perf.get("3m", {})
            if p3m:
                st.metric("3月期均收益", f"{p3m.get('mean_return',0):+.2f}%")
                st.metric("3月胜率",     f"{p3m.get('win_rate',0):.1f}%")
                st.metric("3月IR",       f"{p3m.get('ir',0):+.3f}")
        except Exception:
            pass


# ── 右栏：执行区域 + 结果展示 ────────────────────────────────────────────────
with col_right:
    st.markdown("### 📊 执行结果 & AI 分析")

    # ── 执行命令（如有待执行命令）──────────────────────────────────────────
    if st.session_state.exec_cmd and api_status:
        cmd_key = st.session_state.exec_cmd
        st.session_state.exec_cmd = None  # 清除

        cmd_label = COMMANDS_META.get(cmd_key, ("", cmd_key, ""))[1]

        with st.status(f"正在执行：{cmd_label}...", expanded=True) as status:
            st.write(f"**命令**：`{cmd_key}`")
            st.write("⏳ 正在执行，请稍候...")
            t0 = time.time()

            try:
                resp = requests.post(
                    f"{API_BASE}/api/execute",
                    json={
                        "cmd_key": cmd_key,
                        "provider": st.session_state.provider,
                        "analyze_result": analyze_result,
                    },
                    timeout=900,  # 15分钟超时
                )
                data = resp.json()
                elapsed = data.get("elapsed", time.time() - t0)

                if data.get("success"):
                    status.update(label=f"✅ 执行成功（{elapsed:.1f}s）", state="complete")
                else:
                    status.update(label=f"❌ 执行失败（{elapsed:.1f}s）", state="error")

                st.session_state.exec_result = data

            except requests.exceptions.Timeout:
                status.update(label="⏰ 执行超时", state="error")
                st.session_state.exec_result = {"success": False, "output": "执行超时", "ai_analysis": ""}
            except Exception as e:
                status.update(label=f"❌ 请求失败：{e}", state="error")
                st.session_state.exec_result = None

    # ── 显示执行结果 ─────────────────────────────────────────────────────────
    if st.session_state.exec_result:
        res = st.session_state.exec_result
        cmd_label = res.get("cmd_label", "命令")

        # AI 分析
        ai_analysis = res.get("ai_analysis", "")
        if ai_analysis:
            st.markdown(f"""
<div style='background:rgba(9,132,227,0.08);border:1px solid rgba(9,132,227,0.3);
            border-radius:10px;padding:16px;margin-bottom:16px'>
  <div style='font-size:.9rem;color:#0984E3;font-weight:600;margin-bottom:8px'>
    🤖 AI 分析 · {cmd_label}
  </div>
  <div style='font-size:.9rem;line-height:1.6'>
    {ai_analysis.replace(chr(10), '<br>')}
  </div>
</div>
""", unsafe_allow_html=True)

        # 执行日志
        output = res.get("output", "")
        if output:
            with st.expander(f"📋 执行日志（{res.get('elapsed',0):.1f}s）", expanded=not ai_analysis):
                st.code(output[-5000:] if len(output) > 5000 else output, language="text")

        # 执行后刷新图表
        st.markdown("#### 📈 最新数据可视化")
        st.cache_data.clear()

    # ── 实时数据看板（无论是否刚执行）────────────────────────────────────────
    with st.spinner("加载最新数据..."):
        try:
            days    = load_sim30d_days()
            sr      = load_sim30d_stock_returns()
            ic_data = load_ic_analysis()
            rpnl    = load_realized_pnl()
            navs    = load_nav_curves()
            cfg     = load_strategy_config()
        except Exception as e:
            st.error(f"数据加载失败：{e}")
            days, sr, ic_data, rpnl, navs, cfg = [], pd.DataFrame(), {}, pd.DataFrame(), {}, {}

    if days:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 收益曲线", "🔬 因子IC", "📊 PnL分布", "⚖️ 系统权重"
        ])

        with tab1:
            # 四期累计收益
            HZ_COLORS = {"1w": "#D63031", "2w": "#E17055", "21d": "#FDCB6E", "3m": "#00B894"}
            fig_cum = go.Figure()
            perf_summary = {}
            for hz, color in HZ_COLORS.items():
                ts = horizon_portfolio_ts(days, hz)
                if ts.empty:
                    continue
                mean_r = float(ts["mean"].mean())
                win_r  = float((ts["mean"] > 0).mean())
                ir_v   = mean_r / ts["mean"].std() if ts["mean"].std() > 0 else 0
                perf_summary[hz] = (mean_r, win_r, ir_v)
                fig_cum.add_trace(go.Scatter(
                    x=ts["date"].dt.strftime("%m-%d"), y=ts["cum_return"] * 100,
                    name={"1w":"1周","2w":"2周","21d":"21天","3m":"3月"}[hz],
                    line=dict(color=color, width=2.5), mode="lines+markers", marker_size=4,
                ))
            fig_cum.add_hline(y=0, line_dash="dot", line_color="#B2BEC3")
            fig_cum.update_layout(
                height=320, margin=dict(t=10, b=40, l=50, r=20),
                yaxis_title="累计收益(%)",
                legend=dict(orientation="h", y=1.08),
                hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            # 绩效摘要
            cols_perf = st.columns(len(perf_summary))
            labels = {"1w":"1周","2w":"2周","21d":"21天","3m":"3月"}
            for i, (hz, (mean_r, win_r, ir_v)) in enumerate(perf_summary.items()):
                with cols_perf[i]:
                    delta_color = "normal" if mean_r > 0 else "inverse"
                    st.metric(labels[hz], f"{mean_r*100:+.2f}%",
                              f"胜率{win_r*100:.0f}% IR={ir_v:+.2f}",
                              delta_color=delta_color)

            # AI 解读（如果 API 可用）
            if api_status and perf_summary:
                p3m = perf_summary.get("3m", (0, 0, 0))
                try:
                    r = requests.post(
                        f"{API_BASE}/api/analyze/chart",
                        json={"title": "四持仓期累计收益",
                              "data": {hz: {"mean_return": f"{v[0]*100:.2f}%",
                                            "win_rate": f"{v[1]*100:.1f}%"}
                                       for hz, v in perf_summary.items()},
                              "provider": st.session_state.provider},
                        timeout=20,
                    )
                    c = r.json().get("commentary", "")
                    if c:
                        st.caption(f"🤖 {c}")
                except Exception:
                    pass

        with tab2:
            ic_final = ic_data.get("ic_final_picks", {})
            if ic_final:
                ic_items = [
                    (k, v.get("ic_3m") or 0, v.get("p_3m") or 1)
                    for k, v in ic_final.items()
                    if v.get("ic_3m") is not None
                ]
                ic_items.sort(key=lambda x: x[1], reverse=True)
                factors = [x[0] for x in ic_items]
                ic_vals = [x[1] for x in ic_items]
                p_vals  = [x[2] for x in ic_items]
                sig_marks = ["**" if p < 0.05 else ("*" if p < 0.10 else "") for p in p_vals]
                colors = ["#00B894" if v > 0 else "#D63031" for v in ic_vals]

                fig_ic = go.Figure(go.Bar(
                    x=factors, y=ic_vals, marker_color=colors,
                    text=[f"{v:+.4f}{s}" for v, s in zip(ic_vals, sig_marks)],
                    textposition="outside",
                ))
                fig_ic.add_hline(y=0, line_dash="dot", line_color="#B2BEC3")
                fig_ic.update_layout(
                    title="因子 3月 IC（**=p<0.05，*=p<0.10）",
                    height=320, margin=dict(t=40, b=80, l=10, r=10),
                    yaxis_title="IC", xaxis=dict(tickangle=-30),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_ic, use_container_width=True)

                # AI 解读
                if api_status and ic_items:
                    try:
                        top3 = {f: v for f, v, _ in ic_items[:3]}
                        r = requests.post(
                            f"{API_BASE}/api/analyze/chart",
                            json={"title": "因子IC分析",
                                  "data": {"top_factors": top3, "total": len(ic_items)},
                                  "provider": st.session_state.provider},
                            timeout=20,
                        )
                        c = r.json().get("commentary", "")
                        if c:
                            st.caption(f"🤖 {c}")
                    except Exception:
                        pass
            else:
                st.info("IC 数据不可用。")

        with tab3:
            if not rpnl.empty:
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    if "pnl_pct" in rpnl.columns:
                        fig_hist = go.Figure(go.Histogram(
                            x=rpnl["pnl_pct"] * 100, nbinsx=30,
                            marker_color="#0984E3", opacity=0.75,
                        ))
                        mean_pnl = float(rpnl["pnl_pct"].mean()) * 100
                        fig_hist.add_vline(x=mean_pnl, line_dash="dash",
                                           line_color="#D63031",
                                           annotation_text=f"均值{mean_pnl:+.2f}%")
                        fig_hist.update_layout(
                            title=f"PnL 分布（{len(rpnl)} 条）",
                            height=300, margin=dict(t=40, b=40),
                            xaxis_title="收益率(%)", plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                with col_p2:
                    win_rate = float((rpnl["pnl_pct"] > 0).mean()) * 100 if "pnl_pct" in rpnl.columns else 0
                    mean_win = float(rpnl[rpnl["pnl_pct"] > 0]["pnl_pct"].mean()) * 100 if "pnl_pct" in rpnl.columns else 0
                    mean_loss = float(rpnl[rpnl["pnl_pct"] < 0]["pnl_pct"].mean()) * 100 if "pnl_pct" in rpnl.columns else 0
                    st.metric("记录总数", f"{len(rpnl)} 条")
                    st.metric("胜率", f"{win_rate:.1f}%")
                    st.metric("平均盈利", f"{mean_win:+.2f}%")
                    st.metric("平均亏损", f"{mean_loss:+.2f}%")
                    if win_rate > 0 and mean_loss != 0:
                        profit_factor = abs(mean_win * win_rate / (mean_loss * (100 - win_rate)))
                        st.metric("盈亏比", f"{profit_factor:.2f}x")
            else:
                st.info("PnL 数据不可用。")

        with tab4:
            s2_updates = cfg.get("system2_updates", {})
            new_weights = s2_updates.get("weights_calibrated",
                {"value": 0.242, "momentum": 0.223, "quality": 0.333, "technical": 0.202})
            cur_weights = s2_updates.get("weights_previous",
                {"value": 0.30, "momentum": 0.25, "quality": 0.25, "technical": 0.20})

            DIM_LABELS = {"value": "价值", "momentum": "动量",
                          "quality": "质量", "technical": "技术"}
            DIM_COLORS = {"value": "#FDCB6E", "momentum": "#E17055",
                          "quality": "#00B894", "technical": "#74B9FF"}

            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.markdown("**v1 权重（启发式）**")
                fig_cur = go.Figure(go.Pie(
                    labels=[DIM_LABELS[k] for k in cur_weights],
                    values=list(cur_weights.values()), hole=0.5,
                    marker_colors=[DIM_COLORS[k] for k in cur_weights],
                ))
                fig_cur.update_layout(height=220, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_cur, use_container_width=True)

            with col_w2:
                st.markdown("**v2 权重（IC 驱动校准）**")
                fig_new = go.Figure(go.Pie(
                    labels=[DIM_LABELS[k] for k in new_weights],
                    values=list(new_weights.values()), hole=0.5,
                    marker_colors=[DIM_COLORS[k] for k in new_weights],
                ))
                fig_new.update_layout(height=220, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_new, use_container_width=True)

            # 权重变化表
            weight_rows = []
            for k in cur_weights:
                delta = new_weights.get(k, 0) - cur_weights[k]
                weight_rows.append({
                    "维度": DIM_LABELS[k],
                    "v1权重": f"{cur_weights[k]*100:.1f}%",
                    "v2权重": f"{new_weights.get(k,0)*100:.1f}%",
                    "变化":   f"{delta*100:+.1f}pp",
                    "方向":   "⬆️" if delta > 0 else "⬇️",
                })
            st.dataframe(pd.DataFrame(weight_rows), hide_index=True, use_container_width=True)

            # AI 解读
            if api_status:
                try:
                    r = requests.post(
                        f"{API_BASE}/api/analyze/chart",
                        json={"title": "System2权重校准",
                              "data": {"v1": cur_weights, "v2": new_weights},
                              "provider": st.session_state.provider},
                        timeout=20,
                    )
                    c = r.json().get("commentary", "")
                    if c:
                        st.caption(f"🤖 {c}")
                except Exception:
                    pass
    else:
        st.info("暂无模拟数据。请先运行「30日全A股模拟」。")
