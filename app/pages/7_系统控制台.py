"""7_系统控制台.py — 交互式运行中心：一键执行 + 实时日志 + AI解读 + 结果可视化."""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.services.executor import COMMANDS, COMMAND_GROUPS
from app.utils.sim_data import (
    load_sim30d_days, load_sim30d_stock_returns,
    load_ic_analysis, load_realized_pnl,
    horizon_portfolio_ts, load_strategy_config,
)

st.set_page_config(
    page_title="系统控制台 · QuantMind",
    page_icon="🖥️",
    layout="wide",
)

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "exec_history" not in st.session_state:
    st.session_state.exec_history = []   # list[dict]
if "pending_cmd"  not in st.session_state:
    st.session_state.pending_cmd  = None
if "ai_provider"  not in st.session_state:
    st.session_state.ai_provider  = "dashscope"


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def get_health() -> dict:
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).json()
    except Exception:
        return {}

@st.cache_data(ttl=30)
def get_summary() -> dict:
    try:
        return requests.get(f"{API_BASE}/api/data/summary", timeout=8).json()
    except Exception:
        return {}


def call_execute(cmd_key: str, provider: str) -> dict:
    timeout_sec = COMMANDS[cmd_key].get("timeout", 600) + 30
    resp = requests.post(
        f"{API_BASE}/api/execute",
        json={"cmd_key": cmd_key, "provider": provider, "analyze_result": True},
        timeout=timeout_sec,
    )
    return resp.json()


def call_analyze_chart(title: str, data: dict, provider: str) -> str:
    try:
        r = requests.post(
            f"{API_BASE}/api/analyze/chart",
            json={"title": title, "data": data, "provider": provider},
            timeout=25,
        )
        return r.json().get("commentary", "")
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
health = get_health()
api_ok = bool(health)

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("## 🖥️ 系统控制台")
    st.caption("在此触发任意系统功能 · 实时查看执行日志 · AI自动解读结果")
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if api_ok:
        st.success("✅ API 在线", icon="🟢")
    else:
        st.error("❌ API 离线", icon="🔴")
        st.info("请先运行:\n```\nbash start_api.sh\n```")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 主布局：左侧控制面板 | 右侧内容区
# ─────────────────────────────────────────────────────────────────────────────
sidebar_col, main_col = st.columns([1, 2.5], gap="large")

# ╔══════════════════════════════╗
# ║       左侧：控制面板          ║
# ╚══════════════════════════════╝
with sidebar_col:

    # ── AI 模型 ──────────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ AI 设置")
    provider = st.radio(
        "分析模型",
        options=["dashscope", "ollama", "deepseek"],
        format_func=lambda x: {
            "dashscope": "🌐 百炼 qwen-turbo",
            "ollama":    "💻 Ollama qwen2.5:7b",
            "deepseek":  "🔮 DeepSeek",
        }[x],
        index=["dashscope", "ollama", "deepseek"].index(st.session_state.ai_provider),
        key="provider_radio",
    )
    st.session_state.ai_provider = provider
    ai_analyze = st.toggle("执行后AI自动分析", value=True)

    st.divider()

    # ── 命令按钮（按组分类）──────────────────────────────────────────────────
    st.markdown("#### 🎛️ 可用操作")

    for group_name, cmd_keys in COMMAND_GROUPS.items():
        with st.expander(f"**{group_name}**", expanded=(group_name in ("模拟盘", "分析"))):
            for key in cmd_keys:
                if key not in COMMANDS:
                    continue
                info = COMMANDS[key]
                col_btn, col_info = st.columns([3, 1])
                with col_btn:
                    disabled = not api_ok
                    clicked  = st.button(
                        info["label"],
                        key=f"run_{key}",
                        use_container_width=True,
                        disabled=disabled,
                        type="primary" if info.get("quick") else "secondary",
                    )
                    if clicked:
                        st.session_state.pending_cmd = key
                with col_info:
                    est = info.get("timeout", 600)
                    mins = est // 60
                    st.caption(f"⏱ ~{mins}分" if mins >= 1 else "⏱ <1分")
                st.caption(f"&nbsp;&nbsp;{info['desc']}")

    st.divider()

    # ── 快速系统状态 ──────────────────────────────────────────────────────────
    st.markdown("#### 📋 系统快照")
    summary = get_summary()
    if summary:
        perf = summary.get("performance", {})
        p3m  = perf.get("3m", {})
        p1w  = perf.get("1w", {})
        st.metric("模拟天数",  f"{summary.get('simulation_days', '—')} 天")
        st.metric("3月均收益", f"{p3m.get('mean_return', '—')}%",
                  f"胜率 {p3m.get('win_rate', '—')}%")
        st.metric("1周均收益", f"{p1w.get('mean_return', '—')}%")
        st.metric("PnL记录",   f"{summary.get('realized_pnl_count', '—')} 条")
    else:
        st.caption("API离线或数据未加载")

    if st.button("🔄 刷新缓存", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ╔══════════════════════════════════════╗
# ║       右侧：主内容区                  ║
# ╚══════════════════════════════════════╝
with main_col:

    # ── 执行挂起的命令 ────────────────────────────────────────────────────────
    if st.session_state.pending_cmd and api_ok:
        cmd_key   = st.session_state.pending_cmd
        cmd_info  = COMMANDS[cmd_key]
        st.session_state.pending_cmd = None

        st.markdown(f"### ▶ 正在执行：{cmd_info['label']}")
        prog_bar = st.progress(0, text="准备中...")
        log_box  = st.empty()

        t_start = time.time()
        log_lines: list[str] = []

        # 模拟进度（不知道实际进度，用时间估算）
        def fake_progress(elapsed: float, total: float) -> float:
            return min(0.95, elapsed / max(total, 1))

        with st.spinner(f"执行 {cmd_info['short']} 中，请勿关闭页面..."):
            try:
                result = call_execute(cmd_key, provider)
            except requests.exceptions.Timeout:
                result = {"success": False, "output": "请求超时", "ai_analysis": "", "elapsed": 0}
            except Exception as e:
                result = {"success": False, "output": str(e), "ai_analysis": "", "elapsed": 0}

        prog_bar.progress(1.0, text="完成")
        elapsed   = result.get("elapsed", 0)
        success   = result.get("success", False)
        output    = result.get("output", "")
        ai_text   = result.get("ai_analysis", "")

        # 显示执行日志
        log_box.code(output[-4000:] if len(output) > 4000 else output,
                     language="text")

        # 状态条
        if success:
            st.success(f"✅ 执行成功  |  耗时 {elapsed:.1f}s  |  {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.error(f"❌ 执行失败  |  耗时 {elapsed:.1f}s")

        # AI 分析
        if ai_analyze and ai_text:
            st.markdown(f"""
<div style='background:rgba(0,184,148,0.08);border-left:4px solid #00B894;
            border-radius:0 8px 8px 0;padding:14px 18px;margin:12px 0'>
  <div style='color:#00B894;font-weight:700;font-size:.85rem;margin-bottom:6px'>
    🤖 AI 解读 · {provider}
  </div>
  <div style='font-size:.9rem;line-height:1.7'>{ai_text.replace(chr(10),'<br>')}</div>
</div>
""", unsafe_allow_html=True)

        # 写入历史
        st.session_state.exec_history.append({
            "ts":       datetime.now().strftime("%H:%M:%S"),
            "cmd_key":  cmd_key,
            "label":    cmd_info["short"],
            "success":  success,
            "elapsed":  elapsed,
            "output":   output,
            "ai_text":  ai_text,
        })

        # 刷新数据缓存，立刻反映最新结果
        st.cache_data.clear()
        st.divider()

    # ── Tab 区域：实时数据 + 执行历史 ──────────────────────────────────────
    tab_data, tab_perf, tab_ic, tab_history = st.tabs(
        ["📊 实时数据总览", "📈 收益曲线", "🔬 因子分析", "📋 执行历史"]
    )

    # ── Tab1: 实时数据总览 ──────────────────────────────────────────────────
    with tab_data:
        try:
            days    = load_sim30d_days()
            sr      = load_sim30d_stock_returns()
            rpnl    = load_realized_pnl()
            cfg     = load_strategy_config()
        except Exception as e:
            st.error(f"数据加载失败：{e}")
            days, sr, rpnl, cfg = [], pd.DataFrame(), pd.DataFrame(), {}

        if days:
            # KPI 行
            kpi_cols = st.columns(4)
            perf_all = {}
            for hz in ["1w", "21d", "3m"]:
                ts = horizon_portfolio_ts(days, hz)
                if not ts.empty:
                    m = float(ts["mean"].mean())
                    w = float((ts["mean"] > 0).mean())
                    ir = m / ts["mean"].std() if ts["mean"].std() > 0 else 0
                    perf_all[hz] = (m, w, ir)

            labels = {"1w": "1周均收益", "21d": "21天均收益", "3m": "3月均收益"}
            for i, (hz, label) in enumerate(labels.items()):
                v = perf_all.get(hz)
                with kpi_cols[i]:
                    if v:
                        st.metric(label, f"{v[0]*100:+.2f}%",
                                  f"胜率{v[1]*100:.0f}% · IR{v[2]:+.2f}")
                    else:
                        st.metric(label, "—")
            with kpi_cols[3]:
                st.metric("realized PnL", f"{len(rpnl)} 条",
                          f"分析 {len(sr)} 只")

            st.divider()

            # 最新推荐
            latest = days[-1]
            date_str = f"{latest['date'][:4]}-{latest['date'][4:6]}-{latest['date'][6:8]}"
            st.markdown(f"**📋 最新推荐（{date_str}）**")
            from app.utils.sim_data import sim_day_to_df
            final_df = sim_day_to_df(latest)
            if not final_df.empty:
                show_cols = [c for c in ["rank","ticker","name","industry",
                                          "composite_score","rating","risk_level"]
                             if c in final_df.columns]
                disp = final_df[show_cols].rename(columns={
                    "rank":"#","ticker":"代码","name":"名称","industry":"行业",
                    "composite_score":"综合分","rating":"评级","risk_level":"风险"
                })
                if "综合分" in disp.columns:
                    disp["综合分"] = disp["综合分"].apply(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—")
                st.dataframe(disp, use_container_width=True, hide_index=True, height=260)

                # AI 解读最新推荐
                if api_ok and ai_analyze:
                    top_tickers = final_df["ticker"].tolist()[:5] if "ticker" in final_df.columns else []
                    commentary = call_analyze_chart(
                        f"最新推荐（{date_str}）",
                        {"tickers": top_tickers,
                         "count": len(final_df),
                         "date": date_str},
                        provider,
                    )
                    if commentary:
                        st.caption(f"🤖 {commentary}")
        else:
            st.info("暂无模拟数据，请先运行「绩效评估」或「30日模拟」。")

    # ── Tab2: 收益曲线 ──────────────────────────────────────────────────────
    with tab_perf:
        try:
            days = load_sim30d_days()
        except Exception:
            days = []

        if days:
            HZ_META = {
                "1w":  ("1周",  "#D63031"),
                "2w":  ("2周",  "#E17055"),
                "21d": ("21天", "#FDCB6E"),
                "3m":  ("3月",  "#00B894"),
            }
            selected_hz = st.multiselect(
                "选择持仓期",
                options=list(HZ_META.keys()),
                default=["21d", "3m"],
                format_func=lambda x: HZ_META[x][0],
            )

            fig = go.Figure()
            perf_data_for_ai: dict = {}
            for hz in selected_hz:
                ts = horizon_portfolio_ts(days, hz)
                if ts.empty:
                    continue
                label, color = HZ_META[hz]
                fig.add_trace(go.Scatter(
                    x=ts["date"].dt.strftime("%m-%d"),
                    y=ts["cum_return"] * 100,
                    name=label,
                    line=dict(color=color, width=2.5),
                    mode="lines+markers",
                    marker_size=4,
                    hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
                ))
                mean_r = float(ts["mean"].mean())
                perf_data_for_ai[hz] = {
                    "mean": f"{mean_r*100:+.2f}%",
                    "win":  f"{(ts['mean']>0).mean()*100:.0f}%",
                    "cum":  f"{ts['cum_return'].iloc[-1]*100:+.2f}%",
                }

            fig.add_hline(y=0, line_dash="dot", line_color="rgba(178,190,195,0.5)")
            fig.update_layout(
                height=360,
                margin=dict(t=20, b=40, l=50, r=20),
                yaxis_title="累计收益 (%)",
                legend=dict(orientation="h", y=1.05),
                hovermode="x unified",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(178,190,195,0.15)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(178,190,195,0.15)"),
            )
            st.plotly_chart(fig, use_container_width=True)

            if api_ok and perf_data_for_ai:
                commentary = call_analyze_chart("四持仓期累计收益曲线", perf_data_for_ai, provider)
                if commentary:
                    st.caption(f"🤖 **AI解读**：{commentary}")

            # 每日收益热力图
            st.markdown("**每日收益（所有持仓期）**")
            hz_list = [h for h in ["1w","2w","21d","3m"] if not horizon_portfolio_ts(days, h).empty]
            z_data, date_labels = [], []
            for hz in hz_list:
                ts = horizon_portfolio_ts(days, hz)
                z_data.append((ts["mean"] * 100).tolist())
                if not date_labels:
                    date_labels = ts["date"].dt.strftime("%m-%d").tolist()

            if z_data:
                fig_hm = go.Figure(go.Heatmap(
                    z=z_data,
                    x=date_labels,
                    y=[HZ_META[h][0] for h in hz_list],
                    colorscale="RdYlGn", zmid=0,
                    text=[[f"{v:+.1f}%" for v in row] for row in z_data],
                    texttemplate="%{text}",
                    colorbar=dict(title="收益%", thickness=12),
                ))
                fig_hm.update_layout(
                    height=200, margin=dict(t=10, b=40, l=60, r=20),
                    xaxis=dict(tickfont_size=9),
                )
                st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("暂无数据。")

    # ── Tab3: 因子分析 ──────────────────────────────────────────────────────
    with tab_ic:
        try:
            ic_data = load_ic_analysis()
            cfg     = load_strategy_config()
        except Exception:
            ic_data, cfg = {}, {}

        ic_final = ic_data.get("ic_final_picks", {})
        if ic_final:
            # IC 柱状图
            items = [
                (k, v.get("ic_3m") or 0, v.get("p_3m") or 1)
                for k, v in ic_final.items()
                if v.get("ic_3m") is not None
            ]
            items.sort(key=lambda x: x[1], reverse=True)
            factors   = [x[0] for x in items]
            ic_vals   = [x[1] for x in items]
            sig_marks = ["**" if x[2]<0.05 else ("*" if x[2]<0.10 else "") for x in items]
            colors    = ["#00B894" if v > 0 else "#D63031" for v in ic_vals]

            fig_ic = go.Figure(go.Bar(
                x=factors, y=ic_vals, marker_color=colors,
                text=[f"{v:+.4f}{s}" for v, s in zip(ic_vals, sig_marks)],
                textposition="outside",
            ))
            fig_ic.add_hline(y=0, line_dash="dot", line_color="rgba(178,190,195,0.5)")
            fig_ic.update_layout(
                title="因子 3月 IC（**=p<0.05，*=p<0.10）",
                height=300, margin=dict(t=40, b=90, l=10, r=10),
                yaxis_title="Spearman IC",
                xaxis=dict(tickangle=-35, tickfont_size=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ic, use_container_width=True)

            if api_ok:
                top3 = {f: v for f, v, _ in items[:3]}
                commentary = call_analyze_chart("因子IC分析", {"top_factors": top3}, provider)
                if commentary:
                    st.caption(f"🤖 **AI解读**：{commentary}")

            # System2 权重对比
            st.markdown("**⚖️ System2 权重（v1 vs v2校准）**")
            s2 = cfg.get("system2_updates", {})
            cur = s2.get("weights_previous", {"value":0.30,"momentum":0.25,"quality":0.25,"technical":0.20})
            new = s2.get("weights_calibrated", {"value":0.242,"momentum":0.223,"quality":0.333,"technical":0.202})
            DIM = {"value":"价值","momentum":"动量","quality":"质量","technical":"技术"}
            COLS_DIM = {"value":"#FDCB6E","momentum":"#E17055","quality":"#00B894","technical":"#74B9FF"}

            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(
                name="v1启发式",
                x=[DIM[k] for k in cur], y=[v*100 for v in cur.values()],
                marker_color="rgba(178,190,195,0.6)",
            ))
            fig_w.add_trace(go.Bar(
                name="v2 IC校准",
                x=[DIM[k] for k in new], y=[v*100 for v in new.values()],
                marker_color=[COLS_DIM[k] for k in new],
            ))
            fig_w.update_layout(
                barmode="group", height=260,
                margin=dict(t=10, b=40, l=40, r=10),
                yaxis_title="权重(%)",
                legend=dict(orientation="h", y=1.05),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("IC 数据不可用，请先运行「IC优化分析」。")

    # ── Tab4: 执行历史 ──────────────────────────────────────────────────────
    with tab_history:
        history = st.session_state.exec_history
        if not history:
            st.info("本次会话尚未执行任何命令，在左侧点击操作按钮开始。")
        else:
            # 倒序显示
            for rec in reversed(history):
                icon = "✅" if rec["success"] else "❌"
                with st.expander(
                    f"{icon} {rec['ts']}  {rec['label']}  ({rec['elapsed']:.1f}s)",
                    expanded=False,
                ):
                    if rec.get("ai_text"):
                        st.markdown(f"**🤖 AI解读：** {rec['ai_text']}")
                        st.divider()
                    st.code(
                        rec["output"][-3000:] if len(rec["output"]) > 3000 else rec["output"],
                        language="text",
                    )

            # 汇总统计
            st.divider()
            total   = len(history)
            success = sum(1 for r in history if r["success"])
            st.markdown(f"**本次会话**：共执行 {total} 次 · 成功 {success} 次 · 失败 {total-success} 次")
