"""QuantMind Streamlit 主页 — 综合看板."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.sim_data import (
    RATING_COLORS,
    get_latest_sim_day,
    horizon_portfolio_ts,
    load_ic_analysis,
    load_nav_curves,
    load_realized_pnl,
    load_sim30d_days,
    load_sim30d_stock_returns,
    load_strategy_config,
    sim_day_to_df,
)

st.set_page_config(
    page_title="QuantMind 投资系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 暗色模式 CSS 注入 ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: rgba(9,132,227,0.08);
    border: 1px solid rgba(9,132,227,0.2);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
  }
  .metric-card h1 { margin: 4px 0; font-size: 2rem; }
  .metric-card p  { margin: 0; color: #636E72; font-size: .85rem; }
  .badge-green { background:#00B894;color:white;padding:2px 8px;border-radius:4px;font-size:.8rem; }
  .badge-red   { background:#D63031;color:white;padding:2px 8px;border-radius:4px;font-size:.8rem; }
  .badge-blue  { background:#0984E3;color:white;padding:2px 8px;border-radius:4px;font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

# ── 加载数据 ─────────────────────────────────────────────────────────────────
with st.spinner("加载系统数据..."):
    days  = load_sim30d_days()
    sr    = load_sim30d_stock_returns()
    navs  = load_nav_curves()
    cfg   = load_strategy_config()
    ic    = load_ic_analysis()
    rpnl  = load_realized_pnl()

latest_day = get_latest_sim_day(days) if days else None

# ── 侧栏 ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 系统状态")
    st.markdown(f"📅 **最新模拟日**：{days[-1]['date'][:4]}-{days[-1]['date'][4:6]}-{days[-1]['date'][6:8]}" if days else "📅 未加载")
    st.markdown(f"📊 **模拟天数**：{len(days)} 天（Alpha池约1418只）")
    st.markdown(f"🤖 **主模型**：`lgbm_v6_alpha`（38特征）")
    st.markdown(f"📈 **realized_pnl**：{len(rpnl)} 条")

    ts3m = horizon_portfolio_ts(days, "3m") if days else pd.DataFrame()
    if not ts3m.empty:
        ir_3m  = ts3m["mean"].mean() / ts3m["mean"].std() if ts3m["mean"].std() > 0 else 0
        wr_3m  = (ts3m["mean"] > 0).mean()
        st.markdown(f"📊 **3月期**：IR=`{ir_3m:+.3f}` | 胜率=`{wr_3m:.1%}`")

    st.divider()
    hp = cfg.get("holding_period", {})
    st.markdown("### 📋 当前策略")
    st.success(f"**推荐持仓期**：{hp.get('recommended', '3m')}")
    s2 = cfg.get("system2_updates", {}).get("weights_calibrated", {})
    if s2:
        st.caption(f"质量:{s2.get('quality',0):.1%} | 价值:{s2.get('value',0):.1%} | 动量:{s2.get('momentum',0):.1%} | 技术:{s2.get('technical',0):.1%}")

    if st.button("🔄 刷新缓存"):
        st.cache_data.clear()
        st.success("缓存已清除")

# ── 顶部 Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0984E3 0%,#00B894 60%,#6C5CE7 100%);
            padding:28px 32px;border-radius:14px;color:white;margin-bottom:24px;
            box-shadow:0 4px 20px rgba(9,132,227,0.3)'>
  <h1 style='margin:0;font-size:2rem'>📊 QuantMind</h1>
  <p style='margin:8px 0 0 0;font-size:1.05rem;opacity:.9'>
    AI 增强量化投资系统 &nbsp;·&nbsp; Alpha池三系统选股 &nbsp;·&nbsp;
    LGBM v6 + 四维分析 + 历史回测验证
  </p>
</div>
""", unsafe_allow_html=True)

# ── 核心指标卡 ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)

ts_3m  = horizon_portfolio_ts(days, "3m")  if days else pd.DataFrame()
ts_1w  = horizon_portfolio_ts(days, "1w")  if days else pd.DataFrame()

with c1:
    v = ts_3m["mean"].mean() if not ts_3m.empty else None
    st.metric("3月期均收益", f"{v*100:+.2f}%" if v else "—", "30日模拟", delta_color="normal")
with c2:
    v = (ts_3m["mean"] > 0).mean() if not ts_3m.empty else None
    st.metric("3月期胜率", f"{v*100:.1f}%" if v else "—", "30日×10只")
with c3:
    if not ts_3m.empty and ts_3m["mean"].std() > 0:
        ir = ts_3m["mean"].mean() / ts_3m["mean"].std()
    else:
        ir = None
    st.metric("3月 IR", f"{ir:+.3f}" if ir else "—", "信息比率")
with c4:
    v = ts_1w["mean"].mean() if not ts_1w.empty else None
    st.metric("1周期均收益", f"{v*100:+.2f}%" if v else "—", "", delta_color="inverse")
with c5:
    st.metric("模拟覆盖", f"{len(days)} 交易日", "5535只全A股")
with c6:
    st.metric("realized PnL", f"{len(rpnl)} 条", "历史持仓记录")

st.divider()

# ── 最新推荐 + 行业分布 ───────────────────────────────────────────────────────
col_latest, col_ind = st.columns([3, 2])

with col_latest:
    if latest_day:
        date_str = f"{latest_day['date'][:4]}-{latest_day['date'][4:6]}-{latest_day['date'][6:8]}"
        st.markdown(f"#### 📋 最新推荐（{date_str}，Day {latest_day['day_index']}）")
        final_df = sim_day_to_df(latest_day)
        if not final_df.empty:
            show_cols = [c for c in ["rank","ticker","name","industry","composite_score","rating","risk_level"]
                         if c in final_df.columns]
            disp = final_df[show_cols].copy()
            disp = disp.rename(columns={
                "rank":"#","ticker":"代码","name":"名称","industry":"行业",
                "composite_score":"综合分","rating":"评级","risk_level":"风险"
            })
            if "综合分" in disp.columns:
                disp["综合分"] = disp["综合分"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
            st.dataframe(disp, use_container_width=True, hide_index=True, height=350)
            rets = latest_day.get("returns", {})
            r3m  = rets.get("3m", {}).get("mean")
            wr3m = rets.get("3m", {}).get("win_rate")
            if r3m is not None:
                st.caption(f"该日组合收益：1周{rets.get('1w',{}).get('mean',0)*100:+.1f}% | "
                           f"21天{rets.get('21d',{}).get('mean',0)*100:+.1f}% | "
                           f"**3月{r3m*100:+.1f}%（胜率{wr3m*100:.0f}%）**")
    else:
        st.info("未加载模拟盘数据。")

with col_ind:
    st.markdown("#### 🏭 行业归因（30日，3月收益）")
    if not sr.empty and "industry" in sr.columns:
        final_all = sr[sr["in_final"]].copy()
        if "return_3m" in final_all.columns:
            ind_agg = final_all.groupby("industry")["return_3m"].mean().sort_values(ascending=False).head(10)
            colors_bar = ["#00B894" if v > 0 else "#D63031" for v in ind_agg.values]
            fig_ind = go.Figure(go.Bar(
                x=ind_agg.index, y=ind_agg.values * 100,
                marker_color=colors_bar,
                text=[f"{v*100:+.0f}%" for v in ind_agg.values],
                textposition="outside",
            ))
            fig_ind.update_layout(
                height=360, margin=dict(t=10, b=80, l=10, r=10),
                xaxis=dict(tickangle=-40, tickfont_size=10),
                yaxis_title="3月均值(%)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ind, use_container_width=True)

st.divider()

# ── 累计收益曲线 ──────────────────────────────────────────────────────────────
st.markdown("#### 📈 四持仓期累计收益曲线（30日模拟）")

HZ_LABELS  = {"1w":"1周","2w":"2周","21d":"21天","3m":"3月"}
HZ_COLORS  = {"1w":"#D63031","2w":"#E17055","21d":"#FDCB6E","3m":"#00B894"}

fig_cum = go.Figure()
for hz in ["1w", "2w", "21d", "3m"]:
    ts = horizon_portfolio_ts(days, hz)
    if ts.empty:
        continue
    fig_cum.add_trace(go.Scatter(
        x=ts["date"].dt.strftime("%Y-%m-%d"), y=ts["cum_return"] * 100,
        name=HZ_LABELS[hz], line=dict(color=HZ_COLORS[hz], width=2.5),
        mode="lines+markers", marker_size=4,
    ))
fig_cum.add_hline(y=0, line_dash="dot", line_color="#B2BEC3")
fig_cum.update_layout(
    height=320, margin=dict(t=10, b=40, l=50, r=20),
    yaxis_title="累计收益 (%)",
    legend=dict(orientation="h", y=1.05),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_cum, use_container_width=True)

st.divider()

# ── 系统架构 + 快速导航 ───────────────────────────────────────────────────────
st.markdown("#### 🏗️ 三系统架构与页面导航")

nav_items = [
    ("📋", "今日推荐", "查看30日每日三系统选股结果、雷达图、收益热力图", "#0984E3"),
    ("🔻", "漏斗选股", "6层筛选漏斗可视化、行业归因、System1候选列表", "#6C5CE7"),
    ("📈", "单股分析", "个股四维评分、历史追踪、跨日收益散点", "#00B894"),
    ("📊", "回测表现", "NAV曲线、逐日绩效明细、realized PnL分布分析", "#E17055"),
    ("🧠", "模型管理", "LGBM特征重要性、IC热力图、System2权重校准对比", "#FDCB6E"),
    ("💬", "智能问答", "基于真实数据自动回答关于绩效、行业、因子的问题", "#636E72"),
]

cols = st.columns(3)
for i, (icon, title, desc, color) in enumerate(nav_items):
    with cols[i % 3]:
        st.markdown(f"""
<div style='border:1px solid {color}33;border-radius:10px;padding:14px;
            background:{color}0d;margin-bottom:10px'>
  <div style='font-size:1.6rem'>{icon}</div>
  <div style='font-weight:600;margin:4px 0'>{title}</div>
  <div style='font-size:.82rem;color:#636E72'>{desc}</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── System2 权重可视化 ────────────────────────────────────────────────────────
st.markdown("#### ⚖️ System2 v2 校准权重（IC 数据驱动）")

s2_weights = cfg.get("system2_updates", {}).get("weights_calibrated",
    {"value": 0.242, "momentum": 0.223, "quality": 0.333, "technical": 0.202})
DIM_LABELS = {"value": "价值(24.2%)", "momentum": "动量(22.3%)", "quality": "质量(33.3%)", "technical": "技术(20.2%)"}
DIM_COLORS_PIE = {"value": "#FDCB6E", "momentum": "#E17055", "quality": "#00B894", "technical": "#74B9FF"}

col_pie, col_ic = st.columns([1, 2])
with col_pie:
    fig_w = go.Figure(go.Pie(
        labels=[DIM_LABELS.get(k, k) for k in s2_weights],
        values=list(s2_weights.values()),
        hole=0.5,
        marker_colors=[DIM_COLORS_PIE.get(k, "#B2BEC3") for k in s2_weights],
    ))
    fig_w.update_layout(height=240, margin=dict(t=0, b=0, l=0, r=0),
                        showlegend=True, legend=dict(font_size=11))
    st.plotly_chart(fig_w, use_container_width=True)

with col_ic:
    ic_final = ic.get("ic_final_picks", {})
    ic_summary = [
        {"因子": k,
         "IC_3月": round(v.get("ic_3m", 0), 4),
         "显著性": "**" if (v.get("p_3m") or 1) < 0.05 else ("*" if (v.get("p_3m") or 1) < 0.10 else "—"),
         "p值": round(v.get("p_3m") or 1, 3)}
        for k, v in ic_final.items()
        if v.get("ic_3m") is not None
    ]
    if ic_summary:
        ic_s_df = pd.DataFrame(ic_summary).sort_values("IC_3月", ascending=False)
        colors_ic = ["#00B894" if v > 0 else "#D63031" for v in ic_s_df["IC_3月"]]
        fig_ic = go.Figure(go.Bar(
            x=ic_s_df["因子"], y=ic_s_df["IC_3月"],
            marker_color=colors_ic,
            text=[f"{v:+.4f}{s}" for v, s in zip(ic_s_df["IC_3月"], ic_s_df["显著性"])],
            textposition="outside",
        ))
        fig_ic.add_hline(y=0, line_dash="dot", line_color="#B2BEC3")
        fig_ic.update_layout(
            height=240, margin=dict(t=10, b=80, l=10, r=10),
            yaxis_title="3月 IC",
            xaxis=dict(tickangle=-30, tickfont_size=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ic, use_container_width=True)
