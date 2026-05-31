"""4_回测表现.py — 30日模拟盘 & 季度 NAV 回测绩效分析."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.sim_data import (
    horizon_portfolio_ts,
    load_nav_curves,
    load_realized_pnl,
    load_sim30d_days,
    load_sim30d_stock_returns,
    load_strategy_config,
)

st.set_page_config(page_title="回测表现 · QuantMind", page_icon="📊", layout="wide")

st.markdown("""
<div style='background:linear-gradient(90deg,#D63031,#E17055);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:20px'>
  <h2 style='margin:0'>📊 回测绩效分析</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    30日全A股模拟盘 + 季度 Paper Trading NAV | 持仓期：1w / 2w / 21d / 3m
  </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("加载数据..."):
    days   = load_sim30d_days()
    sr     = load_sim30d_stock_returns()
    navs   = load_nav_curves()
    rpnl   = load_realized_pnl()
    cfg    = load_strategy_config()

if not days:
    st.error("未找到模拟盘数据。")
    st.stop()

# ── 侧栏 ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 视图设置")
    view_hz = st.multiselect(
        "持仓期", ["1w", "2w", "21d", "3m"],
        default=["21d", "3m"], key="hz_select"
    )
    show_quarterly = st.checkbox("显示季度 NAV 曲线", value=True)
    st.divider()
    st.markdown("### 📋 当前策略参数")
    hp = cfg.get("holding_period", {})
    st.info(f"**推荐持仓期**：{hp.get('recommended', '3m')}\n\n"
            f"{hp.get('rationale', '')}")
    risk = cfg.get("system1_updates", {})
    st.caption(f"LGBM 3m IC：{risk.get('lgbm_ic_3m', '—')}")

HZ_LABELS = {"1w": "1周", "2w": "2周", "21d": "21天", "3m": "3个月"}
HZ_COLORS = {"1w": "#D63031", "2w": "#E17055", "21d": "#FDCB6E", "3m": "#00B894"}

# ── Tab 布局 ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 累计收益曲线", "📋 逐日绩效", "💰 realized PnL", "🔢 统计汇总"])

# ── Tab1: 累计收益曲线 ────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### 30日等权组合累计收益（各持仓期）")

    fig_cum = go.Figure()
    for hz in (view_hz or ["1w", "2w", "21d", "3m"]):
        ts = horizon_portfolio_ts(days, hz)
        if ts.empty:
            continue
        fig_cum.add_trace(go.Scatter(
            x=ts["date"].dt.strftime("%Y-%m-%d"),
            y=ts["cum_return"] * 100,
            name=HZ_LABELS.get(hz, hz),
            line=dict(color=HZ_COLORS.get(hz, "#0984E3"), width=2.5),
            mode="lines+markers", marker_size=5,
        ))

    fig_cum.add_hline(y=0, line_dash="dot", line_color="#B2BEC3")
    fig_cum.update_layout(
        yaxis_title="累计收益 (%)",
        height=380, margin=dict(t=20, b=40, l=50, r=20),
        legend=dict(orientation="h", y=1.05),
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    if show_quarterly and navs:
        st.markdown("#### 季度 Paper Trading NAV 曲线（9期回测）")
        fig_nav = go.Figure()
        for hz, nav_df in navs.items():
            if "nav" not in nav_df.columns:
                continue
            nav_df = nav_df.copy()
            nav_df["as_of"] = pd.to_datetime(nav_df["as_of"])
            fig_nav.add_trace(go.Scatter(
                x=nav_df["as_of"], y=nav_df["nav"],
                name=HZ_LABELS.get(hz, hz),
                line=dict(color=HZ_COLORS.get(hz, "#0984E3"), width=2),
            ))
        fig_nav.add_hline(y=1.0, line_dash="dash", line_color="#636E72",
                          annotation_text="初始净值=1.0")
        fig_nav.update_layout(
            yaxis_title="净值", height=320, margin=dict(t=20, b=40, l=50, r=20),
            legend=dict(orientation="h", y=1.05),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_nav, use_container_width=True)

# ── Tab2: 逐日绩效 ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### 逐日组合绩效明细")

    rows = []
    for d in days:
        r = d.get("returns", {})
        date_str = f"{d['date'][:4]}-{d['date'][4:6]}-{d['date'][6:8]}"
        row = {"日期": date_str, "Day": d["day_index"]}
        for hz in ["1w", "2w", "21d", "3m"]:
            rd = r.get(hz, {})
            row[f"{HZ_LABELS[hz]}均值"] = rd.get("mean")
            row[f"{HZ_LABELS[hz]}胜率"] = rd.get("win_rate")
        rows.append(row)

    daily_df = pd.DataFrame(rows)

    # 格式化
    disp_df = daily_df.copy()
    for col in disp_df.columns:
        if "均值" in col:
            disp_df[col] = disp_df[col].apply(
                lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—"
            )
        elif "胜率" in col:
            disp_df[col] = disp_df[col].apply(
                lambda x: f"{x*100:.0f}%" if pd.notna(x) else "—"
            )

    st.dataframe(disp_df, use_container_width=True, hide_index=True, height=520)

    st.divider()
    st.markdown("#### 期均收益 vs 期均胜率")
    hz_summary = []
    for hz in ["1w", "2w", "21d", "3m"]:
        ts = horizon_portfolio_ts(days, hz)
        if ts.empty:
            continue
        hz_summary.append({
            "持仓期": HZ_LABELS[hz],
            "均值收益": ts["mean"].mean() * 100,
            "胜率": ts["win_rate"].mean() * 100,
            "IR": ts["mean"].mean() / ts["mean"].std() if ts["mean"].std() > 0 else 0,
            "最大单期": ts["mean"].max() * 100,
            "最小单期": ts["mean"].min() * 100,
        })
    if hz_summary:
        hz_df = pd.DataFrame(hz_summary)
        fig_hz = make_subplots(specs=[[{"secondary_y": True}]])
        colors_hz = [HZ_COLORS.get({"1周":"1w","2周":"2w","21天":"21d","3个月":"3m"}.get(r,""),
                                     "#0984E3") for r in hz_df["持仓期"]]
        fig_hz.add_trace(go.Bar(
            x=hz_df["持仓期"], y=hz_df["均值收益"],
            name="均值收益(%)", marker_color=colors_hz,
            text=[f"{v:+.2f}%" for v in hz_df["均值收益"]], textposition="outside",
        ))
        fig_hz.add_trace(go.Scatter(
            x=hz_df["持仓期"], y=hz_df["胜率"],
            name="胜率(%)", line=dict(color="#6C5CE7", width=2),
            mode="lines+markers", marker_size=8,
        ), secondary_y=True)
        fig_hz.update_layout(height=300, margin=dict(t=20, b=40, l=40, r=40),
                              plot_bgcolor="rgba(0,0,0,0)")
        fig_hz.update_yaxes(title_text="均值收益(%)")
        fig_hz.update_yaxes(title_text="胜率(%)", secondary_y=True, range=[0, 120])
        st.plotly_chart(fig_hz, use_container_width=True)

# ── Tab3: realized PnL ────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"#### 💰 Realized PnL 分析（{len(rpnl):,}条实际持仓记录）")

    if rpnl.empty:
        st.info("暂无 realized_pnl 数据。")
    else:
        ret_col = "actual_return_63d"
        valid = rpnl[rpnl[ret_col].notna()].copy() if ret_col in rpnl.columns else pd.DataFrame()

        if not valid.empty:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("总记录数", f"{len(rpnl):,} 条")
            with col_m2:
                mean_ret = valid[ret_col].mean()
                st.metric("平均3月收益", f"{mean_ret*100:+.2f}%")
            with col_m3:
                win_rate = (valid[ret_col] > 0).mean()
                st.metric("胜率", f"{win_rate*100:.1f}%")
            with col_m4:
                # Distinguish 30d sim vs quarterly
                n_sim = valid[valid[ret_col] > 0.05].shape[0]
                st.metric("高盈利记录(>5%)", f"{n_sim:,} 条")

            st.divider()
            col_hist, col_ts = st.columns([2, 1])
            with col_hist:
                st.markdown("**收益率分布直方图**")
                fig_hist = px.histogram(
                    valid, x=ret_col, nbins=40,
                    color_discrete_sequence=["#0984E3"],
                    labels={ret_col: "3月收益率"},
                    height=300,
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="#D63031",
                                   annotation_text="0%")
                fig_hist.add_vline(x=mean_ret, line_dash="dot", line_color="#00B894",
                                   annotation_text=f"均值{mean_ret*100:+.1f}%")
                fig_hist.update_layout(margin=dict(t=10, b=40, l=40, r=10),
                                       plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_ts:
                st.markdown("**盈亏统计**")
                profit_df = pd.DataFrame({
                    "类别": ["盈利", "亏损"],
                    "数量": [(valid[ret_col] > 0).sum(), (valid[ret_col] <= 0).sum()],
                    "平均收益": [
                        valid[valid[ret_col] > 0][ret_col].mean(),
                        valid[valid[ret_col] <= 0][ret_col].mean(),
                    ]
                })
                profit_df["平均收益"] = profit_df["平均收益"].apply(
                    lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—"
                )
                st.dataframe(profit_df, use_container_width=True, hide_index=True)

                st.metric("最大单笔盈利", f"{valid[ret_col].max()*100:+.1f}%")
                st.metric("最大单笔亏损", f"{valid[ret_col].min()*100:+.1f}%")
                st.metric("标准差", f"{valid[ret_col].std()*100:.2f}%")

            # 时序散点
            st.divider()
            st.markdown("**实际收益时序散点**")
            ts_valid = valid.copy()
            ts_valid["entry_date"] = pd.to_datetime(ts_valid["entry_date"], errors="coerce")
            ts_valid = ts_valid.dropna(subset=["entry_date"])
            if not ts_valid.empty:
                ts_valid["color"] = ts_valid[ret_col].apply(
                    lambda x: "#00B894" if x > 0 else "#D63031"
                )
                fig_sc = go.Figure(go.Scatter(
                    x=ts_valid["entry_date"],
                    y=ts_valid[ret_col] * 100,
                    mode="markers",
                    marker=dict(
                        color=ts_valid["color"].tolist(),
                        size=6, opacity=0.7,
                    ),
                    text=ts_valid["ticker"] if "ticker" in ts_valid.columns else None,
                    hovertemplate="<b>%{text}</b><br>收益: %{y:.2f}%<extra></extra>",
                ))
                fig_sc.add_hline(y=0, line_dash="dash", line_color="#636E72")
                fig_sc.update_layout(
                    height=280, margin=dict(t=10, b=40, l=40, r=10),
                    yaxis_title="3月收益(%)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab4: 统计汇总 ────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### 📊 各持仓期汇总统计")

    summary_rows = []
    for hz in ["1w", "2w", "21d", "3m"]:
        ts = horizon_portfolio_ts(days, hz)
        if ts.empty:
            continue
        m = ts["mean"]
        pnl_col = "actual_return_63d"
        summary_rows.append({
            "持仓期": HZ_LABELS[hz],
            "期数": len(m),
            "均值收益": f"{m.mean()*100:+.2f}%",
            "中位数": f"{m.median()*100:+.2f}%",
            "胜率": f"{(m>0).mean()*100:.1f}%",
            "最大单期": f"{m.max()*100:+.2f}%",
            "最小单期": f"{m.min()*100:+.2f}%",
            "标准差": f"{m.std()*100:.2f}%",
            "IR": f"{m.mean()/m.std():+.3f}" if m.std() > 0 else "—",
        })

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 🏆 最佳/最差单日表现（3月持仓期）")
    ts_3m = horizon_portfolio_ts(days, "3m")
    if not ts_3m.empty:
        best = ts_3m.nlargest(3, "mean")
        worst = ts_3m.nsmallest(3, "mean")
        col_b, col_w = st.columns(2)
        with col_b:
            st.markdown("**最佳3日**")
            for _, r in best.iterrows():
                st.metric(r["date"].strftime("%Y-%m-%d"),
                          f"{r['mean']*100:+.2f}%",
                          f"胜率{r['win_rate']*100:.0f}%")
        with col_w:
            st.markdown("**最差3日**")
            for _, r in worst.iterrows():
                st.metric(r["date"].strftime("%Y-%m-%d"),
                          f"{r['mean']*100:+.2f}%",
                          f"胜率{r['win_rate']*100:.0f}%")

    if navs:
        st.divider()
        st.markdown("#### 📋 季度 Paper Trading NAV 明细（9期×4持仓期）")
        nav_summary = []
        for hz, nav_df in navs.items():
            if "nav" in nav_df.columns:
                nav_df = nav_df.copy()
                nav_df["as_of"] = pd.to_datetime(nav_df["as_of"])
                final_nav = nav_df["nav"].iloc[-1]
                max_nav   = nav_df["nav"].max()
                mdd       = (nav_df["nav"].min() / max_nav - 1)
                nav_summary.append({
                    "持仓期": HZ_LABELS.get(hz, hz),
                    "最终NAV": f"{final_nav:.4f}",
                    "累计收益": f"{(final_nav-1)*100:+.2f}%",
                    "最大回撤": f"{mdd*100:.2f}%",
                    "期数": len(nav_df),
                })
        if nav_summary:
            st.dataframe(pd.DataFrame(nav_summary), use_container_width=True, hide_index=True)
