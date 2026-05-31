"""3_单股分析.py — 个股四维评分 + 历史持仓收益 + 30日追踪."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.sim_data import (
    RATING_COLORS,
    RISK_COLORS,
    load_sim30d_days,
    load_sim30d_stock_returns,
)

st.set_page_config(page_title="单股分析 · QuantMind", page_icon="📈", layout="wide")

st.markdown("""
<div style='background:linear-gradient(90deg,#00B894,#6C5CE7);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:20px'>
  <h2 style='margin:0'>📈 单股深度分析</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    30日模拟盘中每只股票的四维评分、历史表现 & 跨期收益分析
  </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("加载数据..."):
    days = load_sim30d_days()
    sr   = load_sim30d_stock_returns()

if not days or sr.empty:
    st.error("未找到模拟盘数据。")
    st.stop()

# ── 构建全股票索引 ────────────────────────────────────────────────────────────
all_stock_scores: dict[str, dict] = {}
for d in days:
    s2 = d.get("system2_analysis", {})
    s3_map = {s["ticker"]: s for s in d.get("system3_final_list", [])}
    for ticker, scores in s2.items():
        if ticker not in all_stock_scores:
            all_stock_scores[ticker] = {
                "ticker": ticker,
                "name": s3_map.get(ticker, {}).get("name", ""),
                "industry": s3_map.get(ticker, {}).get("industry", ""),
                **scores,
            }

# 候选股：在 final list 中出现过的
final_tickers = sr[sr["in_final"]]["ticker"].unique().tolist() if not sr.empty else []

with st.sidebar:
    st.markdown("### 🔍 股票选择")
    view_mode = st.radio("查看范围", ["最终推荐股票（推荐）", "全部分析股票"])
    if view_mode.startswith("最终"):
        ticker_pool = sorted(final_tickers)
    else:
        ticker_pool = sorted(all_stock_scores.keys())

    name_map = {t: all_stock_scores.get(t, {}).get("name", t) for t in ticker_pool}
    sel_ticker = st.selectbox(
        "选择股票",
        options=ticker_pool,
        format_func=lambda t: f"{t}  {name_map.get(t, '')}",
    )

if not sel_ticker:
    st.info("请在侧栏选择一只股票。")
    st.stop()

# ── 股票基本信息 ──────────────────────────────────────────────────────────────
stock_info = all_stock_scores.get(sel_ticker, {})
st.markdown(f"### {sel_ticker} — {stock_info.get('name', '')} · {stock_info.get('industry', '')}")

# 该股票在30天中的所有记录
ticker_rows = sr[sr["ticker"] == sel_ticker].copy() if not sr.empty else pd.DataFrame()
final_rows  = ticker_rows[ticker_rows["in_final"]] if not ticker_rows.empty else pd.DataFrame()

appear_days = len(ticker_rows)
final_days  = len(final_rows)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("出现在分析日", f"{appear_days}/30 天")
with c2:
    st.metric("进入最终名单", f"{final_days} 天")
with c3:
    avg_3m = final_rows["return_3m"].mean() if not final_rows.empty else None
    st.metric("3月平均收益", f"{avg_3m*100:+.1f}%" if avg_3m is not None else "—")
with c4:
    wr3m = (final_rows["return_3m"] > 0).mean() if not final_rows.empty else None
    st.metric("3月胜率", f"{wr3m*100:.0f}%" if wr3m is not None else "—")
with c5:
    avg_comp = ticker_rows["composite_score"].mean() if not ticker_rows.empty else None
    st.metric("平均综合分", f"{avg_comp:.1f}" if avg_comp is not None else "—")

st.divider()

# ── 四维雷达图（最近一次评分）+ 收益柱图 ────────────────────────────────────
col_radar, col_returns = st.columns([1, 1])

with col_radar:
    st.markdown("#### 🎯 四维评分（最新）")
    # 找最近一次在 system2 中的评分
    latest_scores = None
    for d in reversed(days):
        if sel_ticker in d.get("system2_analysis", {}):
            latest_scores = d["system2_analysis"][sel_ticker]
            break

    if latest_scores:
        dims = ["价值", "动量", "质量", "技术"]
        vals = [
            latest_scores.get("value_score", 50),
            latest_scores.get("momentum_score", 50),
            latest_scores.get("quality_score", 50),
            latest_scores.get("technical_score", 50),
        ]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims + [dims[0]],
            fill="toself", fillcolor="rgba(0,184,148,0.18)",
            line=dict(color="#00B894", width=2.5),
        ))
        # Industry benchmark (fixed at 50)
        fig_r.add_trace(go.Scatterpolar(
            r=[50, 50, 50, 50, 50], theta=dims + [dims[0]],
            name="均值基准(50)", line=dict(color="#B2BEC3", width=1, dash="dot"),
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=320, margin=dict(t=20, b=10),
            legend=dict(font_size=10, orientation="h", y=-0.05),
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption(f"评级：**{latest_scores.get('rating','—')}** | 建议持仓：**{latest_scores.get('suggested_horizon','—')}** | 综合分：**{latest_scores.get('composite_score',0):.1f}**")
    else:
        st.info("未找到该股票的评分数据。")

with col_returns:
    st.markdown("#### 📊 各持仓期实际收益（最终名单内）")
    if not final_rows.empty:
        ret_cols = ["return_1w", "return_2w", "return_21d", "return_3m"]
        ret_labels = ["1周", "2周", "21天", "3月"]
        ret_means = [final_rows[c].mean() for c in ret_cols if c in final_rows.columns]
        ret_stds  = [final_rows[c].std() for c in ret_cols if c in final_rows.columns]

        colors_ret = ["#00B894" if v > 0 else "#D63031" for v in ret_means]
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Bar(
            x=ret_labels[:len(ret_means)],
            y=[v * 100 for v in ret_means],
            marker_color=colors_ret,
            error_y=dict(type="data", array=[s * 100 for s in ret_stds], visible=True),
            text=[f"{v*100:+.1f}%" for v in ret_means],
            textposition="outside",
            name="均值收益",
        ))
        fig_ret.update_layout(
            yaxis_title="收益率 (%)",
            height=320, margin=dict(t=20, b=40, l=40, r=10),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(zeroline=True, zerolinecolor="#B2BEC3", zerolinewidth=1),
        )
        st.plotly_chart(fig_ret, use_container_width=True)
        st.caption(f"基于该股票进入最终名单的 {len(final_rows)} 次记录，误差棒为标准差")
    else:
        st.info("该股票未进入任何日期的最终推荐名单。")

st.divider()

# ── 时序：评分变化 & 收益 ─────────────────────────────────────────────────────
st.markdown("#### 📅 30日评分时序追踪")

if not ticker_rows.empty:
    score_ts = ticker_rows[["date", "composite_score", "value_score", "momentum_score",
                              "quality_score", "technical_score", "in_final"]].copy()
    score_ts["date"] = pd.to_datetime(score_ts["date"]).dt.strftime("%Y-%m-%d")

    fig_ts = go.Figure()
    for col, label, color in [
        ("composite_score", "综合分", "#0984E3"),
        ("quality_score", "质量", "#00B894"),
        ("value_score", "价值", "#FDCB6E"),
        ("momentum_score", "动量", "#E17055"),
        ("technical_score", "技术", "#6C5CE7"),
    ]:
        if col not in score_ts.columns:
            continue
        visible = True if col == "composite_score" else "legendonly"
        fig_ts.add_trace(go.Scatter(
            x=score_ts["date"], y=score_ts[col],
            name=label, line=dict(color=color, width=2 if col == "composite_score" else 1.5),
            visible=visible,
        ))

    # 标注进入 final 的日期
    final_ts = score_ts[score_ts["in_final"]]
    if not final_ts.empty:
        fig_ts.add_trace(go.Scatter(
            x=final_ts["date"], y=final_ts["composite_score"],
            mode="markers", marker=dict(symbol="star", size=12, color="#D63031"),
            name="入选最终名单",
        ))

    fig_ts.update_layout(
        height=280, margin=dict(t=10, b=40, l=40, r=10),
        yaxis=dict(title="评分", range=[0, 110]),
        xaxis=dict(tickfont_size=9),
        legend=dict(orientation="h", y=1.08, font_size=11),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    st.caption("★ 标注：该日期进入 System3 最终推荐名单")

st.divider()

# ── 各日期收益散点 ────────────────────────────────────────────────────────────
st.markdown("#### 💹 综合分 vs 3月实际收益（该股票所有记录）")

if not final_rows.empty and "composite_score" in final_rows.columns and "return_3m" in final_rows.columns:
    plot_df = final_rows[["date", "composite_score", "return_3m", "rating"]].dropna()
    if not plot_df.empty:
        plot_df["date_str"] = pd.to_datetime(plot_df["date"]).dt.strftime("%Y-%m-%d")
        plot_df["return_3m_pct"] = plot_df["return_3m"] * 100

        fig_sc = px.scatter(
            plot_df, x="composite_score", y="return_3m_pct",
            color="rating",
            color_discrete_map=RATING_COLORS,
            text="date_str",
            labels={"composite_score": "综合分", "return_3m_pct": "3月收益(%)"},
            height=300,
        )
        fig_sc.update_traces(textposition="top center", marker_size=10)
        fig_sc.add_hline(y=0, line_dash="dot", line_color="#B2BEC3")
        fig_sc.update_layout(margin=dict(t=10, b=40, l=40, r=10),
                             plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc, use_container_width=True)
else:
    st.info("该股票进入最终名单次数不足，无散点图数据。")

# ── 同行业对比 ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 🏭 同行业股票对比（30日平均3月收益）")

industry = stock_info.get("industry", "")
if industry and not sr.empty and "industry" in sr.columns:
    # 同行业对比：功能待实现（行业维度需跨日聚合）
    st.info("同行业横向对比功能正在开发中，敬请期待。")

# Simple: compare across all final picks
if not sr.empty:
    final_all = sr[sr["in_final"]].copy()
    if "return_3m" in final_all.columns:
        ticker_avg = final_all.groupby("ticker").agg(
            ret_3m=("return_3m", "mean"),
            count=("ticker", "count"),
        ).sort_values("ret_3m", ascending=False).reset_index()

        # Highlight selected ticker
        ticker_avg["is_selected"] = ticker_avg["ticker"] == sel_ticker
        ticker_avg["color"] = ticker_avg["is_selected"].map({True: "#D63031", False: "#74B9FF"})

        top20 = ticker_avg.head(20)
        sel_in_top = sel_ticker in top20["ticker"].values

        fig_comp = go.Figure(go.Bar(
            x=top20["ticker"],
            y=top20["ret_3m"] * 100,
            marker_color=top20["color"],
            text=[f"{v*100:+.1f}%" for v in top20["ret_3m"]],
            textposition="outside",
        ))
        fig_comp.update_layout(
            title=f"全部最终推荐股票 3月均值收益 Top20（🔴 = {sel_ticker}）",
            height=300, margin=dict(t=40, b=60, l=40, r=10),
            xaxis=dict(tickangle=-45, tickfont_size=9),
            yaxis_title="3月收益(%)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_comp, use_container_width=True)
