"""1_今日推荐.py — 30日模拟盘每日选股推荐展示."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.sim_data import (
    RATING_COLORS,
    get_latest_sim_day,
    load_sim30d_days,
    sim_day_to_df,
)

st.set_page_config(page_title="今日推荐 · QuantMind", page_icon="📋", layout="wide")

# ── 页面标题 ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,#0984E3,#00B894);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:20px'>
  <h2 style='margin:0'>📋 每日选股推荐</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    全A股 5535只 → 三系统筛选 → 最终10只 &nbsp;|&nbsp; 回测窗口：2025-10-09 ~ 2025-11-19
  </p>
</div>
""", unsafe_allow_html=True)

# ── 加载数据 ─────────────────────────────────────────────────────────────────
with st.spinner("加载模拟盘数据..."):
    days = load_sim30d_days()

if not days:
    st.error("未找到 `data/sim30d/daily/` 数据，请先运行 `scripts/run_30day_sim.py`。")
    st.stop()

# ── 侧栏 ─────────────────────────────────────────────────────────────────────
date_options = [d["date"] for d in days]

def _fmt_date(s: str) -> str:
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

with st.sidebar:
    st.markdown("### 📅 选择交易日")
    sel_date = st.selectbox(
        "交易日（共30个）",
        options=date_options[::-1],
        format_func=_fmt_date,
        index=0,
    )
    st.divider()
    st.markdown("### 🎯 筛选条件")
    rating_filter = st.multiselect(
        "投资评级",
        ["强烈买入", "买入", "持有", "观望"],
        default=["强烈买入", "买入", "持有"],
    )
    risk_filter = st.multiselect(
        "风险等级", ["低", "中", "高"], default=["低", "中"]
    )

sel_day  = next(d for d in days if d["date"] == sel_date)
final_df = sim_day_to_df(sel_day)

# 过滤
if not final_df.empty:
    if "rating" in final_df.columns:
        final_df = final_df[final_df["rating"].isin(rating_filter)]
    if "risk_level" in final_df.columns:
        final_df = final_df[final_df["risk_level"].isin(risk_filter)]

# ── KPI 卡 ───────────────────────────────────────────────────────────────────
rets = sel_day.get("returns", {})

def _pct(v, default="—"):
    try:
        return f"{float(v)*100:+.2f}%" if v is not None else default
    except Exception:
        return default

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("📅 模拟日", _fmt_date(sel_date), f"Day {sel_day['day_index']}/30")
with c2:
    v = rets.get("1w", {}).get("mean")
    st.metric("1周等权收益", _pct(v), f"胜率 {rets.get('1w',{}).get('win_rate',0)*100:.0f}%")
with c3:
    v = rets.get("21d", {}).get("mean")
    st.metric("21天等权收益", _pct(v))
with c4:
    v = rets.get("3m", {}).get("mean")
    wr = rets.get("3m", {}).get("win_rate", 0)
    st.metric("3月等权收益", _pct(v), f"胜率 {wr*100:.0f}%")
with c5:
    n = len(sel_day.get("system3_final_list", []))
    st.metric("最终持仓", f"{n} 只", "全A → 三系统")

st.divider()

# ── 主内容：表格 + 饼图 ───────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown(f"#### 🏆 最终推荐名单（{len(final_df)} 只）")
    if final_df.empty:
        st.info("当前过滤条件下无推荐股票，请放宽筛选。")
    else:
        show_cols = [c for c in
            ["rank", "ticker", "name", "industry", "composite_score",
             "rating", "risk_level", "hist_win_rate", "hist_sharpe", "suggested_horizon"]
            if c in final_df.columns]
        disp = final_df[show_cols].copy()
        rename = {
            "rank": "排名", "ticker": "代码", "name": "名称", "industry": "行业",
            "composite_score": "综合分", "rating": "评级", "risk_level": "风险",
            "hist_win_rate": "历史胜率", "hist_sharpe": "历史Sharpe", "suggested_horizon": "建议持仓"
        }
        disp = disp.rename(columns={k: v for k, v in rename.items() if k in disp.columns})
        if "历史胜率" in disp.columns:
            disp["历史胜率"] = disp["历史胜率"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
        if "历史Sharpe" in disp.columns:
            disp["历史Sharpe"] = disp["历史Sharpe"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        if "综合分" in disp.columns:
            disp["综合分"] = disp["综合分"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        st.dataframe(disp, use_container_width=True, hide_index=True)

with col_right:
    st.markdown("#### 📊 行业分布")
    orig = sim_day_to_df(sel_day)
    if "industry" in orig.columns and not orig.empty:
        ind_counts = orig["industry"].value_counts()
        colors_pie = ["#0984E3","#00B894","#FDCB6E","#D63031","#6C5CE7",
                      "#E17055","#74B9FF","#55EFC4","#FD79A8","#B2BEC3"]
        fig_pie = go.Figure(go.Pie(
            labels=ind_counts.index, values=ind_counts.values,
            hole=0.42, marker_colors=colors_pie[:len(ind_counts)],
        ))
        fig_pie.update_layout(
            showlegend=True, margin=dict(t=10, b=0, l=0, r=0), height=240,
            legend=dict(font_size=11, orientation="v"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### ⭐ 评级分布")
    if "rating" in orig.columns and not orig.empty:
        rat_counts = orig["rating"].value_counts().reindex(
            ["强烈买入", "买入", "持有", "观望"], fill_value=0
        )
        fig_bar = go.Figure(go.Bar(
            x=rat_counts.index, y=rat_counts.values,
            marker_color=[RATING_COLORS.get(r, "#636E72") for r in rat_counts.index],
            text=rat_counts.values, textposition="outside",
        ))
        fig_bar.update_layout(
            margin=dict(t=10, b=10, l=10, r=10), height=200,
            yaxis=dict(title="数量", range=[0, max(rat_counts.values)+2]),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── 雷达图：个股四维评分 ─────────────────────────────────────────────────────
st.markdown("#### 🎯 个股四维评分雷达图")

s2 = sel_day.get("system2_analysis", {})
orig_df = sim_day_to_df(sel_day)
tickers_with_scores = [t for t in orig_df.get("ticker", pd.Series()).tolist() if t in s2] if not orig_df.empty else []

if tickers_with_scores and "ticker" in orig_df.columns and "name" in orig_df.columns:
    name_map = dict(zip(orig_df["ticker"], orig_df["name"]))
    sel_ticker = st.selectbox(
        "选择股票",
        options=tickers_with_scores,
        format_func=lambda t: f"{t}  {name_map.get(t, '')}",
    )
    scores = s2.get(sel_ticker, {})
    dims   = ["价值", "动量", "质量", "技术"]
    vals   = [
        scores.get("value_score", 50),
        scores.get("momentum_score", 50),
        scores.get("quality_score", 50),
        scores.get("technical_score", 50),
    ]

    col_r, col_info = st.columns([2, 1])
    with col_r:
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims + [dims[0]],
            fill="toself", fillcolor="rgba(9,132,227,0.18)",
            line=dict(color="#0984E3", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont_size=10),
                angularaxis=dict(tickfont_size=13),
            ),
            showlegend=False, height=300, margin=dict(t=20, b=10),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_info:
        st.markdown(f"**{sel_ticker}** · {name_map.get(sel_ticker, '')}")
        comp = scores.get("composite_score", 0)
        st.metric("综合评分", f"{comp:.1f} / 100")
        st.metric("投资评级", scores.get("rating", "—"))
        st.metric("建议持仓", scores.get("suggested_horizon", "—"))
        st.progress(min(int(comp), 100))
        st.caption("四维分数")
        for dim, val in zip(dims, vals):
            col_a, col_b = st.columns([2, 3])
            col_a.caption(dim)
            col_b.progress(int(val))
else:
    st.info("暂无个股四维评分数据。")

st.divider()

# ── 30日收益热力图 ───────────────────────────────────────────────────────────
st.markdown("#### 🌡️ 30日等权组合收益热力图（%）")

hm_rows = []
for d in days:
    r = d.get("returns", {})
    hm_rows.append({
        "日期": _fmt_date(d["date"]),
        "1周": (r.get("1w") or {}).get("mean"),
        "2周": (r.get("2w") or {}).get("mean"),
        "21天": (r.get("21d") or {}).get("mean"),
        "3月": (r.get("3m") or {}).get("mean"),
    })
hm_df = pd.DataFrame(hm_rows).set_index("日期")

z = hm_df.values.T * 100
text = [[f"{v:.1f}%" if v is not None else "" for v in row] for row in hm_df.values.T]

fig_hm = go.Figure(go.Heatmap(
    z=z, x=hm_df.index.tolist(), y=hm_df.columns.tolist(),
    colorscale="RdYlGn", zmid=0,
    text=text, texttemplate="%{text}",
    colorbar=dict(title="%", thickness=12),
))
fig_hm.update_layout(
    height=240, margin=dict(t=10, b=60, l=50, r=10),
    xaxis=dict(tickangle=-45, tickfont_size=9),
    yaxis=dict(tickfont_size=11),
)
st.plotly_chart(fig_hm, use_container_width=True)
