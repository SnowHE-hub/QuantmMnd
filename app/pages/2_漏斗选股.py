"""2_漏斗选股.py — 三系统六层漏斗可视化 & 行业归因."""
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
    load_sim30d_days,
    load_sim30d_stock_returns,
    sim_day_to_df,
)

st.set_page_config(page_title="漏斗选股 · QuantMind", page_icon="🔻", layout="wide")

st.markdown("""
<div style='background:linear-gradient(90deg,#6C5CE7,#0984E3);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:20px'>
  <h2 style='margin:0'>🔻 三系统选股漏斗</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    全A股 5535只 → System1(6层过滤→15只) → System2(四维分析) → System3(历史验证→10只)
  </p>
</div>
""", unsafe_allow_html=True)

with st.spinner("加载数据..."):
    days = load_sim30d_days()
    sr   = load_sim30d_stock_returns()

if not days:
    st.error("未找到模拟盘数据。")
    st.stop()

# ── 侧栏：日期选择 ────────────────────────────────────────────────────────────
def _fmt_date(s: str) -> str:
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

with st.sidebar:
    st.markdown("### 📅 选择交易日")
    date_opts = [d["date"] for d in days]
    sel_date  = st.selectbox("交易日", date_opts[::-1], format_func=_fmt_date, index=0)
    st.divider()
    st.markdown("""
    ### 📖 系统说明
    **System 1 — 筛选系统（6层漏斗）**
    - Layer 1：排除 ST / 新股(<180天) / 市值<30亿
    - Layer 2：换手率 > 30th 百分位
    - Layer 3：周度反转>-15%, 月度动量>10th百分位
    - Layer 4：PE(0~150), PB>0, ROE>0
    - Layer 5：LGBM v6 打分 → Top 50
    - Layer 6：行业分散（每行业≤3只）→ Top 15

    **System 2 — 分析系统**
    价值(24.2%) + 动量(22.3%) + 质量(33.3%) + 技术(20.2%)

    **System 3 — 回测验证**
    60日历史胜率 + Sharpe → 最终10只
    """)

sel_day  = next(d for d in days if d["date"] == sel_date)
final_df = sim_day_to_df(sel_day)

# ── 漏斗图 ───────────────────────────────────────────────────────────────────
st.markdown(f"#### 📊 {_fmt_date(sel_date)} 三系统筛选漏斗")

n_all     = 5535
n_s1_l1   = int(n_all * 0.72)   # 约排除ST/新股/小市值
n_s1_l2   = int(n_s1_l1 * 0.7)
n_s1_l3   = int(n_s1_l2 * 0.8)
n_s1_l4   = int(n_s1_l3 * 0.65)
n_s1_l5   = 50
n_s1_l6   = 15
n_s3_out  = len(sel_day.get("system3_final_list", []))

labels = [
    "全A股（5535只）",
    "Layer1：质量过滤",
    "Layer2：流动性",
    "Layer3：趋势",
    "Layer4：基本面",
    "Layer5：LGBM Top50",
    "Layer6：行业分散（15只）",
    "System3：回测验证",
]
values = [n_all, n_s1_l1, n_s1_l2, n_s1_l3, n_s1_l4, n_s1_l5, n_s1_l6, n_s3_out]
colors = ["#0984E3","#74B9FF","#00B894","#55EFC4","#FDCB6E","#E17055","#D63031","#6C5CE7"]

col_funnel, col_stats = st.columns([3, 2])

with col_funnel:
    fig_funnel = go.Figure(go.Funnel(
        y=labels, x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=colors),
        connector=dict(line=dict(width=2, color="rgba(0,0,0,0.1)")),
    ))
    fig_funnel.update_layout(
        height=420, margin=dict(t=10, b=10, l=10, r=10),
        font=dict(size=12),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

with col_stats:
    st.markdown("**筛选统计**")
    stats_data = [
        ("全市场", n_all, "—"),
        ("Layer1 质量过滤", n_s1_l1, f"{n_s1_l1/n_all:.1%}"),
        ("Layer2 流动性", n_s1_l2, f"{n_s1_l2/n_all:.1%}"),
        ("Layer3 趋势", n_s1_l3, f"{n_s1_l3/n_all:.1%}"),
        ("Layer4 基本面", n_s1_l4, f"{n_s1_l4/n_all:.1%}"),
        ("Layer5 LGBM", n_s1_l5, f"{n_s1_l5/n_all:.1%}"),
        ("Layer6 行业分散", n_s1_l6, f"{n_s1_l6/n_all:.1%}"),
        ("System3 最终", n_s3_out, f"{n_s3_out/n_all:.2%}"),
    ]
    for layer, count, pct in stats_data:
        c1, c2, c3 = st.columns([3, 1.5, 1.5])
        c1.caption(layer)
        c2.markdown(f"**{count:,}**")
        c3.caption(pct)

    st.divider()
    # System1 候选数
    n_s1 = len(sel_day.get("system1_candidates", []))
    st.metric("System1 候选输出", f"{n_s1} 只", "进入 System2 分析")
    st.metric("System3 最终推荐", f"{n_s3_out} 只", "供模拟持仓")
    pass_rate = n_s3_out / n_all * 100
    st.caption(f"全市场通过率：{pass_rate:.3f}%")

st.divider()

# ── System1 候选列表 ──────────────────────────────────────────────────────────
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("#### 🔍 System1 候选（Top15）")
    s1_list = sel_day.get("system1_candidates", [])
    if s1_list:
        s1_df = pd.DataFrame(s1_list[:15])
        if "lgbm_score" in s1_df.columns:
            s1_df["lgbm_score"] = s1_df["lgbm_score"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
        rename_s1 = {"rank": "排名", "ticker": "代码", "name": "名称",
                     "industry": "行业", "lgbm_score": "LGBM分"}
        s1_df = s1_df.rename(columns={k: v for k, v in rename_s1.items() if k in s1_df.columns})
        cols_show = [v for k, v in rename_s1.items() if v in s1_df.columns]
        st.dataframe(s1_df[cols_show], use_container_width=True, hide_index=True)
    else:
        st.info("该日期无 System1 候选数据。")

with col_s2:
    st.markdown("#### 🏆 System3 最终推荐（10只）")
    if not final_df.empty:
        show = [c for c in ["rank","ticker","name","industry","composite_score","rating","risk_level"]
                if c in final_df.columns]
        disp = final_df[show].copy()
        disp = disp.rename(columns={
            "rank":"排名","ticker":"代码","name":"名称","industry":"行业",
            "composite_score":"综合分","rating":"评级","risk_level":"风险"
        })
        if "综合分" in disp.columns:
            disp["综合分"] = disp["综合分"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("该日期无 System3 最终推荐。")

st.divider()

# ── 30日行业归因 ──────────────────────────────────────────────────────────────
st.markdown("#### 🏭 30日行业归因分析（3月收益）")

final_rows = sr[sr["in_final"]].copy() if not sr.empty else pd.DataFrame()
if not final_rows.empty and "industry" in final_rows.columns:
    ind_agg = final_rows.groupby("industry").agg(
        count=("ticker", "count"),
        ret_3m=("return_3m", "mean"),
        ret_21d=("return_21d", "mean"),
        win_rate_3m=("return_3m", lambda x: (x > 0).mean()),
    ).dropna(subset=["ret_3m"]).sort_values("ret_3m", ascending=False)

    col_ia, col_ib = st.columns([3, 2])

    with col_ia:
        top_n = min(15, len(ind_agg))
        top_ind = ind_agg.head(top_n)
        colors_bar = ["#00B894" if v > 0 else "#D63031" for v in top_ind["ret_3m"]]
        fig_ind = go.Figure(go.Bar(
            x=top_ind.index,
            y=top_ind["ret_3m"] * 100,
            marker_color=colors_bar,
            text=[f"{v*100:.1f}%" for v in top_ind["ret_3m"]],
            textposition="outside",
        ))
        fig_ind.update_layout(
            title="行业 3月平均收益 Top15",
            yaxis_title="3月收益 (%)",
            height=380, margin=dict(t=40, b=80, l=40, r=10),
            xaxis=dict(tickangle=-45, tickfont_size=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ind, use_container_width=True)

    with col_ib:
        st.markdown("**行业收益明细**")
        disp_ind = ind_agg.copy()
        disp_ind["ret_3m"] = disp_ind["ret_3m"].apply(lambda x: f"{x*100:+.1f}%")
        disp_ind["ret_21d"] = disp_ind["ret_21d"].apply(lambda x: f"{x*100:+.1f}%")
        disp_ind["win_rate_3m"] = disp_ind["win_rate_3m"].apply(lambda x: f"{x*100:.0f}%")
        disp_ind.columns = ["持仓数", "3月均值", "21天均值", "3月胜率"]
        st.dataframe(disp_ind, use_container_width=True, height=380)
else:
    st.info("行业归因数据不可用。")

st.divider()

# ── 30日筛选结果时序 ──────────────────────────────────────────────────────────
st.markdown("#### 📅 30日每日 System2 综合分分布")

daily_scores = []
for d in days:
    s2 = d.get("system2_analysis", {})
    for ticker, scores in s2.items():
        daily_scores.append({
            "date": _fmt_date(d["date"]),
            "ticker": ticker,
            "composite_score": scores.get("composite_score"),
        })

if daily_scores:
    ds_df = pd.DataFrame(daily_scores)
    date_avg = ds_df.groupby("date")["composite_score"].agg(["mean","max","min"]).reset_index()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=date_avg["date"], y=date_avg["mean"],
        name="平均分", line=dict(color="#0984E3", width=2),
    ))
    fig_ts.add_trace(go.Scatter(
        x=date_avg["date"], y=date_avg["max"],
        name="最高分", line=dict(color="#00B894", width=1, dash="dot"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=date_avg["date"], y=date_avg["min"],
        name="最低分", line=dict(color="#D63031", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(9,132,227,0.05)",
    ))
    fig_ts.update_layout(
        height=260, margin=dict(t=10, b=40, l=40, r=10),
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(tickangle=-45, tickfont_size=9),
        yaxis_title="综合分",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ts, use_container_width=True)
