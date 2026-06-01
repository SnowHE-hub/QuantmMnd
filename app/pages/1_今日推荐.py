"""1_今日推荐.py — 真实每日选股推荐展示（读 data/recommendations/）.

数据源：data/recommendations/{date}.json（daily_update.py 产出的真实推荐）。
与"30日模拟"（data/sim30d/，回测验证用）区分开——本页是真实当日选股。
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.rec_data import load_all_recommendations, load_name_map

st.set_page_config(page_title="今日推荐 · QuantMind", page_icon="📋", layout="wide")

# ── 加载真实推荐 ──────────────────────────────────────────────────────────────
with st.spinner("加载真实每日推荐..."):
    recs = load_all_recommendations()      # 按日期降序
    name_map = load_name_map()

if not recs:
    st.error(
        "未找到任何真实推荐（`data/recommendations/*.json`）。\n\n"
        "请先运行 `python scripts/daily_update.py` 生成当日推荐，"
        "或在「系统控制台」点击运行每日更新。"
    )
    st.stop()

# ── 侧栏：选择推荐日期 ────────────────────────────────────────────────────────
rec_dates = [r.get("as_of", "") for r in recs]      # 已降序
today_str = date.today().isoformat()

with st.sidebar:
    st.markdown("### 📅 推荐日期")
    sel_date = st.selectbox(
        "选择交易日",
        options=rec_dates,
        index=0,
        help="默认展示最近一个交易日的真实推荐",
    )
    st.caption(f"共 {len(rec_dates)} 个推荐日")
    st.divider()
    st.markdown("### 🎯 行业过滤")

sel_rec = next((r for r in recs if r.get("as_of") == sel_date), recs[0])
top10 = sel_rec.get("top10", [])

# 补全名称（历史文件可能未回填）
for it in top10:
    if not it.get("name"):
        it["name"] = name_map.get(it.get("ticker", ""), it.get("ticker", ""))

industries = sorted({it.get("industry", "") for it in top10 if it.get("industry")})
with st.sidebar:
    sel_inds = st.multiselect("行业", options=industries, default=[])

# ── 顶部 banner（明确标注：真实推荐，非回测）──────────────────────────────────
is_today = sel_date == today_str
freshness = "✅ 今日推荐" if is_today else f"📌 最近交易日推荐（{sel_date}）"
banner_color = "#00B894" if is_today else "#0984E3"

st.markdown(f"""
<div style='background:linear-gradient(90deg,{banner_color},#6C5CE7);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:6px'>
  <h2 style='margin:0'>📋 真实每日推荐 · {sel_date}</h2>
  <p style='margin:6px 0 0 0;opacity:.9'>
    Alpha池 → LGBM 粗排 → LLM 重排 → 最终 {len(top10)} 只 &nbsp;|&nbsp; {freshness}
  </p>
</div>
""", unsafe_allow_html=True)

if not is_today:
    st.caption(
        f"ℹ️ 今天（{today_str}）暂无新推荐，展示最近交易日 {sel_date} 的结果。"
        "工作日 16:30 自动更新。"
    )

# ── 市场摘要 ──────────────────────────────────────────────────────────────────
st.info(f"📊 {sel_rec.get('market_summary', '（无摘要）')}")

# 应用行业过滤
view = [it for it in top10 if not sel_inds or it.get("industry") in sel_inds]

# ── KPI 卡 ───────────────────────────────────────────────────────────────────
def _avg(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None

avg_pe = _avg([it.get("raw_pe_ttm") for it in top10
               if isinstance(it.get("raw_pe_ttm"), (int, float)) and it["raw_pe_ttm"] > 0])
avg_roe = _avg([it.get("raw_roe") for it in top10])
avg_entry = _avg([it.get("entry_price") for it in top10])
gen_at = sel_rec.get("generated_at", "")[:16].replace("T", " ")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📋 推荐数", f"{len(top10)} 只")
c2.metric("平均 PE", f"{avg_pe:.1f}x" if avg_pe else "—")
c3.metric("平均 ROE", f"{avg_roe:.1f}%" if avg_roe is not None else "—")
c4.metric("平均入场价", f"¥{avg_entry:.2f}" if avg_entry else "—")
c5.metric("生成时间", gen_at or "—")

st.divider()

# ── 主内容：推荐表 + 行业分布 ─────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown(f"#### 🏆 推荐名单（{len(view)} 只）")
    if not view:
        st.info("当前行业过滤下无股票，请放宽筛选。")
    else:
        rows = []
        for it in view:
            rows.append({
                "排名":     it.get("rank", it.get("lgbm_rank", "")),
                "代码":     it.get("ticker", ""),
                "名称":     it.get("name", ""),
                "行业":     it.get("industry", "") or "—",
                "入场价":   f"¥{it['entry_price']:.2f}" if it.get("entry_price") else "—",
                "PE":      f"{it['raw_pe_ttm']:.1f}" if isinstance(it.get("raw_pe_ttm"), (int, float)) else "—",
                "ROE%":    f"{it['raw_roe']:.1f}" if isinstance(it.get("raw_roe"), (int, float)) else "—",
                "LGBM得分": f"{it['lgbm_score']:.3f}" if isinstance(it.get("lgbm_score"), (int, float)) else "—",
                "LLM排名": it.get("llm_rank", "—"),
            })
        df_disp = pd.DataFrame(rows).sort_values("排名")
        st.dataframe(df_disp, use_container_width=True, hide_index=True, height=400)

    # 推荐理由（折叠）
    with st.expander("💬 各股推荐理由", expanded=False):
        for it in view:
            reason = it.get("reason", "") or "（无理由）"
            st.markdown(f"**{it.get('name','')}（{it.get('ticker','')}）** — {reason}")

with col_right:
    st.markdown("#### 📊 行业分布")
    ind_counts = pd.Series(
        [it.get("industry", "") or "未知" for it in top10]
    ).value_counts()
    if not ind_counts.empty:
        colors_pie = ["#0984E3", "#00B894", "#FDCB6E", "#D63031", "#6C5CE7",
                      "#E17055", "#74B9FF", "#55EFC4", "#FD79A8", "#B2BEC3"]
        fig_pie = go.Figure(go.Pie(
            labels=ind_counts.index, values=ind_counts.values,
            hole=0.42, marker_colors=colors_pie[:len(ind_counts)],
        ))
        fig_pie.update_layout(
            showlegend=True, margin=dict(t=10, b=0, l=0, r=0), height=260,
            legend=dict(font_size=11, orientation="v"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 📈 LGBM 得分分布")
    scores = [(it.get("name", it.get("ticker", "")), it.get("lgbm_score"))
              for it in view if isinstance(it.get("lgbm_score"), (int, float))]
    if scores:
        scores.sort(key=lambda x: x[1], reverse=True)
        fig_bar = go.Figure(go.Bar(
            x=[s[1] for s in scores], y=[s[0] for s in scores],
            orientation="h", marker_color="#0984E3",
            text=[f"{s[1]:.3f}" for s in scores], textposition="outside",
        ))
        fig_bar.update_layout(
            height=max(200, len(scores) * 28),
            margin=dict(t=10, b=20, l=10, r=40),
            xaxis_title="LGBM 得分（分位）",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── 说明：与 30 日模拟的区别 ─────────────────────────────────────────────────
with st.expander("ℹ️ 这是真实推荐还是回测？", expanded=False):
    st.markdown("""
- **本页（今日推荐）**：`data/recommendations/{date}.json` — daily_update.py 每日真实选股，
  反映**当天**模型对全市场的实盘推荐。
- **「30日模拟」**：`data/sim30d/` — 历史回测窗口（2025-10~11）的**验证性**模拟，
  用于评估策略 IR/胜率，**不是**实时推荐。请在「回测表现」「持仓详情」等页查看。
- **「持仓跟踪」**：`forward_positions.json` — 真实推荐的前向持仓，待到期结算。
""")
