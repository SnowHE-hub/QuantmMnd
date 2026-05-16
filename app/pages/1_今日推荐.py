"""今日推荐 — 概览、热力图、AI 综述与可展开个股卡片。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import charts
from app.utils.data_loader import (
    AGENT_ORDER,
    DISPLAY_NAMES,
    agent_signal_matrix,
    list_top10_dates,
    load_agent_registry,
    load_final_recommendations_md,
    load_funnel_candidates,
    load_price_history,
    load_strategies_json,
    load_top10,
    load_validations_json,
    merge_top10_with_md,
    parse_tri_classification,
)
from app.utils.llm_commentator import cached_llm_comment, stock_report_cached

st.set_page_config(page_title="今日推荐", layout="wide")

st.markdown("### 📌 今日推荐")
dates = list_top10_dates()
if not dates:
    st.warning("暂无 `data/recommendations/*/top10.json`。")
    st.stop()

pick = st.selectbox("📅 选择分析日期", dates, index=len(dates) - 1)

md = load_final_recommendations_md(pick)
tri = parse_tri_classification(md)

na, nw, nv = len(tri["accept"]), len(tri["watch"]), len(tri["avoid"])
st.markdown("### 📊 本次推荐概览")
b1, b2, b3 = st.columns(3)
with b1:
    st.metric("✅ 可接受", f"{na} 只")
with b2:
    st.metric("👀 观察", f"{nw} 只")
with b3:
    st.metric("❌ 回避", f"{nv} 只")

if md.strip():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### ✅ 可接受")
        st.dataframe(pd.DataFrame(tri["accept"]) if tri["accept"] else pd.DataFrame([{"提示": "无"}]))
    with c2:
        st.markdown("#### 👀 观察")
        st.dataframe(pd.DataFrame(tri["watch"]) if tri["watch"] else pd.DataFrame([{"提示": "无"}]))
    with c3:
        st.markdown("#### ❌ 回避")
        st.dataframe(pd.DataFrame(tri["avoid"]) if tri["avoid"] else pd.DataFrame([{"提示": "无"}]))
else:
    st.info(f"未找到 `reports/investment_pipeline/{pick}/final_recommendations.md`，三类清单不可用。")

raw = load_top10(pick)
tbl = merge_top10_with_md(raw, tri)

fc_data = load_funnel_candidates(pick)
layer_stats = (fc_data.get("funnel_stats") or fc_data) if isinstance(fc_data, dict) else {}
n_candidates = layer_stats.get("layer6_out") or layer_stats.get("layer1_in")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 推荐股票列表（Top10）")
    if tbl.empty:
        st.warning("top10.json 不可读或为空。")
    else:
        st.dataframe(tbl, use_container_width=True, hide_index=True)

strategies = load_strategies_json(pick)
tickers, composites, matrix = agent_signal_matrix(strategies)

hover_txt = []
if tickers and matrix:
    by_t = {str(r.get("ticker")): r for r in strategies}
    for t in tickers:
        one = by_t.get(t, {})
        sigs = one.get("agent_signals") or {}
        hs: list[str] = []
        for ag in AGENT_ORDER[:-1]:
            block = sigs.get(ag) or {}
            hs.append(str(block.get("summary", "") or "")[:120])
        comp = float(one.get("composite_signal", 0.0) or 0.0)
        hs.append(f"综合:{comp:+.2f} · {one.get('rating', '')}"[:120])
        hover_txt.append(hs)

with col_right:
    st.subheader("🕸️ Agent 信号热力图")
    if tickers and matrix:
        fig_h = charts.heatmap_agents(matrix, tickers, DISPLAY_NAMES, hover_text=hover_txt or None)
        st.plotly_chart(fig_h, use_container_width=True)
        summ_blob = json.dumps(
            {"tickers": tickers[:10], "composite": composites[:10]},
            ensure_ascii=False,
        )[:1200]
        land = cached_llm_comment(
            f"请用 2~3 句中文概括下列候选在六列 Agent/综合上的多空格局与集中度风险：\n{summ_blob}",
            cache_key=f"heatmap_landscape_{pick}",
            max_tokens=220,
        )
        st.caption("🤖 信号格局解读")
        st.info(land)
    else:
        st.info("无 `strategies.json` 或无法解析 Agent 信号矩阵。")

overview_payload = {
    "date": pick,
    "top10_tickers": tbl["ticker"].tolist() if not tbl.empty and "ticker" in tbl.columns else [],
    "tri_counts": {"accept": na, "watch": nw, "avoid": nv},
    "funnel_final": n_candidates,
}
st.markdown("### 🤖 AI 市场综述")
overview = cached_llm_comment(
    "基于以下结构化摘要，写 2~3 段中文「市场综述」，语气专业克制，提示风险：\n"
    + json.dumps(overview_payload, ensure_ascii=False),
    cache_key=f"overview_{pick}_{len(tickers)}",
    max_tokens=380,
)
with st.expander("展开查看完整分析", expanded=False):
    st.markdown(overview)

if tri["accept"]:
    st.markdown("### 🏆 重点推荐（可接受标的）")
    by_ticker = {str(r.get("ticker")): r for r in strategies}
    val_rows = {str(v.get("ticker")): v for v in load_validations_json(pick)}
    for stock in tri["accept"]:
        tk = str(stock.get("股票", "")).strip()
        if not tk:
            continue
        row = by_ticker.get(tk, {})
        sigs = row.get("agent_signals") or {}
        rating = stock.get("评级", row.get("rating", "—"))
        with st.expander(f"📈 {tk} — {rating}", expanded=False):
            e1, e2, e3 = st.columns(3)
            px = load_price_history(tk, days=40)
            with e1:
                st.plotly_chart(charts.plot_sparkline(px, tk, days=30), use_container_width=True)
            with e2:
                st.plotly_chart(charts.plot_agent_hbars(sigs), use_container_width=True)
            with e3:
                st.markdown("**策略参数**")
                er = row.get("entry_price_range") or []
                if len(er) >= 2:
                    st.markdown(f"- 入场：{er[0]:.2f} ~ {er[1]:.2f}")
                st.markdown(f"- 1月目标：{row.get('target_price_1m', '—')}")
                st.markdown(f"- 止损：{row.get('stop_loss_price', '—')}")
                st.markdown(f"- 仓位：{row.get('position_size', '—')}")
            val_stub = val_rows.get(tk, {"win_rate": 0.0, "avg_return": 0.0})
            report = stock_report_cached(tk, sigs, row, val_stub)
            st.markdown("**🤖 AI 分析报告**")
            st.markdown(report)

with st.expander("投资逻辑（StrategyAgent / thesis）"):
    for t in tickers[:12]:
        st.markdown(f"**{t}**")
        row = next((x for x in strategies if x.get("ticker") == t), {})
        st.write(row.get("investment_thesis") or "—")

reg = load_agent_registry()
if reg:
    with st.expander("Agent 注册表快照（registry.json）"):
        st.json({k: ("versions=%d" % len(v)) for k, v in reg.items()})
