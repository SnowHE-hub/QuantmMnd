"""11_自选股.py — 用户自选股追踪与 6-Agent 每日分析."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantmind.watchlist.manager import WatchlistManager
from quantmind.watchlist.daily_scorer import WatchlistDailyScorer, StockScore

st.set_page_config(page_title="自选股 · QuantMind", page_icon="⭐", layout="wide")

# ── 常量 ──────────────────────────────────────────────────────────────────────
_MGMT    = WatchlistManager()
_SCORER  = WatchlistDailyScorer()
TODAY    = date.today().isoformat()

_REC_COLOR = {
    "强烈买入": "#00B894",
    "买入":     "#55EFC4",
    "持有":     "#FDCB6E",
    "观察":     "#E17055",
    "观望":     "#E17055",
    "回避":     "#D63031",
    "卖出":     "#D63031",
}

# ── 顶部 Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,#6C5CE7,#0984E3);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:20px'>
  <h2 style='margin:0'>⭐ 自选股追踪</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    手动添加任意 A 股，每日获取因子评分 + 6-Agent 辩论结论
  </p>
</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "scores" not in st.session_state:
    st.session_state["scores"] = {}      # ticker → StockScore
if "score_date" not in st.session_state:
    st.session_state["score_date"] = ""
if "expanded_ticker" not in st.session_state:
    st.session_state["expanded_ticker"] = None

# ── 区域 0：操作栏 ────────────────────────────────────────────────────────────
with st.container():
    col_add, col_ctrl = st.columns([3, 2])

    with col_add:
        st.markdown("##### ➕ 添加自选股")
        inp_col1, inp_col2, inp_col3 = st.columns([2, 2, 1])
        with inp_col1:
            new_ticker = st.text_input(
                "股票代码", placeholder="如 600519.SH", label_visibility="collapsed"
            )
        with inp_col2:
            new_note = st.text_input(
                "备注（可选）", placeholder="备注（可选）", label_visibility="collapsed"
            )
        with inp_col3:
            add_clicked = st.button("＋ 添加", use_container_width=True)

        if add_clicked and new_ticker.strip():
            with st.spinner(f"查询 {new_ticker.strip().upper()} 信息…"):
                res = _MGMT.add_stock(new_ticker.strip(), new_note.strip())
            if res["success"]:
                st.success(res["message"])
                st.session_state["scores"] = {}   # 清空缓存，下次重算
            else:
                st.warning(res["message"])

    with col_ctrl:
        st.markdown("##### 🔄 分析控制")
        ctrl1, ctrl2 = st.columns([1, 1])
        with ctrl1:
            agent_mode = st.selectbox(
                "Agent 模式", ["fast（规则）", "full（LLM）"],
                index=0, label_visibility="collapsed"
            )
            mode_str = "full" if "full" in agent_mode else "fast"
        with ctrl2:
            refresh_btn = st.button("🔄 更新分析", use_container_width=True)

        if st.session_state.get("score_date"):
            st.caption(f"上次分析：{st.session_state['score_date']}")

# ── 加载自选股列表 ────────────────────────────────────────────────────────────
watchlist = _MGMT.get_all()

if not watchlist:
    st.info("暂无自选股，请在上方输入框添加股票代码（如 600519.SH）。")
    st.stop()

tickers   = [s["ticker"] for s in watchlist]
name_map  = {s["ticker"]: s["name"]      for s in watchlist}
price_map = {s["ticker"]: s.get("add_price") for s in watchlist}

# ── 触发/恢复分析结果 ─────────────────────────────────────────────────────────
needs_score = (
    refresh_btn
    or not st.session_state["scores"]
    or st.session_state["score_date"] != TODAY
)

if needs_score:
    total = len(tickers)
    prog  = st.progress(0, text="准备分析…")
    if mode_str == "full":
        st.caption(f"⏳ full 模式每只约 3-5 分钟，共 {total} 只，预计 {total*4} 分钟。")

    scores_tmp: dict[str, StockScore] = {}
    for i, tk in enumerate(tickers):
        prog.progress((i) / total, text=f"分析 {tk}（{i+1}/{total}）…")
        try:
            result = _SCORER.score_all(
                [tk], TODAY, mode=mode_str, use_cache=(not refresh_btn),
                name_map={tk: name_map.get(tk, tk)},
            )
            scores_tmp.update(result)
        except Exception as e:
            st.warning(f"{tk} 分析失败: {e}")
    prog.progress(1.0, text="分析完成")
    st.session_state["scores"]     = scores_tmp
    st.session_state["score_date"] = TODAY

scores: dict[str, StockScore] = st.session_state["scores"]

st.divider()

# ── 区域 1：概览卡片行 ────────────────────────────────────────────────────────
st.markdown("#### 📊 持仓概览")

def _pct_fmt(v: float) -> str:
    return f"{v*100:+.2f}%"

def _color(v: float) -> str:
    return "#D63031" if v > 0 else "#00B894"   # A股：涨红跌绿

# 每行 3 只
cols_per_row = 3
for row_start in range(0, len(tickers), cols_per_row):
    row_tickers = tickers[row_start: row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, tk in zip(cols, row_tickers):
        sc = scores.get(tk)
        wl = next((w for w in watchlist if w["ticker"] == tk), {})
        with col:
            # 加入以来涨跌幅
            add_px = wl.get("add_price") or 0
            cur_px = sc.current_price if sc else 0
            since_pct = (cur_px / add_px - 1.0) if (add_px and cur_px) else 0.0

            rec   = sc.agent_recommendation if sc else "—"
            rec_c = _REC_COLOR.get(rec, "#636E72")
            comp  = sc.composite_score if sc else 50
            d1    = sc.price_change_1d if sc else 0.0
            univ_warn = "" if (sc and sc.in_alpha_universe) else " ⚠️"

            st.markdown(f"""
<div style='border:1px solid rgba(108,92,231,0.3);border-radius:10px;
            padding:14px 16px;margin-bottom:4px;background:rgba(108,92,231,0.04)'>
  <div style='font-size:1.05rem;font-weight:700'>{tk}{univ_warn}</div>
  <div style='font-size:.85rem;color:#636E72;margin-bottom:8px'>{name_map.get(tk, '')}</div>
  <div style='font-size:1.5rem;font-weight:700'>
    ¥{cur_px:.2f}
    <span style='font-size:.9rem;color:{_color(d1)};margin-left:8px'>{_pct_fmt(d1)}</span>
  </div>
  <div style='margin:6px 0 4px 0;font-size:.8rem;color:#636E72'>综合评分</div>
  <div style='background:#eee;border-radius:4px;height:6px;overflow:hidden'>
    <div style='width:{min(100,max(0,int(comp)))}%;height:100%;
                background:{"#00B894" if comp>=60 else "#FDCB6E" if comp>=40 else "#D63031"};
                border-radius:4px'></div>
  </div>
  <div style='display:flex;justify-content:space-between;font-size:.8rem;margin-top:2px'>
    <span>{comp:.0f}/100</span>
    <span style='background:{rec_c};color:white;padding:1px 7px;
                 border-radius:3px;font-size:.75rem'>{rec}</span>
  </div>
  <div style='font-size:.8rem;margin-top:6px;color:#636E72'>
    加入以来：<span style='color:{_color(since_pct)};font-weight:600'>{_pct_fmt(since_pct)}</span>
    &nbsp;·&nbsp; 加入价：¥{add_px:.2f if add_px else "—"}
  </div>
</div>
""", unsafe_allow_html=True)
            if not (sc and sc.in_alpha_universe):
                st.caption("⚠️ 不在 Alpha 宇宙（LGBM未覆盖，得分仅供参考）")
            if st.button(f"详细分析 ▼", key=f"expand_{tk}"):
                if st.session_state["expanded_ticker"] == tk:
                    st.session_state["expanded_ticker"] = None
                else:
                    st.session_state["expanded_ticker"] = tk

st.divider()

# ── 区域 2：详细分析（展开） ──────────────────────────────────────────────────
exp_tk = st.session_state.get("expanded_ticker")
if exp_tk and exp_tk in scores:
    sc = scores[exp_tk]
    st.markdown(f"#### 🔍 详细分析：{exp_tk}  {name_map.get(exp_tk, '')}")

    det_left, det_right = st.columns([1, 1])

    with det_left:
        st.markdown("##### 📡 四维因子雷达图")
        dims = ["价值", "动量", "质量", "技术"]
        vals = [sc.value_score, sc.momentum_score, sc.quality_score, sc.technical_score]
        ref  = [50, 50, 50, 50]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=ref + [ref[0]], theta=dims + [dims[0]],
            fill="toself", fillcolor="rgba(200,200,200,0.15)",
            line=dict(color="rgba(150,150,150,0.5)", width=1, dash="dot"),
            name="市场均值(50)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims + [dims[0]],
            fill="toself", fillcolor="rgba(108,92,231,0.18)",
            line=dict(color="#6C5CE7", width=2),
            name=exp_tk,
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont_size=9),
                angularaxis=dict(tickfont_size=12),
            ),
            showlegend=True, height=300,
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=20, b=30, l=30, r=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # 价格相关指标
        st.markdown("**📈 价格指标**")
        pi1, pi2 = st.columns(2)
        pi1.metric("当前价格", f"¥{sc.current_price:.2f}", f"{_pct_fmt(sc.price_change_1d)}")
        pi2.metric("5日涨跌", _pct_fmt(sc.price_change_5d))
        pi3, pi4 = st.columns(2)
        pi3.metric("MA20", f"¥{sc.ma20:.2f}" if sc.ma20 else "—")
        pi4.metric("MA60", f"¥{sc.ma60:.2f}" if sc.ma60 else "—")
        pi5, pi6 = st.columns(2)
        pi5.metric("距52周高点", _pct_fmt(sc.drawdown_from_high))
        pi6.metric("量比", f"{sc.volume_ratio:.2f}")

    with det_right:
        st.markdown("##### 🤖 6-Agent 辩论结论")
        rec_c = _REC_COLOR.get(sc.agent_recommendation, "#636E72")
        st.markdown(f"""
<div style='background:{rec_c};color:white;text-align:center;
            padding:10px;border-radius:8px;font-size:1.1rem;font-weight:700;
            margin-bottom:12px'>
  {sc.agent_recommendation}
</div>
""", unsafe_allow_html=True)

        conf_pct = min(int(sc.agent_confidence * 100), 100)
        st.markdown(f"**置信度** {conf_pct}%")
        st.progress(conf_pct)

        if sc.agent_bull_reasons:
            st.markdown("🟢 **看多理由**")
            for r in sc.agent_bull_reasons[:3]:
                st.markdown(f"- {r}")

        if sc.agent_bear_reasons:
            st.markdown("🔴 **看空理由**")
            for r in sc.agent_bear_reasons[:3]:
                st.markdown(f"- {r}")

        if sc.agent_key_debate:
            st.markdown(f"⚖️ **关键争议**：{sc.agent_key_debate}")

        if sc.agent_debate_summary:
            with st.expander("完整辩论摘要"):
                st.write(sc.agent_debate_summary)

        if sc.lgbm_percentile is not None:
            st.metric("LGBM 在宇宙中的百分位", f"{sc.lgbm_percentile:.1f}%")
        elif not sc.in_alpha_universe:
            st.caption("⚠️ 该股不在 Alpha 1418 宇宙，LGBM 百分位不可用")

        if sc.data_freshness:
            st.caption(f"📅 因子数据日期：{sc.data_freshness}")

    # 价格走势迷你图
    st.markdown("##### 📉 近60日收盘价走势")
    hist = _SCORER.get_price_history(exp_tk, n=90)
    if not hist.empty:
        wl_entry = next((w for w in watchlist if w["ticker"] == exp_tk), {})
        add_date = wl_entry.get("add_date")
        add_price = wl_entry.get("add_price")

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=hist["trade_date"], y=hist["close"],
            name="收盘价", line=dict(color="#6C5CE7", width=2),
        ))
        if sc.ma20 and len(hist) >= 20:
            ma20_series = hist["close"].rolling(20).mean()
            fig_line.add_trace(go.Scatter(
                x=hist["trade_date"], y=ma20_series,
                name="MA20", line=dict(color="#E17055", width=1, dash="dot"),
            ))
        if sc.ma60 and len(hist) >= 60:
            ma60_series = hist["close"].rolling(60).mean()
            fig_line.add_trace(go.Scatter(
                x=hist["trade_date"], y=ma60_series,
                name="MA60", line=dict(color="#0984E3", width=1, dash="dot"),
            ))
        # 加入价水平线
        if add_price:
            fig_line.add_hline(
                y=add_price, line_dash="dash", line_color="#D63031",
                annotation_text=f"加入价 ¥{add_price:.2f}",
            )
        # 加入日期竖线
        if add_date:
            fig_line.add_vline(
                x=add_date, line_dash="dash", line_color="#D63031", opacity=0.6,
            )
        fig_line.update_layout(
            height=240, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", y=1.1, font_size=11),
            xaxis=dict(tickangle=-30, tickfont_size=9),
            yaxis_title="价格",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("价格历史数据不可用。")

    st.divider()

# ── 区域 3：对比分析表格 ──────────────────────────────────────────────────────
st.markdown("#### 📋 自选股对比（综合分降序）")

if scores:
    rows = []
    for tk in tickers:
        sc = scores.get(tk)
        wl = next((w for w in watchlist if w["ticker"] == tk), {})
        add_px = wl.get("add_price") or 0
        cur_px = sc.current_price if sc else 0
        since_pct = (cur_px / add_px - 1.0) if (add_px and cur_px) else None
        rows.append({
            "代码":       tk,
            "名称":       name_map.get(tk, tk),
            "当前价":     f"¥{sc.current_price:.2f}" if sc and sc.current_price else "—",
            "1日%":      f"{sc.price_change_1d*100:+.2f}%" if sc else "—",
            "5日%":      f"{sc.price_change_5d*100:+.2f}%" if sc else "—",
            "加入以来%":  f"{since_pct*100:+.2f}%" if since_pct is not None else "—",
            "综合分":     f"{sc.composite_score:.1f}" if sc else "—",
            "估值分":     f"{sc.value_score:.1f}" if sc else "—",
            "动量分":     f"{sc.momentum_score:.1f}" if sc else "—",
            "质量分":     f"{sc.quality_score:.1f}" if sc else "—",
            "Agent推荐":  sc.agent_recommendation if sc else "—",
            "宇宙内":    "✅" if (sc and sc.in_alpha_universe) else "⚠️",
        })
    df_cmp = pd.DataFrame(rows)
    # 按综合分降序（字符串排序安全处理）
    try:
        df_cmp["_sort"] = pd.to_numeric(df_cmp["综合分"], errors="coerce")
        df_cmp = df_cmp.sort_values("_sort", ascending=False).drop("_sort", axis=1)
    except Exception:
        pass
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)
else:
    st.info("请点击「更新分析」加载分析结果。")

st.divider()

# ── 区域 4：自选股管理（添加/删除记录） ───────────────────────────────────────
st.markdown("#### 🗂️ 自选股管理")

for wl in watchlist:
    tk = wl["ticker"]
    mgmt_cols = st.columns([1.5, 1.5, 1.5, 1.5, 3, 1])
    mgmt_cols[0].caption(tk)
    mgmt_cols[1].caption(wl.get("name", "—"))
    mgmt_cols[2].caption(f"加入：{wl.get('add_date', '—')}")
    mgmt_cols[3].caption(f"¥{wl['add_price']:.2f}" if wl.get("add_price") else "—")
    mgmt_cols[4].caption(wl.get("note", "") or "—")
    if mgmt_cols[5].button("🗑️", key=f"del_{tk}", help=f"删除 {tk}"):
        _MGMT.remove_stock(tk)
        if tk in st.session_state["scores"]:
            del st.session_state["scores"][tk]
        st.rerun()

if not watchlist:
    st.info("自选股列表为空。")
