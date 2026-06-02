"""12_历史推荐.py — 历史推荐记录追踪 & P&L 归因.

数据来源
--------
- data/recommendations/*.json        推荐列表（每日一文件）
- data/feedback/realized_pnl.parquet 已结算收益
- data/paper_trading/forward_positions.json 进行中持仓
- data/raw/daily_prices_panel.parquet 实时价格（计算浮盈）
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.data_service import get_data_service
from app.utils.rec_data import (
    load_all_recommendations,
    load_forward_positions,
    load_name_map,
    get_latest_prices,
    get_prices_for_dates,
    build_flat_records,
    compute_summary_stats,
    compute_attribution,
)

_SVC = get_data_service()

st.set_page_config(
    page_title="历史推荐 · QuantMind",
    page_icon="📜",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# 缓存数据加载
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def _load_recs() -> list[dict]:
    return load_all_recommendations()

@st.cache_data(ttl=600)
def _load_pnl() -> pd.DataFrame | None:
    # 统一走 DataService（去重 realized_pnl loader）
    df = _SVC.get_realized_pnl()
    return df if not df.empty else None

@st.cache_data(ttl=600)
def _load_fwd() -> list[dict]:
    return load_forward_positions()

@st.cache_data(ttl=600)
def _load_names() -> dict[str, str]:
    return load_name_map()

@st.cache_data(ttl=300)
def _load_latest_prices(tickers_key: tuple) -> dict[str, float | None]:
    return get_latest_prices(list(tickers_key))

@st.cache_data(ttl=600)
def _load_entry_prices(pairs_key: tuple) -> dict[tuple, float | None]:
    return get_prices_for_dates(list(pairs_key))


# ─────────────────────────────────────────────────────────────────────────────
# 标题
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📜 历史推荐记录")
st.caption("每日选股结果 · P&L 追踪 · 行业归因 · 评级效果分析")

# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────
recs    = _load_recs()
pnl_df  = _load_pnl()
fwd_pos = _load_fwd()
name_map = _load_names()

if not recs:
    st.warning(
        "📂 暂无推荐记录。请先运行每日更新（`scripts/daily_update.py`）"
        "生成 `data/recommendations/*.json`。"
    )
    st.stop()

# 提取所有 ticker / 日期 准备批量价格查询
all_tickers: list[str] = []
all_pairs:   list[tuple[str, str]] = []
for rec in recs:
    as_of = str(rec.get("as_of", ""))
    for item in rec.get("top10", []):
        ticker = item.get("ticker", "")
        if ticker:
            all_tickers.append(ticker)
            all_pairs.append((ticker, as_of))

unique_tickers = tuple(sorted(set(all_tickers)))
unique_pairs   = tuple(sorted(set(all_pairs)))

with st.spinner("加载价格数据..."):
    latest_prices = _load_latest_prices(unique_tickers)
    entry_prices  = _load_entry_prices(unique_pairs)

# 构建扁平记录
flat = build_flat_records(recs, pnl_df, fwd_pos, name_map, latest_prices, entry_prices)
df_all = pd.DataFrame(flat) if flat else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# 区域 0：筛选栏
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔍 筛选条件", expanded=True):
    fc1, fc2, fc3 = st.columns(3)

    all_dates = sorted({rec.get("as_of", "") for rec in recs}, reverse=True)
    date_options = ["全部"] + all_dates

    with fc1:
        sel_dates = st.multiselect(
            "推荐日期",
            options=all_dates,
            default=all_dates[:5] if len(all_dates) >= 5 else all_dates,
            help="可多选，默认展示最近 5 天",
        )
        if not sel_dates:
            sel_dates = all_dates  # 空选 = 全选

    with fc2:
        all_ratings = sorted(
            {r.get("评级", "") for r in flat if r.get("评级")}
        )
        sel_ratings = st.multiselect(
            "推荐评级",
            options=all_ratings if all_ratings else ["（无评级数据）"],
            default=[],
            help="空选 = 不过滤",
        )

    with fc3:
        status_opts = ["全部", "持仓中", "已结算盈利 ✅", "已结算亏损 ❌", "未追踪"]
        sel_status = st.selectbox("持仓状态", options=status_opts, index=0)

# 应用筛选
df_filtered = df_all.copy() if not df_all.empty else df_all
if not df_filtered.empty:
    df_filtered = df_filtered[df_filtered["推荐日期"].isin(sel_dates)]
    if sel_ratings:
        df_filtered = df_filtered[df_filtered["评级"].isin(sel_ratings)]
    if sel_status != "全部":
        df_filtered = df_filtered[df_filtered["状态"] == sel_status]

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 区域 1：汇总 KPI 卡片
# ─────────────────────────────────────────────────────────────────────────────
stats = compute_summary_stats(flat)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📊 总推荐次数",  f"{stats['total_recs']:,}")
k2.metric("✅ 持仓中",       f"{stats['n_open']}")
k3.metric(
    "🎯 3m胜率",
    f"{stats['win_rate_3m']:.1%}" if stats["win_rate_3m"] is not None else "—",
    help=f"基于 {stats['n_settled']} 笔已结算记录",
)
k4.metric(
    "📈 3m平均收益",
    f"{stats['avg_return_3m']:+.2f}%" if stats["avg_return_3m"] is not None else "—",
)
k5.metric(
    "🏆 最佳单笔",
    f"{stats['best_trade']:+.2f}%" if stats["best_trade"] is not None else "—",
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 区域 2：推荐历史时间线
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📅 推荐历史时间线")

recs_filtered = [r for r in recs if r.get("as_of") in sel_dates]
if not recs_filtered:
    st.info("当前筛选条件下无推荐记录。")
else:
    for rec in recs_filtered:
        as_of = rec.get("as_of", "")
        top10 = rec.get("top10", [])
        mkt   = rec.get("market_summary", "")

        # 计算该日整体表现
        day_rows = [r for r in flat if r["推荐日期"] == as_of]
        settled  = [r for r in day_rows if r["3m实际收益%"] is not None]
        open_cnt = sum(1 for r in day_rows if r["状态"] == "持仓中")

        if settled:
            day_rets  = [r["3m实际收益%"] for r in settled]
            day_avg   = sum(day_rets) / len(day_rets)
            day_win   = sum(1 for x in day_rets if x > 0) / len(day_rets)
            perf_str  = f"3m均值 **{day_avg:+.2f}%** · 胜率 **{day_win:.0%}** ({len(settled)}笔已结算)"
            icon = "🟢" if day_avg >= 0 else "🔴"
        elif open_cnt > 0:
            perf_str = f"🔵 {open_cnt} 笔持仓中"
            icon = "🔵"
        else:
            perf_str = "📋 暂无追踪数据"
            icon = "⚪"

        with st.expander(
            f"{icon} **{as_of}**  ·  推荐 {len(top10)} 只  ·  {perf_str}",
            expanded=(as_of == all_dates[0]),  # 最新日期默认展开
        ):
            col_info, col_stocks = st.columns([2, 3])

            with col_info:
                st.markdown(f"**市场概况**")
                st.caption(mkt or "—")
                if open_cnt > 0:
                    st.info(f"🔵 {open_cnt} 只持仓中")
                if settled:
                    avg_s = sum(r["3m实际收益%"] for r in settled) / len(settled)
                    st.metric("3m均值收益", f"{avg_s:+.2f}%")

            with col_stocks:
                ticker_badges = []
                for item in top10[:10]:
                    ticker  = item.get("ticker", "")
                    nm      = name_map.get(ticker) or ticker
                    row_r   = next((r for r in flat if r["推荐日期"] == as_of and r["代码"] == ticker), {})
                    ret_val = row_r.get("3m实际收益%")
                    status  = row_r.get("状态", "")

                    if ret_val is not None:
                        color = "#27ae60" if ret_val >= 0 else "#e74c3c"
                        label = f"{nm} {ret_val:+.1f}%"
                    elif "持仓" in status:
                        color = "#2980b9"
                        label = f"{nm} 持仓中"
                    else:
                        color = "#95a5a6"
                        label = nm

                    ticker_badges.append(
                        f'<span style="background:{color};color:white;'
                        f'padding:2px 8px;border-radius:10px;font-size:12px;'
                        f'margin:2px;display:inline-block">{label}</span>'
                    )

                st.markdown(" ".join(ticker_badges), unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 区域 3：个股追踪表格
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📋 个股追踪明细")

if df_filtered.empty:
    st.info("当前筛选无数据。")
else:
    # 显示列
    disp_cols = [
        "推荐日期", "代码", "名称", "状态",
        "入场价", "当前价", "浮盈%",
        "3m实际收益%", "盈利?",
        "评级", "LGBM排名", "行业",
    ]
    disp_cols = [c for c in disp_cols if c in df_filtered.columns]
    df_disp = df_filtered[disp_cols].copy()

    # 格式化
    for pct_col in ("浮盈%", "3m实际收益%"):
        if pct_col in df_disp.columns:
            df_disp[pct_col] = df_disp[pct_col].apply(
                lambda x: f"{x:+.2f}%" if x is not None and not pd.isna(x) else "—"
            )
    for price_col in ("入场价", "当前价"):
        if price_col in df_disp.columns:
            df_disp[price_col] = df_disp[price_col].apply(
                lambda x: f"¥{x:.2f}" if x is not None and not pd.isna(x) else "—"
            )
    if "盈利?" in df_disp.columns:
        df_disp["盈利?"] = df_disp["盈利?"].apply(
            lambda x: "✅" if x is True else ("❌" if x is False else "—")
        )

    st.dataframe(
        df_disp.sort_values("推荐日期", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    # 下载按钮
    csv = df_filtered.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇ 下载完整数据（CSV）",
        data=csv,
        file_name=f"quantmind_recs_{date.today()}.csv",
        mime="text/csv",
        use_container_width=False,
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 区域 3.5：6-Agent 六维信号（DataService.get_agent_analysis）
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🤖 6-Agent 六维信号")
_agent_dates = _SVC.get_recommendation_dates()
if _agent_dates:
    _ad_col1, _ad_col2 = st.columns([1, 3])
    with _ad_col1:
        sel_agent_date = st.selectbox("分析日期", options=_agent_dates, index=0,
                                      key="agent_date_sel")
    all_agents = _SVC.get_all_agent_analysis(sel_agent_date)
    if not all_agents:
        st.info(f"{sel_agent_date} 暂无 6-Agent 分析（reports/investment_pipeline/）。")
    else:
        with _ad_col2:
            sel_agent_tk = st.selectbox(
                "选择股票",
                options=list(all_agents.keys()),
                format_func=lambda t: f"{t}  {name_map.get(t, '')}",
                key="agent_tk_sel",
            )
        a = all_agents.get(sel_agent_tk, {})
        if a:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("综合评级", a.get("rating", "—"))
            cs = a.get("composite_signal")
            mc2.metric("综合信号", f"{cs:+.3f}" if isinstance(cs, (int, float)) else "—")
            cf = a.get("confidence")
            mc3.metric("置信度", f"{cf:.0%}" if isinstance(cf, (int, float)) else "—")

            agents = a.get("agents", {})
            if agents:
                import plotly.graph_objects as _go
                dim_order = ["Valuation", "Momentum", "Quality", "Sentiment", "Risk"]
                dim_cn = {"Valuation": "估值", "Momentum": "动量", "Quality": "质量",
                          "Sentiment": "情绪", "Risk": "风险"}
                dims = [d for d in dim_order if d in agents]
                vals = [agents[d].get("signal") or 0 for d in dims]
                radar_col, txt_col = st.columns([1, 1])
                with radar_col:
                    fig = _go.Figure(_go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=[dim_cn[d] for d in dims] + [dim_cn[dims[0]]],
                        fill="toself", fillcolor="rgba(108,92,231,0.18)",
                        line=dict(color="#6C5CE7", width=2),
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
                        showlegend=False, height=300, margin=dict(t=30, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with txt_col:
                    for d in dims:
                        sig = agents[d].get("signal")
                        summ = agents[d].get("summary", "")
                        sig_str = f"{sig:+.2f}" if isinstance(sig, (int, float)) else "—"
                        st.markdown(f"**{dim_cn[d]}** `{sig_str}` {summ[:60]}")
            thesis = a.get("investment_thesis", "")
            if thesis:
                with st.expander("📝 投资论证全文", expanded=False):
                    st.text(thesis)
else:
    st.info("暂无推荐日期。")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# 区域 4：归因分析（折叠）
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔬 归因分析（点击展开）", expanded=False):
    attribution = compute_attribution(flat)

    ind_df = attribution.get("industry", pd.DataFrame())
    rat_df = attribution.get("rating",   pd.DataFrame())

    if ind_df.empty and rat_df.empty:
        st.info(
            "归因分析需要已结算的推荐记录。"
            "当前尚无已结算数据（持仓到期后自动更新）。"
        )
    else:
        ac1, ac2 = st.columns(2)

        with ac1:
            st.markdown("#### 🏭 行业归因（3m）")
            if not ind_df.empty:
                # 柱状图
                fig_ind = go.Figure(go.Bar(
                    y=ind_df["行业"].tolist(),
                    x=ind_df["均值"].tolist(),
                    orientation="h",
                    marker_color=[
                        "#27ae60" if v >= 0 else "#e74c3c"
                        for v in ind_df["均值"]
                    ],
                    text=[f"{v:.1%}" for v in ind_df["均值"]],
                    textposition="outside",
                ))
                fig_ind.update_layout(
                    height=max(200, len(ind_df) * 35),
                    margin=dict(t=10, b=30, l=10, r=60),
                    xaxis_title="3m均值收益",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_ind, use_container_width=True)
            else:
                st.caption("暂无行业数据")

        with ac2:
            st.markdown("#### 📊 评级归因（3m）")
            if not rat_df.empty:
                st.dataframe(rat_df, use_container_width=True, hide_index=True)
                # 快速对比柱图
                if len(rat_df) > 1:
                    fig_rat = go.Figure(go.Bar(
                        x=rat_df["评级"].tolist(),
                        y=rat_df["均值"].tolist(),
                        marker_color=[
                            "#27ae60" if v >= 0 else "#e74c3c"
                            for v in rat_df["均值"]
                        ],
                        text=[f"{v:.1%}" for v in rat_df["均值"]],
                        textposition="outside",
                    ))
                    fig_rat.update_layout(
                        height=260,
                        margin=dict(t=10, b=50, l=10, r=10),
                        yaxis_title="3m均值收益",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_rat, use_container_width=True)
            else:
                st.caption(
                    "推荐记录中暂无评级字段（较早的记录未包含 agent_recommendation）"
                )

        # 整体胜率趋势（按日期）
        if not df_all.empty:
            settled_grp = (
                df_all[df_all["3m实际收益%"].notna()]
                .groupby("推荐日期")["3m实际收益%"]
                .agg(均值="mean", 胜率=lambda x: (x > 0).mean())
                .reset_index()
                .sort_values("推荐日期")
            )
            if not settled_grp.empty:
                st.markdown("#### 📈 逐日胜率趋势")
                fig_wr = go.Figure()
                fig_wr.add_trace(go.Bar(
                    x=settled_grp["推荐日期"],
                    y=settled_grp["均值"],
                    name="3m均值收益",
                    marker_color=[
                        "#27ae60" if v >= 0 else "#e74c3c"
                        for v in settled_grp["均值"]
                    ],
                    yaxis="y",
                ))
                fig_wr.add_trace(go.Scatter(
                    x=settled_grp["推荐日期"],
                    y=settled_grp["胜率"],
                    name="胜率",
                    line=dict(color="#f39c12", width=2, dash="dot"),
                    yaxis="y2",
                    mode="lines+markers",
                ))
                fig_wr.update_layout(
                    height=280,
                    margin=dict(t=10, b=50, l=10, r=60),
                    yaxis=dict(title="3m均值收益", tickformat=".1%"),
                    yaxis2=dict(
                        title="胜率", overlaying="y", side="right",
                        tickformat=".0%", range=[0, 1.1],
                    ),
                    legend=dict(orientation="h", y=1.1),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_wr, use_container_width=True)
