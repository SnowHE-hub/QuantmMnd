"""14_执行管理.py — E3 执行层完整可视化.

5 个区域：
  1. KPI 卡片（订单总数 / OPEN / CLOSED / 胜率 / 平均收益 / 平均持仓天数）
  2. 当前持仓监控（含距止损/止盈距离 + 进度条）
  3. 平仓历史（含 close_reason 分布饼图）
  4. 执行 vs 死扛对比（核心价值）
  5. 单股订单详情（K 线 + target/stop_loss 横线）
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.data_service import get_data_service

st.set_page_config(page_title="执行管理 · QuantMind", page_icon="🎯", layout="wide")

_SVC = get_data_service()

st.markdown("""
<div style='background:linear-gradient(90deg,#6c5ce7,#a29bfe);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:18px'>
  <h2 style='margin:0'>🎯 执行管理（E3 模拟执行层）</h2>
  <p style='margin:6px 0 0 0;opacity:.9'>
    Agent 建议 → 开仓 → 每日维护（高/低/回撤）→ 止损/止盈/到期 → 平仓结算
  </p>
</div>
""", unsafe_allow_html=True)

if st.button("🔄 刷新", type="secondary"):
    _SVC.clear_cache()
    st.cache_data.clear()
    st.rerun()

# 一次性拉取
all_orders = _SVC.get_simulated_orders(status="all")
stats = _SVC.get_execution_stats(days=365)

# ═════════════════════════════════════════════════════════════════════════════
# 区域 1：KPI 卡片
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 区域1 · 执行概览")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("总订单数", f"{stats.get('total_orders', 0):,}")
k2.metric("持仓中", f"{stats.get('open_orders', 0):,}")
k3.metric("已平仓", f"{stats.get('closed_orders', 0):,}")
wr = stats.get("win_rate")
k4.metric("胜率", f"{wr * 100:.1f}%" if wr is not None else "—")
ar = stats.get("avg_return")
k5.metric("平均收益", f"{ar * 100:+.2f}%" if ar is not None else "—")
ahd = stats.get("avg_holding_days")
k6.metric("平均持仓", f"{ahd:.1f}天" if ahd is not None else "—")

best = stats.get("best_trade")
worst = stats.get("worst_trade")
if best and worst:
    bw1, bw2 = st.columns(2)
    bw1.success(
        f"🏆 **最佳交易**: {best['ticker']} {best.get('name', '')} "
        f"{best['pnl_pct']*100:+.2f}% · {best['close_reason']} · "
        f"{best['open_date']} → {best['close_date']}"
    )
    bw2.error(
        f"💔 **最差交易**: {worst['ticker']} {worst.get('name', '')} "
        f"{worst['pnl_pct']*100:+.2f}% · {worst['close_reason']} · "
        f"{worst['open_date']} → {worst['close_date']}"
    )

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 2：当前持仓监控
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 📍 区域2 · 当前持仓监控")
open_df = all_orders[all_orders["status"] == "OPEN"].copy() if not all_orders.empty else pd.DataFrame()

if open_df.empty:
    st.info("当前无 OPEN 订单。")
else:
    # 当前价：用 high_price（持仓期高点）/ low_price 估算最新（DB 还没有价格快照接口）
    # 这里临时用 (high + low) / 2 模拟最近价
    open_df["current_price"] = ((open_df["high_price"].fillna(open_df["open_price"]) +
                                   open_df["low_price"].fillna(open_df["open_price"])) / 2)
    open_df["pnl_pct_live"] = (open_df["current_price"] - open_df["open_price"]) / open_df["open_price"]
    # 距止损/止盈
    open_df["dist_to_stop"] = (open_df["current_price"] - open_df["stop_loss_price"]) / open_df["current_price"]
    open_df["dist_to_target"] = (open_df["target_price"] - open_df["current_price"]) / open_df["current_price"]
    # 持仓天数 + 进度
    today = date.today()
    open_df["days_held"] = open_df["open_date"].apply(
        lambda d: (today - pd.to_datetime(d).date()).days if d is not None else 0)
    open_df["progress"] = (open_df["days_held"] / open_df["holding_period"].clip(lower=1)).clip(upper=1.0)

    show_cols = {
        "ticker": "代码", "name": "名称", "industry": "行业",
        "open_date": "开仓日", "open_price": "开仓价",
        "current_price": "当前价",
        "pnl_pct_live": "浮盈%", "dist_to_stop": "距止损",
        "dist_to_target": "距目标",
        "days_held": "持仓天数", "holding_period": "计划",
        "progress": "进度",
    }
    df_show = open_df[list(show_cols.keys())].rename(columns=show_cols)
    df_show["开仓价"] = df_show["开仓价"].map(lambda v: f"{v:.2f}")
    df_show["当前价"] = df_show["当前价"].map(lambda v: f"{v:.2f}")
    for col in ("浮盈%", "距止损", "距目标"):
        df_show[col] = df_show[col].map(
            lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—"
        )

    st.dataframe(
        df_show, use_container_width=True, hide_index=True,
        column_config={
            "进度": st.column_config.ProgressColumn(
                "进度", min_value=0.0, max_value=1.0, format="%.0f%%"),
        },
    )

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 3：平仓历史
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 📜 区域3 · 平仓历史")
closed_df = all_orders[all_orders["status"] == "CLOSED"].copy() if not all_orders.empty else pd.DataFrame()

if closed_df.empty:
    st.info("暂无 CLOSED 订单。")
else:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**最近 30 笔平仓**")
        recent = closed_df.head(30).copy()
        recent_show = recent[[
            "ticker", "name", "open_date", "close_date",
            "close_reason", "pnl_pct", "holding_days",
        ]].rename(columns={
            "ticker": "代码", "name": "名称",
            "open_date": "开仓", "close_date": "平仓",
            "close_reason": "原因", "pnl_pct": "收益%",
            "holding_days": "天数",
        })
        recent_show["收益%"] = recent_show["收益%"].map(
            lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—")
        # 颜色映射
        def _color_reason(val):
            if val == "stop_loss":
                return "background-color: #ffe0e0"
            if val == "target_hit":
                return "background-color: #d4edda"
            if val == "trailing_stop":
                return "background-color: #fff3cd"
            return ""
        st.dataframe(
            recent_show.style.applymap(_color_reason, subset=["原因"]),
            use_container_width=True, hide_index=True, height=420,
        )

    with c2:
        st.markdown("**Close Reason 分布**")
        reason_counts = closed_df["close_reason"].value_counts()
        if not reason_counts.empty:
            reason_color = {
                "stop_loss":     "#e74c3c",
                "target_hit":    "#27ae60",
                "trailing_stop": "#f39c12",
                "time_expired":  "#3498db",
                "regime_change": "#9b59b6",
            }
            fig = go.Figure(go.Pie(
                labels=reason_counts.index.tolist(),
                values=reason_counts.values.tolist(),
                marker=dict(colors=[reason_color.get(r, "#95a5a6")
                                    for r in reason_counts.index]),
                hole=0.4,
                textinfo="label+percent+value",
            ))
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20),
                              showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

            # 每个 reason 的平均收益
            st.markdown("**按 reason 平均收益**")
            avg_by = closed_df.groupby("close_reason").agg(
                n=("pnl_pct", "count"),
                avg_pnl=("pnl_pct", "mean"),
                avg_days=("holding_days", "mean"),
            ).reset_index()
            avg_by["avg_pnl"] = avg_by["avg_pnl"].map(lambda v: f"{v*100:+.2f}%")
            avg_by["avg_days"] = avg_by["avg_days"].map(lambda v: f"{v:.1f}")
            avg_by.columns = ["原因", "笔数", "平均收益", "平均天数"]
            st.dataframe(avg_by, use_container_width=True, hide_index=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 4：执行 vs 死扛对比（核心价值）
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### ⚖️ 区域4 · 执行 vs 死扛对比")
cmp = _SVC.get_execution_vs_hold_comparison()

if cmp.get("error"):
    st.info(f"对比数据不可用：{cmp['error']}")
else:
    exec_ = cmp["execute"]
    hold = cmp["hold_to_expiry"]

    # 累计净值曲线
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exec_["dates"], y=exec_["curve"],
        mode="lines+markers", name="按规则执行（止损/止盈/到期）",
        line=dict(color="#0984E3", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=hold["dates"], y=hold["curve"],
        mode="lines+markers", name="死扛到 63 天",
        line=dict(color="#95a5a6", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="累计净值曲线（按订单平仓顺序）",
        xaxis_title="平仓日期", yaxis_title="累计净值（初始 1.0）",
        height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 指标对比表
    st.markdown("**关键指标对比**")
    rows = []
    for label, src in [("按规则执行", exec_), ("死扛到 63 天", hold)]:
        rows.append({
            "策略": label,
            "样本数": src.get("n"),
            "胜率": f"{src['win_rate']*100:.1f}%" if src.get("win_rate") is not None else "—",
            "平均收益": f"{src['avg_return']*100:+.2f}%" if src.get("avg_return") is not None else "—",
            "累计收益": f"{src['total_return']*100:+.2f}%" if src.get("total_return") is not None else "—",
            "最大回撤": f"{src['max_dd']*100:.2f}%" if src.get("max_dd") is not None else "—",
            "Sharpe": f"{src['sharpe']:.2f}" if src.get("sharpe") is not None else "—",
            "平均持仓": f"{src['avg_holding_days']:.0f}天" if src.get("avg_holding_days") is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 一句话结论
    if (exec_.get("total_return") is not None and
            hold.get("total_return") is not None):
        diff = exec_["total_return"] - hold["total_return"]
        if diff > 0:
            st.success(
                f"💡 **执行规则 vs 死扛**: 累计收益差 **{diff*100:+.2f}%** "
                f"(执行 {exec_['total_return']*100:+.2f}% vs 死扛 {hold['total_return']*100:+.2f}%) — "
                f"止损/止盈规则**改善了**整体表现。"
            )
        else:
            st.warning(
                f"💡 **执行规则 vs 死扛**: 累计收益差 **{diff*100:+.2f}%** "
                f"(执行 {exec_['total_return']*100:+.2f}% vs 死扛 {hold['total_return']*100:+.2f}%) — "
                f"在此样本下，止损规则**削弱了**总收益（但通常 MaxDD 更优）。"
            )

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 5：单股订单详情
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔍 区域5 · 单股订单详情")
if all_orders.empty:
    st.info("暂无订单。")
else:
    options = all_orders.apply(
        lambda r: f"#{r['order_id']} {r['ticker']} {r.get('name', '')} "
                  f"({r['status']}, open={r['open_date']})", axis=1,
    ).tolist()
    selected = st.selectbox("选择订单", options, key="ord_select")
    selected_id = int(selected.split()[0].lstrip("#"))
    order = all_orders[all_orders["order_id"] == selected_id].iloc[0].to_dict()

    # 摘要
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("代码", order["ticker"])
    cc2.metric("状态", order["status"])
    cc3.metric("开仓价", f"{order['open_price']:.2f}")
    if pd.notna(order.get("pnl_pct")):
        cc4.metric("最终收益", f"{order['pnl_pct']*100:+.2f}%",
                   delta=order.get("close_reason"))
    else:
        cc4.metric("浮动收益", "持仓中")

    # 完整字段
    with st.expander("📋 订单完整字段", expanded=False):
        st.json({k: (str(v) if pd.notna(v) else None) for k, v in order.items()})

    # K 线（从 PG 拉日线数据）
    try:
        from sqlalchemy import text
        from app.db.postgres import get_pg_engine
        end_date = (order.get("close_date") if pd.notna(order.get("close_date"))
                    else date.today())
        with get_pg_engine().connect() as conn:
            kdf = pd.read_sql(
                text("""
                    SELECT trade_date, open, high, low, close
                    FROM daily_prices_panel
                    WHERE ts_code=:t AND trade_date >= :s AND trade_date <= :e
                    ORDER BY trade_date
                """),
                conn, params={"t": order["ticker"],
                              "s": order["open_date"], "e": end_date},
            )
        if not kdf.empty:
            kdf["trade_date"] = pd.to_datetime(kdf["trade_date"])
            fig = go.Figure(data=[go.Candlestick(
                x=kdf["trade_date"],
                open=kdf["open"], high=kdf["high"],
                low=kdf["low"], close=kdf["close"],
                name="K线",
            )])
            # target / stop_loss 横线
            if order.get("target_price"):
                fig.add_hline(y=order["target_price"], line_dash="dash",
                              line_color="#27ae60",
                              annotation_text=f"目标 {order['target_price']:.2f}")
            if order.get("stop_loss_price"):
                fig.add_hline(y=order["stop_loss_price"], line_dash="dash",
                              line_color="#e74c3c",
                              annotation_text=f"止损 {order['stop_loss_price']:.2f}")
            # 开仓 / 平仓标记
            fig.add_vline(x=pd.to_datetime(order["open_date"]),
                          line_color="#0984E3", line_dash="dot",
                          annotation_text="开仓")
            if pd.notna(order.get("close_date")):
                fig.add_vline(x=pd.to_datetime(order["close_date"]),
                              line_color="#6c5ce7", line_dash="dot",
                              annotation_text=f"平仓({order.get('close_reason','')})")
            fig.update_layout(
                height=480, xaxis_rangeslider_visible=False,
                title=f"{order['ticker']} {order.get('name', '')} K线 + 退出条件",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无该期间日线数据")
    except Exception as e:  # noqa: BLE001
        st.warning(f"K 线绘制失败: {e}")
