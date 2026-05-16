"""单股价格、雷达、Agent 明细与 LLM 报告。"""

from __future__ import annotations

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
    list_top10_dates,
    load_price_history_with_vol,
    load_strategies_json,
    load_strategy_detail,
    load_validations_json,
)
from app.utils.llm_commentator import cached_llm_comment, stock_report_cached

AGENT_ICONS = {
    "ValuationAgent": "💰",
    "MomentumAgent": "📈",
    "QualityAgent": "⭐",
    "SentimentAgent": "💬",
    "RiskAgent": "⚠️",
}

st.set_page_config(page_title="单股分析", layout="wide")
st.title("📈 单股分析")

dates = list_top10_dates()
pick = st.selectbox("推荐批次日期（用于策略 JSON）", dates, index=len(dates) - 1) if dates else None

strategies = load_strategies_json(pick) if pick else []
tickers = sorted({str(x.get("ticker")) for x in strategies if x.get("ticker")})

manual = st.text_input("或直接输入 ticker（如 000651.SZ）", value=tickers[0] if tickers else "")
choices = sorted(set(tickers + ([manual.strip()] if manual.strip() else [])))
t_sel = st.selectbox("选择股票", choices) if choices else None

if not t_sel:
    st.warning("无可选股票。")
    st.stop()

rec_day = pd.Timestamp(pick) if pick else None

px_df = load_price_history_with_vol(t_sel, days=66)
st.subheader("📈 价格走势（约 3 个月，含均线与成交量）")
if px_df.empty:
    st.info("未找到价格数据（尝试 `data/raw/daily_prices_panel.parquet`）。")
else:
    st.plotly_chart(charts.price_volume_ma(px_df, rec_day), use_container_width=True)
    ma20_s = px_df["close"].rolling(20, min_periods=5).mean()
    ma20_last = float(ma20_s.iloc[-1]) if len(ma20_s) else float("nan")
    ma_txt = f"{ma20_last:.2f}" if ma20_last == ma20_last else "—"
    trend_txt = cached_llm_comment(
        f"用 1~2 句中文描述股票 {t_sel} 近3个月价格与均线位置关系（基于本地收盘价，非投资建议）："
        f"最新收盘 {float(px_df['close'].iloc[-1]):.2f}，MA20≈{ma_txt}。",
        cache_key=f"px_trend_{t_sel}_{pick}",
        max_tokens=120,
    )
    st.caption(f"🤖 {trend_txt}")

detail = load_strategy_detail(pick, t_sel) if pick else {}
row = detail if detail else next((x for x in strategies if x.get("ticker") == t_sel), {})

sig = row.get("agent_signals") or {}
comp = float(row.get("composite_signal", 0.0) or 0.0)

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("🕸️ Agent 雷达（五维 + 风险反向）")
    st.plotly_chart(charts.plot_agent_radar_themed(sig, t_sel, comp), use_container_width=True)
with c2:
    st.subheader("📊 五维得分条")
    st.plotly_chart(charts.plot_agent_hbars(sig), use_container_width=True)

st.subheader("📋 投资策略参数")
z1, z2, z3, z4, z5 = st.columns(5)
er = row.get("entry_price_range") or []
with z1:
    st.metric("综合信号", f"{comp:.3f}")
with z2:
    st.metric("评级", str(row.get("rating", "—")))
with z3:
    st.metric("入场区间", f"{er[0]:.2f}~{er[1]:.2f}" if len(er) >= 2 else "—")
with z4:
    st.metric("1月 / 3月目标", f"{row.get('target_price_1m', '—')} / {row.get('target_price_3m', '—')}")
with z5:
    st.metric("止损 / 仓位", f"{row.get('stop_loss_price', '—')} / {row.get('position_size', '—')}")

val_one = next((v for v in load_validations_json(pick) if str(v.get("ticker")) == t_sel), {}) if pick else {}
full_rep = stock_report_cached(t_sel, sig, row, val_one if val_one else {"win_rate": 0.0, "avg_return": 0.0})

st.markdown("### 🤖 完整投资分析报告")
st.markdown(full_rep)

st.subheader("📉 历史回测验证（流水线 validations.json）")
val_all = load_validations_json(pick) if pick else []
if val_all:
    st.plotly_chart(charts.validation_scatter(val_all), use_container_width=True)
    vtxt = cached_llm_comment(
        f"标的 {t_sel}：胜率 {float(val_one.get('win_rate', 0) or 0):.0%}，"
        f"平均收益 {float(val_one.get('avg_return', 0) or 0):.2%}，"
        f"验证结论摘录：{str(val_one.get('final_recommendation', ''))[:400]}。\n"
        "请用 2 句中文解读该历史样本的代表性与局限。",
        cache_key=f"val_interp_{t_sel}_{pick}",
        max_tokens=180,
    )
    st.caption(f"🤖 {vtxt}")
else:
    st.info("无 validations.json。")

st.subheader("🔍 Agent 详细分析")
for ag in AGENT_ORDER[:-1]:
    signal_data = sig.get(ag) or {}
    if not isinstance(signal_data, dict):
        signal_data = {}
    s_val = float(signal_data.get("signal", 0.0) or 0.0)
    icon = AGENT_ICONS.get(ag, "🔍")
    emoji = "🟢" if s_val > 0.2 else "🔴" if s_val < -0.2 else "⚪"
    with st.expander(f"{icon} {ag}: 信号={s_val:+.2f} {emoji}"):
        u1, u2 = st.columns([1, 2])
        conf_f = float(signal_data.get("confidence", 0.0) or 0.0)
        with u1:
            st.progress(min(1.0, max(0.0, (s_val + 1) / 2)))
            st.caption(f"信号强度 {s_val:+.2f}")
            st.markdown(f"**置信度**: {conf_f:.0%}")
        with u2:
            st.markdown(f"**摘要**: {signal_data.get('summary', '—')}")
            ev = signal_data.get("evidence") or {}
            if isinstance(ev, dict):
                for k, v in list(ev.items())[:12]:
                    if isinstance(v, float):
                        st.markdown(f"- {k}: `{v:.4f}`")
                    else:
                        st.markdown(f"- {k}: `{v}`")
            elif ev:
                st.write(ev)
        summ = str(signal_data.get("summary", ""))[:500]
        agent_commentary = cached_llm_comment(
            f"用 2 句话解读 {ag} 对 {t_sel} 的结论：{summ}",
            cache_key=f"agent_{pick}_{t_sel}_{ag}",
            max_tokens=120,
        )
        st.caption(f"🤖 {agent_commentary}")

with st.expander("StrategyAgent — 投资逻辑"):
    st.write(row.get("investment_thesis") or "—")
