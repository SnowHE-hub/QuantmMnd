"""9_持仓跟踪.py — 模拟盘持仓实时跟踪与 P&L 监控."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── 页面配置 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="持仓跟踪 · QuantMind",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<div style='background:linear-gradient(90deg,#00B894,#55EFC4);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:20px'>
  <h2 style='margin:0'>📊 模拟盘持仓跟踪</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    实时监控开仓持仓的浮动盈亏、持仓天数与 Regime 状态
  </p>
</div>
""", unsafe_allow_html=True)

# ─── 常量 ─────────────────────────────────────────────────────────────────────
POSITIONS_JSON  = ROOT / "data" / "paper_trading" / "forward_positions.json"
PRICES_PARQUET  = ROOT / "data" / "raw" / "daily_prices_panel.parquet"
TODAY           = date.today()
REGIME_COLOR    = {"bull": "#00B894", "neutral": "#FDCB6E", "bear": "#D63031"}

# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def _load_positions() -> dict[str, Any]:
    """加载 forward_positions.json，缺少文件时返回空结构。"""
    if not POSITIONS_JSON.exists():
        return {"positions": [], "n_open_positions": 0, "generated_at": None,
                "tracking_horizon": "3m", "next_review": None}
    with POSITIONS_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_current_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """从本地 parquet 取各 ticker 最新收盘价（最多追溯 60 交易日）。"""
    if not PRICES_PARQUET.exists() or not tickers:
        return {}
    try:
        import pyarrow.parquet as pq
        schema_names = pq.ParquetFile(str(PRICES_PARQUET)).schema_arrow.names
        price_col    = "adj_close" if "adj_close" in schema_names else "close"
        date_col     = "trade_date" if "trade_date" in schema_names else "date"

        df = pd.read_parquet(
            PRICES_PARQUET,
            filters=[("ts_code", "in", list(tickers))],
            columns=[date_col, "ts_code", price_col],
        )
        df = df.rename(columns={date_col: "trade_date", price_col: "close"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        result = (
            df.sort_values("trade_date")
              .groupby("ts_code")["close"]
              .last()
              .to_dict()
        )
        return result
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_entry_prices(tickers: tuple[str, ...], as_of_str: str) -> dict[str, float]:
    """取建仓日期附近的价格作为参考入场价。"""
    if not PRICES_PARQUET.exists() or not tickers:
        return {}
    try:
        import pyarrow.parquet as pq
        schema_names = pq.ParquetFile(str(PRICES_PARQUET)).schema_arrow.names
        price_col    = "adj_close" if "adj_close" in schema_names else "close"
        date_col     = "trade_date" if "trade_date" in schema_names else "date"

        df = pd.read_parquet(
            PRICES_PARQUET,
            filters=[("ts_code", "in", list(tickers))],
            columns=[date_col, "ts_code", price_col],
        )
        df = df.rename(columns={date_col: "trade_date", price_col: "close"})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        as_of = pd.Timestamp(as_of_str)
        # 取 as_of 日或其之后最近的价格作为"入场价"
        near = df[df["trade_date"] >= as_of]
        if near.empty:
            return {}
        result = (
            near.sort_values("trade_date")
                .groupby("ts_code")["close"]
                .first()
                .to_dict()
        )
        return result
    except Exception:
        return {}


def _get_regime() -> str:
    """尝试从 HMM 获取当前 Regime，失败时读 strategy_config_v2.json 兜底。"""
    try:
        from quantmind.regime import RegimeHMM, build_observations
        hmm = RegimeHMM()
        obs_path = ROOT / "data" / "raw" / "index_daily_panel.parquet"
        if obs_path.exists():
            obs = build_observations(str(obs_path))
            if obs is not None and len(obs) > 0:
                regime = hmm.predict_regime(obs, pd.Timestamp(TODAY))
                return str(regime)
    except Exception:
        pass
    try:
        cfg_path = ROOT / "data" / "paper_trading" / "strategy_config_v2.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        # strategy_config_v2 不含 current_regime；从 ensemble_regime_weights 推断
        if "ensemble_regime_weights" in cfg:
            return "neutral"
    except Exception:
        pass
    return "neutral"


def _holding_days(as_of_str: str) -> int:
    try:
        entry = date.fromisoformat(as_of_str)
        return (TODAY - entry).days
    except Exception:
        return 0


# ─── 主逻辑 ───────────────────────────────────────────────────────────────────
raw = _load_positions()
positions_raw: list[dict] = raw.get("positions", [])
open_pos = [p for p in positions_raw if p.get("status") == "OPEN"]

if not open_pos:
    st.warning("当前无开仓持仓。请先运行 `scripts/run_paper_trade.py` 生成持仓。")
    st.stop()

# ── 刷新按钮 ─────────────────────────────────────────────────────────────────
col_r1, col_r2, col_r3 = st.columns([6, 1, 1])
with col_r2:
    if st.button("🔄 刷新价格", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_r3:
    gen_at = raw.get("generated_at", "")
    if gen_at:
        try:
            gen_dt = datetime.fromisoformat(gen_at)
            st.caption(f"更新: {gen_dt.strftime('%m-%d %H:%M')}")
        except Exception:
            pass

# ── 获取价格 ─────────────────────────────────────────────────────────────────
tickers   = tuple(p["ticker"] for p in open_pos)
as_of_str = open_pos[0].get("as_of", str(TODAY))

with st.spinner("加载价格数据…"):
    entry_prices   = _load_entry_prices(tickers, as_of_str)
    current_prices = _load_current_prices(tickers)
    current_regime = _get_regime()

# ── 构建 DataFrame ─────────────────────────────────────────────────────────────
rows = []
for p in open_pos:
    tk          = p["ticker"]
    entry_px    = p.get("entry_price") or entry_prices.get(tk)
    current_px  = current_prices.get(tk)
    pnl_pct: float | None = None
    if entry_px and current_px and entry_px > 0:
        pnl_pct = (current_px / entry_px - 1.0) * 100.0
    rows.append({
        "代码":       tk,
        "建仓日期":   p.get("as_of", ""),
        "预计退出":   p.get("estimated_exit_date", ""),
        "持仓天数":   _holding_days(p.get("as_of", "")),
        "预测排名":   p.get("predicted_rank"),
        "预测分":     p.get("predicted_score"),
        "入场价":     entry_px,
        "当前价":     current_px,
        "浮动盈亏%":  pnl_pct,
        "状态":       p.get("status", "OPEN"),
    })

df = pd.DataFrame(rows)
# 按浮动盈亏降序排（无数据的排末尾）
df = df.sort_values("浮动盈亏%", ascending=False, na_position="last")

# ── KPI 卡片 ──────────────────────────────────────────────────────────────────
n_open   = len(open_pos)
has_pnl  = df["浮动盈亏%"].notna()
avg_pnl  = df.loc[has_pnl, "浮动盈亏%"].mean() if has_pnl.any() else None
avg_days = int(df["持仓天数"].mean()) if not df.empty else 0
regime_c = REGIME_COLOR.get(current_regime, "#636E72")
regime_emoji = {"bull": "🐂", "neutral": "⚖️", "bear": "🐻"}.get(current_regime, "❓")

st.markdown("---")
kc1, kc2, kc3, kc4 = st.columns(4)

with kc1:
    st.metric("📦 开仓数量", f"{n_open} 只")

with kc2:
    if avg_pnl is not None:
        delta_str = f"{avg_pnl:+.2f}%"
        st.metric("💰 平均浮盈", delta_str, delta=delta_str)
    else:
        st.metric("💰 平均浮盈", "待价格", help="需本地价格数据计算")

with kc3:
    st.metric("📅 平均持仓", f"{avg_days} 天")

with kc4:
    st.markdown(
        f"""<div style='background:{regime_c};border-radius:8px;
                        padding:10px 16px;color:white;text-align:center'>
              <div style='font-size:1.5rem'>{regime_emoji}</div>
              <div style='font-weight:600;font-size:1.1rem'>
                {current_regime.upper()}</div>
              <div style='font-size:.8rem;opacity:.9'>当前 Regime</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── 持仓明细表 ────────────────────────────────────────────────────────────────
st.subheader("📋 持仓明细")

def _color_pnl(val: float | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return "color: #00B894; font-weight:600" if val >= 0 else "color: #D63031; font-weight:600"


display_df = df.copy()
# 格式化
display_df["预测分"]    = display_df["预测分"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
display_df["入场价"]    = display_df["入场价"].map(lambda x: f"¥{x:.2f}" if pd.notna(x) else "—")
display_df["当前价"]    = display_df["当前价"].map(lambda x: f"¥{x:.2f}" if pd.notna(x) else "—")
display_df["浮动盈亏%"] = display_df["浮动盈亏%"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")

# 原始数值列用于着色
pnl_raw = df["浮动盈亏%"].copy()

styled = (
    display_df.style
    .apply(
        lambda col: [_color_pnl(pnl_raw.iloc[i]) for i in range(len(pnl_raw))],
        subset=["浮动盈亏%"],
        axis=0,
    )
)

st.dataframe(styled, use_container_width=True, height=420)

# ── 图表区 ───────────────────────────────────────────────────────────────────
st.markdown("---")
gc1, gc2 = st.columns([1, 1])

with gc1:
    st.subheader("🏦 交易所分布")
    exchange_map = {tk: ("沪市" if tk.endswith(".SH") else "深市") for tk in df["代码"]}
    ex_df = df.copy()
    ex_df["交易所"] = ex_df["代码"].map(exchange_map)
    ex_counts = ex_df.groupby("交易所").size().reset_index(name="数量")
    fig_pie = px.pie(
        ex_counts,
        values="数量",
        names="交易所",
        color="交易所",
        color_discrete_map={"沪市": "#0984E3", "深市": "#00B894"},
        hole=0.4,
    )
    fig_pie.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=True,
        height=300,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with gc2:
    st.subheader("📈 预测分布")
    fig_hist = go.Figure()
    if has_pnl.any():
        # 如果有 P&L 数据，优先展示 P&L 分布
        pnl_vals = df.loc[has_pnl, "浮动盈亏%"].values
        fig_hist.add_trace(go.Histogram(
            x=pnl_vals,
            nbinsx=10,
            marker_color="#00B894",
            name="浮动盈亏%",
        ))
        fig_hist.update_layout(
            xaxis_title="浮动盈亏 (%)",
            yaxis_title="股票数",
            bargap=0.08,
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
        )
    else:
        # 无 P&L 时展示预测分
        score_vals = df["预测分"].map(
            lambda x: float(x.replace(",", "")) if isinstance(x, str) and x != "—" else np.nan
        )
        score_vals = pd.to_numeric(score_vals, errors="coerce").dropna()
        fig_hist.add_trace(go.Histogram(
            x=score_vals.values,
            nbinsx=10,
            marker_color="#6C5CE7",
            name="预测分",
        ))
        fig_hist.update_layout(
            xaxis_title="预测分",
            yaxis_title="股票数",
            bargap=0.08,
            height=300,
            margin=dict(t=20, b=20, l=20, r=20),
        )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── 底部信息 ──────────────────────────────────────────────────────────────────
st.markdown("---")
info_c1, info_c2, info_c3 = st.columns(3)
info_c1.info(f"📅 下次审查：{raw.get('next_review', 'N/A')}")
info_c2.info(f"⏱ 跟踪周期：{raw.get('tracking_horizon', '3m')}")
info_c3.info(f"💡 入场价注：{'已计算（本地价格）' if entry_prices else '待补充（entry_price=null）'}")
