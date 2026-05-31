"""10_持仓详情.py — 30 日模拟盘持仓 NAV 曲线 + 止损模拟对比.

三个区域：
  1. 批次选择器
  2. 单股逐日 NAV 折线图（止损触发点标记）+ 数据表
  3. 止损 vs 不止损组合对比（含实时止损阈值滑块）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── 页面配置 ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="持仓详情 · QuantMind",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<div style='background:linear-gradient(90deg,#6C5CE7,#A29BFE);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:20px'>
  <h2 style='margin:0'>📈 持仓 NAV 曲线 & 止损模拟</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    逐日净值还原 · 固定/追踪止损对比 · 止损有效性分析
  </p>
</div>
""", unsafe_allow_html=True)

# ─── 常量 ─────────────────────────────────────────────────────────────────────
NAV_DIR      = ROOT / "data" / "sim30d" / "nav"
POSITIONS_PQ = ROOT / "data" / "sim30d" / "positions.parquet"

_PALETTE = [
    "#6C5CE7", "#00B894", "#E17055", "#0984E3", "#FDCB6E",
    "#D63031", "#00CEC9", "#E84393", "#55EFC4", "#636E72",
]

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_nav_json(date_str: str) -> dict[str, Any] | None:
    path = NAV_DIR / f"{date_str}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(ttl=600)
def _load_positions() -> pd.DataFrame:
    if not POSITIONS_PQ.exists():
        return pd.DataFrame()
    return pd.read_parquet(POSITIONS_PQ)


@st.cache_data(ttl=600)
def _available_dates() -> list[str]:
    if not NAV_DIR.exists():
        return []
    return sorted(p.stem for p in NAV_DIR.glob("*.json"))


# ─── 实时止损重算（不缓存，依赖 prices） ────────────────────────────────────

@st.cache_data(ttl=600)
def _load_prices_cached() -> pd.DataFrame:
    """加载合并后的价格宽表（含合成）。"""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_position_nav import load_prices  # noqa: PLC0415
        return load_prices()
    except Exception:
        return pd.DataFrame()


def _recompute_batch(
    date_str: str,
    stop_loss: float,
    trailing: bool,
) -> dict[str, Any] | None:
    """用新止损参数重算某批次 NAV（Streamlit slider 用）。"""
    try:
        import json as _json
        import glob as _glob

        sys.path.insert(0, str(ROOT / "scripts"))
        from build_position_nav import build_nav_for_batch, load_prices  # noqa

        daily_path = ROOT / "data" / "sim30d" / "daily" / f"{date_str}.json"
        if not daily_path.exists():
            return None
        with daily_path.open() as fh:
            day = _json.load(fh)

        picks = [s["ticker"] for s in day.get("system3_final_list", [])]
        prices_df = load_prices()
        return build_nav_for_batch(
            entry_date   = date_str,
            stock_list   = picks,
            prices_df    = prices_df,
            stop_loss    = stop_loss,
            trailing     = trailing,
            holding_days = 63,
        )
    except Exception as e:
        st.warning(f"重算失败: {e}")
        return None


# ─── 辅助：把 [(date_str, nav_val), ...] 变成 DataFrame ─────────────────────

def _nav_list_to_df(nav_list: list) -> pd.DataFrame:
    if not nav_list:
        return pd.DataFrame(columns=["date", "nav"])
    df = pd.DataFrame(nav_list, columns=["date", "nav"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    df["day"] = range(len(df))
    return df


# ─── 区域 1：批次选择器 ───────────────────────────────────────────────────────

dates = _available_dates()
if not dates:
    st.error("⚠️ 未找到 nav/ 目录下的 JSON 文件。请先运行：\n"
             "```\npython scripts/build_position_nav.py --entry-date all\n```")
    st.stop()

col_sel, col_info = st.columns([2, 3])
with col_sel:
    selected_date = st.selectbox(
        "选择建仓批次",
        options=dates,
        format_func=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]} (第{dates.index(d)+1}批)",
    )
with col_info:
    st.markdown(f"""
    <div style='padding:10px;background:#F8F9FA;border-radius:8px;margin-top:8px'>
      <b>📅 共 {len(dates)} 个批次</b> &nbsp;·&nbsp;
      {dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]} 至
      {dates[-1][:4]}-{dates[-1][4:6]}-{dates[-1][6:]}
    </div>
    """, unsafe_allow_html=True)

nav_data = _load_nav_json(selected_date)
if nav_data is None:
    st.error(f"无法加载 {selected_date}.json")
    st.stop()

stocks_data = nav_data.get("stocks", {})
tickers     = [t for t, v in stocks_data.items() if v.get("cumulative_nav")]

st.divider()

# ─── 区域 2：单股 NAV 折线图 ──────────────────────────────────────────────────

st.subheader("📊 单只股票逐日净值曲线")

if not tickers:
    st.warning("该批次无可用逐日价格数据（所有股票均使用插值合成）。")
else:
    positions_df = _load_positions()

    fig = go.Figure()

    # 0% 参考线
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        opacity=0.5,
        annotation_text="0%",
        annotation_position="left",
    )

    table_rows = []

    for i, ticker in enumerate(tickers):
        stock = stocks_data[ticker]
        nav_list = stock.get("cumulative_nav", [])
        if not nav_list:
            continue

        nav_df   = _nav_list_to_df(nav_list)
        color    = _PALETTE[i % len(_PALETTE)]
        final_r  = stock.get("final_return", 0.0)
        stopped  = stock.get("stop_loss_triggered", False)
        sl_date  = stock.get("stop_loss_date")
        mdd      = stock.get("max_drawdown", 0.0)
        hd       = stock.get("holding_days_actual", 0)

        # 查询建仓信息
        name = ticker
        conf = None
        if not positions_df.empty:
            row_mask = (positions_df["date"] == selected_date) & \
                       (positions_df["ticker"] == ticker)
            if row_mask.any():
                prow = positions_df[row_mask].iloc[0]
                name = prow.get("name", ticker)
                conf = prow.get("composite_score", None)

        label = f"{name}({ticker})"

        # 主 NAV 折线
        fig.add_trace(go.Scatter(
            x          = nav_df["day"],
            y          = nav_df["nav"],
            name       = label,
            mode       = "lines",
            line       = dict(color=color, width=2),
            hovertemplate = (
                f"<b>{label}</b><br>"
                "第 %{x} 日<br>"
                "净值: %{y:.4f}<extra></extra>"
            ),
        ))

        # 止损点标记
        if stopped and sl_date:
            sl_ts = pd.Timestamp(sl_date)
            sl_row = nav_df[nav_df["date"] <= sl_ts]
            if not sl_row.empty:
                sl_pt = sl_row.iloc[-1]
                fig.add_trace(go.Scatter(
                    x    = [sl_pt["day"]],
                    y    = [sl_pt["nav"]],
                    mode = "markers",
                    marker = dict(symbol="x", size=14, color="red", line_width=2),
                    name   = f"{ticker} 止损",
                    showlegend = False,
                    hovertemplate = (
                        f"<b>止损触发：{ticker}</b><br>"
                        f"日期: {sl_date}<br>"
                        f"净值: {sl_pt['nav']:.4f}<extra></extra>"
                    ),
                ))

        # 末端标注
        last = nav_df.iloc[-1]
        fig.add_annotation(
            x    = last["day"],
            y    = last["nav"],
            text = f"{final_r*100:+.1f}%",
            showarrow = False,
            xanchor   = "left",
            font      = dict(size=10, color=color),
        )

        table_rows.append({
            "股票":    label,
            "建仓价":  "—",
            "最终净值": f"{1.0 + final_r:.4f}",
            "收益":    f"{final_r*100:+.2f}%",
            "MaxDD":   f"{mdd*100:.2f}%",
            "实际持仓(日)": hd,
            "止损触发": "🔴 是" if stopped else "🟢 否",
            "综合评分": f"{conf:.1f}" if conf else "—",
        })

    fig.update_layout(
        title      = f"批次 {selected_date} — 持仓逐日净值（共 {len(tickers)} 只有价格数据）",
        xaxis_title = "持仓交易日",
        yaxis_title = "累计净值（入场=1.0）",
        height     = 520,
        hovermode  = "x unified",
        legend     = dict(orientation="h", yanchor="bottom", y=-0.3),
        yaxis      = dict(tickformat=".3f"),
        plot_bgcolor = "#FAFAFA",
        paper_bgcolor = "white",
    )

    st.plotly_chart(fig, use_container_width=True)

    if table_rows:
        st.markdown("**持仓明细表**")
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width = True,
            hide_index = True,
        )

st.divider()

# ─── 区域 3：止损 vs 不止损对比 ──────────────────────────────────────────────

st.subheader("⚖️ 止损效果对比")

left_col, right_col = st.columns(2)

# 止损参数控件
with st.container():
    ctl1, ctl2, ctl3 = st.columns([2, 2, 1])
    with ctl1:
        sl_threshold = st.slider(
            "止损阈值",
            min_value = -0.30,
            max_value = -0.05,
            value     = float(nav_data.get("stop_loss", -0.15)),
            step      = 0.01,
            format    = "%.0f%%",
            help      = "触发止损的最大亏损比例（从入场价或峰值计算）",
        )
    with ctl2:
        trailing_mode = st.radio(
            "止损类型",
            options = ["固定止损", "追踪止损"],
            index   = 1 if nav_data.get("trailing", False) else 0,
            horizontal = True,
        )
    with ctl3:
        recompute = st.button("⚙️ 重新计算", use_container_width=True)

# 判断是否需要重算（参数变化时）
_use_cached = (
    abs(sl_threshold - float(nav_data.get("stop_loss", -0.15))) < 0.001
    and (trailing_mode == "追踪止损") == nav_data.get("trailing", False)
    and not recompute
)

if _use_cached:
    display_data = nav_data
else:
    with st.spinner("重新计算中…"):
        display_data = _recompute_batch(
            selected_date,
            stop_loss = sl_threshold,
            trailing  = (trailing_mode == "追踪止损"),
        ) or nav_data

# 提取数据
port_nav_raw = display_data.get("portfolio_nav", [])
port_ret     = display_data.get("portfolio_final_return", float("nan"))
port_mdd     = display_data.get("portfolio_max_drawdown", float("nan"))

sl_data = display_data.get("with_stop_loss", {})
if not sl_data:
    st.info("该批次暂无止损模拟数据")
    sl_nav_raw   = []
    sl_ret       = float("nan")
    sl_mdd       = float("nan")
    stopped_list = []
    improvement  = float("nan")
else:
    sl_nav_raw   = sl_data.get("portfolio_nav", [])
    sl_ret       = sl_data.get("portfolio_final_return", float("nan"))
    sl_mdd       = sl_data.get("portfolio_max_drawdown", float("nan"))
    stopped_list = sl_data.get("stopped_stocks", [])
    improvement  = sl_data.get("improvement_vs_no_stop", float("nan"))

port_df = _nav_list_to_df(port_nav_raw)
sl_df   = _nav_list_to_df(sl_nav_raw)

def _nav_fig(df: pd.DataFrame, title: str, color: str) -> go.Figure:
    f = go.Figure()
    f.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    if not df.empty:
        f.add_trace(go.Scatter(
            x    = df["day"],
            y    = df["nav"],
            mode = "lines+markers",
            line = dict(color=color, width=3),
            marker = dict(size=4),
            name = title,
            hovertemplate = "第 %{x} 日<br>净值: %{y:.4f}<extra></extra>",
        ))
    f.update_layout(
        title          = title,
        xaxis_title    = "持仓交易日",
        yaxis_title    = "等权组合净值",
        height         = 320,
        showlegend     = False,
        yaxis          = dict(tickformat=".3f"),
        plot_bgcolor   = "#FAFAFA",
        paper_bgcolor  = "white",
    )
    return f


with left_col:
    st.plotly_chart(
        _nav_fig(port_df, "不止损组合 NAV", "#0984E3"),
        use_container_width=True,
    )

with right_col:
    st.plotly_chart(
        _nav_fig(sl_df, f"{'追踪' if trailing_mode=='追踪止损' else '固定'}止损 {sl_threshold*100:.0f}% 组合 NAV", "#E17055"),
        use_container_width=True,
    )

# 指标卡片
m1, m2, m3, m4 = st.columns(4)
_na = lambda v: f"{v*100:+.2f}%" if not pd.isna(v) else "N/A"
_na_neg = lambda v: f"{v*100:.2f}%" if not pd.isna(v) else "N/A"

with m1:
    st.metric("不止损 · 最终收益", _na(port_ret))
with m2:
    st.metric("不止损 · MaxDD",    _na_neg(abs(port_mdd) if not pd.isna(port_mdd) else float("nan")))
with m3:
    st.metric("有止损 · 最终收益", _na(sl_ret),
              delta=f"{improvement*100:+.2f}%" if not pd.isna(improvement) else None)
with m4:
    st.metric("有止损 · MaxDD",    _na_neg(abs(sl_mdd) if not pd.isna(sl_mdd) else float("nan")))

# 结论标签
if not pd.isna(improvement):
    if improvement > 0.005:
        st.success(f"🟢 止损**改善**了收益（相对不止损 {improvement*100:+.2f}%），止损策略有效")
    elif improvement < -0.005:
        st.error(f"🔴 止损**损害**了收益（相对不止损 {improvement*100:+.2f}%），止损过于激进")
    else:
        st.info("⬜ 止损对收益影响微小（< ±0.5%）")

# 止损触发股票列表
if stopped_list:
    st.markdown(f"**触发止损的股票（{len(stopped_list)} 只）：** "
                + "  ".join(f"`{t}`" for t in stopped_list))
