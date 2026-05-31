"""8_Regime状态.py — HMM 市场状态实时监控与动态权重展示."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantmind.regime import DynamicWeightManager, RegimeHMM

# ─── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regime 状态 · QuantMind",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<div style='background:linear-gradient(90deg,#6C5CE7,#A29BFE);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:20px'>
  <h2 style='margin:0'>🎯 市场 Regime 状态监控</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    HMM 3态（Bull / Neutral / Bear）+ 动态权重自适应配置
  </p>
</div>
""", unsafe_allow_html=True)

# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def _regime_switch_impact(fr: str, to: str) -> str:
    """生成一句话切换影响描述。"""
    impacts = {
        ("bull",    "bear"):    "⚠️ 防御切换：LGBM↑，CNN↓，价值权重↑",
        ("bear",    "bull"):    "📈 进攻切换：CNN↑，质量权重↑，EM因子↑",
        ("neutral", "bull"):    "📈 乐观切换：质量因子权重提升",
        ("neutral", "bear"):    "⚠️ 谨慎切换：价值因子防御，动量降权",
        ("bull",    "neutral"): "➡️ 回归中性：均衡配置",
        ("bear",    "neutral"): "➡️ 缓和信号：适度增配 CNN",
    }
    return impacts.get((fr, to), "→ 权重自动调整")


def _hex_rgba(hex6: str, alpha: float) -> str:
    """将 6 位 hex 颜色转为 rgba() 字符串。

    Plotly 不接受 8 位 hex（#RRGGBBAA），必须用 rgba(r,g,b,a) 格式。
    alpha 取值 0.0~1.0。
    """
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ─── 颜色常量 ─────────────────────────────────────────────────────────────────
REGIME_COLOR = {
    "bull":    "#00B894",   # 绿
    "neutral": "#FDCB6E",   # 黄
    "bear":    "#D63031",   # 红
}
REGIME_EMOJI = {"bull": "🟢", "neutral": "🟡", "bear": "🔴"}
REGIME_CN    = {"bull": "牛市", "neutral": "中性", "bear": "熊市"}

# ─── 数据加载 ─────────────────────────────────────────────────────────────────
DATA_DIR     = ROOT / "data"
# Bug#6 fix: config 在 data/paper_trading/ 下，不在项目根目录
CONFIG_PATH  = ROOT / "data" / "paper_trading" / "strategy_config_v2.json"
REGIME_CACHE = DATA_DIR / "regime" / "regime_history.parquet"

# 尝试使用正确工厂方法拟合 HMM
@st.cache_data(ttl=3600, show_spinner="加载 HMM Regime 模型…")
def load_regime_history() -> tuple[pd.DataFrame, float]:
    """返回 (regime_df, fit_ll)。
    regime_df columns: date, regime, bull_prob, neutral_prob, bear_prob
    """
    # 优先读取缓存
    if REGIME_CACHE.exists():
        try:
            df = pd.read_parquet(REGIME_CACHE)
            df["date"] = pd.to_datetime(df["date"])
            return df, float("nan")
        except Exception:
            pass

    # Bug#7 fix: 使用工厂方法 + 正确数据源（CSI300 指数日线）
    hist_path = ROOT / "data" / "raw" / "index_daily_panel.parquet"
    sim_path  = ROOT / "data" / "sim30d" / "raw" / "index_000300_SH.parquet"

    try:
        model = RegimeHMM.fit_from_file(
            hist_path=hist_path if hist_path.exists() else None,
            sim_path=sim_path  if sim_path.exists()  else None,
        )

        if not len(model._dates):
            return pd.DataFrame(), float("nan")

        label_series = [model._label_map[s] for s in model._states]
        df = pd.DataFrame({
            "date":         model._dates,
            "regime":       label_series,
            # Viterbi 确定性预测：用独热概率代替平滑概率
            "bull_prob":    [1.0 if lbl == "bull"    else 0.0 for lbl in label_series],
            "neutral_prob": [1.0 if lbl == "neutral" else 0.0 for lbl in label_series],
            "bear_prob":    [1.0 if lbl == "bear"    else 0.0 for lbl in label_series],
        })
        # 确保缓存目录存在
        REGIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(REGIME_CACHE, index=False)
        return df, float("nan")
    except Exception as e:
        st.warning(f"HMM 拟合失败：{e}")
        return pd.DataFrame(), float("nan")


@st.cache_data(ttl=300)
def get_weight_manager() -> DynamicWeightManager:
    return DynamicWeightManager(
        config_path=CONFIG_PATH if CONFIG_PATH.exists() else None,
        auto_reload=False,
    )


# ─── 加载数据 ──────────────────────────────────────────────────────────────────
regime_df, fit_ll = load_regime_history()
weight_mgr = get_weight_manager()

# ─── 如无数据，给出演示模式 ────────────────────────────────────────────────────
DEMO_MODE = regime_df.empty
if DEMO_MODE:
    st.info("⚠️ 未检测到价格面板数据，以**演示模式**展示权重配置（Regime 历史为模拟数据）。")
    rng = np.random.default_rng(2024)
    n = 252
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    probs = rng.dirichlet([2, 1, 0.5], size=n)   # bias toward bull
    labels_int = probs.argmax(axis=1)
    label_map = {0: "bull", 1: "neutral", 2: "bear"}
    regime_df = pd.DataFrame({
        "date":         dates,
        "regime":       [label_map[i] for i in labels_int],
        "bull_prob":    probs[:, 0],
        "neutral_prob": probs[:, 1],
        "bear_prob":    probs[:, 2],
    })

# ─── 当前 Regime ────────────────────────────────────────────────────────────────
latest = regime_df.iloc[-1]
current_regime = str(latest["regime"]).lower()
current_probs = {
    "bull":    float(latest.get("bull_prob",    0.0)),
    "neutral": float(latest.get("neutral_prob", 0.0)),
    "bear":    float(latest.get("bear_prob",    0.0)),
}
current_confidence = max(current_probs.values())
current_date = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")

# ─── Section 1: 当前状态卡片 ──────────────────────────────────────────────────
st.markdown("### 📡 当前 Regime 状态")
col_main, col_prob1, col_prob2, col_prob3 = st.columns([2, 1, 1, 1])

with col_main:
    color = REGIME_COLOR.get(current_regime, "#636e72")
    emoji = REGIME_EMOJI.get(current_regime, "⚪")
    cn    = REGIME_CN.get(current_regime, current_regime.upper())
    st.markdown(f"""
    <div style='background:{color}22;border:2px solid {color};
                border-radius:12px;padding:20px 24px;text-align:center'>
      <div style='font-size:48px'>{emoji}</div>
      <div style='font-size:28px;font-weight:700;color:{color}'>{cn.upper()}</div>
      <div style='color:#636e72;margin-top:6px'>截至 {current_date}
        {"（演示）" if DEMO_MODE else ""}
      </div>
      <div style='margin-top:10px;font-size:18px'>
        置信度：<b>{current_confidence:.1%}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

for r, label in [("bull", "🟢 牛市"), ("neutral", "🟡 中性"), ("bear", "🔴 熊市")]:
    p = current_probs.get(r, 0.0)
    col = [col_prob1, col_prob2, col_prob3][["bull", "neutral", "bear"].index(r)]
    with col:
        st.metric(label, f"{p:.1%}")

st.divider()

# ─── Section 2: Regime 历史时间线 ─────────────────────────────────────────────
st.markdown("### 📅 历史 Regime 时间线")

# 计算最近 N 天
lookback_days = st.slider("展示近 N 个交易日", 60, 504, 252, step=21, key="timeline_days")
plot_df = regime_df.tail(lookback_days).copy()
plot_df["date"] = pd.to_datetime(plot_df["date"])

# 概率堆叠面积图
fig_timeline = go.Figure()
for r, label_cn, color in [
    ("bull",    "牛市概率",  "#00B894"),
    ("neutral", "中性概率",  "#FDCB6E"),
    ("bear",    "熊市概率",  "#D63031"),
]:
    col_key = f"{r}_prob"
    if col_key in plot_df.columns:
        fig_timeline.add_trace(go.Scatter(
            x=plot_df["date"],
            y=plot_df[col_key],
            mode="lines",
            name=label_cn,
            stackgroup="prob",
            line=dict(width=0.5, color=color),
            fillcolor=_hex_rgba(color, 0.6),
            hovertemplate="%{y:.1%}<extra>" + label_cn + "</extra>",
        ))

# 在当前日期打一条垂线
fig_timeline.add_vline(
    x=pd.Timestamp(latest["date"]).timestamp() * 1000,
    line_dash="dash",
    line_color="#2d3436",
    annotation_text="今日",
    annotation_position="top right",
)
fig_timeline.update_layout(
    height=280,
    margin=dict(l=20, r=20, t=20, b=20),
    yaxis=dict(tickformat=".0%", title="Regime 概率", range=[0, 1]),
    xaxis=dict(title=""),
    legend=dict(orientation="h", y=-0.15),
    plot_bgcolor="white",
    paper_bgcolor="white",
    yaxis_showgrid=True,
    xaxis_showgrid=False,
)
fig_timeline.update_xaxes(showline=True, linecolor="#dfe6e9")
fig_timeline.update_yaxes(showline=True, linecolor="#dfe6e9")
st.plotly_chart(fig_timeline, use_container_width=True)

# 离散 Regime 条带（按连续段上色）
fig_band = go.Figure()
prev_r, start_i = plot_df.iloc[0]["regime"], 0
segments = []
for i, row in enumerate(plot_df.itertuples()):
    if row.regime != prev_r or i == len(plot_df) - 1:
        end_i = i if row.regime != prev_r else i + 1
        segments.append((prev_r, plot_df.iloc[start_i]["date"], plot_df.iloc[min(end_i, len(plot_df)-1)]["date"]))
        start_i = i
        prev_r = row.regime

for regime, x0, x1 in segments:
    fig_band.add_vrect(
        x0=x0, x1=x1,
        fillcolor=_hex_rgba(REGIME_COLOR.get(str(regime).lower(), "#b2bec3"), 0.33),
        line_width=0,
        layer="below",
    )
    fig_band.add_trace(go.Scatter(
        x=[x0], y=[0.5],
        mode="markers",
        marker=dict(color=REGIME_COLOR.get(str(regime).lower(), "#b2bec3"), size=1),
        name=REGIME_CN.get(str(regime).lower(), str(regime)),
        showlegend=False,
        hoverinfo="skip",
    ))

fig_band.update_layout(
    height=50,
    margin=dict(l=20, r=20, t=0, b=0),
    yaxis=dict(visible=False, range=[0, 1]),
    xaxis=dict(showticklabels=False, range=[plot_df["date"].min(), plot_df["date"].max()]),
    plot_bgcolor="white",
    paper_bgcolor="white",
)
st.plotly_chart(fig_band, use_container_width=True)

# Regime 占比统计
r_counts = plot_df["regime"].value_counts()
col_c1, col_c2, col_c3 = st.columns(3)
for col, r in zip([col_c1, col_c2, col_c3], ["bull", "neutral", "bear"]):
    cnt = int(r_counts.get(r, 0))
    pct = cnt / max(len(plot_df), 1)
    with col:
        st.metric(
            f"{REGIME_EMOJI[r]} {REGIME_CN[r]}天数",
            f"{cnt}日",
            f"{pct:.1%}",
            delta_color="off",
        )

st.divider()

# ─── Section 3: 动态权重表 ──────────────────────────────────────────────────────
st.markdown("### ⚖️ 动态权重配置（当前 Schema）")

wt_df = weight_mgr.summary_dataframe()

# 高亮当前 regime 所在行
def _highlight_row(row):
    if row.name.lower() == current_regime:
        return [f"background-color:{REGIME_COLOR.get(current_regime,'#DFE6E9')}33"] * len(row)
    return [""] * len(row)

st.dataframe(
    wt_df.style
         .apply(_highlight_row, axis=1)
         .format("{:.3f}"),
    use_container_width=True,
)

# 当前 regime 权重详情
st.markdown(f"#### {REGIME_EMOJI[current_regime]} 当前 {REGIME_CN[current_regime]} 权重明细")
w = weight_mgr.get_weights(current_regime)

col_s2, col_ens, col_supp = st.columns(3)
with col_s2:
    st.markdown("**System2 四维权重**")
    s2 = w["system2"]
    for k, v in s2.items():
        k_cn = {"value": "价值", "momentum": "动量", "quality": "质量", "technical": "技术"}.get(k, k)
        st.progress(float(v), text=f"{k_cn}: {v:.3f}")

with col_ens:
    st.markdown("**Ensemble（LGBM / CNN）**")
    ens = w["ensemble"]
    st.progress(float(ens["lgbm"]), text=f"LGBM: {ens['lgbm']:.2f}")
    st.progress(float(ens["cnn"]),  text=f"CNN:  {ens['cnn']:.2f}")

with col_supp:
    st.markdown("**补充因子权重**")
    st.metric("EM 基本面因子", f"{w['em_factor']:.2f}")
    st.metric("文本情绪因子",  f"{w['text_factor']:.2f}")

st.divider()

# ─── Section 4: 近期 Regime 切换记录 ─────────────────────────────────────────
st.markdown("### 🔄 近期 Regime 切换记录（最近 5 次）")

# 找切换点
regime_series = regime_df[["date", "regime"]].copy()
regime_series["prev_regime"] = regime_series["regime"].shift(1)
changes = regime_series[regime_series["regime"] != regime_series["prev_regime"]].copy()
changes = changes.dropna(subset=["prev_regime"]).tail(5)

if changes.empty:
    st.info("暂无 Regime 切换记录。")
else:
    rows = []
    for _, row in changes.iterrows():
        fr = str(row["prev_regime"]).lower()
        to = str(row["regime"]).lower()
        rows.append({
            "日期":     pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
            "从":       f"{REGIME_EMOJI.get(fr,'⚪')} {REGIME_CN.get(fr, fr.upper())}",
            "切换到":   f"{REGIME_EMOJI.get(to,'⚪')} {REGIME_CN.get(to, to.upper())}",
            "影响":     _regime_switch_impact(fr, to),
        })
    st.table(pd.DataFrame(rows).set_index("日期"))

st.divider()

# ─── Section 5: 侧栏信息 (moved helper to top of file) ────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Regime 快览")
    st.markdown(f"""
    **当前状态**：{REGIME_EMOJI[current_regime]} {REGIME_CN[current_regime]}
    **置信度**：{current_confidence:.1%}
    **截至**：{current_date}
    """)
    st.divider()
    st.markdown("### ⚙️ 模型配置")
    st.caption(f"HMM 状态数：3")
    st.caption(f"配置文件：{'✅ 已加载' if CONFIG_PATH.exists() else '⬜ 默认值'}")
    if not DEMO_MODE:
        st.caption(f"样本量：{len(regime_df)} 交易日")
    st.divider()
    if st.button("🔄 强制刷新"):
        st.cache_data.clear()
        st.rerun()
    if DEMO_MODE:
        st.warning("演示模式 — 配置真实数据路径后重启")
