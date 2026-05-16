"""漏斗选股统计与可视化。"""

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
    list_top10_dates,
    load_funnel_candidates,
    load_stock_basic,
    load_top10,
)

st.set_page_config(page_title="漏斗选股", layout="wide")
st.title("🔻 漏斗选股")

dates = list_top10_dates()
if not dates:
    st.warning("暂无推荐日期目录。")
    st.stop()

pick = st.selectbox("选择日期", dates, index=len(dates) - 1)
fc = load_funnel_candidates(pick)
stats = fc.get("funnel_stats") or {}

with st.sidebar.expander("各层过滤逻辑（占位说明）", expanded=False):
    st.markdown(
        """
- **Layer1**：全市场 CSI300 候选  
- **Layer2**：基础过滤 / ST / 上市天数  
- **Layer3**：流动性 + 趋势过滤  
- **Layer4**：基本面硬闸  
- **Layer5**：LGBM 粗排保留池  
- **Layer6**：LLM / 规则精选输出 Top10  
具体规则以管线代码为准；此处仅展示统计 JSON。
"""
    )

if not stats:
    st.info("未找到 `funnel_candidates.json` 或无 `funnel_stats`。")
    st.stop()

labels = [
    "全市场",
    "基础过滤",
    "流动性",
    "趋势",
    "基本面",
    "LGBM池",
    "LLM精选",
]
xs = [
    stats.get("layer1_in", 0),
    stats.get("layer1_out", 0),
    stats.get("layer2_out", 0),
    stats.get("layer3_out", 0),
    stats.get("layer4_out", 0),
    stats.get("layer5_out", 0),
    stats.get("layer6_out", 0),
]
fig = charts.funnel_chart(labels, [float(x) for x in xs])
st.plotly_chart(fig, use_container_width=True)

cs = fc.get("cache_stats") or {}
if cs:
    st.caption(
        f"缓存：disk_hits={cs.get('disk_hits','—')} misses={cs.get('misses','—')} "
        f"hit_rate={float(cs.get('hit_rate',0))*100:.1f}%"
    )

basic = load_stock_basic(pick)
strategies_tickers = [x.get("ticker") for x in (fc.get("candidates") or []) if x.get("ticker")]
top10 = load_top10(pick).get("top10") or []
if isinstance(top10, list):
    strategies_tickers = [str(x.get("ticker")) for x in top10 if x.get("ticker")]

if not basic.empty and strategies_tickers:
    sub = basic[basic["ticker"].astype(str).isin(strategies_tickers)]
    if not sub.empty:
        vc = sub.groupby(sub["industry"].fillna("未知")).size().to_dict()
        st.plotly_chart(charts.industry_pie(vc), use_container_width=True)
    else:
        st.info("候选 ticker 在当日 stock_basic 中无行业映射。")
else:
    st.info("无法绘制行业饼图：缺少 `stock_basic.parquet` 或候选 ticker。")

rows = load_top10(pick).get("top10") or []
if isinstance(rows, list) and rows:
    rows_sorted = sorted(rows, key=lambda x: float(x.get("lgbm_score") or 0), reverse=True)[:20]
    tk = [str(x.get("ticker")) for x in rows_sorted]
    sc = [float(x.get("lgbm_score") or 0) for x in rows_sorted]
    st.plotly_chart(charts.lgbm_bar(tk, sc), use_container_width=True)
else:
    st.info("无 Top10 LGBM 得分柱图数据。")
