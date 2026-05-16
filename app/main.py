"""QuantMind Streamlit 主页 — 仅从本地 reports / data 读取。"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.data_loader import (
    latest_top10_date_or_none,
    load_agent_registry,
    load_backtest_metrics,
    load_final_recommendations_md,
    load_holdout_metrics,
    load_top10,
    merge_top10_with_md,
    parse_tri_classification,
)
from app.utils.llm_commentator import LLMCommentator

st.set_page_config(
    page_title="QuantMind 投资系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

fe = load_backtest_metrics("reports/final_eval/backtest_metrics.json")
ho = load_holdout_metrics()
seg24 = fe.get("segment_2024") or {}
seg_oos_h = ho.get("segment_oos") or {}

_commentator = LLMCommentator()
_reg = load_agent_registry()
_kb_root = ROOT / ".cache" / "chromadb"
try:
    kb_count = sum(1 for _ in _kb_root.rglob("*")) if _kb_root.is_dir() else 0
except OSError:
    kb_count = 0

_active_label = "—"
_v_labels: list[str] = []
for agent_name, versions in (_reg or {}).items():
    if not isinstance(versions, list):
        continue
    for v in versions:
        if v.get("is_active"):
            mv = v.get("model_version") or "?"
            _v_labels.append(f"{agent_name}:{mv}")
if _v_labels:
    _active_label = ", ".join(_v_labels[:8])
elif _reg:
    try:
        first = next(iter(_reg.values()))
        if isinstance(first, list) and first:
            _active_label = str(first[-1].get("model_version") or first[-1].get("model_path") or "active")
    except (StopIteration, TypeError, KeyError):
        pass

ld_sidebar = latest_top10_date_or_none()
ic_snap = seg24.get("ic_mean")

with st.sidebar:
    st.markdown("## ⚙️ 系统状态")
    st.markdown(f"📅 **最新数据**：{ld_sidebar or '—'}")
    st.markdown(f"🤖 **LLM 引擎**：`{_commentator.provider}`")
    st.markdown(f"📊 **知识库索引条目（近似文件计数）**：{kb_count}")
    st.markdown(f"🧠 **模型快照**：`{_active_label}`")
    if ic_snap is not None and seg24.get("ic_ir") is not None and seg24.get("ls_sharpe") is not None:
        st.markdown(
            f"快照：**IC**=`{float(ic_snap):.3f}` | **ICIR**=`{float(seg24['ic_ir']):.3f}` | "
            f"**多空夏普**=`{float(seg24['ls_sharpe']):.2f}`"
        )
    if st.button("🔄 清空缓存（LLM/加载）"):
        st.cache_data.clear()
        st.success("已调用 `st.cache_data.clear()`。")

st.markdown(
    """
<div style='background: linear-gradient(90deg, #0984E3 0%, #00B894 100%);
            padding: 20px; border-radius: 10px; color: white;'>
    <h2 style='margin:0;'>📊 QuantMind</h2>
    <p style='margin:0.5rem 0 0 0;'>AI 增强量化投资系统 · 漏斗选股 · 多 Agent 研究 · 回测验证</p>
</div>
""",
    unsafe_allow_html=True,
)

ic_test = seg24.get("ic_mean")
ic_hold = seg_oos_h.get("ic_mean")
delta_txt = "—"
try:
    if ic_test is not None and ic_hold is not None:
        delta_txt = f"{float(ic_test) - float(ic_hold):+.4f} vs 保留集"
except (TypeError, ValueError):
    pass


def _fmt_pct(x, nd=2):
    try:
        if x is None:
            return "—"
        return f"{float(x)*100:.{nd}f}%"
    except (TypeError, ValueError):
        return "—"


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "OOS IC_mean（Test 2024）",
        f"{float(ic_test):.4f}" if ic_test is not None else "—",
        delta_txt if delta_txt != "—" else None,
    )
with col2:
    st.metric(
        "Q5 年化收益（无成本）",
        _fmt_pct(seg24.get("q5_ann")),
        "Test 2024",
    )
with col3:
    st.metric(
        "多空夏普（无成本）",
        f"{float(seg24.get('ls_sharpe')):.3f}" if seg24.get("ls_sharpe") is not None else "—",
        "因子分层口径",
    )
with col4:
    st.metric(
        "保留集 IC_mean",
        f"{float(ic_hold):.4f}" if ic_hold is not None else "—",
        "2025–2026（一次性）",
    )

st.subheader("📋 最新推荐预览")
ld = ld_sidebar
if not ld:
    st.info("未找到 `data/recommendations/*/top10.json`，请先运行每日流水线生成推荐。")
else:
    st.caption(f"最近可用日期：**{ld}**（`data/recommendations/{ld}/top10.json`）")
    raw = load_top10(ld)
    md = load_final_recommendations_md(ld)
    tri = parse_tri_classification(md)
    tbl = merge_top10_with_md(raw, tri)
    if tbl.empty:
        st.warning("top10.json 为空或格式不符。")
    else:
        st.dataframe(tbl, use_container_width=True, hide_index=True)
