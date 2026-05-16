"""quantmind.ui.streamlit_app — QuantMind 主入口（多页面 Streamlit 应用）.

启动命令：
    streamlit run quantmind/ui/streamlit_app.py

页面：
  Overview      — 项目架构 + 进度 + 最新回测
  个股研究       — 多 Agent 深度研究（Orchestrator）
  策略回测       — 量化策略回测（BacktestEngine）
  Agent 回测     — Agent 决策历史回测
  知识库         — KB 检索 + 文档管理
  模型管理       — LightGBM + DPO 模型状态
"""

import sys
from pathlib import Path

import streamlit as st

# 确保项目根目录在 sys.path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantMind",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 深色金融主题 CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* 主背景 */
    .stApp { background-color: #0e1117; }
    /* 侧边栏 */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    /* 卡片样式 */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #58a6ff; }
    .metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
    /* 状态徽章 */
    .badge-complete { background: #238636; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .badge-pending  { background: #6e7681; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    /* 标题 */
    h1, h2, h3 { color: #e6edf3 !important; }
    p { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

# ── 侧边栏导航 ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 QuantMind")
    st.markdown("*AI Agent 量化研究系统*")
    st.divider()

    page = st.radio(
        "导航",
        [
            "🏠 Overview",
            "🔍 个股研究",
            "📊 策略回测",
            "🤖 Agent 回测",
            "📚 知识库",
            "🧠 模型管理",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"根目录: `{ROOT}`")

# ── 路由到各页面 ──────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    from quantmind.ui.pages import p01_overview as _p
    _p.render()

elif page == "🔍 个股研究":
    from quantmind.ui.pages import p02_stock_research as _p
    _p.render()

elif page == "📊 策略回测":
    from quantmind.ui.pages import p03_backtest as _p
    _p.render()

elif page == "🤖 Agent 回测":
    from quantmind.ui.pages import p04_agent_backtest as _p
    _p.render()

elif page == "📚 知识库":
    from quantmind.ui.pages import p05_knowledge_base as _p
    _p.render()

elif page == "🧠 模型管理":
    from quantmind.ui.pages import p06_models as _p
    _p.render()
