"""05 知识库 — 已索引文档浏览 + 检索测试 + 统计."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent


def render() -> None:
    st.title("📚 知识库管理")
    st.markdown("浏览已索引文档，测试语义检索效果（BGE-M3 + BM25 混合检索）。")

    # ── Tab 布局 ──────────────────────────────────────────────────────────────
    tab_search, tab_browse, tab_stats = st.tabs(["🔍 检索测试", "📄 文档浏览", "📊 统计"])

    with tab_search:
        _render_search()

    with tab_browse:
        _render_browse()

    with tab_stats:
        _render_stats()


# ─── 检索测试 ──────────────────────────────────────────────────────────────────

def _render_search() -> None:
    st.subheader("语义检索测试")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "查询语句",
            value="贵州茅台 2023 年营收增长",
            placeholder="输入自然语言查询...",
        )
    with col2:
        top_k = st.slider("Top-K", 1, 10, 5)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        search_btn = st.button("🔎 检索", type="primary", use_container_width=True)

    if search_btn and query:
        _run_search(query, top_k)
    elif not search_btn:
        st.info("输入查询语句后点击「检索」")


def _run_search(query: str, top_k: int) -> None:
    with st.spinner("正在检索..."):
        try:
            from quantmind.knowledge.retriever import HybridRetriever
            retriever = HybridRetriever()
            results = retriever.search(query, top_k=top_k)
            _display_search_results(results)
        except Exception as e:
            st.warning(f"检索引擎未就绪：{e}，展示 Demo 结果")
            _display_demo_search_results(query, top_k)


def _display_search_results(results: list) -> None:
    if not results:
        st.info("未找到相关文档")
        return

    st.success(f"找到 {len(results)} 条结果")
    for i, r in enumerate(results, 1):
        with st.expander(f"#{i}  {r.get('title', r.get('source', '未知来源'))}  —  相关度 {r.get('score', 0):.3f}"):
            st.markdown(f"**来源：** `{r.get('source', 'N/A')}`")
            st.markdown(f"**日期：** {r.get('date', 'N/A')}  |  **股票：** {r.get('ticker', 'N/A')}")
            st.divider()
            st.markdown(r.get("content", r.get("text", "（内容缺失）")))


def _display_demo_search_results(query: str, top_k: int) -> None:
    demo_results = [
        {
            "title": "贵州茅台 2023 年年度报告摘要",
            "source": "reports/600519_2023_annual.txt",
            "ticker": "600519.SH",
            "date": "2024-03-28",
            "score": 0.923,
            "content": "公司 2023 年实现营业收入 1476.94 亿元，同比增长 18.04%；实现归母净利润 747.34 亿元，同比增长 19.16%。",
        },
        {
            "title": "贵州茅台 2023Q4 业绩点评",
            "source": "research/600519_q4_2023.txt",
            "ticker": "600519.SH",
            "date": "2024-01-15",
            "score": 0.871,
            "content": "Q4 单季收入约 430 亿元，同比 +17%，量价齐升趋势延续，系列酒增速高于飞天茅台。",
        },
        {
            "title": "白酒行业 2024 年展望",
            "source": "industry/baijiu_2024_outlook.txt",
            "ticker": "N/A",
            "date": "2024-01-05",
            "score": 0.745,
            "content": "高端白酒量价有望继续双升，茅台/五粮液/泸州老窖受益春节旺季，渠道库存健康。",
        },
    ][:top_k]

    st.success(f"（Demo）找到 {len(demo_results)} 条结果")
    for i, r in enumerate(demo_results, 1):
        with st.expander(f"#{i}  {r['title']}  —  相关度 {r['score']:.3f}"):
            st.markdown(f"**来源：** `{r['source']}`")
            st.markdown(f"**日期：** {r['date']}  |  **股票：** {r['ticker']}")
            st.divider()
            st.markdown(r["content"])


# ─── 文档浏览 ──────────────────────────────────────────────────────────────────

def _render_browse() -> None:
    st.subheader("已索引文档")

    docs = _load_doc_list()

    if docs.empty:
        st.info("暂无已索引文档。请运行：`python -m quantmind.knowledge.indexer --ingest`")
        return

    # 筛选器
    col1, col2, col3 = st.columns(3)
    with col1:
        tickers = ["全部"] + sorted(docs["ticker"].dropna().unique().tolist())
        ticker_f = st.selectbox("股票代码", tickers, key="kb_ticker")
    with col2:
        doc_types = ["全部"] + sorted(docs["doc_type"].dropna().unique().tolist())
        type_f = st.selectbox("文档类型", doc_types, key="kb_type")
    with col3:
        date_range = st.date_input(
            "日期范围",
            value=[],
            key="kb_date",
        )

    filtered = docs.copy()
    if ticker_f != "全部":
        filtered = filtered[filtered["ticker"] == ticker_f]
    if type_f != "全部":
        filtered = filtered[filtered["doc_type"] == type_f]

    st.markdown(f"**共 {len(filtered)} 篇文档**")
    st.dataframe(
        filtered[["ticker", "doc_type", "date", "title", "chunks", "source"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


@st.cache_data(ttl=60)
def _load_doc_list() -> pd.DataFrame:
    """尝试从知识库读取文档列表，否则返回 Demo 数据."""
    try:
        from quantmind.knowledge.retriever import HybridRetriever
        retriever = HybridRetriever()
        docs = retriever.list_documents()
        if docs:
            return pd.DataFrame(docs)
    except Exception:
        pass

    # Demo 数据
    return pd.DataFrame([
        {"ticker": "600519.SH", "doc_type": "annual_report", "date": "2024-03-28",
         "title": "贵州茅台 2023 年年报", "chunks": 142, "source": "600519_2023_annual.txt"},
        {"ticker": "600519.SH", "doc_type": "research_note", "date": "2024-01-15",
         "title": "茅台 Q4 点评", "chunks": 38, "source": "600519_q4_2023.txt"},
        {"ticker": "000858.SZ", "doc_type": "annual_report", "date": "2024-04-10",
         "title": "五粮液 2023 年年报", "chunks": 118, "source": "000858_2023_annual.txt"},
        {"ticker": "000001.SZ", "doc_type": "quarterly_report", "date": "2024-04-30",
         "title": "平安银行 2024Q1 季报", "chunks": 67, "source": "000001_2024q1.txt"},
        {"ticker": "N/A", "doc_type": "industry_report", "date": "2024-01-05",
         "title": "白酒行业 2024 年展望", "chunks": 54, "source": "baijiu_2024_outlook.txt"},
        {"ticker": "N/A", "doc_type": "macro_report", "date": "2024-03-01",
         "title": "2024 年 A 股市场展望", "chunks": 89, "source": "astock_2024_outlook.txt"},
    ])


# ─── 统计 ──────────────────────────────────────────────────────────────────────

def _render_stats() -> None:
    st.subheader("知识库统计")
    docs = _load_doc_list()

    if docs.empty:
        st.info("暂无文档数据")
        return

    # 摘要指标
    total_docs = len(docs)
    total_chunks = int(docs["chunks"].sum()) if "chunks" in docs.columns else 0
    unique_tickers = docs["ticker"].nunique()
    doc_types = docs["doc_type"].nunique() if "doc_type" in docs.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("文档总数", total_docs)
    c2.metric("文本块总数", total_chunks)
    c3.metric("覆盖股票数", unique_tickers)
    c4.metric("文档类型数", doc_types)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**按文档类型分布**")
        if "doc_type" in docs.columns:
            type_counts = docs["doc_type"].value_counts()
            fig = go.Figure(go.Bar(
                x=type_counts.index,
                y=type_counts.values,
                marker_color="#58a6ff",
            ))
            fig.update_layout(
                height=260,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                font=dict(color="#c9d1d9"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**按股票文档数 Top-10**")
        if "ticker" in docs.columns:
            ticker_counts = docs[docs["ticker"] != "N/A"]["ticker"].value_counts().head(10)
            fig2 = go.Figure(go.Bar(
                x=ticker_counts.values,
                y=ticker_counts.index,
                orientation="h",
                marker_color="#3fb950",
            ))
            fig2.update_layout(
                height=260,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                font=dict(color="#c9d1d9"),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)
