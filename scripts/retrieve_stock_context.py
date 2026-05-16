"""retrieve_stock_context.py — 面向投资分析的结构化 KB 分层检索工具.

按 source_type / doc_type 分层检索，保证 news、reports、snapshot 各类材料都能拿到。

用法示例：
    python scripts/retrieve_stock_context.py \\
        --ticker 600519.SH \\
        --as-of 2024-12-31 \\
        --top-k-news 5 \\
        --top-k-reports 5 \\
        --top-k-per-doc-type 3 \\
        --output reports/stock_context_600519.md \\
        --json-output reports/stock_context_600519.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_CHROMA = ".cache/chromadb"
_DEFAULT_COLLECTION = "default"

SNAPSHOT_DOC_TYPES = (
    "company_profile",
    "latest_market_metrics",
    "financial_indicator_summary",
    "northbound_summary",
    "margin_summary",
)


def _preview(text: str, max_len: int = 150) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Structured KB retrieval for investment analysis"
    )
    p.add_argument("--ticker", required=True, help="Stock ticker, e.g. 600519.SH")
    p.add_argument("--as-of", default="", help="PIT cutoff date YYYY-MM-DD (optional)")
    p.add_argument("--top-k-news", type=int, default=5)
    p.add_argument("--top-k-reports", type=int, default=5)
    p.add_argument("--top-k-per-doc-type", type=int, default=3)
    p.add_argument("--output", type=Path, default=None, help="Markdown output path")
    p.add_argument("--json-output", type=Path, default=None, help="JSON output path")
    p.add_argument("--collection-name", default=_DEFAULT_COLLECTION)
    p.add_argument("--chroma-dir", default=_DEFAULT_CHROMA)
    return p.parse_args()


# ── 检索辅助 ────────────────────────────────────────────────────────────────


def _post_filter(
    results: list[Any],
    ticker: str | None = None,
    source_type: str | None = None,
    doc_type: str | None = None,
) -> list[Any]:
    """在 Python 层对检索结果做精确过滤（补充 HybridRetriever 的 filter 能力不足）."""
    out = []
    for r in results:
        meta = r.metadata or {}
        if ticker:
            t = (meta.get("ticker") or "").strip()
            if t != ticker:
                continue
        if source_type:
            st = (meta.get("source_type") or "").strip()
            if st and st != source_type:
                continue
        if doc_type:
            dt = (meta.get("doc_type") or "").strip()
            if dt and dt != doc_type:
                continue
            if not dt:
                continue
        out.append(r)
    return out


def _result_dict(rank: int, r: Any) -> dict[str, Any]:
    """将 RetrievalResult 序列化为字典（JSON 输出保留完整 text）."""
    meta = r.metadata or {}
    return {
        "rank": rank,
        "score": round(float(r.score), 6),
        "ticker": meta.get("ticker", ""),
        "source_type": meta.get("source_type", ""),
        "doc_type": meta.get("doc_type", ""),
        "as_of": meta.get("as_of", ""),
        "published_date": meta.get("published_date", ""),
        "source": meta.get("source", ""),
        "title": meta.get("title", ""),
        "text_preview": _preview(r.text, 150),
        "text": r.text,
    }


# ── 各层检索 ────────────────────────────────────────────────────────────────


def _ticker_variants(ticker: str) -> list[str]:
    """生成 ticker 的可能变体（带后缀/不带后缀），用于兼容 KB 中格式不统一的问题."""
    variants = [ticker]
    code6 = ticker.split(".")[0] if "." in ticker else ticker
    if code6 != ticker:
        variants.append(code6)
    return variants


def retrieve_news(
    retriever: Any,
    ticker: str,
    as_of: date | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """检索新闻。

    注意：KB 中的新闻由 AkShare 实时抓取，published_date 为抓取时的当前日期，
    因此 PIT 过滤（as_of=2024-12-31）会将所有新闻过滤掉。
    本函数不传 as_of，直接拉取所有与 ticker 相关的新闻。
    """
    all_results: list[Any] = []
    for variant in _ticker_variants(ticker):
        where: dict = {
            "$and": [
                {"ticker": {"$eq": variant}},
                {"source_type": {"$eq": "news"}},
            ]
        }
        # news 不做 PIT 过滤（KB 中新闻时间为抓取时间，非历史时间）
        items = retriever.fetch_by_metadata(where=where, as_of=None, limit=top_k * 5)
        all_results.extend(items)

    # dedup by source
    seen: set[str] = set()
    deduped: list[Any] = []
    for r in all_results:
        key = (r.metadata or {}).get("source", r.text[:80])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    deduped = deduped[:top_k]
    return [_result_dict(i + 1, r) for i, r in enumerate(deduped)]


def retrieve_reports(
    retriever: Any,
    ticker: str,
    as_of: date | None,
    top_k: int,
) -> list[dict[str, Any]]:
    """检索研究报告。

    KB 中 report ticker 可能为 '600519'（无后缀），需兼容匹配。
    """
    all_results: list[Any] = []
    for variant in _ticker_variants(ticker):
        for st in ("report", "reports"):
            where: dict = {
                "$and": [
                    {"ticker": {"$eq": variant}},
                    {"source_type": {"$eq": st}},
                ]
            }
            items = retriever.fetch_by_metadata(where=where, as_of=as_of, limit=top_k * 5)
            all_results.extend(items)

    # dedup by source
    seen: set[str] = set()
    deduped: list[Any] = []
    for r in all_results:
        key = (r.metadata or {}).get("source", r.text[:80])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    deduped.sort(key=lambda x: (x.metadata.get("published_date") or ""), reverse=True)
    deduped = deduped[:top_k]
    return [_result_dict(i + 1, r) for i, r in enumerate(deduped)]


def retrieve_snapshot_by_doc_type(
    retriever: Any,
    ticker: str,
    as_of: date | None,
    doc_type: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """优先用 metadata WHERE filter 精确拉取，再用向量检索补充.

    company_profile 在向量空间中可能排名极低（公司特定词汇拉低相似度），
    因此必须走 metadata 直接拉取路径。
    """
    # 主路径：metadata WHERE 直接拉取（精确，不依赖向量相似度）
    where: dict = {
        "$and": [
            {"ticker": {"$eq": ticker}},
            {"doc_type": {"$eq": doc_type}},
            {"source_type": {"$eq": "snapshot"}},
        ]
    }
    direct = retriever.fetch_by_metadata(where=where, as_of=as_of, limit=top_k * 5)
    if direct:
        direct = direct[:top_k]
        return [_result_dict(i + 1, r) for i, r in enumerate(direct)]

    # 备用路径：向量检索 + 后过滤（候选池扩大至全集）
    doc_type_queries: dict[str, str] = {
        "company_profile": f"{ticker} 公司概览 行业 上市",
        "latest_market_metrics": f"{ticker} 市场指标 估值 市值 PE PB",
        "financial_indicator_summary": f"{ticker} 财务指标 ROE 净利润率 资产负债",
        "northbound_summary": f"{ticker} 北向持股 陆股通 外资持仓",
        "margin_summary": f"{ticker} 融资融券 融资余额 两融",
    }
    query = doc_type_queries.get(doc_type, f"{ticker} {doc_type}")
    total = retriever.count()
    raw = retriever.retrieve(
        query,
        top_k=min(total, 500),
        as_of=as_of,
        ticker_filter=ticker,
        source_type_filter="snapshot",
    )
    filtered = _post_filter(raw, ticker=ticker, source_type="snapshot", doc_type=doc_type)
    filtered = filtered[:top_k]
    return [_result_dict(i + 1, r) for i, r in enumerate(filtered)]


def retrieve_market_context(
    retriever: Any,
    as_of: date | None,
    top_k: int = 1,
) -> list[dict[str, Any]]:
    """market_index_context 只有一个 ticker=__MARKET__，直接用 metadata 拉取."""
    where: dict = {
        "$and": [
            {"ticker": {"$eq": "__MARKET__"}},
            {"doc_type": {"$eq": "market_index_context"}},
        ]
    }
    direct = retriever.fetch_by_metadata(where=where, as_of=as_of, limit=top_k * 3)
    if direct:
        direct = direct[:top_k]
        return [_result_dict(i + 1, r) for i, r in enumerate(direct)]

    # 备用：向量检索
    query = "主要指数 市场环境 沪深300 中证500 创业板"
    raw = retriever.retrieve(
        query,
        top_k=top_k * 10,
        as_of=as_of,
        ticker_filter="__MARKET__",
    )
    filtered = _post_filter(raw, ticker="__MARKET__", doc_type="market_index_context")
    filtered = filtered[:top_k]
    return [_result_dict(i + 1, r) for i, r in enumerate(filtered)]


# ── Markdown 渲染 ────────────────────────────────────────────────────────────


def _section_md(
    title: str,
    items: list[dict[str, Any]],
    returned_count: int,
    note: str = "",
) -> str:
    lines = [f"## {title}", f"returned_count: {returned_count}"]
    if note:
        lines.append(f"> {note}")
    if not items:
        lines.append("\n_（无结果）_\n")
        return "\n".join(lines) + "\n"
    lines.append("")
    for item in items:
        lines.append(f"### rank {item['rank']} | score={item['score']:.4f}")
        lines.append(f"- **ticker**: {item['ticker']}")
        lines.append(f"- **source_type**: {item['source_type']}")
        lines.append(f"- **doc_type**: {item['doc_type'] or '(absent)'}")
        lines.append(f"- **as_of**: {item['as_of'] or item['published_date']}")
        lines.append(f"- **source**: {item['source']}")
        lines.append(f"- **text_preview**: {item['text_preview']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_markdown(
    ticker: str,
    as_of: date | None,
    ctx: dict[str, Any],
) -> str:
    lines = [
        f"# Stock Context: {ticker}",
        f"as_of: {as_of or '(none)'}",
        "",
    ]

    lines.append(_section_md("1. News Context", ctx["news_context"], ctx["news_count"]))
    lines.append(_section_md("2. Report Context", ctx["report_context"], ctx["report_count"]))

    lines.append("## 3. Snapshot Context\n")
    for dt in SNAPSHOT_DOC_TYPES:
        key = f"snapshot_{dt}"
        items = ctx.get(key, [])
        cnt = ctx.get(f"{key}_count", 0)
        lines.append(_section_md(f"3.{SNAPSHOT_DOC_TYPES.index(dt) + 1} {dt}", items, cnt))

    lines.append(
        _section_md("4. Market Index Context", ctx["market_context"], ctx["market_count"])
    )
    return "\n".join(lines)


# ── 主流程 ───────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    ticker = args.ticker.strip()
    as_of: date | None = None
    if args.as_of:
        as_of = date.fromisoformat(args.as_of)

    print(f"[retrieve_stock_context] ticker={ticker}  as_of={as_of}")
    print(f"[retrieve_stock_context] collection={args.collection_name}  chroma_dir={args.chroma_dir}")

    from quantmind.kb.retriever import HybridRetriever

    retriever = HybridRetriever(
        collection_name=args.collection_name,
        chroma_dir=args.chroma_dir,
        use_reranker=False,
    )
    total_docs = retriever.count()
    print(f"[retrieve_stock_context] collection_count={total_docs}")
    if total_docs == 0:
        print("ERROR: Chroma collection is empty. Check chroma_dir and collection_name.")
        raise SystemExit(2)

    # ── 1. news ──────────────────────────────────────────────────────────────
    print(f"\n[retrieve_stock_context] === news (top_k={args.top_k_news}) ===")
    news_items = retrieve_news(retriever, ticker, as_of, args.top_k_news)
    print(f"[retrieve_stock_context] news returned_count={len(news_items)}")

    # ── 2. reports ───────────────────────────────────────────────────────────
    print(f"\n[retrieve_stock_context] === reports (top_k={args.top_k_reports}) ===")
    report_items = retrieve_reports(retriever, ticker, as_of, args.top_k_reports)
    print(f"[retrieve_stock_context] reports returned_count={len(report_items)}")

    # ── 3. snapshot by doc_type ───────────────────────────────────────────────
    snapshot_results: dict[str, list[dict[str, Any]]] = {}
    for dt in SNAPSHOT_DOC_TYPES:
        print(f"\n[retrieve_stock_context] === snapshot/{dt} (top_k={args.top_k_per_doc_type}) ===")
        items = retrieve_snapshot_by_doc_type(retriever, ticker, as_of, dt, args.top_k_per_doc_type)
        snapshot_results[dt] = items
        print(f"[retrieve_stock_context] snapshot/{dt} returned_count={len(items)}")

    # ── 4. market context ─────────────────────────────────────────────────────
    print("\n[retrieve_stock_context] === market_index_context ===")
    market_items = retrieve_market_context(retriever, as_of)
    print(f"[retrieve_stock_context] market_index_context returned_count={len(market_items)}")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    ctx: dict[str, Any] = {
        "ticker": ticker,
        "as_of": as_of.isoformat() if as_of else "",
        "collection_count": total_docs,
        "news_context": news_items,
        "news_count": len(news_items),
        "report_context": report_items,
        "report_count": len(report_items),
        "market_context": market_items,
        "market_count": len(market_items),
    }
    for dt in SNAPSHOT_DOC_TYPES:
        key = f"snapshot_{dt}"
        ctx[key] = snapshot_results[dt]
        ctx[f"{key}_count"] = len(snapshot_results[dt])

    # ── 打印摘要 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"SUMMARY  ticker={ticker}  as_of={as_of or '(none)'}")
    print("=" * 60)
    print(f"  news                        : {ctx['news_count']}")
    print(f"  report                      : {ctx['report_count']}")
    for dt in SNAPSHOT_DOC_TYPES:
        cnt = ctx[f"snapshot_{dt}_count"]
        status = "OK" if cnt > 0 else "EMPTY"
        print(f"  snapshot/{dt:<28}: {cnt}  [{status}]")
    print(f"  market_index_context        : {ctx['market_count']}")
    print("=" * 60)

    # ── Markdown 输出 ────────────────────────────────────────────────────────
    if args.output:
        md_path = Path(args.output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_text = build_markdown(ticker, as_of, ctx)
        md_path.write_text(md_text, encoding="utf-8")
        print(f"\n[retrieve_stock_context] Markdown written → {md_path}")

    # ── JSON 输出 ────────────────────────────────────────────────────────────
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[retrieve_stock_context] JSON written → {json_path}")

    print("\n[retrieve_stock_context] DONE")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
