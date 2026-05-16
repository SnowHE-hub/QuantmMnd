"""generate_investment_report.py — 单股票 RAG 投资分析报告 CLI.

用法（provider=none 模板版，无需 LLM）：
    python scripts/generate_investment_report.py \\
        --ticker 600519.SH \\
        --as-of 2024-12-31 \\
        --provider none \\
        --output reports/investment_report_600519_template.md \\
        --context-output reports/stock_context_600519_for_report.json

用法（LLM 版，需要对应 API key）：
    python scripts/generate_investment_report.py \\
        --ticker 600519.SH \\
        --as-of 2024-12-31 \\
        --provider dashscope \\
        --model qwen-plus \\
        --output reports/investment_report_600519_llm.md

API key 缺失时自动 fallback → provider=none。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_CHROMA = ".cache/chromadb"
_DEFAULT_COLLECTION = "default"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG Investment Report Generator — QuantMind"
    )
    p.add_argument("--ticker", required=True, help="股票代码，如 600519.SH")
    p.add_argument("--as-of", default="", help="数据截止日期 YYYY-MM-DD")
    p.add_argument(
        "--context-json",
        type=Path,
        default=None,
        help="已有 context JSON 路径（跳过检索直接读取）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown 报告输出路径",
    )
    p.add_argument(
        "--context-output",
        type=Path,
        default=None,
        help="context JSON 输出路径（保存检索结果）",
    )
    p.add_argument(
        "--provider",
        default="none",
        choices=["none", "dashscope", "openai", "anthropic", "deepseek"],
        help="LLM provider，默认 none（模板生成）",
    )
    p.add_argument("--model", default=None, help="LLM 模型名称（可选）")
    p.add_argument("--language", default="zh", help="报告语言（当前仅支持 zh）")
    p.add_argument("--top-k-news", type=int, default=5)
    p.add_argument("--top-k-reports", type=int, default=5)
    p.add_argument("--top-k-per-doc-type", type=int, default=3)
    p.add_argument("--collection-name", default=_DEFAULT_COLLECTION)
    p.add_argument("--chroma-dir", default=_DEFAULT_CHROMA)
    return p.parse_args()


# ── Context 检索 ─────────────────────────────────────────────────────────────


def retrieve_context(
    args: argparse.Namespace,
    *,
    silent: bool = False,
    on_empty_kb: str = "exit",
) -> dict:
    """调用 retrieve_stock_context 逻辑，生成 context dict.

    Args:
        silent:      为 True 时不 print（供批量脚本使用）。
        on_empty_kb: ``exit`` — KB 为空时打印错误并 ``SystemExit(2)``；
                     ``empty`` — 返回空检索结构，不退出（供流水线降级）。
    """
    from datetime import date as date_cls

    from scripts.retrieve_stock_context import (
        SNAPSHOT_DOC_TYPES,
        retrieve_market_context,
        retrieve_news,
        retrieve_reports,
        retrieve_snapshot_by_doc_type,
    )
    from quantmind.kb.retriever import HybridRetriever

    def _log(msg: str) -> None:
        if not silent:
            print(msg)

    ticker = args.ticker.strip()
    as_of: date_cls | None = None
    if args.as_of:
        as_of = date_cls.fromisoformat(args.as_of)

    retriever = HybridRetriever(
        collection_name=args.collection_name,
        chroma_dir=args.chroma_dir,
        use_reranker=False,
    )
    total = retriever.count()
    _log(f"[generate_investment_report] KB collection_count={total}")

    empty_ctx_base: dict = {
        "ticker": ticker,
        "as_of": as_of.isoformat() if as_of else "",
        "collection_count": 0,
        "news_context": [],
        "news_count": 0,
        "report_context": [],
        "report_count": 0,
        "market_context": [],
        "market_count": 0,
    }
    for dt in SNAPSHOT_DOC_TYPES:
        empty_ctx_base[f"snapshot_{dt}"] = []
        empty_ctx_base[f"snapshot_{dt}_count"] = 0

    if total == 0:
        if on_empty_kb == "empty":
            return dict(empty_ctx_base)
        print("ERROR: Chroma collection is empty.")
        raise SystemExit(2)

    news_items = retrieve_news(retriever, ticker, as_of, args.top_k_news)
    report_items = retrieve_reports(retriever, ticker, as_of, args.top_k_reports)

    snapshot_results = {}
    for dt in SNAPSHOT_DOC_TYPES:
        items = retrieve_snapshot_by_doc_type(retriever, ticker, as_of, dt, args.top_k_per_doc_type)
        snapshot_results[dt] = items

    market_items = retrieve_market_context(retriever, as_of)

    ctx: dict = {
        "ticker": ticker,
        "as_of": as_of.isoformat() if as_of else "",
        "collection_count": total,
        "news_context": news_items,
        "news_count": len(news_items),
        "report_context": report_items,
        "report_count": len(report_items),
        "market_context": market_items,
        "market_count": len(market_items),
    }
    for dt in SNAPSHOT_DOC_TYPES:
        ctx[f"snapshot_{dt}"] = snapshot_results[dt]
        ctx[f"snapshot_{dt}_count"] = len(snapshot_results[dt])

    return ctx


# ── 主流程 ───────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    ticker = args.ticker.strip()

    print(f"[generate_investment_report] ticker={ticker}  as_of={args.as_of or '(none)'}")
    print(f"[generate_investment_report] provider={args.provider}  model={args.model or '(default)'}")

    # ── Step 1: 获取 context ──────────────────────────────────────────────────
    if args.context_json and args.context_json.exists():
        print(f"[generate_investment_report] 读取已有 context: {args.context_json}")
        with open(args.context_json, encoding="utf-8") as f:
            context = json.load(f)
    else:
        print("[generate_investment_report] 调用 retrieve_stock_context 生成 context ...")
        context = retrieve_context(args)

    # ── Step 2: 保存 context JSON ─────────────────────────────────────────────
    if args.context_output:
        ctx_path = Path(args.context_output)
        ctx_path.parent.mkdir(parents=True, exist_ok=True)
        ctx_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[generate_investment_report] context JSON → {ctx_path}")

    # ── Step 3: 生成报告 ─────────────────────────────────────────────────────
    from quantmind.agents.rag_report_agent import RAGReportAgent

    agent = RAGReportAgent()
    result = agent.generate_report(
        context=context,
        language=args.language,
        use_llm=(args.provider != "none"),
        llm_provider=args.provider,
        llm_model=args.model,
    )

    # ── Step 4: 打印摘要 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"REPORT SUMMARY  ticker={result['ticker']}  as_of={result['as_of']}")
    print("=" * 60)
    print(f"  llm_used        : {result['llm_used']}")
    print(f"  provider        : {result['provider']}")
    print(f"  model           : {result['model']}")
    cov = result.get("data_coverage", {})
    print(f"  news_count      : {cov.get('news_count', 0)}")
    print(f"  report_count    : {cov.get('report_count', 0)}")
    for key in [
        "snapshot_company_profile_count",
        "snapshot_latest_market_metrics_count",
        "snapshot_financial_indicator_summary_count",
        "snapshot_northbound_summary_count",
        "snapshot_margin_summary_count",
        "market_context_count",
    ]:
        print(f"  {key:<40}: {cov.get(key, 0)}")

    warns = result.get("warnings", [])
    if warns:
        print(f"\n  WARNINGS ({len(warns)}):")
        for w in warns:
            print(f"    ⚠️  {w}")
    print("=" * 60)

    # ── Step 5: 写出 Markdown 报告 ────────────────────────────────────────────
    out_path: Path | None = args.output
    if out_path is None:
        safe = ticker.replace(".", "_")
        out_path = Path(f"reports/investment_report_{safe}_template.md")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result["report_markdown"], encoding="utf-8")
    print(f"\n[generate_investment_report] Markdown 报告 → {out_path}")

    md_len = len(result["report_markdown"])
    print(f"[generate_investment_report] 报告长度：{md_len} 字符")
    print("[generate_investment_report] DONE")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
