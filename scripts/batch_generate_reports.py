"""批量生成多只股票投资分析报告。

用法见 parse_args()；依赖 generate_investment_report.retrieve_context 与 RAGReportAgent。
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_CHROMA = ".cache/chromadb"
_DEFAULT_COLLECTION = "default"

from scripts.retrieve_stock_context import SNAPSHOT_DOC_TYPES  # noqa: E402

# 用于 batch_report_summary.md 行业标注（覆盖本批 tickers；未知则「其他」）
INDUSTRY_BY_TICKER: dict[str, str] = {
    "600519.SH": "白酒（消费）",
    "300750.SZ": "新能源电池",
    "000858.SZ": "白酒（消费）",
    "600036.SH": "银行",
    "601318.SH": "保险",
    "300059.SZ": "金融科技",
    "600276.SH": "医药",
    "000333.SZ": "家电（消费）",
    "600887.SH": "乳制品（消费）",
    "601012.SH": "光伏新能源",
    "600030.SH": "券商",
    "000002.SZ": "地产",
    "601899.SH": "有色金属",
    "002475.SZ": "消费电子",
}


def _safe_filename_ticker(ticker: str) -> str:
    return ticker.replace(".", "_")


def parse_ticker_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="*", default=[], help="股票代码，空格分隔")
    p.add_argument("--ticker-file", type=Path, default=None, help="每行一个 ticker，支持 # 注释与空行")
    p.add_argument("--as-of", default="2024-12-31")
    p.add_argument("--provider", default="dashscope", help="LLM provider；none=模板")
    p.add_argument("--model", default="qwen-plus")
    p.add_argument("--output-dir", type=Path, default=Path("reports/batch"))
    p.add_argument("--context-dir", type=Path, default=Path("reports/batch/contexts"))
    p.add_argument("--language", default="zh")
    p.add_argument("--top-k-news", type=int, default=5)
    p.add_argument("--top-k-reports", type=int, default=5)
    p.add_argument("--top-k-per-doc-type", type=int, default=3)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--context-only",
        action="store_true",
        help="仅重新检索并写入 context JSON（不调用 LLM、不覆盖已有报告 Markdown）",
    )
    p.add_argument("--collection-name", default=_DEFAULT_COLLECTION)
    p.add_argument("--chroma-dir", default=_DEFAULT_CHROMA)
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("reports/batch/batch_summary.json"),
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=Path("reports/batch/batch_summary.md"),
    )
    p.add_argument(
        "--batch-report-md",
        type=Path,
        default=Path("reports/batch/batch_report_summary.md"),
    )
    return p.parse_args()


def build_arg_namespace(
    ticker: str,
    as_of: str,
    *,
    top_k_news: int,
    top_k_reports: int,
    top_k_per_doc_type: int,
    collection_name: str,
    chroma_dir: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        ticker=ticker,
        as_of=as_of or "",
        top_k_news=top_k_news,
        top_k_reports=top_k_reports,
        top_k_per_doc_type=top_k_per_doc_type,
        collection_name=collection_name,
        chroma_dir=chroma_dir,
    )


def retrieve_context_for_batch(ns: SimpleNamespace) -> dict[str, Any]:
    from scripts.generate_investment_report import retrieve_context

    return retrieve_context(ns)


@dataclass
class TickerJobResult:
    ticker: str
    status: str  # success | failed | skipped
    llm_used: bool = False
    llm_attempted: bool = False
    news_count: int = 0
    report_count: int = 0
    snapshot_counts: dict[str, int] = field(default_factory=dict)
    snapshot_five_ok: bool = False
    market_count: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def _process_one_ticker(
    ticker: str,
    *,
    as_of: str,
    provider: str,
    model: str | None,
    language: str,
    output_dir: Path,
    context_dir: Path,
    skip_existing: bool,
    context_only: bool,
    top_k_news: int,
    top_k_reports: int,
    top_k_per_doc_type: int,
    collection_name: str,
    chroma_dir: str,
    retrieve_fn: Callable[[SimpleNamespace], dict[str, Any]] | None = None,
    agent_factory: Callable[[], Any] | None = None,
) -> TickerJobResult:
    retrieve_fn = retrieve_fn or retrieve_context_for_batch
    safe = _safe_filename_ticker(ticker)
    report_path = output_dir / f"{safe}_report.md"
    ctx_path = context_dir / f"{safe}_context.json"

    if context_only:
        ns = build_arg_namespace(
            ticker,
            as_of,
            top_k_news=top_k_news,
            top_k_reports=top_k_reports,
            top_k_per_doc_type=top_k_per_doc_type,
            collection_name=collection_name,
            chroma_dir=chroma_dir,
        )
        try:
            context = retrieve_fn(ns)
        except Exception as e:
            return TickerJobResult(
                ticker=ticker,
                status="failed",
                error=f"context 检索失败: {e}",
            )
        context_dir.mkdir(parents=True, exist_ok=True)
        ctx_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
        snap_counts = {dt: int(context.get(f"snapshot_{dt}_count") or 0) for dt in SNAPSHOT_DOC_TYPES}
        five_ok = all(snap_counts[dt] > 0 for dt in SNAPSHOT_DOC_TYPES)
        news_c = int(context.get("news_count") or 0)
        report_c = int(context.get("report_count") or 0)
        mkt = int(context.get("market_count") or 0)
        return TickerJobResult(
            ticker=ticker,
            status="success",
            llm_used=False,
            llm_attempted=False,
            news_count=news_c,
            report_count=report_c,
            snapshot_counts=snap_counts,
            snapshot_five_ok=five_ok,
            market_count=mkt,
        )

    if skip_existing and report_path.is_file():
        # 仍读取 context 路径若存在，用于汇总计数；否则填 zeros
        snap: dict[str, int] = {dt: 0 for dt in SNAPSHOT_DOC_TYPES}
        news_c = report_c = mkt = 0
        w: list[str] = []
        if ctx_path.is_file():
            try:
                ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                news_c = int(ctx.get("news_count") or 0)
                report_c = int(ctx.get("report_count") or 0)
                mkt = int(ctx.get("market_count") or 0)
                for dt in SNAPSHOT_DOC_TYPES:
                    snap[dt] = int(ctx.get(f"snapshot_{dt}_count") or 0)
            except Exception:
                pass
        five_ok = all(snap[dt] > 0 for dt in SNAPSHOT_DOC_TYPES)
        return TickerJobResult(
            ticker=ticker,
            status="skipped",
            llm_used=False,
            llm_attempted=False,
            news_count=news_c,
            report_count=report_c,
            snapshot_counts=snap,
            snapshot_five_ok=five_ok,
            market_count=mkt,
            warnings=w,
        )

    ns = build_arg_namespace(
        ticker,
        as_of,
        top_k_news=top_k_news,
        top_k_reports=top_k_reports,
        top_k_per_doc_type=top_k_per_doc_type,
        collection_name=collection_name,
        chroma_dir=chroma_dir,
    )

    try:
        context = retrieve_fn(ns)
    except Exception as e:
        return TickerJobResult(
            ticker=ticker,
            status="failed",
            error=f"context 检索失败: {e}",
        )

    context_dir.mkdir(parents=True, exist_ok=True)
    ctx_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")

    snap_counts = {dt: int(context.get(f"snapshot_{dt}_count") or 0) for dt in SNAPSHOT_DOC_TYPES}
    five_ok = all(snap_counts[dt] > 0 for dt in SNAPSHOT_DOC_TYPES)
    news_c = int(context.get("news_count") or 0)
    report_c = int(context.get("report_count") or 0)
    mkt = int(context.get("market_count") or 0)

    use_llm = provider.strip().lower() != "none"
    if agent_factory:
        agent = agent_factory()
    else:
        from quantmind.agents.rag_report_agent import RAGReportAgent

        agent = RAGReportAgent()

    llm_attempted = use_llm
    try:
        result = agent.generate_report(
            context=context,
            language=language,
            use_llm=use_llm,
            llm_provider=provider if use_llm else None,
            llm_model=model if use_llm else None,
        )
    except Exception as e:
        return TickerJobResult(
            ticker=ticker,
            status="failed",
            news_count=news_c,
            report_count=report_c,
            snapshot_counts=snap_counts,
            snapshot_five_ok=five_ok,
            market_count=mkt,
            error=f"报告生成失败: {e}",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result["report_markdown"], encoding="utf-8")

    warns = list(result.get("warnings") or [])
    return TickerJobResult(
        ticker=ticker,
        status="success",
        llm_used=bool(result.get("llm_used")),
        llm_attempted=llm_attempted,
        news_count=news_c,
        report_count=report_c,
        snapshot_counts=snap_counts,
        snapshot_five_ok=five_ok,
        market_count=mkt,
        warnings=warns,
    )


def _result_to_dict(r: TickerJobResult) -> dict[str, Any]:
    return {
        "ticker": r.ticker,
        "status": r.status,
        "llm_used": r.llm_used,
        "llm_attempted": r.llm_attempted,
        "news_count": r.news_count,
        "report_count": r.report_count,
        "market_count": r.market_count,
        "snapshot_counts": r.snapshot_counts,
        "snapshot_five_doc_types_complete": r.snapshot_five_ok,
        "error": r.error,
        "warnings": r.warnings,
    }


def write_summary_files(
    results: list[TickerJobResult],
    *,
    summary_json: Path,
    summary_md: Path,
    batch_report_md: Path,
    as_of: str,
    provider: str,
    model: str,
) -> dict[str, Any]:
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    attempted = sum(1 for r in results if r.llm_attempted)
    llm_ok = sum(1 for r in results if r.llm_used)

    summary = {
        "as_of": as_of,
        "provider": provider,
        "model": model,
        "counts": {
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "llm_attempted": attempted,
            "llm_succeeded": llm_ok,
        },
        "per_ticker": [_result_to_dict(r) for r in results],
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Batch report — summary",
        "",
        f"- as_of: `{as_of}`",
        f"- provider: `{provider}`  model: `{model}`",
        "",
        "## Counts",
        "",
        f"- success: **{success}**",
        f"- failed: **{failed}**",
        f"- skipped: **{skipped}**",
        f"- llm_attempted: **{attempted}**",
        f"- llm_succeeded: **{llm_ok}**",
        "",
        "## Per ticker",
        "",
        "| ticker | status | llm_used | news | reports | 5×snapshot | error |",
        "| --- | --- | --- | ---: | ---: | :--- | --- |",
    ]
    for r in results:
        err = (r.error or "").replace("|", "\\|")[:120]
        lines.append(
            f"| {r.ticker} | {r.status} | {r.llm_used} | {r.news_count} | {r.report_count} | "
            f"{'yes' if r.snapshot_five_ok else 'no'} | {err} |"
        )
    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    # 扩展 narrative：batch_report_summary.md
    empty_snap_stats: dict[str, int] = {dt: 0 for dt in SNAPSHOT_DOC_TYPES}
    for r in results:
        if r.status != "success":
            continue
        for dt in SNAPSHOT_DOC_TYPES:
            if r.snapshot_counts.get(dt, 0) == 0:
                empty_snap_stats[dt] += 1

    industries = [f"- `{t}` — {INDUSTRY_BY_TICKER.get(t, '其他')}" for t in sorted({r.ticker for r in results})]
    llm_rate = f"{100.0 * llm_ok / attempted:.1f}%" if attempted else "n/a"
    br_lines = [
        "# 批量投资报告 — 总览（自动生成）",
        "",
        "## 1. 覆盖股票与行业",
        "",
        *industries,
        "",
        "## 2. 运行统计",
        "",
        f"- 成功 / 失败 / 跳过: **{success}** / **{failed}** / **{skipped}**",
        f"- LLM 尝试次数: **{attempted}**；LLM 实际生成正文 (`llm_used=true`): **{llm_ok}**",
        f"- **LLM 调用成功率**（相对尝试次数）: **{llm_rate}**",
        f"- provider=`{provider}` model=`{model}` as_of=`{as_of}`",
        "",
        "## 3. 数据覆盖摘要（逐股）",
        "",
        "| ticker | news | 研报 | 5×snapshot | llm_used |",
        "| --- | ---: | ---: | :--- | :--- |",
    ]
    for r in results:
        br_lines.append(
            f"| {r.ticker} | {r.news_count} | {r.report_count} | "
            f"{'yes' if r.snapshot_five_ok else 'no'} | {r.llm_used} |"
        )
    br_lines.extend([
        "",
        "各 `snapshot_*` 条数见 `batch_summary.json` 的 `snapshot_counts`。**market_context** 每成功股约 1 条。",
        "",
        "### 快照五类在成功样本中的空缺计数",
        "",
    ])
    for dt, c in empty_snap_stats.items():
        br_lines.append(f"- `{dt}` 为空出现的股票数: **{c}**")
    br_lines.extend([
        "",
        "## 4. 共性问题与说明",
        "",
        "- 若某 ticker `snapshot_five_doc_types_complete=false`，多为 KB 中该类型 chunk 缺失或为 0。",
        "- **news / report 大量为 0**：多为 KB 未入库对应舆情或研报，而非脚本错误。",
        "- **DashScope 超时 / 连接错误**时 `llm_used=false` 但任务仍 `success`（模板回退）。",
        "- 新闻为抓取时间语义，非严格历史 PIT。",
        "",
        "## 5. 下一阶段建议",
        "",
        "- 扩充 KB 研报与新闻覆盖，或 Agent 侧并行联网工具。",
        "- LLM 调用增加重试/backoff，降低超时导致的模板回退比例。",
        "- Investment Agent 侧可增加按 `doc_type` 的工具检索，减轻混合排序偏置。",
        "- 批量 context 已写入 `reports/batch/contexts/`，可只重跑 LLM（`--skip-existing` + 删除 `llm/` 下报告）降低成本。",
        "",
    ])
    batch_report_md.write_text("\n".join(br_lines), encoding="utf-8")

    return summary


def collect_tickers(args: argparse.Namespace) -> list[str]:
    if args.skip_existing and args.context_only:
        raise SystemExit("--skip-existing 与 --context-only 语义冲突")
    if args.ticker_file and args.tickers:
        raise SystemExit("请只指定 --ticker-file 或 --tickers 之一")
    if args.ticker_file:
        if not args.ticker_file.is_file():
            raise SystemExit(f"文件不存在: {args.ticker_file}")
        return parse_ticker_file(args.ticker_file)
    if args.tickers:
        return [t.strip() for t in args.tickers if t.strip()]
    raise SystemExit("必须提供 --tickers 或 --ticker-file")


def run_batch(
    args: argparse.Namespace | None = None,
    *,
    retrieve_fn: Callable[[SimpleNamespace], dict[str, Any]] | None = None,
    agent_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    args = args or parse_args()
    tickers = collect_tickers(args)
    if not tickers:
        raise SystemExit("ticker 列表为空")

    as_of = args.as_of.strip()
    if as_of:
        date.fromisoformat(as_of)

    workers = max(1, int(args.workers))
    results: list[TickerJobResult] = []

    def job(t: str) -> TickerJobResult:
        return _process_one_ticker(
            t,
            as_of=as_of,
            provider=args.provider,
            model=args.model,
            language=args.language,
            output_dir=args.output_dir,
            context_dir=args.context_dir,
            skip_existing=args.skip_existing,
            context_only=args.context_only,
            top_k_news=args.top_k_news,
            top_k_reports=args.top_k_reports,
            top_k_per_doc_type=args.top_k_per_doc_type,
            collection_name=args.collection_name,
            chroma_dir=args.chroma_dir,
            retrieve_fn=retrieve_fn,
            agent_factory=agent_factory,
        )

    total = len(tickers)
    if workers == 1:
        for i, t in enumerate(tickers, start=1):
            r = job(t)
            results.append(r)
            print(f"[{i}/{total}] {t}  {r.status}" + (f"  err={r.error}" if r.error else ""))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(job, t): t for t in tickers}
            done = 0
            for fut in as_completed(fut_map):
                t = fut_map[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = TickerJobResult(ticker=t, status="failed", error=str(e))
                results.append(r)
                done += 1
                print(f"[{done}/{total}] {t}  {r.status}" + (f"  err={r.error}" if r.error else ""))
        # 按输入顺序排序
        order = {t: i for i, t in enumerate(tickers)}
        results.sort(key=lambda x: order.get(x.ticker, 999))

    summ = write_summary_files(
        results,
        summary_json=args.summary_json,
        summary_md=args.summary_md,
        batch_report_md=args.batch_report_md,
        as_of=as_of,
        provider=args.provider,
        model=args.model,
    )

    print("")
    print("=" * 60)
    print(
        f"BATCH DONE  success={summ['counts']['success']}  failed={summ['counts']['failed']}  "
        f"skipped={summ['counts']['skipped']}"
    )
    print(f"summary JSON → {args.summary_json}")
    print(f"summary MD   → {args.summary_md}")
    print(f"report MD    → {args.batch_report_md}")
    print("=" * 60)
    return summ


def main() -> None:
    run_batch(parse_args())


if __name__ == "__main__":
    main()
