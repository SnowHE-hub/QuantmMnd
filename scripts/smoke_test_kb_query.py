"""Smoke test: query KB via HybridRetriever (same defaults as build_kb.py).

Does not print secrets or full document bodies.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_CHROMA = ".cache/chromadb"
_DEFAULT_COLLECTION = "default"


def _preview(text: str, max_len: int = 120) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KB retrieval smoke test (HybridRetriever)")
    p.add_argument("--query", required=True)
    p.add_argument("--ticker", default="", help="If set, passed as ticker_filter (e.g. 600519.SH)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--collection-name", default=_DEFAULT_COLLECTION)
    p.add_argument("--chroma-dir", default=_DEFAULT_CHROMA)
    p.add_argument("--as-of", default="", help="Optional PIT date YYYY-MM-DD")
    p.add_argument("--log-file", type=Path, default=Path("reports/smoke_test_kb_query.log"))
    return p.parse_args()


def _log(lines: list[str], log_file: Path | None) -> None:
    block = "\n".join(lines) + "\n"
    print(block, end="")
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(block, encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_path: Path = args.log_file

    from quantmind.kb.retriever import HybridRetriever

    chroma_dir = args.chroma_dir
    collection_user = args.collection_name
    chroma_collection_full = f"quantmind_kb_{collection_user}"

    ticker_filter = args.ticker.strip() or None
    as_of: date | None = None
    if args.as_of:
        as_of = date.fromisoformat(args.as_of)

    lines: list[str] = [
        "=" * 60,
        "KB query smoke test",
        f"query          : {args.query}",
        f"ticker_filter  : {ticker_filter or '(none)'}",
        f"top_k          : {args.top_k}",
        f"collection_name: {collection_user}  (Chroma: {chroma_collection_full})",
        f"chroma_dir     : {chroma_dir}",
        f"as_of          : {as_of or '(none)'}",
        f"use_reranker   : False",
        "=" * 60,
    ]

    try:
        r = HybridRetriever(
            collection_name=collection_user,
            chroma_dir=chroma_dir,
            use_reranker=False,
        )
        n = r.count()
    except Exception as e:
        lines.append(f"ERROR: cannot init HybridRetriever or count: {e}")
        _log(lines, log_path)
        raise SystemExit(1)

    lines.append(f"collection_count (Chroma): {n}")

    if n == 0:
        lines.extend([
            "",
            "EMPTY: Chroma collection has 0 documents.",
            "Check:",
            "  - chroma_dir matches the path used by build_kb.py (default .cache/chromadb)",
            "  - collection_name matches build_kb --collection (default default → quantmind_kb_default)",
        ])
        _log(lines, log_path)
        raise SystemExit(2)

    results = r.retrieve(
        args.query,
        top_k=args.top_k,
        as_of=as_of,
        ticker_filter=ticker_filter,
    )

    lines.append(f"returned_count: {len(results)}")
    lines.append("")

    if not results:
        lines.extend([
            "EMPTY RESULTS (collection is non-empty). Possible causes:",
            "  - ticker_filter too strict vs metadata['ticker'] (e.g. code format mismatch)",
            "  - as_of excludes all published_date values",
            "  - query + hybrid retrieval returned nothing after merge/filter (rare if count>0)",
            "  - Note: new process has empty BM25 RAM index; dense Chroma path still runs.",
        ])
        _log(lines, log_path)
        raise SystemExit(3)

    for i, res in enumerate(results, start=1):
        meta = res.metadata or {}
        ticker_m = meta.get("ticker", "")
        src = meta.get("source", "")
        st = meta.get("source_type", "")
        doc_type = meta.get("doc_type", "")
        pub = meta.get("published_date", "")
        score = res.score
        finite = isinstance(score, (int, float)) and math.isfinite(float(score))

        lines.append(f"--- rank {i} ---")
        lines.append(f"score            : {score}  (finite={finite})")
        lines.append(f"ticker (meta)    : {ticker_m}")
        lines.append(f"source           : {_preview(str(src), 80)}")
        lines.append(f"source_type      : {st}")
        lines.append(f"doc_type         : {doc_type or '(absent)'}")
        lines.append(f"published_date   : {pub}")
        lines.append(f"text_preview     : {_preview(res.text, 120)}")
        lines.append("")

    lines.append("DONE")
    _log(lines, log_path)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
