"""验证 KB 中 news 覆盖（按 ticker + source_type=news）或 snapshot（按 as_of）。

``--ticker-file`` 可为每行 ticker 的 .txt，或含 ``ts_code``/``ticker`` 列的 ``universe.parquet``。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_ticker_file(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def load_tickers_for_verify(path: Path) -> list[str]:
    """每行 ticker 文本，或 ``universe.parquet``（读取 ts_code / ticker 列）。"""
    if path.suffix.lower() != ".parquet":
        return parse_ticker_file(path)
    import pandas as pd

    from quantmind.kb.snapshot_parquet import _normalize_ticker_code

    df = pd.read_parquet(path)
    col = "ts_code" if "ts_code" in df.columns else "ticker" if "ticker" in df.columns else None
    if col is None:
        raise SystemExit(f"--ticker-file parquet 需含 ts_code 或 ticker 列；实际={list(df.columns)}")
    return sorted({_normalize_ticker_code(str(x)) for x in df[col].tolist() if str(x).strip()})


def parse_dates_file(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        datetime.strptime(line, "%Y-%m-%d")
        out.append(line)
    return out


def snapshot_aggregate_for_as_of(
    collection_name_full: str,
    chroma_dir: str | Path,
    as_of_iso: str,
    *,
    batch: int = 8000,
) -> tuple[int, dict[str, int]]:
    """Paginate snapshot rows at ``as_of``; return total count + doc_type histogram."""
    import chromadb
    from collections import Counter

    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll = client.get_collection(collection_name_full)
    where: dict = {
        "$and": [
            {"as_of": {"$eq": as_of_iso}},
            {"source_type": {"$eq": "snapshot"}},
        ]
    }
    total = 0
    offset = 0
    dc: Counter[str] = Counter()
    while True:
        res = coll.get(where=where, limit=batch, offset=offset, include=["metadatas"])
        metas = res.get("metadatas") or []
        ids = res.get("ids") or []
        n = len(ids)
        total += n
        for m in metas:
            if not m:
                continue
            dt = str(m.get("doc_type") or "")
            if dt:
                dc[dt] += 1
        if n < batch:
            break
        offset += batch
    return total, dict(sorted(dc.items()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ticker-file", type=Path, default=Path("reports/batch/tickers.txt"))
    p.add_argument("--collection-name", default="default")
    p.add_argument("--chroma-dir", default=".cache/chromadb")
    p.add_argument("--limit", type=int, default=500, help="fetch_by_metadata limit（单 ticker 最多拉取条数）")
    p.add_argument(
        "--check-snapshots",
        action="store_true",
        help="按 as_of 统计 source_type=snapshot 的文档数（需配合 --as-of-list）",
    )
    p.add_argument(
        "--as-of-list",
        type=Path,
        default=None,
        help="每行一个 YYYY-MM-DD（与 --check-snapshots 共用）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    coll_full = f"quantmind_kb_{args.collection_name}"

    if args.check_snapshots:
        if args.as_of_list is None or not args.as_of_list.is_file():
            raise SystemExit("--check-snapshots 需要有效的 --as-of-list 文件路径")
        dates = parse_dates_file(args.as_of_list)
        from quantmind.kb.retriever import HybridRetriever

        r = HybridRetriever(
            collection_name=args.collection_name,
            chroma_dir=args.chroma_dir,
            use_reranker=False,
        )
        kb_total = r.count()
        print(f"KB collection count: {kb_total}")
        print("-" * 72)
        for d in dates:
            n, histogram = snapshot_aggregate_for_as_of(coll_full, args.chroma_dir, d)
            hist_s = ", ".join(f"{k}={histogram[k]}" for k in histogram) if histogram else "(none)"
            print(f"as_of={d}  snapshot_count={n}  by_doc_type: {hist_s}")
        print("-" * 72)
        return

    if not args.ticker_file.is_file():
        raise SystemExit(f"文件不存在: {args.ticker_file}")

    tickers = load_tickers_for_verify(args.ticker_file)
    from quantmind.kb.retriever import HybridRetriever

    r = HybridRetriever(
        collection_name=args.collection_name,
        chroma_dir=args.chroma_dir,
        use_reranker=False,
    )
    kb_total = r.count()
    print(f"KB collection count: {kb_total}")
    print("-" * 72)

    with_news = 0
    rows: list[tuple[str, int, str]] = []
    for tk in tickers:
        where: dict = {
            "$and": [
                {"ticker": {"$eq": tk}},
                {"source_type": {"$eq": "news"}},
            ]
        }
        items = r.fetch_by_metadata(where=where, as_of=None, limit=args.limit)
        n = len(items)
        latest = ""
        if items:
            latest = (items[0].metadata or {}).get("published_date") or ""
        if n > 0:
            with_news += 1
        rows.append((tk, n, latest))
        print(f"{tk}  news_count={n}  latest_published_date={latest or '(none)'}")

    print("-" * 72)
    print(f"news 覆盖率: {with_news}/{len(tickers)} tickers 至少 1 条 news")


if __name__ == "__main__":
    main()
