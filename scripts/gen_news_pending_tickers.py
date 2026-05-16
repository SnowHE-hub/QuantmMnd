#!/usr/bin/env python3
"""列出「universe 内有、KB 尚无 news」的 ticker，写入 pending_tickers.txt。

KB 已有 news ticker 通过对 Chroma 中 ``source_type=news`` 分页扫描 metadatas
收集（比单次 fetch_by_metadata(limit=…) 更能覆盖全量）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.kb.snapshot_parquet import _normalize_ticker_code  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--universe-parquet",
        type=Path,
        default=Path("data/snapshots/2024-12-31/universe.parquet"),
        help="Universe parquet（默认 2024-12-31）",
    )
    p.add_argument("--collection-name", default="default")
    p.add_argument("--chroma-dir", default=".cache/chromadb")
    p.add_argument(
        "--batch",
        type=int,
        default=8000,
        help="ChromaDB get pagination size",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/news_fullbatch/pending_tickers.txt"),
        help="待摄入 ticker，每行一个",
    )
    return p.parse_args()


def tickers_with_news_from_kb(
    collection_name_full: str,
    chroma_dir: Path | str,
    *,
    batch: int,
) -> set[str]:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll = client.get_collection(collection_name_full)
    where: dict = {"source_type": {"$eq": "news"}}
    seen: set[str] = set()
    offset = 0
    while True:
        res = coll.get(where=where, limit=batch, offset=offset, include=["metadatas"])
        metas = res.get("metadatas") or []
        ids = res.get("ids") or []
        n = len(ids)
        for m in metas:
            if not m:
                continue
            tk = str(m.get("ticker") or "").strip()
            if tk:
                seen.add(_normalize_ticker_code(tk))
        if n < batch:
            break
        offset += batch
    return seen


def universe_tickers_from_parquet(parquet_path: Path) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    col = "ts_code" if "ts_code" in df.columns else "ticker" if "ticker" in df.columns else None
    if col is None:
        raise SystemExit(f"universe parquet 缺少 ts_code/ticker 列，实际列={list(df.columns)}")
    codes = {_normalize_ticker_code(str(x)) for x in df[col].tolist() if str(x).strip()}
    return sorted(codes)


def main() -> None:
    args = parse_args()
    if not args.universe_parquet.is_file():
        raise SystemExit(f"universe parquet 不存在: {args.universe_parquet}")

    universe = universe_tickers_from_parquet(args.universe_parquet)
    coll_full = f"quantmind_kb_{args.collection_name}"
    kb_news_tickers = tickers_with_news_from_kb(coll_full, args.chroma_dir, batch=args.batch)
    pending = [t for t in universe if t not in kb_news_tickers]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(pending) + ("\n" if pending else ""), encoding="utf-8")

    print(f"universe 总数           : {len(universe)}")
    print(f"KB 已有 news 股票数（去重）: {len(kb_news_tickers & set(universe))}")
    print(f"待写入 pending 数       : {len(pending)}")
    print(f"输出 → {args.out.resolve()}")

    sample_in = sorted(kb_news_tickers & set(universe))[:8]
    if sample_in:
        print(f"(示例 KB 已有 news 的 ticker: {', '.join(sample_in)} ...)")


if __name__ == "__main__":
    main()
