"""Debug KB data sources and chunk counts without embedding / Chroma.

Does not load .env; does not print secrets. Does not invoke EmbeddingService.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.kb.builder import (
    KBBuilder,
    analyze_raw_news_records,
)
from quantmind.kb.chunker import SmartChunker


def _sample(text: str, max_len: int = 80) -> str:
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _parse_as_of(s: str | None) -> date:
    if not s:
        return date.today()
    return datetime.strptime(s, "%Y-%m-%d").date()


def run_news(tickers: list[str], as_of: date, days_back: int) -> None:
    builder = KBBuilder(chunker=SmartChunker())
    cutoff = as_of - timedelta(days=days_back)
    chunker = builder.chunker

    print("\n## News diagnostics\n")
    for ticker in tickers:
        t0 = time.perf_counter()
        news_list = builder._fetch_news(ticker, as_of)
        stats = analyze_raw_news_records(
            news_list,
            ticker,
            cutoff,
            as_of,
            chunker,
            seen_hashes=None,
            enable_dedup=True,
        )
        elapsed = time.perf_counter() - t0
        fields = stats["raw_field_names"]
        print(f"### `{ticker}`  (fetch+analyze {elapsed:.2f}s)\n")
        print(f"- **provider rows**: {stats['raw_docs']}")
        print(f"- **raw field names**: `{fields}`")
        print(f"- **resolved orders**: title={stats['resolved_title_fields']}, "
              f"body={stats['resolved_body_fields']}, "
              f"time={stats['resolved_time_fields']}, "
              f"url={stats['resolved_url_fields']}")
        print("\n**Non-empty counts (per column)**\n")
        print(_md_table(
            ["group", "column", "nonempty_n"],
            [["title-like", k, str(v)] for k, v in stats["title_like_nonempty"].items()]
            + [["content-like", k, str(v)] for k, v in stats["content_like_nonempty"].items()]
            + [["summary-like", k, str(v)] for k, v in stats["summary_like_nonempty"].items()]
            + [["url-like", k, str(v)] for k, v in stats["url_like_nonempty"].items()]
            + [["time-like", k, str(v)] for k, v in stats["publish_time_like_nonempty"].items()],
        ))
        print("\n**Pipeline**\n")
        print(_md_table(
            ["metric", "value"],
            [
                ["date_parse_success_rows", str(stats["date_parse_success_rows"])],
                ["drop_before_cutoff", str(stats["drop_before_cutoff"])],
                ["drop_after_asof", str(stats["drop_after_asof"])],
                ["docs_after_date_filter", str(stats["docs_after_date_filter"])],
                ["empty_text_docs", str(stats["empty_text_docs"])],
                ["docs_before_dedup", str(stats["docs_before_dedup"])],
                ["dedup_dropped", str(stats["dedup_dropped"])],
                ["docs_after_dedup", str(stats["docs_after_dedup"])],
                ["len_chunks", str(stats["len_chunks"])],
            ],
        ))
        if news_list and fields:
            row0 = news_list[0]
            preview_bits = []
            for k in fields[:5]:
                preview_bits.append(f"{k}={_sample(str(row0.get(k, '')))}")
            print(f"\n*Row0 preview (truncated)*: {', '.join(preview_bits)}")
        print()


def run_reports(reports_dir: Path, as_of: date) -> None:
    builder = KBBuilder(chunker=SmartChunker())
    extensions = (".txt", ".md", ".pdf")
    print("\n## Reports diagnostics\n")
    if not reports_dir.exists():
        print(f"*Reports dir missing*: `{reports_dir}`\n")
        return

    rows_out: list[list[str]] = []
    n_files = 0
    total_chars = 0
    total_chunks = 0
    for fpath in sorted(reports_dir.rglob("*")):
        if fpath.suffix.lower() not in extensions:
            continue
        n_files += 1
        text = builder._read_file(fpath)
        n_chars = len(text or "")
        total_chars += n_chars
        ticker, pub_date = builder._infer_from_filename(fpath.name)
        pit_skip = ""
        if pub_date and pub_date > as_of:
            pit_skip = "yes"
            n_chunks = 0
        elif not text:
            n_chunks = 0
        else:
            docs = [{
                "text": text,
                "source": str(fpath),
                "source_type": "report",
                "ticker": ticker or "",
                "published_date": pub_date.isoformat() if pub_date else "",
                "title": fpath.stem,
            }]
            n_chunks = len(builder.chunker.chunk_documents(docs))
        total_chunks += n_chunks
        rows_out.append([
            fpath.name,
            str(n_chars),
            str(n_chunks),
            ticker or "",
            str(pub_date or ""),
            pit_skip,
            _sample(text, 60),
        ])

    print(f"- **directory**: `{reports_dir}`")
    print(f"- **matching files**: {n_files}")
    print(f"- **total text chars (summed)**: {total_chars}")
    print(f"- **total chunks (post-PIT filter, no dedup)**: {total_chunks}\n")
    if rows_out:
        print(_md_table(
            ["file", "chars", "chunks", "ticker", "pub_date", "pit_skip", "sample"],
            rows_out,
        ))
    print()


def run_snapshot(
    snapshot_path: Path | None,
    tickers: list[str],
    as_of: date,
) -> None:
    print("\n## Snapshot diagnostics (text + chunk count only)\n")
    if snapshot_path is None or not snapshot_path.is_file():
        print("*No snapshot file* (pass path or place `*.pkl` under `data/snapshots/`)\n")
        return

    builder = KBBuilder(chunker=SmartChunker())
    try:
        with open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)
    except Exception as e:
        print(f"*Failed to load*: `{e}`\n")
        return

    tickers_eff = tickers
    if not tickers_eff:
        tickers_eff = list(getattr(snapshot, "tickers", {}).keys())

    rows: list[list[str]] = []
    sum_chunks = 0
    for ticker in tickers_eff:
        text = builder._snapshot_to_text(snapshot, ticker, as_of)
        n_chars = len(text or "")
        if not text:
            n_chunks = 0
        else:
            docs = [{
                "text": text,
                "source": f"snapshot:{ticker}",
                "source_type": "filing",
                "ticker": ticker,
                "published_date": as_of.isoformat(),
                "title": f"{ticker} 财务摘要 {as_of}",
            }]
            n_chunks = len(builder.chunker.chunk_documents(docs))
        sum_chunks += n_chunks
        rows.append([ticker, str(n_chars), str(n_chunks), _sample(text, 60)])

    print(f"- **path**: `{snapshot_path}`")
    print(f"- **tickers**: {len(tickers_eff)}")
    print(f"- **total chunks**: {sum_chunks}\n")
    if rows:
        print(_md_table(["ticker", "chars", "chunks", "sample"], rows))
    print()


def run_snapshot_parquet(snapshot_dir: Path, tickers: list[str], as_of: date) -> None:
    """Inspect parquet snapshot → documents/chunks without embedding or Chroma."""
    from quantmind.kb.snapshot_parquet import SNAPSHOT_PARQUET_FILES, generate_snapshot_documents, load_snapshot_parquets

    print("\n## Snapshot (parquet) diagnostics\n")
    root = snapshot_dir.resolve()
    print(f"- **snapshot_dir**: `{root}`")
    print(f"- **as_of**: {as_of.isoformat()}")
    print(f"- **tickers**: {tickers!r}\n")

    if not root.is_dir():
        print(f"*Not a directory*: `{root}`\n")
        return

    tables, _, _ = load_snapshot_parquets(root, strict=False)
    table_rows: list[list[str]] = []
    for fname in SNAPSHOT_PARQUET_FILES:
        key = fname.replace(".parquet", "")
        if key in tables:
            table_rows.append([fname, "loaded", str(len(tables[key]))])
        else:
            table_rows.append([fname, "missing", "—"])

    print("**Parquet files**\n")
    print(_md_table(["file", "status", "rows"], table_rows))
    print()

    docs, _stats = generate_snapshot_documents(root, tickers, as_of=as_of.isoformat())
    chunker = SmartChunker()
    chunks = chunker.chunk_documents(docs)

    empty_text = sum(1 for d in docs if not (d.get("text") or "").strip())
    print(f"- **documents**: {len(docs)}")
    print(f"- **chunks**: {len(chunks)}")
    print(f"- **empty_text_docs**: {empty_text}\n")

    by_ticker: dict[str, list[tuple[str, int, str]]] = {}
    for d in docs:
        tk = str(d.get("ticker", ""))
        dt = str(d.get("doc_type", ""))
        text = (d.get("text") or "").strip()
        by_ticker.setdefault(tk, []).append((dt, len(text), _sample(text, 72)))

    print("**Per ticker: doc_type → text length**\n")
    for tk in sorted(by_ticker.keys()):
        rows_tb = [[dt, str(nchar), prv] for dt, nchar, prv in by_ticker[tk]]
        print(f"### `{tk}`\n")
        print(_md_table(["doc_type", "chars", "preview"], rows_tb))
        print()

    if docs:
        d0 = docs[0]
        print("**Metadata sample (first doc)**\n")
        for k in ("ticker", "as_of", "source_type", "doc_type", "table_name", "snapshot_dir"):
            if k in d0:
                print(f"- `{k}`: {_sample(str(d0.get(k, '')), 100)}")
        missing_meta = [k for k in ("ticker", "as_of", "source_type", "doc_type", "table_name") if k not in d0]
        if missing_meta:
            print(f"\n*Missing metadata keys*: {missing_meta}")
        print()


def discover_snapshot_path() -> Path | None:
    snaps = list(Path("data/snapshots").glob("*.pkl"))
    if not snaps:
        return None
    return snaps[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KB source + chunk debug (no embedding / Chroma)")
    p.add_argument("--tickers", default="", help="Comma tickers e.g. 600519.SH,300750.SZ")
    p.add_argument("--source", choices=["news", "reports", "snapshot", "all"], required=True)
    p.add_argument("--reports-dir", default="reports", type=Path)
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD")
    p.add_argument("--days-back", type=int, default=90)
    p.add_argument("--snapshot-path", default=None, type=Path)
    p.add_argument("--snapshot-dir", default=None, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    as_of = _parse_as_of(args.as_of)
    if args.snapshot_dir and args.as_of is None:
        # Align PIT with dated snapshot folder when omitted (e.g. .../2024-12-31)
        try:
            as_of = date.fromisoformat(args.snapshot_dir.name)
        except ValueError:
            pass
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    snap = args.snapshot_path
    if args.source == "all" and snap is None and args.snapshot_dir is None:
        snap = discover_snapshot_path()

    print("# KB source diagnostics\n")
    print(f"- **as_of**: {as_of}")
    print(f"- **days_back** (news): {args.days_back}")
    print(f"- **source**: {args.source}")
    print(f"- **embedding**: not loaded")
    print(f"- **Chroma**: not used\n")

    if args.source in frozenset({"news", "all"}):
        if not tickers:
            print("*News skipped*: provide `--tickers`\n")
        else:
            run_news(tickers, as_of, args.days_back)

    if args.source in frozenset({"reports", "all"}):
        run_reports(args.reports_dir, as_of)

    if args.source == "snapshot":
        if args.snapshot_dir:
            run_snapshot_parquet(args.snapshot_dir, tickers, as_of)
        elif snap:
            run_snapshot(snap, tickers, as_of)
        else:
            print("*Snapshot skipped*: provide `--snapshot-dir` (parquet) or `--snapshot-path` (.pkl)\n")

    if args.source == "all":
        if args.snapshot_dir:
            run_snapshot_parquet(args.snapshot_dir, tickers, as_of)
        else:
            run_snapshot(snap, tickers, as_of)


if __name__ == "__main__":
    main()
