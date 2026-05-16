"""Sequential KB backfill from dated parquet snapshots (no multiprocessing).

Example:
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \\
    python -u scripts/backfill_kb_snapshots.py \\
      --date-file reports/kb_backfill/pending_dates.txt \\
      --sleep 3
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_date_file(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        datetime.strptime(line, "%Y-%m-%d")
        out.append(line)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dates",
        nargs="+",
        metavar="YYYY-MM-DD",
        help="Snapshot as-of dates to ingest",
    )
    g.add_argument(
        "--date-file",
        type=Path,
        help="File with one YYYY-MM-DD per line",
    )
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("data/snapshots"),
        help="Parent directory holding dated dirs (default: data/snapshots)",
    )
    p.add_argument("--sleep", type=float, default=5.0, help="Seconds to wait between periods")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip as_of if KB already has at least one snapshot doc for that as_of",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path("reports/kb_backfill"),
        help="Directory for per-period log files (optional)",
    )
    p.add_argument("--collection", default="default")
    p.add_argument("--chroma-dir", default=".cache/chromadb")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--chunk-overlap", type=int, default=100)
    return p.parse_args()


def _has_snapshot_for_as_of(
    retriever: object,
    as_of_iso: str,
) -> bool:
    where: dict = {
        "$and": [
            {"as_of": {"$eq": as_of_iso}},
            {"source_type": {"$eq": "snapshot"}},
        ]
    }
    items = retriever.fetch_by_metadata(where=where, as_of=None, limit=1)
    return len(items) > 0


def main() -> None:
    args = parse_args()
    if args.date_file is not None:
        if not args.date_file.is_file():
            raise SystemExit(f"date file not found: {args.date_file}")
        dates = parse_date_file(args.date_file)
    else:
        dates = []
        for d in args.dates:
            datetime.strptime(d, "%Y-%m-%d")
            dates.append(d)

    dates = sorted(set(dates))
    n_total = len(dates)
    if n_total == 0:
        raise SystemExit("no dates to ingest")

    args.log_dir.mkdir(parents=True, exist_ok=True)

    from quantmind.kb.builder import KBBuilder
    from quantmind.kb.chunker import SmartChunker

    chunker = SmartChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    builder = KBBuilder(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        chunker=chunker,
    )
    retriever = builder.get_retriever()

    snapshot_root = args.snapshot_dir

    for i, as_of_s in enumerate(dates, start=1):
        snap_dir = snapshot_root / as_of_s
        if not snap_dir.is_dir():
            print(
                f"[{i}/{n_total}] as_of={as_of_s} SKIPPED snapshot_dir_missing={snap_dir}",
                flush=True,
            )
            continue

        if args.skip_existing and _has_snapshot_for_as_of(retriever, as_of_s):
            kb_total = retriever.count()
            print(
                f"[{i}/{n_total}] as_of={as_of_s} SKIPPED docs_written=0 "
                f"elapsed=0s KB_total={kb_total} (already ingested)",
                flush=True,
            )
            continue

        t0 = time.perf_counter()
        res = builder.build_from_snapshot_dir(
            snap_dir,
            tickers=None,
            as_of=date.fromisoformat(as_of_s),
            dry_run=False,
        )
        elapsed = time.perf_counter() - t0
        docs_written = int(res.get("chunks_written", 0))
        kb_total = retriever.count()
        print(
            f"[{i}/{n_total}] as_of={as_of_s} docs_written={docs_written} "
            f"elapsed={elapsed:.1f}s KB_total={kb_total}",
            flush=True,
        )

        if args.sleep > 0 and i < n_total:
            time.sleep(float(args.sleep))


if __name__ == "__main__":
    main()
