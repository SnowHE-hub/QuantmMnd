"""Audit data/snapshots/*/ for KB parquet ingestion (read-only, no embedding/Chroma)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.kb.snapshot_inventory import write_audit_markdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots-root", default="data/snapshots", type=Path)
    p.add_argument(
        "--out",
        default="reports/audit_snapshot_kb_ingestion_candidates.md",
        type=Path,
    )
    p.add_argument("--manifest", default="reports/kb_snapshot_ingestion_manifest.json", type=Path)
    p.add_argument("--min-universe-count", type=int, default=250)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    man = args.manifest if args.manifest.is_file() else None
    write_audit_markdown(
        args.out,
        args.snapshots_root,
        manifest_path=man,
        min_universe=args.min_universe_count,
    )
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
