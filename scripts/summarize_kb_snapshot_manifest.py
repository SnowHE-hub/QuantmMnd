"""Write reports/kb_snapshot_ingestion_full_summary.md from manifest + snapshot dirs + KB count."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.kb.snapshot_inventory import (
    audit_table_markdown_rows,
    ingest_readiness,
    iter_snapshot_date_dirs,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="reports/kb_snapshot_ingestion_manifest.json", type=Path)
    p.add_argument("--snapshots-root", default="data/snapshots", type=Path)
    p.add_argument("--out", default="reports/kb_snapshot_ingestion_full_summary.md", type=Path)
    p.add_argument("--min-universe", type=int, default=250)
    p.add_argument("--collection-name", default="default")
    p.add_argument("--chroma-dir", default=".cache/chromadb")
    args = p.parse_args()

    man_path = args.manifest
    data: dict = {}
    if man_path.is_file():
        data = json.loads(man_path.read_text(encoding="utf-8"))
    runs = data.get("runs") or {}

    success_dates = sorted(
        k for k, v in runs.items() if isinstance(v, dict) and v.get("status") == "success"
    )
    failed_dates = sorted(
        k for k, v in runs.items() if isinstance(v, dict) and v.get("status") == "failed"
    )

    small_dirs: list[tuple[str, int]] = []
    for d in iter_snapshot_date_dirs(args.snapshots_root):
        inv_loose = ingest_readiness(d, min_universe=args.min_universe, include_small=True)
        u = inv_loose["universe_count"]
        if u < args.min_universe:
            small_dirs.append((d.name, u))

    eligible_strict = []
    for d in iter_snapshot_date_dirs(args.snapshots_root):
        inv = ingest_readiness(d, min_universe=args.min_universe, include_small=False)
        if inv["ingest_eligible"] and not inv.get("skip_reason"):
            eligible_strict.append(d.name)
    eligible_strict.sort()

    missing_success = sorted(set(eligible_strict) - set(success_dates))

    kb_count: int | str
    try:
        from quantmind.kb.retriever import HybridRetriever

        r = HybridRetriever(
            collection_name=args.collection_name,
            chroma_dir=args.chroma_dir,
            use_reranker=False,
        )
        kb_count = r.count()
    except Exception as e:  # noqa: BLE001
        kb_count = f"(error: {e})"

    rows, counts = audit_table_markdown_rows(
        args.snapshots_root,
        manifest_path=man_path if man_path.is_file() else None,
        min_universe=args.min_universe,
    )

    lines = [
        "# KB snapshot full ingestion — manifest & coverage summary",
        "",
        f"- **manifest**: `{man_path.resolve()}`",
        f"- **snapshots_root**: `{Path(args.snapshots_root).resolve()}`",
        f"- **min_universe (strict ingest)**: {args.min_universe}",
        "",
        "## Manifest",
        "",
        f"- **success count**: {len(success_dates)}",
        f"- **failed count**: {len(failed_dates)}",
        f"- **eligible strict (can_ingest=yes) dirs**: {counts['ingest_yes']}",
        "",
        "### Success dates",
        "",
        ", ".join(f"`{x}`" for x in success_dates) or "(none)",
        "",
        "### Failed dates",
        "",
    ]
    if failed_dates:
        for fd in failed_dates:
            rec = runs.get(fd) or {}
            err = (rec.get("error") or "").replace("\n", " ")[:500]
            lines.append(f"- `{fd}`: {err or '(no message)'}")
    else:
        lines.append("(none)")
    lines.extend([
        "",
        "### Eligible but not success (gap)",
        "",
        ", ".join(f"`{x}`" for x in missing_success) if missing_success else "(none — all strict-eligible dates have manifest success)",
        "",
        "## Snapshot dir scan",
        "",
        f"- **YYYY-MM-DD dirs**: {counts['total_dirs']}",
        f"- **universe_count >= {args.min_universe}**: {counts['full_universe']}",
        f"- **universe_count < {args.min_universe} (small, default skip)**: {counts['small_universe']}",
        "",
        "### Small-universe snapshot dirs",
        "",
        ", ".join(f"`{n}` ({u})" for n, u in small_dirs) if small_dirs else "(none listed)",
        "",
        "## Chroma collection",
        "",
        f"- **collection**: `quantmind_kb_{args.collection_name}`",
        f"- **chroma_dir**: `{args.chroma_dir}`",
        f"- **count()**: {kb_count}",
        "",
        "## Audit rows (abbrev)",
        "",
    ])
    for r in rows:
        lines.append("- `" + r[0] + "`: univ=" + r[1] + " can_ingest=" + r[11] + " suggest=" + r[13])

    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
