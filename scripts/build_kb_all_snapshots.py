"""Batch-ingest all dated parquet snapshots into KB with manifest + resumability.

Does not run news/reports. Does not use legacy .pkl snapshots.

Default: dry-run (plan only). Use --execute to write Chroma and run embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, TextIO

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL_NAME = "BAAI/bge-m3"


@dataclass
class RunContext:
    log_fp: TextIO
    collection_name: str
    chroma_dir: str

    def log(self, msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        self.log_fp.write(line + "\n")
        self.log_fp.flush()


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
        "collection_name": "default",
        "chroma_dir": ".cache/chromadb",
        "model_name": MODEL_NAME,
        "runs": {},
    }


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = _iso_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots-root", default="data/snapshots", type=Path)
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD inclusive")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive")
    p.add_argument("--tickers", default=None, help="Comma list; omit for full universe per snapshot")
    p.add_argument("--min-universe-count", type=int, default=250)
    p.add_argument("--include-small-snapshots", action="store_true")
    p.add_argument("--max-dates", type=int, default=None)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Explicit dry-run (default if no --execute)")
    p.add_argument("--execute", action="store_true", help="Write KB (Chroma + embeddings)")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_true", help="Ignore manifest success (still not force overwrite same chunks)")
    p.add_argument("--force", action="store_true", help="Re-run dates even if manifest success")
    p.add_argument("--collection-name", default="default")
    p.add_argument("--chroma-dir", default=".cache/chromadb")
    p.add_argument("--manifest", default="reports/kb_snapshot_ingestion_manifest.json", type=Path)
    p.add_argument("--log-file", default="reports/build_kb_all_snapshots.log", type=Path)
    return p.parse_args()


def _select_dates(
    dirs: list[Path],
    *,
    start: date | None,
    end: date | None,
    reverse: bool,
) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        try:
            ad = date.fromisoformat(d.name)
        except ValueError:
            continue
        if start is not None and ad < start:
            continue
        if end is not None and ad > end:
            continue
        out.append(d)
    out.sort(key=lambda x: x.name, reverse=reverse)
    return out


def main() -> None:
    args = parse_args()
    execute = bool(args.execute)
    if args.dry_run:
        execute = False
    skip_existing = args.skip_existing and not args.no_skip_existing

    tickers_arg: list[str] | None = None
    if args.tickers and args.tickers.strip():
        tickers_arg = [t.strip() for t in args.tickers.split(",") if t.strip()]

    from quantmind.kb.snapshot_inventory import ingest_readiness, iter_snapshot_date_dirs

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    manifest["collection_name"] = args.collection_name
    manifest["chroma_dir"] = args.chroma_dir
    manifest["model_name"] = MODEL_NAME

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fp = args.log_file.open("w", encoding="utf-8")
    ctx = RunContext(log_fp=log_fp, collection_name=args.collection_name, chroma_dir=args.chroma_dir)

    try:
        import torch

        ctx.log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    except Exception as e:  # noqa: BLE001
        ctx.log(f"torch check failed: {e}")

    all_dirs = iter_snapshot_date_dirs(args.snapshots_root)
    start_d = date.fromisoformat(args.start_date) if args.start_date else None
    end_d = date.fromisoformat(args.end_date) if args.end_date else None
    dated = _select_dates(all_dirs, start=start_d, end=end_d, reverse=args.reverse)

    min_u = args.min_universe_count
    include_small = args.include_small_snapshots

    plan: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for d in dated:
        inv = ingest_readiness(d, min_universe=min_u, include_small=include_small)
        if not inv["ingest_eligible"]:
            skipped.append((d.name, inv.get("skip_reason") or "not_eligible"))
            continue
        as_of_s = d.name
        if skip_existing and not args.force:
            run_rec = (manifest.get("runs") or {}).get(as_of_s)
            if isinstance(run_rec, dict) and run_rec.get("status") == "success":
                skipped.append((d.name, "manifest_success_skip_existing"))
                continue
        plan.append(d)

    if args.max_dates:
        plan = plan[: args.max_dates]

    ctx.log(
        f"planning: execute={execute} dates_total_scanned={len(dated)} "
        f"to_ingest={len(plan)} skipped={len(skipped)} "
        f"min_universe={min_u} include_small={include_small}"
    )

    total_docs_est = 0
    total_chunks_est = 0

    from quantmind.kb.builder import KBBuilder
    from quantmind.kb.chunker import SmartChunker

    builder: KBBuilder | None = None
    if execute:
        builder = KBBuilder(
            collection_name=args.collection_name,
            chroma_dir=args.chroma_dir,
            chunker=SmartChunker(),
        )

    for snap_dir in plan:
        as_of_s = snap_dir.name
        as_of_d = date.fromisoformat(as_of_s)
        t0 = time.perf_counter()
        try:
            if not execute:
                from quantmind.kb.snapshot_parquet import estimate_snapshot_kb_stats

                res = estimate_snapshot_kb_stats(snap_dir, tickers_arg)
            else:
                assert builder is not None
                res = builder.build_from_snapshot_dir(
                    snap_dir,
                    tickers=tickers_arg,
                    as_of=as_of_d,
                    dry_run=False,
                )

            elapsed = time.perf_counter() - t0
            total_docs_est += int(res.get("n_docs", 0))
            total_chunks_est += int(res.get("n_chunks", 0))

            rec = {
                "snapshot_dir": str(snap_dir.resolve()),
                "universe_count": res.get("universe_count"),
                "n_tickers_ingested": res.get("n_tickers_ingested"),
                "docs": res.get("n_docs"),
                "chunks": res.get("n_chunks"),
                "add_chunks_returned": res.get("chunks_written"),
                "elapsed_seconds": round(elapsed, 3),
                "final_collection_count_after": res.get("final_collection_count"),
                "dry_run": res.get("dry_run", not execute),
                "status": "success",
                "error": None,
            }
            if execute:
                manifest.setdefault("runs", {})[as_of_s] = rec
                save_manifest(manifest_path, manifest)

            ctx.log(
                f"OK as_of={as_of_s} docs={rec['docs']} chunks={rec['chunks']} "
                f"add={rec['add_chunks_returned']} elapsed_s={elapsed:.2f} "
                f"univ={rec['universe_count']} final_count={rec.get('final_collection_count_after')}"
            )
        except Exception as e:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            err = f"{type(e).__name__}: {e}"
            ctx.log(f"FAIL as_of={as_of_s} err={err}\n{traceback.format_exc()}")
            fail = {
                "snapshot_dir": str(snap_dir.resolve()),
                "status": "failed",
                "error": err,
                "elapsed_seconds": round(elapsed, 3),
            }
            if execute:
                manifest.setdefault("runs", {})[as_of_s] = fail
                save_manifest(manifest_path, manifest)

    ctx.log(
        f"summary: ingest_dates_completed={len(plan)} est_total_docs={total_docs_est} "
        f"est_total_chunks={total_chunks_est} manifest={manifest_path}"
    )
    for name, reason in skipped[:50]:
        ctx.log(f"skip {name}: {reason}")
    if len(skipped) > 50:
        ctx.log(f"... {len(skipped) - 50} more skipped")

    log_fp.close()


if __name__ == "__main__":
    main()
