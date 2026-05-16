"""Read-only snapshot directory inventory (no embedding / Chroma)."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from quantmind.kb.snapshot_parquet import SNAPSHOT_PARQUET_FILES, parquet_row_count

_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def iter_snapshot_date_dirs(snapshots_root: str | Path) -> list[Path]:
    root = Path(snapshots_root)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if _DATE_DIR_RE.match(p.name):
            out.append(p)
    return out


def manifest_success_as_ofs(manifest_path: str | Path | None) -> set[str]:
    path = Path(manifest_path) if manifest_path else None
    if not path or not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    runs = data.get("runs") or {}
    ok: set[str] = set()
    for k, v in runs.items():
        if isinstance(v, dict) and v.get("status") == "success":
            ok.add(str(k))
    return ok


def ingest_readiness(
    snapshot_dir: Path,
    *,
    min_universe: int = 0,
    include_small: bool = False,
) -> dict[str, Any]:
    """Lightweight row: file flags + universe count from footer only."""
    d = snapshot_dir.resolve()
    univ_path = d / "universe.parquet"
    univ_n = parquet_row_count(univ_path)
    flags: dict[str, Any] = {"meta_json": (d / "meta.json").is_file()}
    for fname in SNAPSHOT_PARQUET_FILES:
        key = fname.replace(".parquet", "")
        flags[key] = (d / fname).is_file()

    missing_core = []
    if not flags["universe"]:
        missing_core.append("universe.parquet")
    if not flags["stock_basic"]:
        missing_core.append("stock_basic.parquet")
    if not flags["index_daily"]:
        missing_core.append("index_daily.parquet")

    ready = (
        flags["universe"]
        and flags["stock_basic"]
        and flags["index_daily"]
        and univ_n is not None
        and univ_n > 0
    )

    small = univ_n is not None and univ_n < min_universe
    skip_reason: str | None = None
    if not ready:
        skip_reason = f"missing_or_empty: {','.join(missing_core) if missing_core else 'universe'}"
    elif small and not include_small:
        skip_reason = f"universe_count={univ_n} < min_universe={min_universe}"

    rec: dict[str, Any] = {
        "as_of": d.name,
        "snapshot_dir": str(d),
        "universe_count": univ_n if univ_n is not None else 0,
        "flags": flags,
        "ingest_eligible": ready and (include_small or not small),
        "skip_reason": skip_reason,
    }
    return rec


def audit_table_markdown_rows(
    snapshots_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    min_universe: int = 250,
) -> tuple[list[list[str]], dict[str, int]]:
    """Build table rows for markdown audit."""
    done = manifest_success_as_ofs(manifest_path)
    dirs = iter_snapshot_date_dirs(snapshots_root)
    counts = {
        "total_dirs": len(dirs),
        "full_universe": 0,
        "small_universe": 0,
        "ingest_yes": 0,
        "manifest_done": 0,
    }
    rows: list[list[str]] = []
    for d in dirs:
        in_manifest = d.name in done
        if in_manifest:
            counts["manifest_done"] += 1

        row_min = ingest_readiness(d, min_universe=min_universe, include_small=False)
        eligible_strict = row_min["ingest_eligible"] and not row_min["skip_reason"]
        if eligible_strict:
            counts["ingest_yes"] += 1

        inv = ingest_readiness(d, min_universe=min_universe, include_small=True)
        u = inv["universe_count"]
        fl = inv["flags"]
        if u >= min_universe:
            counts["full_universe"] += 1
        else:
            counts["small_universe"] += 1

        suggest = "ingest"
        if not inv["flags"]["universe"] or not inv["flags"]["stock_basic"]:
            suggest = "skip"
        elif u < min_universe:
            suggest = "warning_small_universe"
        elif not eligible_strict:
            suggest = "skip"

        kb_note = "unknown"
        if in_manifest:
            kb_note = "manifest_success"
        else:
            kb_note = "not_in_manifest"

        rows.append([
            d.name,
            str(u),
            "yes" if fl["meta_json"] else "no",
            "yes" if fl["universe"] else "no",
            "yes" if fl["stock_basic"] else "no",
            "yes" if fl["prices"] else "no",
            "yes" if fl["daily_basic"] else "no",
            "yes" if fl["financial_indicators"] else "no",
            "yes" if fl["hk_hold"] else "no",
            "yes" if fl["margin"] else "no",
            "yes" if fl["index_daily"] else "no",
            "yes" if eligible_strict else "no",
            kb_note,
            suggest,
            row_min.get("skip_reason") or inv.get("skip_reason") or "",
        ])
    return rows, counts


def write_audit_markdown(
    out_path: str | Path,
    snapshots_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    min_universe: int = 250,
) -> None:
    root = Path(snapshots_root).resolve()
    man = Path(manifest_path).resolve() if manifest_path else None
    rows, counts = audit_table_markdown_rows(
        root, manifest_path=man, min_universe=min_universe,
    )
    lines = [
        "# Snapshot KB ingestion — candidate audit",
        "",
        f"- **snapshots_root**: `{root}`",
        f"- **manifest** (for ingest history): `{man}`" if man else "- **manifest**: (none)",
        f"- **generated_at**: {datetime.utcnow().isoformat()}Z",
        "",
        "## Summary counts",
        "",
        f"- Dated snapshot directories: **{counts['total_dirs']}**",
        f"- `universe_count >= {min_universe}`: **{counts['full_universe']}**",
        f"- `universe_count < {min_universe}` (small): **{counts['small_universe']}**",
        f"- Eligible for ingest (strict min, files OK): **{counts['ingest_yes']}**",
        f"- Dates already `status=success` in manifest: **{counts['manifest_done']}**",
        "",
        "## Small-universe policy",
        "",
        "Directories with small `universe_count` (e.g. smoke snapshots) are **flagged** as "
        "`warning_small_universe`. Batch ingestion with `--min-universe-count 250` **skips** them "
        "unless `--include-small-snapshots` is set.",
        "",
        "## Manifest vs Chroma",
        "",
        "`manifest_success` only means the JSON manifest recorded a successful run; it does not "
        "inspect Chroma contents. If manifest is missing or you rebuilt Chroma, use `--force` to re-ingest.",
        "",
        "| as_of | univ | meta | uni | sb | prc | db | fin | hk | mrg | idx | can_ingest | kb_hist | suggest | skip_reason |",
        "| --- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
