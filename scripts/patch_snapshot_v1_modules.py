#!/usr/bin/env python3
"""增量补齐已有 snapshot 的 Data Expansion v1 四张表（不触碰 prices / financials / daily_basic）。

默认 **dry-run**：不写盘、不实例化 ``TushareProvider``。

显式 ``--execute`` 才拉数写盘。``--all-existing-snapshots`` 使用 ``list_snapshots()``
枚举**当前磁盘上已有**的全部 snapshot 日期（含非季末）。

审计（可选）::

    python scripts/audit_historical_snapshots_v1.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

from quantmind.data.snapshot import list_snapshots, patch_v1_modules


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> int:
    p = argparse.ArgumentParser(description="Patch Data Expansion v1 modules onto legacy snapshots")
    p.add_argument("--as-of", type=lambda s: date.fromisoformat(s), default=None)
    p.add_argument(
        "--all-existing-snapshots",
        action="store_true",
        help="对 list_snapshots() 的每个日期依次 patch",
    )
    p.add_argument(
        "--all-existing-quarterly",
        action="store_true",
        help="已弃用别名；请用 --all-existing-snapshots（行为：枚举已有目录，非日历季末过滤）",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="真实拉数并写盘；缺省 dry-run",
    )
    p.add_argument("--strict", action="store_true")
    p.add_argument("--overwrite-v1-modules", action="store_true")

    sb = p.add_mutually_exclusive_group(required=False)
    sb.add_argument(
        "--include-stock-basic",
        dest="inc_sb",
        action="store_true",
        help="包含 stock_basic（默认）",
    )
    sb.add_argument("--no-stock-basic", dest="inc_sb", action="store_false", help="跳过 stock_basic")
    p.set_defaults(inc_sb=True)

    hk = p.add_mutually_exclusive_group(required=False)
    hk.add_argument(
        "--include-hk-hold",
        dest="inc_hk",
        action="store_true",
        help="包含 hk_hold（默认）",
    )
    hk.add_argument("--no-hk-hold", dest="inc_hk", action="store_false", help="跳过 hk_hold")
    p.set_defaults(inc_hk=True)

    mg = p.add_mutually_exclusive_group(required=False)
    mg.add_argument(
        "--include-margin",
        dest="inc_mg",
        action="store_true",
        help="包含 margin（默认）",
    )
    mg.add_argument("--no-margin", dest="inc_mg", action="store_false", help="跳过 margin")
    p.set_defaults(inc_mg=True)

    ix = p.add_mutually_exclusive_group(required=False)
    ix.add_argument(
        "--include-index-daily",
        dest="inc_ix",
        action="store_true",
        help="包含 index_daily（默认）",
    )
    ix.add_argument(
        "--no-index-daily",
        dest="inc_ix",
        action="store_false",
        help="跳过 index_daily",
    )
    p.set_defaults(inc_ix=True)
    p.add_argument(
        "--index-codes",
        type=str,
        default=None,
        help="逗号分隔指数代码；默认 settings.data.index_daily_codes",
    )
    p.add_argument("--max-dates", type=int, default=None, help="批量时最多处理前 N 个日期")
    p.add_argument("--start-date", type=lambda s: date.fromisoformat(s), default=None)
    p.add_argument("--end-date", type=lambda s: date.fromisoformat(s), default=None)
    p.add_argument(
        "--run-audit-after",
        action="store_true",
        help="结束后运行 scripts/audit_historical_snapshots_v1.py",
    )
    args = p.parse_args()

    if args.all_existing_quarterly:
        print("warning: `--all-existing-quarterly` 为别名，等价于 `--all-existing-snapshots`。", file=sys.stderr)
        args.all_existing_snapshots = True

    idx_codes = (
        [x.strip() for x in args.index_codes.split(",") if x.strip()]
        if args.index_codes
        else None
    )

    inc_sb = bool(args.inc_sb)
    inc_hk = bool(args.inc_hk)
    inc_mg = bool(args.inc_mg)
    inc_ix = bool(args.inc_ix)

    dry_run = not args.execute

    if args.all_existing_snapshots:
        dates = list_snapshots()
        if args.start_date is not None:
            dates = [d for d in dates if d >= args.start_date]
        if args.end_date is not None:
            dates = [d for d in dates if d <= args.end_date]
        if args.max_dates is not None:
            dates = dates[: args.max_dates]
    else:
        if args.as_of is None:
            print("需要 `--as-of YYYY-MM-DD` 或 `--all-existing-snapshots`。")
            return 2
        dates = [args.as_of]

    print("# Patch Snapshot v1 Modules\n")
    print(f"- mode: **{'dry-run' if dry_run else 'execute'}**")
    if dates:
        print(f"- dates: **{len(dates)}** ({dates[0]} … {dates[-1]})")
    else:
        print("- dates: **empty** (no snapshots in range)")
        return 0

    print(f"- strict: {args.strict}")
    print(f"- overwrite_v1_modules: {args.overwrite_v1_modules}")
    print(
        f"- includes: stock_basic={inc_sb}, hk_hold={inc_hk}, margin={inc_mg}, index_daily={inc_ix}"
    )
    print(f"- index_codes: {idx_codes or 'defaults from settings'}")
    print()

    results: list[dict] = []
    for d in dates:
        try:
            r = patch_v1_modules(
                d,
                include_stock_basic=inc_sb,
                include_hk_hold=inc_hk,
                include_margin=inc_mg,
                include_index_daily=inc_ix,
                index_daily_codes=idx_codes,
                strict=args.strict,
                overwrite_v1_modules=args.overwrite_v1_modules,
                dry_run=dry_run,
            )
            r["_date"] = d.isoformat()
            r["_ok"] = True
            results.append(r)
        except Exception as e:  # noqa: BLE001
            results.append({"_date": d.isoformat(), "_ok": False, "_error": repr(e)})

    hdr = ["as_of", "ok", "dry_run", "patched", "skipped", "failed_ct", "warnings_ct"]
    rows_md = [_md_row(hdr), _md_row(["---"] * len(hdr))]
    for r in results:
        patched = ",".join(r.get("patched_modules", []) or [])
        skipped = ",".join(r.get("skipped_modules", []) or [])
        failed = r.get("failed_modules", []) or []
        warns = r.get("warnings", []) or []
        rows_md.append(
            _md_row(
                [
                    str(r.get("_date", "")),
                    str(r.get("_ok", "")),
                    str(r.get("dry_run", dry_run)),
                    patched or "—",
                    skipped or "—",
                    str(len(failed)),
                    str(len(warns)),
                ]
            )
        )
    print("## Summary table\n")
    print("\n".join(rows_md))

    if dry_run:
        print("\n## Dry-run fetch plan (`would_fetch_modules`)\n")
        for r in results:
            if not r.get("_ok"):
                continue
            wf = r.get("would_fetch_modules")
            cod = r.get("index_daily_codes_planned")
            wf_s = ",".join(wf) if isinstance(wf, list) and wf else "—"
            cod_s = ",".join(cod) if isinstance(cod, list) and cod else "—"
            print(f"- **{r.get('_date')}**: would_fetch=`{wf_s}`; index_codes=`{cod_s}`")

    for r in results:
        if not r.get("_ok"):
            print(f"\n### Error @ {r.get('_date')}\n\n```\n{r.get('_error')}\n```")

    if args.run_audit_after and dry_run:
        print("\n_(skip post audit: dry-run)_")
    elif args.run_audit_after:
        repo = Path(__file__).resolve().parents[1]
        audit = repo / "scripts" / "audit_historical_snapshots_v1.py"
        print("\n## Post-run audit\n")
        subprocess.run([sys.executable, str(audit)], cwd=str(repo), check=False)

    return 0 if all(r.get("_ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
