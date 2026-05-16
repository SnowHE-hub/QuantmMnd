#!/usr/bin/env python3
"""审计 data/snapshots 下全部时点：Data Expansion v1 四表覆盖、legacy / v1 标记.

仅读本地 parquet + meta.json，不触发网络下载，不打印任何密钥。
输出 Markdown 表格到 stdout（可重定向到文档）。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from quantmind.core.config import get_settings

V1_FILES = ("stock_basic.parquet", "hk_hold.parquet", "margin.parquet", "index_daily.parquet")


def _snapshots_root() -> Path:
    return Path(get_settings().data.dir).resolve() / "snapshots"


def _read_meta(p: Path) -> dict | None:
    meta = p / "meta.json"
    if not meta.is_file():
        return None
    return json.loads(meta.read_text(encoding="utf-8"))


def _classify(
    *,
    has_meta: bool,
    meta: dict | None,
    paths: dict[str, bool],
) -> tuple[str, str]:
    """返回 (status_tag, reason)."""
    if not has_meta or meta is None:
        return "no_meta", "missing meta.json"

    files_declared = set(meta.get("files") or [])
    dev = meta.get("data_expansion_version")

    v1_disk = all(paths.get(f, False) for f in V1_FILES)
    v1_declared = all(f in files_declared for f in V1_FILES)

    if dev == "v1" and v1_disk and v1_declared:
        return "v1_complete", ""

    if dev == "v1" and not v1_disk:
        missing = [f for f in V1_FILES if not paths.get(f, False)]
        return "v1_partial", f"meta claims v1 but files missing: {missing}"

    if dev == "v1" and not v1_declared:
        return "v1_partial", "data_expansion_version=v1 but files list incomplete in meta"

    if dev != "v1" and v1_disk:
        return "v1_partial", f"four v1 parquets exist but data_expansion_version={dev!r}"

    if not v1_disk:
        miss = [f for f in V1_FILES if not paths.get(f, False)]
        if dev is None:
            return "legacy", f"no data_expansion_version; missing {miss}"
        return "legacy_or_incomplete", f"missing {miss}; version={dev!r}"

    return "unknown", "unclassified"


def _coverage_stats(snapshot_dir: Path, meta: dict) -> dict[str, object]:
    del meta
    out: dict[str, object] = {}
    uni_p = snapshot_dir / "universe.parquet"
    if uni_p.is_file():
        u = pd.read_parquet(uni_p, columns=["ticker"])
        out["n_universe"] = int(len(u))
    else:
        out["n_universe"] = None

    hk_p = snapshot_dir / "hk_hold.parquet"
    if hk_p.is_file():
        df = pd.read_parquet(hk_p, columns=["ticker"])
        out["n_hk_cov"] = int(df["ticker"].nunique())
    else:
        out["n_hk_cov"] = None

    mg_p = snapshot_dir / "margin.parquet"
    if mg_p.is_file():
        df = pd.read_parquet(mg_p, columns=["ticker"])
        out["n_margin_cov"] = int(df["ticker"].nunique())
    else:
        out["n_margin_cov"] = None

    ix_p = snapshot_dir / "index_daily.parquet"
    if ix_p.is_file():
        df = pd.read_parquet(ix_p, columns=["ts_code"])
        out["index_n_codes"] = int(df["ts_code"].nunique())
    else:
        out["index_n_codes"] = None

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit historical snapshots for Data Expansion v1")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="snapshots 根目录（默认 settings.data.dir/snapshots）",
    )
    args = parser.parse_args()

    root = args.root.resolve() if args.root else _snapshots_root()
    if not root.is_dir():
        print(f"# Error: snapshots root not found: `{root}`")
        return 1

    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda x: x.name)

    rows: list[list[str]] = []
    for d in dirs:
        name = d.name
        meta = _read_meta(d)
        has_meta = meta is not None

        paths = {f: (d / f).is_file() for f in V1_FILES}
        paths["meta.json"] = (d / "meta.json").is_file()

        tag, reason = _classify(has_meta=has_meta, meta=meta, paths=paths)

        dev = ""
        strict = ""
        use_fe = "no"
        if meta:
            dev = str(meta.get("data_expansion_version", ""))
            strict = str(meta.get("snapshot_strict", ""))
            files_ok = all(paths[f] for f in V1_FILES)
            if tag == "v1_complete":
                use_fe = "yes"
            elif files_ok and dev == "v1":
                use_fe = "yes*"
            elif files_ok:
                use_fe = "maybe"

        cov: dict[str, object] = {}
        if meta:
            try:
                cov = _coverage_stats(d, meta)
            except Exception:
                cov = {}

        n_uni = cov.get("n_universe", "")
        row = [
            name,
            "yes" if paths["meta.json"] else "no",
            dev or "—",
            "yes" if paths["stock_basic.parquet"] else "no",
            "yes" if paths["hk_hold.parquet"] else "no",
            "yes" if paths["margin.parquet"] else "no",
            "yes" if paths["index_daily.parquet"] else "no",
            str(n_uni),
            str(cov.get("n_hk_cov", "")),
            str(cov.get("n_margin_cov", "")),
            str(cov.get("index_n_codes", "")),
            use_fe,
            tag,
            reason.replace("|", "/")[:120],
        ]
        rows.append(row)

    headers = [
        "as_of",
        "meta",
        "data_expansion_version",
        "stock_basic",
        "hk_hold",
        "margin",
        "index_daily",
        "n_universe",
        "hk_tickers",
        "margin_tickers",
        "index_codes",
        "feature_expansion_ok",
        "status",
        "notes",
    ]

    print("# Historical Snapshots — Data Expansion v1 Audit\n")
    print(f"_snapshots root_: `{root}`\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")

    print("\n### Legend\n")
    print("- **status**: `v1_complete` | `v1_partial` | `legacy` | `legacy_or_incomplete` | `no_meta`")
    print("- **feature_expansion_ok**: `yes` = 四表齐且 meta 声明 v1；`yes*` = 表齐；`maybe` = 需人工确认")

    c = Counter(r[-2] for r in rows)
    print("\n### Summary (by status)\n")
    for k, v in sorted(c.items()):
        print(f"- **{k}**: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
