#!/usr/bin/env python3
"""审计单个 snapshot 的 Data Expansion v1 模块：覆盖率、PIT、空列 — Markdown 输出。

用法::

    python scripts/audit_data_expansion_v1_snapshot.py --as-of 2024-12-31
    python scripts/audit_data_expansion_v1_snapshot.py --snapshot-dir data/snapshots/2024-12-31

不访问密钥；不发起下载。
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantmind.core.config import get_settings
from quantmind.data.snapshot import load_snapshot


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _ticker_col(df: pd.DataFrame) -> str | None:
    if "ticker" in df.columns:
        return "ticker"
    if "ts_code" in df.columns:
        return "ts_code"
    return None


def _pit_md(df: pd.DataFrame | None, as_of: date, trade_col: str = "trade_date") -> str:
    if df is None or df.empty:
        return "n/a（空表）"
    if trade_col not in df.columns:
        return "n/a（无日期列）"
    s = pd.to_datetime(df[trade_col], errors="coerce")
    mx, mn = s.max(), s.min()
    ok = pd.notna(mx) and bool(mx <= pd.Timestamp(as_of))
    return (
        f"min={mn} max={mx} **{'PASS' if ok else 'FAIL'}** "
        f"(max <= {as_of})"
    )


def _empty_cols(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return []
    return [c for c in df.columns if df[c].isna().all()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Data Expansion v1 snapshot")
    ap.add_argument("--as-of", type=_parse_date, default=None)
    ap.add_argument("--snapshot-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.snapshot_dir is not None:
        snap_dir = args.snapshot_dir.resolve()
        as_of = date.fromisoformat(snap_dir.name)
    elif args.as_of is not None:
        as_of = args.as_of
        snap_dir = Path(get_settings().data.dir) / "snapshots" / as_of.isoformat()
    else:
        print("需要 --as-of 或 --snapshot-dir", file=sys.stderr)
        return 2

    snap = load_snapshot(as_of)
    meta = snap["meta"]
    assert isinstance(meta, dict)
    u = snap.get("universe")
    n_uni = len(u) if u is not None and hasattr(u, "__len__") else 0

    lines: list[str] = []
    lines.append("## Data Expansion v1 — Snapshot 审计")
    lines.append("")
    lines.append(f"| 项 | 值 |")
    lines.append(f"|---|---|")
    lines.append(f"| as_of | `{as_of}` |")
    lines.append(f"| snapshot_dir | `{snap_dir}` |")
    lines.append(f"| universe 行数 | {n_uni} |")
    lines.append(
        f"| data_expansion_version | `{meta.get('data_expansion_version', 'legacy')}` |"
    )
    lines.append(f"| meta.modules | `{list((meta.get('modules') or {}).keys())}` |")
    lines.append("")

    for key, label in (
        ("stock_basic", "stock_basic"),
        ("hk_hold", "hk_hold"),
        ("margin", "margin"),
        ("index_daily", "index_daily"),
    ):
        df = snap.get(key)
        mod = (meta.get("modules") or {}).get(key, {})
        lines.append(f"### {label}")
        if df is None:
            lines.append("- **状态**: 缺失（未加载）")
            lines.append("")
            continue
        lines.append(f"- **行数**: {len(df)}")
        if key == "index_daily" and "ts_code" in df.columns:
            n_ix = df["ts_code"].nunique()
            exp = len(get_settings().data.index_daily_codes)
            lines.append(
                f"- **指数 ts_code 去重**: {n_ix} / 配置 {exp} 只指数（**{100.0 * min(n_ix, exp) / max(exp, 1):.1f}%** 满配）"
            )
        else:
            tc = _ticker_col(df)
            if tc and n_uni:
                n_cov = df[tc].nunique()
                pct = 100.0 * n_cov / max(n_uni, 1)
                lines.append(
                    f"- **{tc} 覆盖**: {n_cov} / {n_uni} 只（**{pct:.1f}%**）"
                )
        if key == "stock_basic" and "industry" in df.columns:
            non_empty = df["industry"].astype(str).str.strip().ne("") & df["industry"].notna()
            lines.append(
                f"- **industry 非空**: {int(non_empty.sum())} / {len(df)} "
                f"（**{100.0 * non_empty.mean():.1f}%**）"
            )
        lines.append(f"- **列名**: `{list(df.columns)}`")
        if key == "stock_basic":
            if "list_date" in df.columns:
                s = pd.to_datetime(df["list_date"], errors="coerce")
                lines.append(f"- **list_date 范围**: min={s.min()} max={s.max()}")
            lines.append("- **说明**: 静态 `stock_basic` 非日频 PIT 事件表；行业为构建时截面。")
        else:
            lines.append(f"- **PIT (trade_date)**: {_pit_md(df, as_of, 'trade_date')}")
        empty = _empty_cols(df)
        lines.append(f"- **全空列**: {empty if empty else '无'}")
        if mod:
            lines.append(
                f"- **manifest**: rows={mod.get('row_count')} "
                f"pit_col={mod.get('pit_date_column')} "
                f"schema={mod.get('schema_version')}"
            )
        lines.append("")

    lines.append("### 兼容性说明")
    lines.append(
        "- 旧版 meta 无 `modules` / `data_expansion_version` 属正常；"
        "新构建应带 `data_expansion_version: v1`。"
    )
    lines.append("")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
