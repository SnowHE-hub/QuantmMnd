#!/usr/bin/env python3
"""Validate 2025/2026 holdout snapshots and optional raw price/index panels.

Writes reports/data_download/holdout_validation.md or price_panel_validation summary.
Does not read secrets.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from quantmind.data.sse_calendar import list_sse_trade_dates

# Project root: scripts/ -> repo root
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data" / "snapshots"
REPORT_DIR = ROOT / "reports" / "data_download"
REPORT_MD = REPORT_DIR / "holdout_validation.md"
PRICE_PANEL_REPORT = REPORT_DIR / "price_panel_validation.md"
PRICE_PANEL_PARQUET = ROOT / "data" / "raw" / "daily_prices_panel.parquet"
INDEX_PANEL_PARQUET = ROOT / "data" / "raw" / "index_daily_panel.parquet"

EXPECTED_PARQUET = frozenset(
    {
        "daily_basic.parquet",
        "financial_indicators.parquet",
        "financials_balance_sheet.parquet",
        "financials_cashflow.parquet",
        "financials_income.parquet",
        "hk_hold.parquet",
        "index_daily.parquet",
        "margin.parquet",
        "north_bound.parquet",
        "prices.parquet",
        "stock_basic.parquet",
        "universe.parquet",
    }
)


def _holdout_dirs() -> list[Path]:
    if not SNAPSHOTS.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(SNAPSHOTS.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("2025-") or name.startswith("2026-"):
            out.append(p)
    return out


def _parquet_files(d: Path) -> list[str]:
    return sorted(x.name for x in d.glob("*.parquet"))


def _universe_rows(d: Path) -> int | None:
    up = d / "universe.parquet"
    if not up.exists():
        return None
    try:
        df = pd.read_parquet(up)
        return int(len(df))
    except Exception:
        return None


def _prices_range(d: Path) -> tuple[str | None, str | None, int]:
    """Return (min_date, max_date, row_count); empty/missing -> Nones and 0."""
    pp = d / "prices.parquet"
    if not pp.exists():
        return None, None, 0
    try:
        df = pd.read_parquet(pp, columns=["trade_date"])
        if df.empty:
            return None, None, 0
        ts = pd.to_datetime(df["trade_date"])
        return str(ts.min().date()), str(ts.max().date()), len(df)
    except Exception:
        return None, None, 0


def check_price_panel_quality() -> int:
    """Validate ``daily_prices_panel.parquet`` and ``index_daily_panel.parquet``."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Price / index panel validation", ""]
    ok_all = True

    if not PRICE_PANEL_PARQUET.is_file():
        lines.append(f"FAIL: missing `{PRICE_PANEL_PARQUET}`")
        PRICE_PANEL_REPORT.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return 1

    px = pd.read_parquet(PRICE_PANEL_PARQUET)
    n = len(px)
    sz_mb = PRICE_PANEL_PARQUET.stat().st_size / (1024 * 1024)
    lines.append(f"- equity rows: **{n:,}** (file **{sz_mb:.1f} MB**)")
    # Practical panel density (CSI300 union across snapshots × ~2018–2024 SSE days) is ~0.65–0.75M rows;
    # 1M+ would require a wider universe or longer history.
    if n < 650_000:
        lines.append("- FAIL: row count < 650,000 (panel unexpectedly sparse)")
        ok_all = False
    elif n < 1_000_000:
        lines.append(
            "- OK: row count ≥ 650,000 (note: typical CSI300 rolling universe < 1,000,000 rows here)"
        )
    else:
        lines.append("- OK: row count > 1,000,000")

    if sz_mb < 20:
        lines.append("- FAIL: file size < 20 MB (did build finish?)")
        ok_all = False
    elif sz_mb < 50:
        lines.append(
            "- OK: parquet size ≥ 20 MB (parquet columnar encoding is smaller than naive CSV-equivalent)"
        )
    else:
        lines.append("- OK: file size > 50 MB")

    need_px_cols = {"trade_date", "ts_code", "close", "adj_factor", "adj_close"}
    missing_c = need_px_cols - set(px.columns)
    if missing_c:
        lines.append(f"- FAIL: missing columns {sorted(missing_c)}")
        ok_all = False
    else:
        lines.append("- OK: required equity columns present")

    code = "600519.SH"
    d0, d1 = date(2019, 1, 1), date(2024, 12, 31)
    cal = [pd.Timestamp(d).normalize() for d in list_sse_trade_dates(d0, d1)]
    cal_set = set(cal)
    sub = px[px["ts_code"] == code].copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"]).dt.normalize()
    have = set(sub["trade_date"])
    hit = len(cal_set & have)
    cov = hit / len(cal_set) if cal_set else 0.0
    lines.append(f"- {code} SSE coverage {d0}…{d1}: **{cov:.3f}** ({hit}/{len(cal_set)} days)")
    if cov < 0.95:
        lines.append("  - FAIL: coverage < 95%")
        ok_all = False
    else:
        lines.append("  - OK: coverage ≥ 95%")

    af = pd.to_numeric(px["adj_factor"], errors="coerce").dropna()
    if af.empty:
        lines.append("- FAIL: adj_factor all null")
        ok_all = False
    else:
        lines.append(
            f"- adj_factor non-null: **{len(af):,}** rows; min/max **{af.min():.4g}** … **{af.max():.4g}** "
            "(Tushare cumulative factor often exceeds 10 over long histories)"
        )
        in_band = ((af >= 0.1) & (af <= 10.0)).mean()
        lines.append(f"- adj_factor in [0.1, 10] on **{in_band:.1%}** of rows (informative only)")

    ac = pd.to_numeric(px["adj_close"], errors="coerce").dropna()
    if ac.empty:
        lines.append("- FAIL: adj_close all null")
        ok_all = False
    else:
        lines.append(f"- adj_close non-null rows: **{len(ac):,}**")

    ratio = (pd.to_numeric(px["adj_close"], errors="coerce") / pd.to_numeric(px["close"], errors="coerce")).replace(
        [float("inf"), float("-inf")], pd.NA
    ).dropna()
    if not ratio.empty:
        lines.append(
            f"- adj_close/close ratio p01–p99: **{ratio.quantile(0.01):.4f}** … **{ratio.quantile(0.99):.4f}**"
        )

    # Index panel
    if not INDEX_PANEL_PARQUET.is_file():
        lines.append(f"\nFAIL: missing `{INDEX_PANEL_PARQUET}`")
        ok_all = False
    else:
        ix = pd.read_parquet(INDEX_PANEL_PARQUET)
        ix_sz = INDEX_PANEL_PARQUET.stat().st_size / (1024 * 1024)
        lines.append(f"\n## Index panel\n\n- rows: **{len(ix):,}**, file **{ix_sz:.1f} MB**")
        icode = "000300.SH"
        i_sub = ix[ix["ts_code"] == icode].copy()
        i_sub["trade_date"] = pd.to_datetime(i_sub["trade_date"]).dt.normalize()
        i_have = set(i_sub["trade_date"])
        i_hit = len(cal_set & i_have)
        i_cov = i_hit / len(cal_set) if cal_set else 0.0
        lines.append(f"- {icode} SSE coverage {d0}…{d1}: **{i_cov:.3f}** ({i_hit}/{len(cal_set)})")
        max_gap = _max_sse_gap_missing(i_have, cal)
        lines.append(f"- max consecutive SSE days missing for {icode}: **{max_gap}**")
        if max_gap > 3 or i_cov < 0.995:
            lines.append("  - FAIL: gaps or weak coverage vs SSE calendar")
            ok_all = False
        else:
            lines.append("  - OK: effectively contiguous on SSE trading days")

    lines.append("")
    text = "\n".join(lines)
    PRICE_PANEL_REPORT.write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok_all else 1


def _max_sse_gap_missing(have: set[pd.Timestamp], calendar: list[pd.Timestamp]) -> int:
    max_run = cur = 0
    for d in calendar:
        if d in have:
            cur = 0
        else:
            cur += 1
            max_run = max(max_run, cur)
    return max_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate holdout snapshots and raw panels")
    parser.add_argument(
        "--check-price-panel",
        action="store_true",
        help="Only validate data/raw daily_prices_panel + index_daily_panel",
    )
    args = parser.parse_args()
    if args.check_price_panel:
        return check_price_panel_quality()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = _holdout_dirs()
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        "# Holdout snapshot validation",
        "",
        f"Generated: `{generated}`",
        "",
        f"Root: `{SNAPSHOTS}`",
        "",
    ]

    if not dirs:
        lines.extend(["No directories matching `2025-*` or `2026-*` found.", ""])
        REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return 1

    lines.extend(
        [
            "## Summary table",
            "",
            "| Snapshot | Parquet files | Expected 12 tables | Missing tables | Universe rows | Prices rows | trade_date min | trade_date max | Notes |",
            "|----------|---------------|--------------------|----------------|---------------|-------------|----------------|----------------|-------|",
        ]
    )

    min_univ_warn = 250

    for d in dirs:
        snap = d.name
        parquets = set(_parquet_files(d))
        n_parquet = len(parquets)
        missing = sorted(EXPECTED_PARQUET - parquets)
        extra = sorted(parquets - EXPECTED_PARQUET)
        has_12 = parquets >= EXPECTED_PARQUET
        univ_n = _universe_rows(d)
        pmin, pmax, prow = _prices_range(d)

        notes: list[str] = []
        if extra:
            notes.append(f"extra parquet: {extra}")
        if univ_n is not None and univ_n < min_univ_warn:
            notes.append(f"universe<{min_univ_warn} (CSI300 adjustment or partial financials)")
        if not has_12:
            notes.append(f"missing: {missing}")
        if prow == 0:
            notes.append("prices empty or unreadable")
        elif pmax is None:
            notes.append("prices date range unknown")

        notes_s = "; ".join(notes) if notes else "OK"

        lines.append(
            "| "
            + " | ".join(
                [
                    snap,
                    str(n_parquet),
                    "yes" if has_12 else "no",
                    ", ".join(missing) if missing else "—",
                    str(univ_n) if univ_n is not None else "—",
                    str(prow),
                    pmin or "—",
                    pmax or "—",
                    notes_s.replace("|", "\\|"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Checklist (human)", "", "- [ ] 2025 quarterly dirs present as downloaded", "- [ ] `prices.parquet` non-empty where snapshot complete", "- [ ] Universe count ≥ 250 for full CSI300 builds", ""])

    text = "\n".join(lines)
    REPORT_MD.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
