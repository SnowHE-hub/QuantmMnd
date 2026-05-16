#!/usr/bin/env python3
"""Phase B1 — Download CSI300 daily prices + index_daily into data/prices/.

Uses ``TUSHARE_TOKEN`` from environment only (never logged).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _tushare_yyyymmdd(s: str) -> str:
    return pd.Timestamp(str(s).strip()).strftime("%Y%m%d")


def _setup_logging(log_file: Path | None) -> None:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _pro_api(timeout: int = 120):
    import tushare as ts  # noqa: PLC0415

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise SystemExit("TUSHARE_TOKEN environment variable is empty")
    ts.set_token(token)
    return ts.pro_api(timeout=timeout)


def _read_universe_tickers(universe_path: Path) -> list[str]:
    df = pd.read_parquet(universe_path)
    if "ts_code" in df.columns:
        col = "ts_code"
    elif "ticker" in df.columns:
        col = "ticker"
    else:
        raise SystemExit("universe.parquet must contain ts_code or ticker column")
    return sorted(df[col].astype(str).unique().tolist())


def _retry_api(fn: Callable[[], Any], log: logging.Logger, label: str, attempts: int = 4, pause: float = 1.2) -> Any:
    """Retry Tushare HTTP calls on timeouts/transient errors."""
    last_exc: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            log.warning("%s attempt %d/%d: %s", label, i + 1, attempts, str(e)[:200])
            time.sleep(pause * (i + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError(label)


def _read_ticker_lines(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").splitlines()
    out = sorted({ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")})
    return out


def _finalize_long_price_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Align columns/types with ``data/raw/daily_prices_panel.parquet`` long format."""
    out = df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out = out.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
    out["pct_chg"] = out.groupby("ts_code", sort=False)["close"].pct_change() * 100.0
    want = [
        "trade_date",
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "pct_chg",
        "adj_factor",
        "adj_close",
    ]
    for c in want:
        if c not in out.columns:
            out[c] = pd.NA if c in ("adj_factor",) else np.nan
    return out[want]


def _merge_daily_adj(daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame()
    df = daily.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    if adj is None or adj.empty:
        df["adj_factor"] = pd.NA
    else:
        a = adj.copy()
        a["trade_date"] = pd.to_datetime(a["trade_date"], format="%Y%m%d", errors="coerce")
        df = df.merge(a[["trade_date", "adj_factor"]], on="trade_date", how="left")
    df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
    df = df.sort_values("trade_date").reset_index(drop=True)
    latest = df["adj_factor"].iloc[-1]
    if pd.notna(latest) and latest != 0:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce") * df["adj_factor"] / float(latest)
    else:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CSI300 daily + index panel (Tushare)")
    parser.add_argument("--universe-parquet", type=Path, default=ROOT / "data/snapshots/2024-12-31/universe.parquet")
    parser.add_argument("--start-date", type=str, default="20190101")
    parser.add_argument("--end-date", type=str, default="20261231")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--http-timeout", type=int, default=120, help="Tushare HTTP read timeout seconds")
    parser.add_argument("--failed-output", type=Path, default=ROOT / "data/prices/failed_tickers.txt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-file", type=Path, default=ROOT / "reports/data_download/daily_price_download.log")
    parser.add_argument("--out-adj-wide", type=Path, default=ROOT / "data/prices/csi300_daily_adj_close.parquet")
    parser.add_argument("--out-ohlcv", type=Path, default=ROOT / "data/prices/csi300_daily_ohlcv.parquet")
    parser.add_argument("--out-index", type=Path, default=ROOT / "data/prices/index_daily.parquet")
    parser.add_argument(
        "--ticker-file",
        type=Path,
        default=None,
        help="仅下载该文件列出的 ts_code（一行一只），合并进已有 OHLCV / adj_close 宽表",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="写出 long-format 日线面板 parquet（与 --ticker-file 配合；不写宽表 / 不改 index_daily）",
    )
    args = parser.parse_args()

    partial_mode = args.ticker_file is not None
    long_panel_only = partial_mode and args.output is not None
    if partial_mode and not args.ticker_file.is_file():
        raise SystemExit(f"--ticker-file not found: {args.ticker_file}")
    if args.output is not None and not partial_mode:
        raise SystemExit("--output requires --ticker-file")

    args.out_adj_wide.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)
    _setup_logging(args.log_file)

    log = logging.getLogger("build_daily_price_panel")

    if (
        not args.overwrite
        and long_panel_only
        and args.output.is_file()
    ):
        log.info("Long panel output exists and --overwrite not set; skipping download.")
        return 0

    if (
        not args.overwrite
        and not partial_mode
        and args.out_adj_wide.is_file()
        and args.out_ohlcv.is_file()
        and args.out_index.is_file()
    ):
        log.info("Output parquet files exist and --overwrite not set; skipping download.")
        return 0

    if partial_mode and not long_panel_only and (not args.out_ohlcv.is_file() or not args.out_adj_wide.is_file()):
        raise SystemExit("ticker-file mode requires existing --out-ohlcv and --out-adj-wide parquet to merge into")

    pro = _pro_api(args.http_timeout)
    if partial_mode:
        tickers = _read_ticker_lines(args.ticker_file)
        log.info("Ticker-file symbols: %d", len(tickers))
    else:
        tickers = _read_universe_tickers(args.universe_parquet)
        log.info("Universe tickers: %d", len(tickers))

    failed: list[str] = []
    parts: list[pd.DataFrame] = []

    start, end = _tushare_yyyymmdd(args.start_date), _tushare_yyyymmdd(args.end_date)

    for bi in range(0, len(tickers), args.batch_size):
        batch = tickers[bi : bi + args.batch_size]
        log.info("Batch %d-%d / %d", bi + 1, bi + len(batch), len(tickers))
        for t in batch:
            try:
                time.sleep(args.sleep)
                daily = _retry_api(
                    lambda: pro.daily(ts_code=t, start_date=start, end_date=end),
                    log,
                    f"daily {t}",
                )
                time.sleep(args.sleep)
                adj = _retry_api(
                    lambda: pro.adj_factor(ts_code=t, start_date=start, end_date=end),
                    log,
                    f"adj_factor {t}",
                )
                merged = _merge_daily_adj(
                    daily if daily is not None else pd.DataFrame(),
                    adj if adj is not None else pd.DataFrame(),
                )
                if merged.empty:
                    failed.append(t)
                    log.warning("Empty daily for %s", t)
                    continue
                merged["ts_code"] = t
                keep_cols = [
                    "trade_date",
                    "ts_code",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "amount",
                    "pre_close",
                    "adj_factor",
                    "adj_close",
                ]
                for c in keep_cols:
                    if c not in merged.columns and c == "pre_close":
                        merged["pre_close"] = pd.NA
                merged = merged[[c for c in keep_cols if c in merged.columns]]
                merged["volume"] = pd.to_numeric(merged.get("vol"), errors="coerce")
                parts.append(merged)
            except Exception as e:  # noqa: BLE001
                log.warning("Failed ticker %s: %s", t, str(e)[:200])
                failed.append(t)
        if bi + args.batch_size < len(tickers):
            log.info("Sleep 2s between batches")
            time.sleep(2.0)

    if not parts:
        log.error("No price data downloaded.")
        args.failed_output.write_text("\n".join(sorted(set(failed))), encoding="utf-8")
        return 1

    new_chunk = pd.concat(parts, ignore_index=True)
    if long_panel_only:
        panel = _finalize_long_price_panel(new_chunk)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(args.output, index=False, compression="snappy")
        args.failed_output.write_text("\n".join(sorted(set(failed))), encoding="utf-8")
        ok = len(tickers) - len(set(failed))
        log.info("Long panel written to %s success=%d failed=%d", args.output, ok, len(set(failed)))
        print(f"success_tickers={ok} failed_tickers={len(set(failed))} output={args.output} failed_file={args.failed_output}")
        return 0

    if partial_mode:
        old = pd.read_parquet(args.out_ohlcv).reset_index()
        ohlcv = pd.concat([old, new_chunk], ignore_index=True)
    else:
        ohlcv = new_chunk

    ohlcv = ohlcv.sort_values(["trade_date", "ts_code"]).drop_duplicates(["trade_date", "ts_code"], keep="last")
    ohlcv_mi = ohlcv.set_index(["trade_date", "ts_code"]).sort_index()
    ohlcv_mi.to_parquet(args.out_ohlcv, compression="snappy")

    wide = ohlcv.pivot(index="trade_date", columns="ts_code", values="adj_close")
    wide = wide.sort_index()
    wide.to_parquet(args.out_adj_wide, compression="snappy")

    idx_parts: list[pd.DataFrame] = []
    if not partial_mode:
        for code in ("000300.SH", "000905.SH"):
            try:
                time.sleep(args.sleep)
                idf = _retry_api(
                    lambda c=code: pro.index_daily(ts_code=c, start_date=start, end_date=end),
                    log,
                    f"index_daily {code}",
                )
                if idf is None or idf.empty:
                    log.warning("Empty index_daily %s", code)
                    continue
                idf = idf.copy()
                idf["trade_date"] = pd.to_datetime(idf["trade_date"], format="%Y%m%d", errors="coerce")
                idf["ts_code"] = code
                cols = [c for c in ["trade_date", "ts_code", "close", "pct_chg", "open", "high", "low", "vol"] if c in idf.columns]
                idx_parts.append(idf[cols])
            except Exception as e:  # noqa: BLE001
                log.warning("index_daily failed %s: %s", code, str(e)[:200])
    else:
        log.info("Skipping index_daily download (ticker-file partial mode)")

    if idx_parts:
        ix = pd.concat(idx_parts, ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
        ix = ix.set_index(["trade_date", "ts_code"]).sort_index()
        ix.to_parquet(args.out_index, compression="snappy")

    args.failed_output.write_text("\n".join(sorted(set(failed))), encoding="utf-8")
    ok = len(tickers) - len(set(failed))
    log.info("Done. success=%d failed=%d", ok, len(set(failed)))
    print(f"success_tickers={ok} failed_tickers={len(set(failed))} failed_file={args.failed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
