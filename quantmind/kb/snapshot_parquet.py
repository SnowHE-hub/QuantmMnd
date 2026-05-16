"""Parquet snapshot → KB documents (short structured summaries, no full-table dump)."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# Optional tables (strict=False → warn and skip if missing)
SNAPSHOT_PARQUET_FILES: tuple[str, ...] = (
    "universe.parquet",
    "stock_basic.parquet",
    "prices.parquet",
    "daily_basic.parquet",
    "financial_indicators.parquet",
    "hk_hold.parquet",
    "margin.parquet",
    "index_daily.parquet",
)

INDEX_CONTEXT_CODES: tuple[str, ...] = (
    "000300.SH",
    "000905.SH",
    "399006.SZ",
    "000001.SH",
)

FINANCIAL_METRIC_COLUMNS: tuple[str, ...] = (
    "roe",
    "roe_dt",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "eps",
    "bps",
    "ocfps",
    "cfps",
    "profit_to_gr",
    "q_gr_yoy",
    "q_profit_yoy",
    "or_yoy",
    "tr_yoy",
    "roe_yoy",
)


def _is_empty_scalar(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return True
    if pd.isna(v):
        return True
    s = str(v).strip().lower()
    return s in ("", "nan", "none", "nat")


def _fmt_val(v: Any) -> str | None:
    if _is_empty_scalar(v):
        return None
    if isinstance(v, (datetime, pd.Timestamp)):
        try:
            return pd.Timestamp(v).strftime("%Y-%m-%d")
        except Exception:
            return str(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    return str(v).strip()


def _normalize_ticker_code(t: str) -> str:
    t = (t or "").strip().upper()
    if not t:
        return ""
    if "." in t:
        return t
    if len(t) == 6 and t.isdigit():
        if t.startswith("6"):
            return f"{t}.SH"
        return f"{t}.SZ"
    return t


def _df_filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    want = _normalize_ticker_code(ticker)
    code6 = want.split(".")[0] if "." in want else want
    if "ts_code" in df.columns:
        ts = df["ts_code"].astype(str).str.strip().str.upper()
        m = (ts == want) | (ts == code6) | ts.str.startswith(f"{code6}.")
        sub = df[m]
        if not sub.empty:
            return sub
    if "ticker" in df.columns:
        tk = df["ticker"].map(lambda x: _normalize_ticker_code(str(x)))
        sub = df[tk == want]
        if not sub.empty:
            return sub
    if "code" in df.columns:
        cd = df["code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        m = (cd == code6) | (cd == want.split(".")[0])
        sub = df[m]
        if not sub.empty:
            return sub
    return df.iloc[0:0]


def _parse_trade_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def parquet_row_count(path: str | Path) -> int | None:
    """Row count from parquet footer only (no full table load)."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        import pyarrow.parquet as pq

        meta = pq.read_metadata(p)
        return int(meta.num_rows)
    except Exception:  # noqa: BLE001
        return None


def read_universe_tickers(snapshot_dir: str | Path) -> list[str]:
    """Load distinct tickers from universe.parquet (``ts_code`` or ``ticker``)."""
    root = Path(snapshot_dir)
    univ = root / "universe.parquet"
    if not univ.is_file():
        return []
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(univ)
        col = "ts_code" if "ts_code" in schema.names else "ticker" if "ticker" in schema.names else None
        if not col:
            logger.warning(f"[snapshot_parquet] universe.parquet has no ts_code/ticker: {schema.names}")
            return []
        df = pd.read_parquet(univ, columns=[col])
        s = df[col].astype(str).str.strip().str.upper()
        return sorted(s.unique().tolist())
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[snapshot_parquet] read_universe_tickers failed {univ}: {e}")
        return []


def load_snapshot_parquets(
    snapshot_dir: str | Path,
    *,
    strict: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Load optional parquet tables from snapshot_dir.

    Returns:
        (name -> dataframe without .parquet suffix, loaded basenames, missing basenames)
    """
    root = Path(snapshot_dir).resolve()
    loaded: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for fname in SNAPSHOT_PARQUET_FILES:
        path = root / fname
        key = fname.replace(".parquet", "")
        if not path.is_file():
            missing.append(fname)
            msg = f"[snapshot_parquet] missing table {fname} under {root}"
            if strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)
            continue
        try:
            loaded[key] = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            msg = f"[snapshot_parquet] failed to read {path}: {e}"
            if strict:
                raise
            logger.warning(msg)
            missing.append(fname)
    return loaded, list(f"{k}.parquet" for k in loaded), missing


def _coerce_as_of_date(as_of: str | date | None, snapshot_dir: Path) -> date:
    if isinstance(as_of, date):
        return as_of
    if isinstance(as_of, str) and as_of.strip():
        return datetime.strptime(as_of.strip(), "%Y-%m-%d").date()
    # Infer from directory name YYYY-MM-DD
    try:
        return date.fromisoformat(snapshot_dir.name)
    except ValueError:
        return date.today()


def _doc_base(
    *,
    text: str,
    ticker: str,
    as_of_d: date,
    doc_type: str,
    table_name: str,
    snapshot_dir: Path,
    title: str,
) -> dict[str, Any]:
    snap_s = str(snapshot_dir.resolve())
    return {
        "text": text.strip(),
        "source": f"snapshot:{ticker}:{doc_type}",
        "source_type": "snapshot",
        "ticker": ticker,
        "published_date": as_of_d.isoformat(),
        "title": title,
        "doc_type": doc_type,
        "table_name": table_name,
        "snapshot_dir": snap_s,
        "as_of": as_of_d.isoformat(),
    }


def _company_profile_row(
    tables: dict[str, pd.DataFrame],
    ticker: str,
    as_of_d: date,
    snapshot_dir: Path,
) -> dict[str, Any] | None:
    sb = tables.get("stock_basic")
    row: pd.DataFrame
    src_table = "stock_basic"
    if sb is None or sb.empty:
        uni = tables.get("universe")
        if uni is None or uni.empty:
            return None
        row = _df_filter_ticker(uni, ticker)
        if row.empty:
            return None
        src_table = "universe"
        r = row.iloc[-1]
        ts_c = _fmt_val(r.get("ts_code", ticker)) or ticker
        lines = [
            f"公司概览（snapshot as_of={as_of_d}）",
            f"股票代码：{_normalize_ticker_code(ts_c)}",
        ]
        for col in ("name", "fullname", "stock_name"):
            if col in r.index:
                nm = _fmt_val(r.get(col))
                if nm:
                    lines.append(f"名称：{nm}")
                    break
        text = "\n".join(lines)
        return _doc_base(
            text=text,
            ticker=_normalize_ticker_code(ticker),
            as_of_d=as_of_d,
            doc_type="company_profile",
            table_name=src_table,
            snapshot_dir=snapshot_dir,
            title=f"{ts_c} 公司概览 {as_of_d}",
        )
    row = _df_filter_ticker(sb, ticker)
    if row.empty:
        # universe-only fallback name
        uni = tables.get("universe")
        if uni is not None and not uni.empty and "ts_code" in uni.columns:
            row = _df_filter_ticker(uni, ticker)
        if row.empty:
            return None
    r = row.iloc[-1]
    name = _fmt_val(r.get("name", r.get("fullname", "")))
    ind = _fmt_val(r.get("industry", ""))
    area = _fmt_val(r.get("area", ""))
    ld = _fmt_val(r.get("list_date", ""))
    ts_c = _fmt_val(r.get("ts_code", ticker)) or ticker
    lines = [
        f"公司概览（snapshot as_of={as_of_d}）",
        f"股票代码：{ts_c}",
    ]
    if name:
        lines.append(f"名称：{name}")
    if ind:
        lines.append(f"行业：{ind}")
    if area:
        lines.append(f"地区：{area}")
    if ld:
        lines.append(f"上市日期：{ld}")
    text = "\n".join(lines)
    return _doc_base(
        text=text,
        ticker=_normalize_ticker_code(ticker),
        as_of_d=as_of_d,
        doc_type="company_profile",
        table_name="stock_basic",
        snapshot_dir=snapshot_dir,
        title=f"{ts_c} 公司概览 {as_of_d}",
    )


def _latest_market_metrics(
    tables: dict[str, pd.DataFrame],
    ticker: str,
    as_of_d: date,
    snapshot_dir: Path,
) -> dict[str, Any] | None:
    prices = tables.get("prices")
    db = tables.get("daily_basic")
    parts: list[str] = [f"市场指标摘要（as_of={as_of_d}）", f"股票：{_normalize_ticker_code(ticker)}"]
    if prices is not None and not prices.empty and "trade_date" in prices.columns:
        sub = _df_filter_ticker(prices, ticker)
        if not sub.empty:
            sub = sub.copy()
            sub["_td"] = _parse_trade_date(sub["trade_date"])
            sub = sub[sub["_td"].notna() & (sub["_td"].dt.date <= as_of_d)]
            if not sub.empty:
                sub = sub.sort_values("_td").tail(3)
                for _, r in sub.iterrows():
                    td = _fmt_val(r.get("trade_date", ""))
                    seg: list[str] = []
                    for col in ("close", "pct_chg", "vol", "amount"):
                        if col in r.index:
                            fv = _fmt_val(r.get(col))
                            if fv:
                                seg.append(f"{col}={fv}")
                    if seg:
                        parts.append(f"行情 {td}: " + ", ".join(seg))
    if db is not None and not db.empty:
        sub = _df_filter_ticker(db, ticker)
        if not sub.empty and "trade_date" in sub.columns:
            sub = sub.copy()
            sub["_td"] = _parse_trade_date(sub["trade_date"])
            sub = sub[sub["_td"].notna() & (sub["_td"].dt.date <= as_of_d)]
            if not sub.empty:
                r = sub.sort_values("_td").iloc[-1]
                extras: list[str] = []
                for col in ("total_mv", "circ_mv", "pe", "pb", "turnover_rate", "volume_ratio"):
                    if col in r.index:
                        fv = _fmt_val(r.get(col))
                        if fv:
                            extras.append(f"{col}={fv}")
                if extras:
                    parts.append("日常估值/市值：" + ", ".join(extras[:8]))
    if len(parts) <= 2:
        return None
    text = "\n".join(parts)
    return _doc_base(
        text=text,
        ticker=_normalize_ticker_code(ticker),
        as_of_d=as_of_d,
        doc_type="latest_market_metrics",
        table_name="prices,daily_basic",
        snapshot_dir=snapshot_dir,
        title=f"{ticker} 市场指标 {as_of_d}",
    )


def _financial_summary(
    tables: dict[str, pd.DataFrame],
    ticker: str,
    as_of_d: date,
    snapshot_dir: Path,
) -> dict[str, Any] | None:
    fi = tables.get("financial_indicators")
    if fi is None or fi.empty:
        return None
    sub = _df_filter_ticker(fi, ticker)
    if sub.empty:
        return None
    date_cols = [c for c in ("end_date", "ann_date", "trade_date") if c in sub.columns]
    if date_cols:
        sub = sub.copy()
        sub["_pit"] = _parse_trade_date(sub[date_cols[0]])
        sub = sub[sub["_pit"].notna() & (sub["_pit"].dt.date <= as_of_d)]
        if sub.empty:
            return None
        sub = sub.sort_values("_pit")
        r = sub.iloc[-1]
        pit_label = _fmt_val(r.get(date_cols[0], "")) or ""
    else:
        r = sub.iloc[-1]
        pit_label = ""
    lines = [
        f"财务指标摘要（as_of={as_of_d}）",
        f"股票：{_normalize_ticker_code(ticker)}",
    ]
    if pit_label:
        lines.append(f"报告期/公告期：{pit_label}")
    picked = 0
    for col in FINANCIAL_METRIC_COLUMNS:
        if col not in r.index:
            continue
        fv = _fmt_val(r.get(col))
        if fv:
            lines.append(f"{col}：{fv}")
            picked += 1
            if picked >= 12:
                break
    if picked == 0:
        return None
    text = "\n".join(lines)
    return _doc_base(
        text=text,
        ticker=_normalize_ticker_code(ticker),
        as_of_d=as_of_d,
        doc_type="financial_indicator_summary",
        table_name="financial_indicators",
        snapshot_dir=snapshot_dir,
        title=f"{ticker} 财务指标 {as_of_d}",
    )


def _northbound_summary(
    tables: dict[str, pd.DataFrame],
    ticker: str,
    as_of_d: date,
    snapshot_dir: Path,
) -> dict[str, Any] | None:
    hk = tables.get("hk_hold")
    if hk is None or hk.empty:
        return None
    sub = _df_filter_ticker(hk, ticker)
    if sub.empty:
        return None
    if "trade_date" not in sub.columns:
        return None
    sub = sub.copy()
    sub["_td"] = _parse_trade_date(sub["trade_date"])
    sub = sub[sub["_td"].notna() & (sub["_td"].dt.date <= as_of_d)]
    if sub.empty:
        return None
    last = sub.sort_values("_td").iloc[-1]
    td = _fmt_val(last.get("trade_date", ""))
    lines = [
        f"北向持股摘要（as_of={as_of_d}）",
        f"股票：{_normalize_ticker_code(ticker)}",
        f"最近交易日：{td}" if td else "",
    ]
    lines = [x for x in lines if x]
    for col in ("hold_ratio", "hold_vol", "hold_amount"):
        if col in last.index:
            fv = _fmt_val(last.get(col))
            if fv:
                lines.append(f"{col}：{fv}")
    if len(lines) <= 2:
        return None
    return _doc_base(
        text="\n".join(lines),
        ticker=_normalize_ticker_code(ticker),
        as_of_d=as_of_d,
        doc_type="northbound_summary",
        table_name="hk_hold",
        snapshot_dir=snapshot_dir,
        title=f"{ticker} 北向 {as_of_d}",
    )


def _margin_summary(
    tables: dict[str, pd.DataFrame],
    ticker: str,
    as_of_d: date,
    snapshot_dir: Path,
) -> dict[str, Any] | None:
    mg = tables.get("margin")
    if mg is None or mg.empty:
        return None
    sub = _df_filter_ticker(mg, ticker)
    if sub.empty:
        return None
    if "trade_date" not in sub.columns:
        return None
    sub = sub.copy()
    sub["_td"] = _parse_trade_date(sub["trade_date"])
    sub = sub[sub["_td"].notna() & (sub["_td"].dt.date <= as_of_d)]
    if sub.empty:
        return None
    last = sub.sort_values("_td").iloc[-1]
    td = _fmt_val(last.get("trade_date", ""))
    lines = [
        f"融资融券摘要（as_of={as_of_d}）",
        f"股票：{_normalize_ticker_code(ticker)}",
    ]
    if td:
        lines.append(f"最近交易日：{td}")
    for col in ("rzye", "rzmre", "rqye", "rqyl", "rzrqye"):
        if col in last.index:
            fv = _fmt_val(last.get(col))
            if fv:
                lines.append(f"{col}：{fv}")
    if len(lines) <= 3:
        return None
    return _doc_base(
        text="\n".join(lines),
        ticker=_normalize_ticker_code(ticker),
        as_of_d=as_of_d,
        doc_type="margin_summary",
        table_name="margin",
        snapshot_dir=snapshot_dir,
        title=f"{ticker} 融资融券 {as_of_d}",
    )


def _market_index_context(
    tables: dict[str, pd.DataFrame],
    as_of_d: date,
    snapshot_dir: Path,
) -> dict[str, Any] | None:
    idx = tables.get("index_daily")
    if idx is None or idx.empty:
        return None
    if "ts_code" not in idx.columns or "trade_date" not in idx.columns:
        return None
    sub_all = idx[idx["ts_code"].astype(str).str.upper().isin(INDEX_CONTEXT_CODES)].copy()
    if sub_all.empty:
        return None
    sub_all["_td"] = _parse_trade_date(sub_all["trade_date"])
    sub_all = sub_all[sub_all["_td"].notna() & (sub_all["_td"].dt.date <= as_of_d)]
    if sub_all.empty:
        return None
    lines = [
        f"主要指数市场环境（snapshot as_of={as_of_d}）",
    ]
    for code in INDEX_CONTEXT_CODES:
        part = sub_all[sub_all["ts_code"].astype(str).str.upper() == code]
        if part.empty:
            continue
        g = part.sort_values("_td")
        last = g.iloc[-1]
        td = _fmt_val(last.get("trade_date", ""))
        cl = _fmt_val(last.get("close", ""))
        pc = _fmt_val(last.get("pct_chg", "")) if "pct_chg" in last.index else None
        seg = f"{code} 最新({td}): close={cl}" if td or cl else f"{code}"
        if pc:
            seg += f", pct_chg={pc}"
        if len(g) >= 5:
            first = g.iloc[-min(20, len(g))]
            try:
                c0 = float(first["close"])
                c1 = float(last["close"])
                if c0 and not math.isnan(c0) and c1 and not math.isnan(c1):
                    ret20 = (c1 / c0 - 1.0) * 100.0
                    seg += f", ~{min(20, len(g))}d return%={ret20:.2f}"
            except Exception:
                pass
        lines.append(seg)
    if len(lines) <= 1:
        return None
    text = "\n".join(lines)
    return _doc_base(
        text=text,
        ticker="__MARKET__",
        as_of_d=as_of_d,
        doc_type="market_index_context",
        table_name="index_daily",
        snapshot_dir=snapshot_dir,
        title=f"指数环境 {as_of_d}",
    )


def estimate_snapshot_kb_stats(
    snapshot_dir: str | Path,
    tickers: list[str] | None,
) -> dict[str, Any]:
    """Cheap doc/chunk upper-bound estimate without loading full tables."""
    root = Path(snapshot_dir).resolve()
    univ_n = parquet_row_count(root / "universe.parquet")
    if tickers is None:
        tickers_eff = read_universe_tickers(root)
    else:
        tickers_eff = [_normalize_ticker_code(t) for t in tickers if str(t).strip()]
    n_sym = len(tickers_eff)
    has_idx = (root / "index_daily.parquet").is_file()
    has_sb = (root / "stock_basic.parquet").is_file()
    k = 0
    if has_sb:
        k += 1
    if (root / "prices.parquet").is_file() and (root / "daily_basic.parquet").is_file():
        k += 1
    if (root / "financial_indicators.parquet").is_file():
        k += 1
    if (root / "hk_hold.parquet").is_file():
        k += 1
    if (root / "margin.parquet").is_file():
        k += 1
    n_docs = (1 if has_idx else 0) + n_sym * k
    return {
        "snapshot_dir": str(root),
        "as_of": _coerce_as_of_date(None, root).isoformat(),
        "parquet_loaded": [],
        "parquet_missing": [],
        "n_docs": n_docs,
        "n_chunks": n_docs,
        "per_ticker": {},
        "universe_count": int(univ_n or 0),
        "n_tickers_ingested": n_sym,
        "chunks_written": 0,
        "dry_run": True,
    }


def generate_snapshot_documents(
    snapshot_dir: str | Path,
    tickers: list[str] | None,
    as_of: str | date | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build document dicts (no chunking, no embedding).

    Args:
        tickers:  Explicit list；``None`` 表示读取 ``universe.parquet`` 全成分股。

    Returns:
        (documents, stats)
    """
    root = Path(snapshot_dir).resolve()
    as_of_d = _coerce_as_of_date(as_of, root)
    tables, loaded_names, missing_names = load_snapshot_parquets(root, strict=False)
    docs: list[dict[str, Any]] = []
    per_ticker: dict[str, dict[str, Any]] = {}

    if tickers is None:
        tickers = read_universe_tickers(root)
    tickers_eff = [_normalize_ticker_code(t) for t in tickers if str(t).strip()]
    universe_count = parquet_row_count(root / "universe.parquet")
    if universe_count is None:
        universe_count = len(tickers_eff)

    mctx = _market_index_context(tables, as_of_d, root)
    if mctx:
        docs.append(mctx)

    for raw_t in tickers_eff:
        t = _normalize_ticker_code(raw_t)
        per_ticker[t] = {"doc_types": [], "n_docs": 0}
        builders = (
            _company_profile_row,
            _latest_market_metrics,
            _financial_summary,
            _northbound_summary,
            _margin_summary,
        )
        for fn in builders:
            d = fn(tables, t, as_of_d, root)
            if d:
                docs.append(d)
                dt = str(d.get("doc_type", ""))
                per_ticker[t]["doc_types"].append(dt)
                per_ticker[t]["n_docs"] += 1

    stats: dict[str, Any] = {
        "snapshot_dir": str(root),
        "as_of": as_of_d.isoformat(),
        "parquet_loaded": loaded_names,
        "parquet_missing": missing_names,
        "n_docs": len(docs),
        "per_ticker": per_ticker,
        "universe_count": int(universe_count) if universe_count is not None else 0,
        "n_tickers_ingested": len(tickers_eff),
    }
    return docs, stats

