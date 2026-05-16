"""Read-only data loaders for Streamlit (cached)."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _maybe_cache_data(ttl: int):
    def deco(fn):
        if st is None:
            return fn
        return st.cache_data(ttl=ttl)(fn)

    return deco


@_maybe_cache_data(300)
def list_top10_dates() -> list[str]:
    root = _project_root() / "data" / "recommendations"
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "top10.json").is_file():
            try:
                date.fromisoformat(p.name)
                out.append(p.name)
            except ValueError:
                continue
    return sorted(out)


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return {}


@_maybe_cache_data(300)
def load_top10(date_str: str) -> dict[str, Any]:
    p = _project_root() / "data" / "recommendations" / date_str / "top10.json"
    return _safe_read_json(p)


@_maybe_cache_data(300)
def load_funnel_candidates(date_str: str) -> dict[str, Any]:
    p = _project_root() / "data" / "recommendations" / date_str / "funnel_candidates.json"
    return _safe_read_json(p)


@_maybe_cache_data(300)
def load_strategies_json(date_str: str) -> list[dict[str, Any]]:
    p = _project_root() / "reports" / "investment_pipeline" / date_str / "strategies.json"
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        pass
    return []


@_maybe_cache_data(300)
def load_strategy_detail(date_str: str, ticker: str) -> dict[str, Any]:
    safe_t = ticker.replace("/", "_")
    p = (
        _project_root()
        / "reports"
        / "investment_pipeline"
        / date_str
        / "strategies"
        / f"{safe_t}_strategy.json"
    )
    return _safe_read_json(p)


@_maybe_cache_data(300)
def load_final_recommendations_md(date_str: str) -> str:
    p = _project_root() / "reports" / "investment_pipeline" / date_str / "final_recommendations.md"
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    except OSError:
        pass
    return ""


@_maybe_cache_data(600)
def load_backtest_metrics(rel_path: str = "reports/final_eval/backtest_metrics.json") -> dict[str, Any]:
    p = _project_root() / rel_path
    return _safe_read_json(p)


@_maybe_cache_data(600)
def load_holdout_metrics() -> dict[str, Any]:
    return load_backtest_metrics("reports/holdout/backtest_metrics.json")


@_maybe_cache_data(600)
def load_engine_metrics() -> dict[str, Any]:
    return load_backtest_metrics("reports/backtest/lgbm_engine_metrics.json")


@_maybe_cache_data(300)
def load_quintile_returns_csv(rel_path: str = "reports/final_eval/quintile_returns_by_period.csv") -> pd.DataFrame:
    candidates = [
        _project_root() / "reports" / "alpha_final" / "quintile_returns_by_period.csv",
        _project_root() / rel_path,
    ]
    for p in candidates:
        try:
            if p.is_file():
                df = pd.read_csv(p)
                if "as_of" in df.columns:
                    df["as_of"] = pd.to_datetime(df["as_of"])
                return df
        except (OSError, ValueError):
            pass
    return pd.DataFrame()


@_maybe_cache_data(300)
def load_validations_json(date_str: str) -> list[dict[str, Any]]:
    p = _project_root() / "reports" / "investment_pipeline" / date_str / "validations.json"
    try:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        pass
    return []


@_maybe_cache_data(60)
def load_price_history(ticker: str, days: int = 126) -> pd.DataFrame:
    root = _project_root()
    # 优先 alpha 宇宙长表（1374只，覆盖最广）
    for longp in [
        root / "data" / "raw" / "alpha_prices_panel.parquet",
        root / "data" / "raw" / "daily_prices_panel.parquet",
    ]:
        try:
            if not longp.is_file():
                continue
            import pyarrow.parquet as _pq
            schema_names = _pq.ParquetFile(str(longp)).schema_arrow.names
            price_col = "adj_close" if "adj_close" in schema_names else "close"
            try:
                df = pd.read_parquet(
                    longp,
                    filters=[("ts_code", "=", ticker)],
                    columns=["trade_date", "ts_code", price_col],
                )
            except Exception:
                df = pd.read_parquet(longp, columns=["trade_date", "ts_code", price_col])
                df = df[df["ts_code"].astype(str) == ticker]
            if not df.empty:
                df = df.sort_values("trade_date")
                df = df.rename(columns={price_col: "close"})
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                return df.tail(days).reset_index(drop=True)
        except (OSError, ValueError, KeyError):
            pass

    # 回退 CSI300 宽表（兼容旧数据）
    wide = root / "data" / "prices" / "csi300_daily_adj_close.parquet"
    try:
        if wide.is_file():
            panel = pd.read_parquet(wide)
            if ticker in panel.columns:
                s = panel[ticker].dropna().tail(days)
                out = s.reset_index()
                if out.shape[1] >= 2:
                    out.columns = ["trade_date", "close"]
                    out["trade_date"] = pd.to_datetime(out["trade_date"])
                    return out
    except (OSError, ValueError, KeyError):
        pass
    return pd.DataFrame()


@_maybe_cache_data(60)
def load_price_history_with_vol(ticker: str, days: int = 66) -> pd.DataFrame:
    """优先返回含成交量的面板（约 3 个月）；否则退回宽表收盘价并填 NaN 成交量。"""
    root = _project_root()
    for longp in [
        root / "data" / "raw" / "alpha_prices_panel.parquet",
        root / "data" / "raw" / "daily_prices_panel.parquet",
    ]:
        try:
            if not longp.is_file():
                continue
            import pyarrow.parquet as _pq
            cols_available = _pq.ParquetFile(str(longp)).schema_arrow.names
            price_col = "adj_close" if "adj_close" in cols_available else "close"
            vol_col = next((c for c in ["vol", "volume"] if c in cols_available), None)
            read_cols = ["trade_date", "ts_code", price_col] + ([vol_col] if vol_col else [])
            try:
                df = pd.read_parquet(longp, filters=[("ts_code", "=", ticker)], columns=read_cols)
            except Exception:
                df = pd.read_parquet(longp, columns=read_cols)
                df = df[df["ts_code"].astype(str) == ticker]
            if df.empty:
                continue
            df = df.sort_values("trade_date").rename(columns={price_col: "close"})
            if vol_col:
                df = df.rename(columns={vol_col: "vol"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            if "vol" not in df.columns:
                df["vol"] = float("nan")
            return df.tail(days).reset_index(drop=True)
        except (OSError, ValueError, KeyError):
            pass
    px = load_price_history(ticker, days=days)
    if px.empty:
        return px
    if "vol" not in px.columns:
        px = px.copy()
        px["vol"] = float("nan")
    return px


@_maybe_cache_data(120)
def load_index_close(ts_code: str = "000300.SH", days: int = 600) -> pd.DataFrame:
    root = _project_root()
    candidates = [
        root / "data" / "prices" / "index_daily.parquet",
        root / "data" / "raw" / "index_daily_panel.parquet",
    ]
    for p in candidates:
        try:
            if not p.is_file():
                continue
            df = pd.read_parquet(p)
            if "ts_code" in df.columns:
                df = df[df["ts_code"].astype(str) == ts_code].sort_values("trade_date")
            elif "ticker" in df.columns:
                df = df[df["ticker"].astype(str) == ts_code].sort_values("trade_date")
            else:
                continue
            col = "close" if "close" in df.columns else None
            if col is None:
                continue
            df = df[["trade_date", col]].rename(columns={col: "close"})
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            return df.tail(days).reset_index(drop=True)
        except (OSError, ValueError, KeyError):
            continue
    return pd.DataFrame()


@_maybe_cache_data(600)
def load_stock_basic(date_str: str) -> pd.DataFrame:
    p = _project_root() / "data" / "snapshots" / date_str / "stock_basic.parquet"
    try:
        if p.is_file():
            return pd.read_parquet(p, columns=["ticker", "industry"])
    except (OSError, ValueError, KeyError):
        pass
    return pd.DataFrame()


@_maybe_cache_data(3600)
def load_agent_registry() -> dict[str, Any]:
    p = _project_root() / "data" / "agent_models" / "registry.json"
    data = _safe_read_json(p)
    return data if isinstance(data, dict) else {}


@_maybe_cache_data(600)
def list_rolling_quarters() -> list[str]:
    base = _project_root() / "reports" / "rolling_update"
    if not base.is_dir():
        return []
    qs = []
    for child in base.iterdir():
        if child.is_dir() and (child / "validation_result.json").is_file():
            qs.append(child.name)
    return sorted(qs)


@_maybe_cache_data(600)
def load_validation_result(quarter_folder: str) -> dict[str, Any]:
    p = _project_root() / "reports" / "rolling_update" / quarter_folder / "validation_result.json"
    return _safe_read_json(p)


def parse_tri_classification(md_text: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {"accept": [], "watch": [], "avoid": []}
    if not md_text.strip():
        return out

    def grab_rows(section_pattern: str) -> list[list[str]]:
        m = re.search(section_pattern, md_text, re.DOTALL)
        if not m:
            return []
        block = m.group(1)
        rows = []
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("|") or line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows.append(cells)
        return rows

    acc = grab_rows(r"## ✅ 可接受的投资机会.*?\n\n(\|.*?\n(?:\|.*\n)+)")
    for cells in acc[1:]:
        if len(cells) >= 8:
            out["accept"].append({
                "股票": cells[0],
                "评级": cells[1],
                "综合信号": cells[2],
                "历史胜率": cells[3],
                "期望月收益": cells[4],
                "建议仓位": cells[5],
                "止损价": cells[6],
                "1月目标价": cells[7],
            })

    watch_tbl = grab_rows(r"## 👀 观察名单.*?\n\n(\|.*?\n(?:\|.*\n)+)")
    for cells in watch_tbl[1:]:
        if len(cells) >= 4:
            out["watch"].append({"股票": cells[0], "评级": cells[1], "综合信号": cells[2], "原因": cells[3]})

    avoid_tbl = grab_rows(r"## ❌ 暂时回避.*?\n\n(\|.*?\n(?:\|.*\n)+)")
    for cells in avoid_tbl[1:]:
        if len(cells) >= 4:
            out["avoid"].append({"股票": cells[0], "评级": cells[1], "综合信号": cells[2], "原因": cells[3]})

    return out


AGENT_ORDER = [
    "ValuationAgent",
    "MomentumAgent",
    "QualityAgent",
    "SentimentAgent",
    "RiskAgent",
    "StrategyAgent",
]

DISPLAY_NAMES = ["估值", "动量", "质量", "情绪", "风险", "综合"]


def agent_signal_matrix(strategies: list[dict[str, Any]]) -> tuple[list[str], list[float], list[list[float]]]:
    matrix: list[list[float]] = []
    tickers: list[str] = []
    composites: list[float] = []
    for row in strategies:
        t = str(row.get("ticker", "")).strip()
        if not t:
            continue
        tickers.append(t)
        comp = float(row.get("composite_signal", 0.0) or 0.0)
        composites.append(comp)
        sigs = row.get("agent_signals") or {}
        vals = []
        for ag in AGENT_ORDER[:-1]:
            v = sigs.get(ag, {})
            vals.append(float(v.get("signal", 0.0) or 0.0))
        vals.append(max(-1.0, min(1.0, comp)))
        matrix.append(vals)
    return tickers, composites, matrix


def latest_top10_date_or_none() -> str | None:
    ds = list_top10_dates()
    return ds[-1] if ds else None


def merge_top10_with_md(top10: dict[str, Any], tri: dict[str, list]) -> pd.DataFrame:
    rows = top10.get("top10") if isinstance(top10, dict) else None
    if not isinstance(rows, list):
        return pd.DataFrame()
    accept_by_ticker = {r["股票"]: r for r in tri.get("accept", [])}
    avoid_by_ticker = {r["股票"]: r for r in tri.get("avoid", [])}
    watch_by_ticker = {r["股票"]: r for r in tri.get("watch", [])}
    out_rows = []
    for item in rows:
        t = str(item.get("ticker", ""))
        extra = accept_by_ticker.get(t) or watch_by_ticker.get(t) or avoid_by_ticker.get(t) or {}
        rating = extra.get("评级") or str(item.get("reason", ""))[:16]
        sig = extra.get("综合信号") or f'{float(item.get("lgbm_score", 0)):.3f}'
        pos = extra.get("建议仓位", "—")
        tgt = extra.get("1月目标价", "—")
        out_rows.append({
            "ticker": t,
            "评级": rating if isinstance(rating, str) else str(rating),
            "综合信号": sig,
            "建议仓位": pos,
            "1月目标价": tgt,
            "LLM排名": item.get("rank") or item.get("llm_rank"),
            "LGBM分": item.get("lgbm_score"),
        })
    return pd.DataFrame(out_rows)
