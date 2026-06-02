"""sim_data.py — 30日全A股模拟盘 & 历史季度数据加载器（带缓存）."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]


# ── 基础辅助 ──────────────────────────────────────────────────────

def _jload(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


# ── 30日模拟盘 ───────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_sim30d_days() -> list[dict]:
    """加载所有30个交易日的 daily JSON."""
    daily_dir = ROOT / "data" / "sim30d" / "daily"
    days = []
    for f in sorted(daily_dir.glob("*.json")):
        d = _jload(f)
        if d:
            days.append(d)
    return days


@st.cache_data(ttl=300)
def load_sim30d_stock_returns() -> pd.DataFrame:
    p = ROOT / "data" / "sim30d" / "stock_returns.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_sim30d_positions() -> pd.DataFrame:
    p = ROOT / "data" / "sim30d" / "positions.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_sim30d_summary() -> dict:
    return _jload(ROOT / "data" / "sim30d" / "summary.json")


# ── Paper Trading / 季度回测 ─────────────────────────────────────

@st.cache_data(ttl=300)
def load_nav_curves() -> dict[str, pd.DataFrame]:
    out = {}
    for hz in ["1w", "2w", "21d", "3m"]:
        p = ROOT / "data" / "paper_trading" / f"nav_curve_{hz}.parquet"
        if p.exists():
            out[hz] = pd.read_parquet(p)
    return out


@st.cache_data(ttl=300)
def load_strategy_config() -> dict:
    return _jload(ROOT / "data" / "paper_trading" / "strategy_config_v2.json")


@st.cache_data(ttl=300)
def load_ic_analysis() -> dict:
    return _jload(ROOT / "data" / "paper_trading" / "ic_analysis_30day.json")


@st.cache_data(ttl=300)
def load_realized_pnl() -> pd.DataFrame:
    """统一委托给 DataService（去重：原 sim_data / rec_data 各有一份）。"""
    try:
        from app.services.data_service import get_data_service
        return get_data_service().get_realized_pnl()
    except Exception:
        p = ROOT / "data" / "feedback" / "realized_pnl.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_quarterly_recs() -> list[dict]:
    """加载所有季度推荐（data/recommendations/<date>/top10.json）."""
    recs_dir = ROOT / "data" / "recommendations"
    recs = []
    if not recs_dir.exists():
        return recs
    for sub in sorted(recs_dir.iterdir()):
        if sub.is_dir():
            d = _jload(sub / "top10.json")
            if d:
                recs.append({"date": sub.name, **d})
        else:
            if sub.suffix == ".json":
                d = _jload(sub)
                if d and "as_of" in d:
                    recs.append({"date": d["as_of"], **d})
    return recs


# ── 工具函数 ─────────────────────────────────────────────────────

def get_latest_sim_day(days: list[dict]) -> dict | None:
    return days[-1] if days else None


def sim_day_to_df(day: dict) -> pd.DataFrame:
    """将 system3_final_list 转换为 DataFrame."""
    rows = day.get("system3_final_list", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 加入 returns
    rets = day.get("returns", {})
    for hz in ["1w", "2w", "21d", "3m"]:
        df[f"portfolio_{hz}"] = rets.get(hz, {}).get("mean")
    return df


def horizon_portfolio_ts(days: list[dict], hz: str) -> pd.DataFrame:
    """从30日 days 提取某持仓期的等权组合时序."""
    rows = []
    for d in days:
        rets = d.get("returns", {}).get(hz, {})
        rows.append({
            "date": pd.Timestamp(d["date"]),
            "mean": rets.get("mean"),
            "win_rate": rets.get("win_rate"),
            "n": rets.get("n", 0),
        })
    df = pd.DataFrame(rows).dropna(subset=["mean"])
    df["cum_return"] = (1 + df["mean"]).cumprod() - 1
    return df


RATING_ORDER = {"强烈买入": 4, "买入": 3, "持有": 2, "观望": 1}
RATING_COLORS = {
    "强烈买入": "#00B894",
    "买入": "#0984E3",
    "持有": "#FDCB6E",
    "观望": "#636E72",
}
RISK_COLORS = {"低": "#00B894", "中": "#FDCB6E", "高": "#D63031"}

HORIZON_LABELS = {"1w": "1周", "2w": "2周", "21d": "21天", "3m": "3个月"}
