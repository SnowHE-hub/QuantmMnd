"""app/utils/rec_data.py — 历史推荐数据加载与整合工具.

对外暴露的函数
--------------
load_all_recommendations()    — 加载所有推荐 JSON，按日期降序
load_realized_pnl()           — 加载 realized_pnl.parquet
load_forward_positions()      — 加载 forward_positions.json
load_name_map()               — ticker → 股票名称映射
get_prices_for_dates()        — 批量查询 (ticker, date) 收盘价
build_flat_records()          — 合并上述数据为扁平 list[dict]，供 DataFrame 展示
compute_summary_stats()       — 汇总胜率/均值等统计指标
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 推荐历史加载
# ─────────────────────────────────────────────────────────────────────────────

def load_all_recommendations(
    rec_dir: Path | str | None = None,
) -> list[dict]:
    """从 data/recommendations/*.json 加载所有推荐记录，按日期降序.

    Returns
    -------
    list[dict], 每项包含 as_of, generated_at, top10, market_summary 等字段。
    top10 每个 item 保证有 ticker / rank / lgbm_score 字段。
    """
    d = Path(rec_dir) if rec_dir else _ROOT / "data" / "recommendations"
    if not d.exists():
        log.debug("[rec_data] 推荐目录不存在: %s", d)
        return []

    records: list[dict] = []
    for fp in sorted(d.glob("*.json"), reverse=True):
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
            # 处理两种格式：顶层 dict（有 as_of）或直接 list
            if isinstance(raw, dict):
                rec = raw
                rec.setdefault("as_of", fp.stem)  # 从文件名推断
            else:
                rec = {"as_of": fp.stem, "top10": raw}

            # 标准化 top10 字段
            rec["top10"] = _normalize_top10(rec.get("top10") or [])
            records.append(rec)
        except Exception as e:
            log.warning("[rec_data] 跳过 %s: %s", fp.name, e)

    return records


def _normalize_top10(items: list[dict]) -> list[dict]:
    """确保每个 top10 item 拥有标准最小字段集."""
    result = []
    for i, item in enumerate(items):
        normalized = {
            "rank":       item.get("rank", i + 1),
            "ticker":     item.get("ticker", ""),
            "lgbm_score": float(item.get("lgbm_score") or item.get("ensemble_score") or 0.0),
            "lgbm_rank":  item.get("lgbm_rank", i + 1),
            "reason":     item.get("reason", ""),
            # 可选字段
            "llm_rank":              item.get("llm_rank"),
            "agent_recommendation":  item.get("agent_recommendation", item.get("rating", "")),
            "agent_confidence":      item.get("agent_confidence"),
            "suggested_horizon":     item.get("suggested_horizon", "3m"),
            "industry":              item.get("industry", ""),
            "name":                  item.get("name", ""),
        }
        result.append(normalized)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. 已结算收益
# ─────────────────────────────────────────────────────────────────────────────

def load_realized_pnl(
    path: Path | str | None = None,
) -> pd.DataFrame | None:
    """加载 realized_pnl.parquet.

    返回含 as_of_date, ticker, actual_return_63d, hit, entry_price, exit_price 的 DataFrame。
    """
    p = Path(path) if path else _ROOT / "data" / "feedback" / "realized_pnl.parquet"
    if not p.exists():
        log.debug("[rec_data] realized_pnl 不存在: %s", p)
        return None
    try:
        df = pd.read_parquet(p)
        # 确保日期列统一为 string YYYY-MM-DD
        for col in ("as_of_date", "entry_date", "exit_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
        return df
    except Exception as e:
        log.warning("[rec_data] realized_pnl 读取失败: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. 进行中持仓
# ─────────────────────────────────────────────────────────────────────────────

def load_forward_positions(
    path: Path | str | None = None,
) -> list[dict]:
    """加载 forward_positions.json，返回 positions 列表."""
    p = Path(path) if path else _ROOT / "data" / "paper_trading" / "forward_positions.json"
    if not p.exists():
        log.debug("[rec_data] forward_positions 不存在: %s", p)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("positions", [])
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        log.warning("[rec_data] forward_positions 读取失败: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. 股票名称映射
# ─────────────────────────────────────────────────────────────────────────────

def load_name_map(
    root: Path | str | None = None,
) -> dict[str, str]:
    """从 stock_basic 快照构建 ticker → 名称 映射.

    查找路径（按优先级）：
      1. data/raw/stock_basic.parquet
      2. data/raw/stock_basic.csv
      3. data/sim30d/raw/trade_cal_30d.json 中的名称（如有）
    """
    r = Path(root) if root else _ROOT
    name_map: dict[str, str] = {}

    for snap_path in [
        r / "data" / "raw" / "stock_basic.parquet",
        r / "data" / "raw" / "stock_basic.csv",
    ]:
        if snap_path.exists():
            try:
                ext = snap_path.suffix
                sb = pd.read_parquet(snap_path) if ext == ".parquet" else pd.read_csv(snap_path)
                code_col = "ts_code" if "ts_code" in sb.columns else None
                name_col = "name" if "name" in sb.columns else None
                if code_col and name_col:
                    for _, row in sb.iterrows():
                        name_map[str(row[code_col])] = str(row[name_col])
                    log.debug("[rec_data] 从 %s 加载 %d 个股票名称", snap_path.name, len(name_map))
                    return name_map
            except Exception as e:
                log.debug("[rec_data] stock_basic 读取失败: %s", e)

    # fallback: 从 positions 文件中提取 name 字段
    fwd_path = r / "data" / "paper_trading" / "forward_positions.json"
    if fwd_path.exists():
        try:
            data = json.loads(fwd_path.read_text(encoding="utf-8"))
            positions = data.get("positions", []) if isinstance(data, dict) else data
            for pos in positions:
                if pos.get("ticker") and pos.get("name"):
                    name_map[pos["ticker"]] = pos["name"]
        except Exception:
            pass

    return name_map


# ─────────────────────────────────────────────────────────────────────────────
# 5. 价格查询
# ─────────────────────────────────────────────────────────────────────────────

def get_prices_for_dates(
    ticker_date_pairs: list[tuple[str, str]],
    prices_path: Path | str | None = None,
) -> dict[tuple[str, str], float | None]:
    """批量查询 (ticker, YYYY-MM-DD) 对应的收盘价.

    Returns
    -------
    dict  {(ticker, date_str): close_price | None}
    """
    if not ticker_date_pairs:
        return {}

    p = Path(prices_path) if prices_path else _ROOT / "data" / "raw" / "daily_prices_panel.parquet"
    if not p.exists():
        return {pair: None for pair in ticker_date_pairs}

    tickers = list({t for t, _ in ticker_date_pairs})
    dates   = list({d for _, d in ticker_date_pairs})

    try:
        df = pd.read_parquet(
            p,
            filters=[("ts_code", "in", tickers)],
            columns=["trade_date", "ts_code", "close"],
        )
        # 统一 trade_date 为 YYYY-MM-DD string
        if pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
            df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
        else:
            df["trade_date"] = df["trade_date"].astype(str)
            # Handle YYYYMMDD format
            if df["trade_date"].str.len().max() == 8:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

        price_map = {
            (row["ts_code"], row["trade_date"]): float(row["close"])
            for _, row in df.iterrows()
            if row["trade_date"] in dates
        }
    except Exception as e:
        log.warning("[rec_data] 价格读取失败: %s", e)
        price_map = {}

    return {pair: price_map.get(pair) for pair in ticker_date_pairs}


def get_latest_prices(
    tickers: list[str],
    prices_path: Path | str | None = None,
) -> dict[str, float | None]:
    """获取每个 ticker 的最新收盘价."""
    p = Path(prices_path) if prices_path else _ROOT / "data" / "raw" / "daily_prices_panel.parquet"
    if not p.exists() or not tickers:
        return {t: None for t in tickers}
    try:
        df = pd.read_parquet(
            p,
            filters=[("ts_code", "in", tickers)],
            columns=["trade_date", "ts_code", "close"],
        )
        if pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
            df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
        else:
            df["trade_date"] = df["trade_date"].astype(str)
            if df["trade_date"].str.len().max() == 8:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

        latest = (
            df.sort_values("trade_date")
            .groupby("ts_code")
            .last()["close"]
            .to_dict()
        )
        return {t: latest.get(t) for t in tickers}
    except Exception as e:
        log.warning("[rec_data] 最新价格读取失败: %s", e)
        return {t: None for t in tickers}


# ─────────────────────────────────────────────────────────────────────────────
# 6. 扁平记录构建（供 DataFrame 展示）
# ─────────────────────────────────────────────────────────────────────────────

def build_flat_records(
    recs:       list[dict],
    pnl_df:     pd.DataFrame | None,
    fwd_pos:    list[dict],
    name_map:   dict[str, str],
    latest_prices: dict[str, float | None],
    entry_prices:  dict[tuple[str, str], float | None],
) -> list[dict]:
    """将推荐列表与 P&L、持仓、价格合并为扁平 list[dict].

    每行代表一次「推荐事件」（一只股票在某日被推荐）。
    """
    # 建立索引：(as_of_date, ticker) → realized pnl row
    pnl_index: dict[tuple[str, str], dict] = {}
    if pnl_df is not None and not pnl_df.empty:
        for _, row in pnl_df.iterrows():
            key = (str(row.get("as_of_date", "")), str(row.get("ticker", "")))
            pnl_index[key] = row.to_dict()

    # 建立前向持仓索引：(as_of, ticker) → position
    fwd_index: dict[tuple[str, str], dict] = {}
    for pos in fwd_pos:
        key = (str(pos.get("as_of", "")), str(pos.get("ticker", "")))
        fwd_index[key] = pos

    rows: list[dict] = []
    for rec in recs:
        as_of = str(rec.get("as_of", ""))
        for item in rec.get("top10", []):
            ticker = item.get("ticker", "")
            if not ticker:
                continue

            # 名称
            name = name_map.get(ticker) or item.get("name") or ticker

            # 入场价（推荐日收盘）
            entry_price = entry_prices.get((ticker, as_of))

            # 当前价
            cur_price = latest_prices.get(ticker)

            # 未结算浮盈
            unrealized_pct: float | None = None
            if entry_price and cur_price and entry_price > 0:
                unrealized_pct = (cur_price - entry_price) / entry_price

            # 已结算收益
            pnl_key = (as_of, ticker)
            realized_row = pnl_index.get(pnl_key)
            realized_ret: float | None = None
            is_hit: bool | None = None
            exit_date_str: str | None = None
            if realized_row:
                realized_ret = realized_row.get("actual_return_63d")
                is_hit       = realized_row.get("hit")
                exit_date_str = realized_row.get("exit_date")

            # 持仓状态
            fwd_row = fwd_index.get(pnl_key)
            if fwd_row:
                status = "持仓中"
                estimated_exit = fwd_row.get("estimated_exit_date", "")
            elif realized_row:
                if is_hit is True:
                    status = "已结算盈利 ✅"
                elif is_hit is False:
                    status = "已结算亏损 ❌"
                else:
                    status = "已结算"
            else:
                status = "未追踪"

            rows.append({
                "推荐日期":   as_of,
                "代码":       ticker,
                "名称":       name,
                "评级":       item.get("agent_recommendation", ""),
                "LGBM排名":  item.get("rank", ""),
                "LGBM评分":  round(item.get("lgbm_score", 0), 4),
                "行业":       item.get("industry", ""),
                "入场价":     entry_price,
                "当前价":     cur_price,
                "浮盈%":      round(unrealized_pct * 100, 2) if unrealized_pct is not None else None,
                "3m实际收益%": round(realized_ret * 100, 2) if realized_ret is not None else None,
                "盈利?":      is_hit,
                "退出日期":   exit_date_str,
                "状态":       status,
                "推荐原因摘要": str(item.get("reason", ""))[:80],
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 7. 汇总统计
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary_stats(flat_records: list[dict]) -> dict[str, Any]:
    """从扁平记录计算汇总指标."""
    if not flat_records:
        return {
            "total_recs":    0,
            "win_rate_3m":   None,
            "avg_return_3m": None,
            "best_trade":    None,
            "worst_trade":   None,
            "n_settled":     0,
            "n_open":        0,
        }

    total  = len(flat_records)
    n_open = sum(1 for r in flat_records if r["状态"] == "持仓中")

    settled = [r for r in flat_records if r["3m实际收益%"] is not None]
    n_settled = len(settled)

    if settled:
        rets     = [r["3m实际收益%"] for r in settled]
        win_rate = sum(1 for x in rets if x > 0) / len(rets)
        avg_ret  = float(np.mean(rets))
        best     = max(rets)
        worst    = min(rets)
    else:
        win_rate = avg_ret = best = worst = None

    return {
        "total_recs":    total,
        "win_rate_3m":   win_rate,
        "avg_return_3m": avg_ret,
        "best_trade":    best,
        "worst_trade":   worst,
        "n_settled":     n_settled,
        "n_open":        n_open,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. 归因分析
# ─────────────────────────────────────────────────────────────────────────────

def compute_attribution(flat_records: list[dict]) -> dict[str, pd.DataFrame]:
    """按行业 / 评级 分组归因，返回统计 DataFrame."""
    settled = [r for r in flat_records if r["3m实际收益%"] is not None]
    if not settled:
        return {"industry": pd.DataFrame(), "rating": pd.DataFrame()}

    df = pd.DataFrame(settled)

    # 行业归因
    if "行业" in df.columns and df["行业"].notna().any():
        ind_grp = (
            df[df["行业"] != ""]
            .groupby("行业")["3m实际收益%"]
            .agg(均值="mean", 次数="count", 胜率=lambda x: (x > 0).mean())
            .reset_index()
            .sort_values("均值", ascending=False)
        )
    else:
        ind_grp = pd.DataFrame()

    # 评级归因
    if "评级" in df.columns and df["评级"].notna().any():
        rat_grp = (
            df[df["评级"] != ""]
            .groupby("评级")["3m实际收益%"]
            .agg(均值="mean", 次数="count", 胜率=lambda x: (x > 0).mean())
            .reset_index()
            .sort_values("均值", ascending=False)
        )
    else:
        rat_grp = pd.DataFrame()

    return {"industry": ind_grp, "rating": rat_grp}
