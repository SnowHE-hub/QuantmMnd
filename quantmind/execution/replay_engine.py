"""quantmind/execution/replay_engine.py — 历史参数回放引擎.

给定一套参数（stop_loss / target / trailing / holding_days），在历史推荐数据上
回放完整的"如果当时按这套参数执行"流程，输出 NAV 曲线 + 关键指标。

设计要点：
  * 价格数据一次性预加载（避免 N×M 次 SQL）
  * 单笔回放复用 backfill 的 _exec_price_next_open（T+1 + 滑点）
  * 参数支持 None：stop_loss=None 表示不止损，target=None 不止盈，trailing=None 不追踪
  * NAV 等权聚合：每笔 1/N 资金，避免 cumprod 复利失真

依赖：scripts/backfill_executions._exec_price_next_open（next-open 成交价 + 滑点）
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# 引入 backfill 中已实现的"次日开盘成交"辅助
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.backfill_executions import _exec_price_next_open, SLIPPAGE_SELL_PCT


# ── 参数结构 ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReplayParams:
    """单次回放的参数组合。None 表示禁用该规则。"""
    stop_loss:     float | None    # 例: -0.10 = 入场价 × 0.90
    target_price:  float | None    # 例: +0.20 = 入场价 × 1.20
    trailing_stop: float | None    # 例: -0.15 = 高点回撤 15%
    holding_days:  int             # 计划持仓天数

    def to_dict(self) -> dict:
        return {
            "stop_loss":     self.stop_loss,
            "target_price":  self.target_price,
            "trailing_stop": self.trailing_stop,
            "holding_days":  self.holding_days,
        }


# ── 历史推荐数据 ─────────────────────────────────────────────────────────────

@dataclass
class HistoricalRecommendation:
    """一条历史推荐（来自 realized_pnl）。"""
    ticker:        str
    recommend_date: date
    entry_date:    date
    entry_price:   float


def load_historical_recommendations(engine) -> list[HistoricalRecommendation]:
    """从 PG realized_pnl 加载所有历史推荐。"""
    from sqlalchemy import text
    with engine.connect() as conn:
        df = pd.read_sql(text(
            "SELECT as_of_date AS recommend_date, ticker, entry_date, entry_price "
            "FROM realized_pnl ORDER BY as_of_date, ticker"), conn)
    recs = []
    for _, r in df.iterrows():
        recs.append(HistoricalRecommendation(
            ticker=str(r["ticker"]),
            recommend_date=pd.to_datetime(r["recommend_date"]).date(),
            entry_date=pd.to_datetime(r["entry_date"]).date(),
            entry_price=float(r["entry_price"]),
        ))
    return recs


def preload_price_history(
    engine,
    recommendations: list[HistoricalRecommendation],
    max_holding_days: int = 252,    # 留足缓冲（最长 1 年）
) -> dict[str, pd.DataFrame]:
    """一次性加载所有 ticker 在持仓窗口内的日线。返回 {ticker: DataFrame}。"""
    from sqlalchemy import text
    tickers = sorted({r.ticker for r in recommendations})
    if not tickers:
        return {}

    # 取所有推荐的最早入场日和最晚入场日 + buffer
    min_date = min(r.entry_date for r in recommendations)
    max_date = max(r.entry_date for r in recommendations) + timedelta(days=max_holding_days + 10)

    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT ts_code, trade_date, open, high, low, close
                FROM daily_prices_panel
                WHERE ts_code = ANY(:tickers)
                  AND trade_date >= :start
                  AND trade_date <= :end
                ORDER BY ts_code, trade_date
            """),
            conn, params={"tickers": tickers, "start": min_date, "end": max_date},
        )
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    out: dict[str, pd.DataFrame] = {}
    for ticker, sub in df.groupby("ts_code"):
        out[str(ticker)] = sub.reset_index(drop=True)
    return out


# ── 单笔回放 ─────────────────────────────────────────────────────────────────

def replay_single_order(
    rec: HistoricalRecommendation,
    bars: pd.DataFrame,
    params: ReplayParams,
) -> dict[str, Any]:
    """对一条推荐用给定参数回放，返回订单结果 dict。

    严格按 backfill 的 T+1 次日开盘 + 滑点规则，但参数可以是 None（禁用）。

    Returns: {ticker, entry_date, entry_price, close_date, close_price,
              close_reason, pnl_pct, holding_days, ...}
    """
    if bars is None or bars.empty:
        # 无价格数据 → 按 entry_price 退化
        return {
            "ticker":        rec.ticker,
            "entry_date":    rec.entry_date,
            "entry_price":   rec.entry_price,
            "close_date":    rec.entry_date,
            "close_price":   rec.entry_price,
            "close_reason":  "no_data",
            "pnl_pct":       0.0,
            "holding_days":  0,
            "high_price":    rec.entry_price,
            "low_price":     rec.entry_price,
        }

    # 计算阈值价
    stop_loss_price = (rec.entry_price * (1 + params.stop_loss)
                       if params.stop_loss is not None else None)
    target_price = (rec.entry_price * (1 + params.target_price)
                    if params.target_price is not None else None)
    trailing_pct = abs(params.trailing_stop) if params.trailing_stop is not None else None

    # 持仓窗口（自然日，给次日开盘留 buffer）
    plan_exit = rec.entry_date + timedelta(days=int(params.holding_days * 1.7))  # ≈ 工作日→自然日
    window = bars[(bars["trade_date"] > rec.entry_date) &
                   (bars["trade_date"] <= plan_exit + timedelta(days=10))].reset_index(drop=True)

    if window.empty:
        return {
            "ticker": rec.ticker, "entry_date": rec.entry_date,
            "entry_price": rec.entry_price,
            "close_date": rec.entry_date, "close_price": rec.entry_price,
            "close_reason": "no_data",
            "pnl_pct": 0.0, "holding_days": 0,
            "high_price": rec.entry_price, "low_price": rec.entry_price,
        }

    running_high = rec.entry_price
    running_low = rec.entry_price
    high_date = rec.entry_date
    low_date = rec.entry_date
    bars_held = 0  # 已持有的交易日数

    for i, bar in window.iterrows():
        bars_held = int(i) + 1
        d = bar["trade_date"]
        h = float(bar["high"]) if pd.notna(bar["high"]) else 0.0
        lo = float(bar["low"]) if pd.notna(bar["low"]) else 0.0
        c = float(bar["close"]) if pd.notna(bar["close"]) else 0.0

        if h > running_high:
            running_high = h
            high_date = d
        if lo < running_low or running_low == rec.entry_price:
            running_low = lo
            low_date = d

        # 1. 止损（最高优先级）
        if stop_loss_price is not None and lo <= stop_loss_price:
            close_px, close_d = _exec_price_next_open(window, i, fallback_price=stop_loss_price)
            return {
                "ticker": rec.ticker, "entry_date": rec.entry_date,
                "entry_price": rec.entry_price,
                "close_date": close_d, "close_price": close_px,
                "close_reason": "stop_loss",
                "pnl_pct": (close_px - rec.entry_price) / rec.entry_price,
                "holding_days": (close_d - rec.entry_date).days,
                "high_price": running_high, "low_price": running_low,
            }
        # 2. 止盈
        if target_price is not None and h >= target_price:
            close_px, close_d = _exec_price_next_open(window, i, fallback_price=target_price)
            return {
                "ticker": rec.ticker, "entry_date": rec.entry_date,
                "entry_price": rec.entry_price,
                "close_date": close_d, "close_price": close_px,
                "close_reason": "target_hit",
                "pnl_pct": (close_px - rec.entry_price) / rec.entry_price,
                "holding_days": (close_d - rec.entry_date).days,
                "high_price": running_high, "low_price": running_low,
            }
        # 3. 追踪止损
        if trailing_pct is not None and running_high > 0:
            drawdown = (running_high - c) / running_high
            if drawdown >= trailing_pct:
                close_px, close_d = _exec_price_next_open(window, i, fallback_price=c)
                return {
                    "ticker": rec.ticker, "entry_date": rec.entry_date,
                    "entry_price": rec.entry_price,
                    "close_date": close_d, "close_price": close_px,
                    "close_reason": "trailing_stop",
                    "pnl_pct": (close_px - rec.entry_price) / rec.entry_price,
                    "holding_days": (close_d - rec.entry_date).days,
                    "high_price": running_high, "low_price": running_low,
                }
        # 4. 到期：达到 holding_days 个交易日
        if bars_held >= params.holding_days:
            close_px = c * (1 - SLIPPAGE_SELL_PCT)
            return {
                "ticker": rec.ticker, "entry_date": rec.entry_date,
                "entry_price": rec.entry_price,
                "close_date": d, "close_price": close_px,
                "close_reason": "time_expired",
                "pnl_pct": (close_px - rec.entry_price) / rec.entry_price,
                "holding_days": (d - rec.entry_date).days,
                "high_price": running_high, "low_price": running_low,
            }

    # 全部窗口扫完仍未触发（极端：数据末尾就是 holding_days 之前）
    last = window.iloc[-1]
    close_px = float(last["close"]) * (1 - SLIPPAGE_SELL_PCT)
    return {
        "ticker": rec.ticker, "entry_date": rec.entry_date,
        "entry_price": rec.entry_price,
        "close_date": last["trade_date"], "close_price": close_px,
        "close_reason": "time_expired",
        "pnl_pct": (close_px - rec.entry_price) / rec.entry_price,
        "holding_days": (last["trade_date"] - rec.entry_date).days,
        "high_price": running_high, "low_price": running_low,
    }


# ── 完整回放（多笔订单）─────────────────────────────────────────────────────

class HistoricalReplayEngine:
    """对一组历史推荐用一套参数完整回放，输出 NAV + 指标。"""

    def __init__(
        self,
        recommendations: list[HistoricalRecommendation],
        price_history: dict[str, pd.DataFrame],
    ) -> None:
        self.recommendations = recommendations
        self.price_history = price_history

    def replay(self, params: ReplayParams) -> dict[str, Any]:
        """回放一套参数，返回 {nav_curve, orders, metrics}。"""
        orders: list[dict] = []
        for rec in self.recommendations:
            bars = self.price_history.get(rec.ticker)
            result = replay_single_order(rec, bars, params)
            orders.append(result)

        if not orders:
            return {"nav_curve": [], "orders": [], "metrics": {}}

        df = pd.DataFrame(orders).sort_values("close_date").reset_index(drop=True)
        n_total = len(df)

        # 等权 NAV（每笔 1/N 资金）
        contrib = df["pnl_pct"].fillna(0.0) / n_total
        nav = (1.0 + contrib.cumsum()).tolist()
        dates = [str(d) for d in df["close_date"]]

        nav_arr = pd.Series(nav)
        roll_max = nav_arr.cummax()
        dd = (nav_arr - roll_max) / roll_max
        avg = df["pnl_pct"].mean()
        std = df["pnl_pct"].std()

        metrics = {
            "n":               int(n_total),
            "cum_return":      float(nav_arr.iloc[-1] - 1.0),
            "maxdd":           float(dd.min()) if not dd.empty else 0.0,
            "sharpe":          (float(avg / std * (252 ** 0.5))
                                if std and std > 0 else None),
            "win_rate":        float((df["pnl_pct"] > 0).mean()),
            "avg_return":      float(avg),
            "std_return":      float(std) if std else 0.0,
            "avg_holding_days": float(df["holding_days"].mean()),
            "exit_reasons":    df["close_reason"].value_counts().to_dict(),
        }

        nav_curve = list(zip(dates, nav))
        return {"nav_curve": nav_curve, "orders": orders, "metrics": metrics}
