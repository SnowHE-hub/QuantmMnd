"""quantmind/execution/manager.py — 执行管理器.

职责：
  1. 根据 Agent 推荐开仓（写 simulated_orders）
  2. 每日更新持仓的 high/low/max_drawdown
  3. 调用 StopLossEngine 决定平仓
  4. 执行平仓并计算 PnL（commission + slippage）

数据库后端：PostgreSQL 的 simulated_orders 表
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from quantmind.execution.exit_decision import ExitDecision
from quantmind.execution.stop_loss_engine import StopLossEngine

log = logging.getLogger(__name__)

# 默认成本模型（A 股双边）
DEFAULT_COMMISSION_RATE = 0.0003   # 万三
DEFAULT_SLIPPAGE_RATE   = 0.001    # 千一

# Agent 未给出 target/stop_loss 时的 fallback（相对开仓价）
DEFAULT_TARGET_PCT      = 0.20     # +20%
DEFAULT_STOP_LOSS_PCT   = 0.10     # -10%
DEFAULT_HOLDING_DAYS    = 63       # 3 个月（按 63 交易日近似）


# ── holding_horizon 字符串解析 ────────────────────────────────────────────────

_HORIZON_PATTERNS = [
    (re.compile(r"^(\d+)\s*[dD]"),       1),    # "21d", "63 D"
    (re.compile(r"^(\d+)\s*[wW]"),       7),    # "4w"
    (re.compile(r"^(\d+)\s*[mM]"),       21),   # "3m" → 63
    (re.compile(r"^(\d+)\s*[yY]"),       252),  # "1y"
]


def parse_holding_horizon(value: Any, default: int = DEFAULT_HOLDING_DAYS) -> int:
    """把 '3m' / '21d' / '63' 等输入解析成 int 天数。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    s = str(value).strip()
    if not s:
        return default
    if s.isdigit():
        return int(s)
    for pat, mult in _HORIZON_PATTERNS:
        m = pat.match(s)
        if m:
            return int(m.group(1)) * mult
    return default


def parse_position_size(value: Any, default: float = 0.05) -> float:
    """把 '5%' / '轻仓(1-3%)' / 0.05 等解析成 0-1 浮点数。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (int, float)) and 0 < value <= 1:
        return float(value)
    s = str(value)
    # 提取百分号数字
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~]?\s*(\d+(?:\.\d+)?)?\s*%", s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
        return (lo + hi) / 2 / 100
    # "轻仓"="3%", "中仓"="6%", "重仓"="10%"
    if "轻" in s:
        return 0.03
    if "重" in s:
        return 0.10
    if "中" in s:
        return 0.06
    return default


# ── ExecutionManager 主类 ────────────────────────────────────────────────────

class ExecutionManager:
    """模拟执行管理器。"""

    def __init__(
        self,
        engine=None,                                # SQLAlchemy engine（测试可注入）
        stop_loss_engine: StopLossEngine | None = None,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        prices_panel: pd.DataFrame | None = None,   # 价格面板（测试可注入）
    ) -> None:
        if engine is None:
            from app.db.postgres import get_pg_engine
            engine = get_pg_engine()
        self._engine = engine
        self.stop_loss_engine = stop_loss_engine or StopLossEngine()
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self._prices_panel = prices_panel  # 可由调用方注入

    # ── 内部：价格查询 ───────────────────────────────────────────────────────

    def _load_prices_panel(self) -> pd.DataFrame:
        if self._prices_panel is not None:
            return self._prices_panel
        # 从 PG 加载 daily_prices_panel（小表，alpha universe 子集）
        with self._engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT ts_code, trade_date, close, high, low "
                     "FROM daily_prices_panel"), conn)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        self._prices_panel = df
        return df

    def get_price_on(self, ticker: str, as_of: date | str) -> float | None:
        """查 ticker 在 as_of 的收盘价；as_of 不是交易日则向前回退 5 天。"""
        df = self._load_prices_panel()
        as_of = pd.to_datetime(as_of).date() if not isinstance(as_of, date) else as_of
        sub = df[df["ts_code"] == ticker]
        if sub.empty:
            return None
        # 取 <= as_of 的最近一条
        sub = sub[sub["trade_date"] <= as_of].sort_values("trade_date", ascending=False)
        if sub.empty:
            return None
        row = sub.iloc[0]
        if pd.isna(row["close"]):
            return None
        return float(row["close"])

    def get_price_range(
        self, ticker: str, start: date | str, end: date | str,
    ) -> pd.DataFrame:
        """取 ticker 在 [start, end] 区间的日线数据。"""
        df = self._load_prices_panel()
        start = pd.to_datetime(start).date() if not isinstance(start, date) else start
        end = pd.to_datetime(end).date() if not isinstance(end, date) else end
        sub = df[(df["ts_code"] == ticker) &
                 (df["trade_date"] >= start) &
                 (df["trade_date"] <= end)].copy()
        return sub.sort_values("trade_date")

    # ── 开仓 ─────────────────────────────────────────────────────────────────

    def has_open_position(self, ticker: str) -> bool:
        """检查某 ticker 是否已有 OPEN 订单。"""
        with self._engine.connect() as conn:
            cnt = conn.execute(
                text("SELECT COUNT(*) FROM simulated_orders "
                     "WHERE ticker=:t AND status='OPEN'"),
                {"t": ticker},
            ).scalar()
        return (cnt or 0) > 0

    def open_position_from_recommendation(
        self,
        recommendation: dict,
        agent_analysis: dict | None = None,
        as_of: str | date | None = None,
        capital_per_position: float = 100_000.0,
        open_reason: str = "agent_recommendation",
    ) -> int | None:
        """从 Agent 推荐开仓，返回新建的 order_id（开仓失败返回 None）。"""
        as_of = (pd.to_datetime(as_of).date()
                 if as_of else date.today())
        ticker = str(recommendation.get("ticker", "")).strip()
        if not ticker:
            return None

        # 已持仓则跳过
        if self.has_open_position(ticker):
            log.info("[Execution] %s 已有 OPEN 订单，跳过开仓", ticker)
            return None

        # 开仓价：优先 recommendation.entry_price，回退到当日 close
        open_price = recommendation.get("entry_price")
        if not isinstance(open_price, (int, float)) or open_price <= 0:
            open_price = self.get_price_on(ticker, as_of)
        if open_price is None or open_price <= 0:
            log.warning("[Execution] %s 无法获取开仓价（as_of=%s）", ticker, as_of)
            return None
        open_price = float(open_price)

        # Agent 退出条件 + fallback
        agent = agent_analysis or {}
        agent_target = agent.get("target_price_3m") or 0
        agent_stop = agent.get("stop_loss_price") or 0
        target_price = (float(agent_target) if agent_target and agent_target > 0
                        else open_price * (1 + DEFAULT_TARGET_PCT))
        stop_loss_price = (float(agent_stop) if agent_stop and agent_stop > 0
                           else open_price * (1 - DEFAULT_STOP_LOSS_PCT))
        holding_period = parse_holding_horizon(
            agent.get("holding_horizon"), DEFAULT_HOLDING_DAYS)
        position_size = parse_position_size(agent.get("position_size"), 0.05)

        # 成本（按单笔金额计算，仅用于记录）
        commission = capital_per_position * self.commission_rate
        slippage = capital_per_position * self.slippage_rate

        row = {
            "ticker":           ticker,
            "name":             recommendation.get("name"),
            "industry":         recommendation.get("industry"),
            "recommend_date":   as_of,
            "recommend_rank":   recommendation.get("rank") or recommendation.get("lgbm_rank"),
            "agent_signal":     agent.get("composite_signal"),
            "agent_confidence": agent.get("confidence"),
            "agent_rating":     str(agent.get("rating", ""))[:20] or None,
            "open_date":        as_of,
            "open_price":       open_price,
            "open_reason":      open_reason,
            "position_size":    position_size,
            "target_price":     target_price,
            "stop_loss_price":  stop_loss_price,
            "holding_period":   holding_period,
            "high_price":       open_price,
            "low_price":        open_price,
            "high_date":        as_of,
            "low_date":         as_of,
            "max_unrealized_gain": 0.0,
            "max_drawdown":     0.0,
            "commission":       commission,
            "slippage":         slippage,
            "status":           "OPEN",
        }

        with self._engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO simulated_orders (
                        ticker, name, industry, recommend_date, recommend_rank,
                        agent_signal, agent_confidence, agent_rating,
                        open_date, open_price, open_reason, position_size,
                        target_price, stop_loss_price, holding_period,
                        high_price, low_price, high_date, low_date,
                        max_unrealized_gain, max_drawdown,
                        commission, slippage, status
                    ) VALUES (
                        :ticker, :name, :industry, :recommend_date, :recommend_rank,
                        :agent_signal, :agent_confidence, :agent_rating,
                        :open_date, :open_price, :open_reason, :position_size,
                        :target_price, :stop_loss_price, :holding_period,
                        :high_price, :low_price, :high_date, :low_date,
                        :max_unrealized_gain, :max_drawdown,
                        :commission, :slippage, :status
                    ) RETURNING order_id
                """),
                row,
            )
            order_id = int(result.scalar())

        log.info("[Execution] 开仓 #%d %s @ %.3f (target=%.3f stop=%.3f hold=%dd)",
                 order_id, ticker, open_price, target_price, stop_loss_price,
                 holding_period)
        return order_id

    # ── 每日维护 + 平仓 ─────────────────────────────────────────────────────

    def get_open_orders(self) -> list[dict]:
        with self._engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM simulated_orders WHERE status='OPEN' "
                     "ORDER BY open_date, order_id"), conn)
        return df.to_dict(orient="records")

    def get_closed_orders(self, days: int = 90) -> list[dict]:
        cutoff = (pd.Timestamp.today() - pd.Timedelta(days=days)).date()
        with self._engine.connect() as conn:
            df = pd.read_sql(
                text("SELECT * FROM simulated_orders "
                     "WHERE status='CLOSED' AND close_date >= :cutoff "
                     "ORDER BY close_date DESC"),
                conn, params={"cutoff": cutoff})
        return df.to_dict(orient="records")

    def _update_extremes(
        self, order: dict, current_price: float, as_of: date,
    ) -> None:
        """更新订单的 high/low/max_drawdown（不平仓）。"""
        order_id = int(order["order_id"])
        open_price = float(order["open_price"])
        high_price = order.get("high_price") or open_price
        low_price = order.get("low_price") or open_price

        new_high = max(float(high_price), current_price)
        new_low = min(float(low_price), current_price)
        max_gain = (new_high - open_price) / open_price
        # max_drawdown：从历史最高回撤到当前价
        max_dd = (new_high - current_price) / new_high if new_high > 0 else 0.0

        params: dict[str, Any] = {
            "order_id": order_id,
            "high_price": new_high,
            "low_price": new_low,
            "max_unrealized_gain": float(max_gain),
            "max_drawdown": float(max_dd),
            "as_of": as_of,
        }
        sets = ["high_price=:high_price", "low_price=:low_price",
                "max_unrealized_gain=:max_unrealized_gain",
                "max_drawdown=:max_drawdown", "updated_at=NOW()"]
        if new_high > float(high_price):
            sets.append("high_date=:as_of")
        if new_low < float(low_price):
            sets.append("low_date=:hl_as_of")
            params["hl_as_of"] = as_of
        with self._engine.begin() as conn:
            conn.execute(
                text(f"UPDATE simulated_orders SET {', '.join(sets)} "
                     f"WHERE order_id=:order_id"),
                params,
            )

    def close_position(
        self,
        order_id: int,
        close_price: float,
        close_reason: str,
        as_of: date | str,
    ) -> dict:
        """执行平仓 + 计算最终 PnL。"""
        as_of = (pd.to_datetime(as_of).date()
                 if not isinstance(as_of, date) else as_of)

        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM simulated_orders WHERE order_id=:i"),
                {"i": order_id},
            ).mappings().first()
        if row is None:
            raise ValueError(f"order_id={order_id} 不存在")
        if row["status"] != "OPEN":
            raise ValueError(f"order_id={order_id} 状态={row['status']}, 无法平仓")

        open_price = float(row["open_price"])
        commission = float(row.get("commission") or 0.0) * 2  # 双边
        slippage = float(row.get("slippage") or 0.0) * 2
        pnl_pct = (close_price - open_price) / open_price
        pnl_abs = (close_price - open_price) - (commission + slippage) / max(open_price, 1)
        # pnl_abs 按单股价格记录（不乘股数），可供后续按 position_size 加权
        open_date = row["open_date"]
        if isinstance(open_date, datetime):
            open_date = open_date.date()
        holding_days = (as_of - open_date).days

        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE simulated_orders SET
                        close_date=:close_date,
                        close_price=:close_price,
                        close_reason=:close_reason,
                        pnl_abs=:pnl_abs,
                        pnl_pct=:pnl_pct,
                        holding_days=:holding_days,
                        status='CLOSED',
                        updated_at=NOW()
                    WHERE order_id=:order_id
                """),
                {
                    "close_date": as_of, "close_price": float(close_price),
                    "close_reason": close_reason, "pnl_abs": float(pnl_abs),
                    "pnl_pct": float(pnl_pct), "holding_days": int(holding_days),
                    "order_id": order_id,
                },
            )
        log.info("[Execution] 平仓 #%d %s @ %.3f (%.2f%%) reason=%s days=%d",
                 order_id, row["ticker"], close_price, pnl_pct * 100,
                 close_reason, holding_days)
        return {
            "order_id": order_id, "ticker": row["ticker"],
            "close_reason": close_reason, "pnl_pct": pnl_pct,
            "holding_days": holding_days,
        }

    def daily_update(self, as_of: date | str | None = None) -> dict:
        """每日维护：更新极值 → 评估退出 → 平仓。返回今日摘要。"""
        as_of = (pd.to_datetime(as_of).date()
                 if as_of else date.today())
        open_orders = self.get_open_orders()
        if not open_orders:
            return {
                "as_of": as_of.isoformat(),
                "n_open": 0, "n_updated": 0,
                "n_closed": 0, "closes": [],
            }

        # 当日价格
        current_prices: dict[str, float] = {}
        for o in open_orders:
            px = self.get_price_on(str(o["ticker"]), as_of)
            if px is not None:
                current_prices[str(o["ticker"])] = px

        # 1. 更新极值
        for o in open_orders:
            t = str(o["ticker"])
            if t in current_prices:
                self._update_extremes(o, current_prices[t], as_of)

        # 2. 评估退出（先重新查一遍订单获得最新 high_price）
        open_orders = self.get_open_orders()
        decisions = self.stop_loss_engine.evaluate_orders(
            open_orders, current_prices, as_of,
        )

        # 3. 执行平仓
        closes: list[dict] = []
        for d in decisions:
            try:
                result = self.close_position(
                    d.order_id, d.exit_price, d.exit_reason, as_of,
                )
                result["note"] = d.note
                closes.append(result)
            except Exception as e:  # noqa: BLE001
                log.warning("[Execution] 平仓 #%d 失败: %s", d.order_id, e)

        return {
            "as_of":      as_of.isoformat(),
            "n_open":     len(open_orders),
            "n_updated":  len(current_prices),
            "n_closed":   len(closes),
            "closes":     closes,
        }
