"""
scripts/backfill_executions.py — 从历史 realized_pnl + forward_positions 反向构造
  simulated_orders 记录。

执行逻辑：
  1. realized_pnl 80 条 → simulated_orders CLOSED
     * 重新模拟止损/止盈：对每个订单的持仓区间（entry_date 到 exit_date），
       从 PG.daily_prices_panel 拉日线数据，按规则引擎重放
     * 如果中间某天触发止损/止盈，close_reason 标记为对应规则，close_price=触发价
     * 否则 close_reason='time_expired', close_price=原 exit_price
  2. forward_positions 20 条 OPEN → simulated_orders OPEN
     * 用 entry_price 开仓，target/stop_loss 用默认规则填充

用法：
  python scripts/backfill_executions.py [--reset] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db.postgres import get_pg_engine
from quantmind.execution.manager import (
    DEFAULT_HOLDING_DAYS, DEFAULT_STOP_LOSS_PCT, DEFAULT_TARGET_PCT,
)


def _load_pnl_records(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM realized_pnl ORDER BY as_of_date"), conn)
    return df


def _load_forward_positions() -> list[dict]:
    fp = ROOT / "data" / "paper_trading" / "forward_positions.json"
    if not fp.exists():
        return []
    data = json.loads(fp.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("positions", [])


def _load_name_industry_map() -> dict:
    """从 alpha_universe 读 (ticker → {name, industry}) 映射。"""
    eng = get_pg_engine()
    with eng.connect() as conn:
        df = pd.read_sql(
            text("SELECT ts_code, name, industry FROM alpha_universe"), conn)
    return {
        row.ts_code: {"name": row.name, "industry": row.industry}
        for row in df.itertuples()
    }


def _simulate_exit_with_intraday(
    engine,
    ticker: str,
    entry_date: date,
    exit_date: date,
    entry_price: float,
    target_price: float,
    stop_loss_price: float,
    trailing_pct: float = 0.15,
) -> dict:
    """模拟在 [entry_date+1, exit_date] 区间内，按日线 high/low 触发止损/止盈。

    返回 {close_date, close_price, close_reason, high_price, low_price, ...}
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT trade_date, open, high, low, close
                FROM daily_prices_panel
                WHERE ts_code = :t AND trade_date > :s AND trade_date <= :e
                ORDER BY trade_date
            """),
            conn, params={"t": ticker, "s": entry_date, "e": exit_date},
        )

    if df.empty:
        # 无日线数据，按到期处理（最后一日用 entry_price）
        return {
            "close_date":   exit_date,
            "close_price":  entry_price,
            "close_reason": "time_expired",
            "high_price":   entry_price,
            "low_price":    entry_price,
            "high_date":    entry_date,
            "low_date":     entry_date,
        }

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    running_high = entry_price
    running_low = entry_price
    high_date = entry_date
    low_date = entry_date

    for _, bar in df.iterrows():
        d = bar["trade_date"]
        h = float(bar["high"]) if pd.notna(bar["high"]) else 0.0
        lo = float(bar["low"]) if pd.notna(bar["low"]) else 0.0
        c = float(bar["close"]) if pd.notna(bar["close"]) else 0.0

        if h > running_high:
            running_high = h
            high_date = d
        if lo < running_low or running_low == entry_price:
            running_low = lo
            low_date = d

        # 优先级 1：止损（日内 low 击穿）
        if stop_loss_price > 0 and lo <= stop_loss_price:
            return {
                "close_date":   d,
                "close_price":  stop_loss_price,
                "close_reason": "stop_loss",
                "high_price":   running_high,
                "low_price":    running_low,
                "high_date":    high_date,
                "low_date":     low_date,
            }
        # 优先级 2：止盈（日内 high 达到）
        if target_price > 0 and h >= target_price:
            return {
                "close_date":   d,
                "close_price":  target_price,
                "close_reason": "target_hit",
                "high_price":   running_high,
                "low_price":    running_low,
                "high_date":    high_date,
                "low_date":     low_date,
            }
        # 优先级 3：追踪止损（从持仓期高点回撤 ≥ trailing_pct）
        if running_high > 0:
            drawdown = (running_high - c) / running_high
            if drawdown >= trailing_pct:
                return {
                    "close_date":   d,
                    "close_price":  c,
                    "close_reason": "trailing_stop",
                    "high_price":   running_high,
                    "low_price":    running_low,
                    "high_date":    high_date,
                    "low_date":     low_date,
                }

    # 全程未触发，按到期处理
    last_bar = df.iloc[-1]
    return {
        "close_date":   last_bar["trade_date"],
        "close_price":  float(last_bar["close"]),
        "close_reason": "time_expired",
        "high_price":   running_high,
        "low_price":    running_low,
        "high_date":    high_date,
        "low_date":     low_date,
    }


def backfill_closed(engine, dry_run: bool = False) -> dict:
    """根据 realized_pnl 80 条构造 CLOSED 订单。"""
    pnl = _load_pnl_records(engine)
    name_map = _load_name_industry_map()

    print(f"\n[backfill_closed] 处理 {len(pnl)} 条 realized_pnl ...")
    reason_counts: dict[str, int] = {}
    rows = []

    for _, r in pnl.iterrows():
        ticker = str(r["ticker"])
        entry_date = r["entry_date"]
        exit_date = r["exit_date"]
        if isinstance(entry_date, str):
            entry_date = pd.to_datetime(entry_date).date()
        if isinstance(exit_date, str):
            exit_date = pd.to_datetime(exit_date).date()
        entry_price = float(r["entry_price"])
        original_exit_price = float(r["exit_price"]) if pd.notna(r["exit_price"]) else None

        # 默认 target/stop_loss（Agent 历史未给出有效值）
        target_price = entry_price * (1 + DEFAULT_TARGET_PCT)
        stop_loss_price = entry_price * (1 - DEFAULT_STOP_LOSS_PCT)

        # 模拟规则触发
        sim = _simulate_exit_with_intraday(
            engine, ticker, entry_date, exit_date,
            entry_price, target_price, stop_loss_price,
        )
        reason = sim["close_reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        close_price = sim["close_price"]
        close_date = sim["close_date"]
        holding_days = (close_date - entry_date).days
        pnl_pct = (close_price - entry_price) / entry_price
        max_gain = (sim["high_price"] - entry_price) / entry_price if sim["high_price"] else 0.0
        max_dd = ((sim["high_price"] - sim["low_price"]) / sim["high_price"]
                  if sim["high_price"] else 0.0)

        info = name_map.get(ticker, {})
        rows.append({
            "ticker":           ticker,
            "name":             info.get("name"),
            "industry":         info.get("industry"),
            "recommend_date":   r["as_of_date"].date() if hasattr(r["as_of_date"], "date") else r["as_of_date"],
            "recommend_rank":   int(r["predicted_rank"]) if pd.notna(r["predicted_rank"]) else None,
            "agent_signal":     float(r["predicted_score"]) if pd.notna(r["predicted_score"]) else None,
            "agent_confidence": None,
            "agent_rating":     None,
            "open_date":        entry_date,
            "open_price":       entry_price,
            "open_reason":      "backfill",
            "position_size":    0.05,  # 假设均仓 5%
            "target_price":     target_price,
            "stop_loss_price":  stop_loss_price,
            "holding_period":   DEFAULT_HOLDING_DAYS,
            "close_date":       close_date,
            "close_price":      close_price,
            "close_reason":     reason,
            "high_price":       sim["high_price"],
            "low_price":        sim["low_price"],
            "high_date":        sim["high_date"],
            "low_date":         sim["low_date"],
            "max_unrealized_gain": max_gain,
            "max_drawdown":     max_dd,
            "pnl_abs":          close_price - entry_price,
            "pnl_pct":          pnl_pct,
            "holding_days":     holding_days,
            "commission":       30.0,    # 默认 0.0003 * 100k
            "slippage":         100.0,   # 默认 0.001 * 100k
            "status":           "CLOSED",
        })

    print(f"[backfill_closed] reason 分布: {reason_counts}")

    if dry_run:
        print(f"[backfill_closed] DRY RUN — 不写入 DB")
        return {"n": len(rows), "reasons": reason_counts}

    # 批量写入
    df = pd.DataFrame(rows)
    df.to_sql("simulated_orders", engine,
              if_exists="append", index=False, method="multi", chunksize=50)
    print(f"[backfill_closed] ✓ 已写入 {len(rows)} 条 CLOSED 订单")
    return {"n": len(rows), "reasons": reason_counts}


def backfill_open(engine, dry_run: bool = False) -> dict:
    """从 forward_positions 20 条构造 OPEN 订单。"""
    positions = _load_forward_positions()
    name_map = _load_name_industry_map()
    print(f"\n[backfill_open] 处理 {len(positions)} 条 forward_positions ...")

    rows = []
    for p in positions:
        if p.get("status") != "OPEN":
            continue
        ticker = str(p["ticker"])
        entry_date = pd.to_datetime(p["as_of"]).date()
        entry_price = float(p["entry_price"])
        target_price = entry_price * (1 + DEFAULT_TARGET_PCT)
        stop_loss_price = entry_price * (1 - DEFAULT_STOP_LOSS_PCT)
        info = name_map.get(ticker, {})
        rows.append({
            "ticker":           ticker,
            "name":             p.get("name") or info.get("name"),
            "industry":         p.get("industry") or info.get("industry"),
            "recommend_date":   entry_date,
            "recommend_rank":   p.get("predicted_rank"),
            "agent_signal":     p.get("predicted_score"),
            "agent_confidence": None,
            "agent_rating":     None,
            "open_date":        entry_date,
            "open_price":       entry_price,
            "open_reason":      "backfill",
            "position_size":    0.05,
            "target_price":     target_price,
            "stop_loss_price":  stop_loss_price,
            "holding_period":   DEFAULT_HOLDING_DAYS,
            "high_price":       entry_price,
            "low_price":        entry_price,
            "high_date":        entry_date,
            "low_date":         entry_date,
            "max_unrealized_gain": 0.0,
            "max_drawdown":     0.0,
            "commission":       30.0,
            "slippage":         100.0,
            "status":           "OPEN",
        })

    if dry_run:
        print(f"[backfill_open] DRY RUN — {len(rows)} OPEN 订单未写入")
        return {"n": len(rows)}

    df = pd.DataFrame(rows)
    df.to_sql("simulated_orders", engine,
              if_exists="append", index=False, method="multi", chunksize=50)
    print(f"[backfill_open] ✓ 已写入 {len(rows)} 条 OPEN 订单")
    return {"n": len(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true",
                   help="清空 simulated_orders 表后重新回填")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    engine = get_pg_engine()

    if args.reset:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE simulated_orders RESTART IDENTITY"))
        print("[reset] simulated_orders 已清空")

    closed_summary = backfill_closed(engine, dry_run=args.dry_run)
    open_summary = backfill_open(engine, dry_run=args.dry_run)

    print("\n" + "=" * 55)
    print("回填汇总：")
    print(f"  CLOSED 订单: {closed_summary['n']} (close_reason: {closed_summary.get('reasons', {})})")
    print(f"  OPEN 订单:   {open_summary['n']}")

    if not args.dry_run:
        with engine.connect() as conn:
            cnt = conn.execute(text("SELECT COUNT(*) FROM simulated_orders")).scalar()
            stat = pd.read_sql(text(
                "SELECT status, COUNT(*) FROM simulated_orders GROUP BY status"
            ), conn)
            reasons = pd.read_sql(text(
                "SELECT close_reason, COUNT(*) FROM simulated_orders "
                "WHERE close_reason IS NOT NULL GROUP BY close_reason ORDER BY 2 DESC"
            ), conn)
        print(f"\nDB 总计: {cnt} 条")
        print(stat.to_string(index=False))
        print(reasons.to_string(index=False))


if __name__ == "__main__":
    main()
