"""scripts/settle_forward_positions.py

前向持仓结算脚本：
- 读取 data/paper_trading/forward_positions.json
- 用 Tushare 拉取每只股票从建仓日到到期日（或最近可用日）的实际收盘价
- 计算实际收益率，保存到 data/paper_trading/forward_settlement_2026Q2.parquet
- 同时追加到 data/feedback/realized_pnl.parquet

两组持仓：
  2025-12-31 组：exit_date=2026-03-30（已过，完整结算）
  2026-03-31 组：exit_date=2026-06-26（未到，用最近交易日结算，标记 interim）
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import tushare as ts

TOKEN = os.getenv("TUSHARE_TOKEN", "")
if not TOKEN:
    raise RuntimeError("TUSHARE_TOKEN 未设置，请在 .env 中配置")

pro = ts.pro_api(TOKEN)

TODAY_STR = datetime.today().strftime("%Y%m%d")

# ─── 读取前向持仓 ──────────────────────────────────────────────────────────────

fwd_path = ROOT / "data" / "paper_trading" / "forward_positions.json"
with open(fwd_path) as f:
    fwd = json.load(f)

positions = fwd.get("positions", [])
print(f"前向持仓共 {len(positions)} 条")

# ─── 逐只拉价格 ───────────────────────────────────────────────────────────────

results = []

for pos in positions:
    ticker = pos["ticker"]                                    # e.g. "000963.SZ"
    as_of  = pos["as_of"]                                    # "2026-03-31" or "2025-12-31"
    est_exit = pos.get("estimated_exit_date", "2026-06-26")  # "2026-06-26" or "2026-03-30"

    entry_dt = as_of.replace("-", "")                        # "20260331"

    # 决定出场日期
    est_exit_ts = datetime.strptime(est_exit, "%Y-%m-%d")
    today_ts    = datetime.strptime(TODAY_STR, "%Y%m%d")

    if est_exit_ts <= today_ts:
        exit_dt_str = est_exit.replace("-", "")
        settlement_type = "final"
    else:
        # 未到期 → 用今天作为临时结算日
        exit_dt_str = TODAY_STR
        settlement_type = "interim"

    print(f"  {ticker}  as_of={as_of}  exit={exit_dt_str}  type={settlement_type}", end=" ... ")

    try:
        df = pro.daily(
            ts_code=ticker,
            start_date=entry_dt,
            end_date=exit_dt_str,
            fields="trade_date,close,adj_factor",
        )
        time.sleep(0.2)  # Tushare 频率限制

        if df is None or len(df) < 2:
            # 尝试不带 adj_factor
            df = pro.daily(
                ts_code=ticker,
                start_date=entry_dt,
                end_date=exit_dt_str,
                fields="trade_date,close",
            )
            time.sleep(0.2)

        if df is None or len(df) < 2:
            print(f"数据不足 ({0 if df is None else len(df)} 行)")
            results.append({**pos, "actual_return_63d": None, "entry_price": None,
                            "exit_price": None, "settlement_type": settlement_type,
                            "entry_date": as_of, "exit_date": exit_dt_str, "error": "insufficient_data"})
            continue

        df = df.sort_values("trade_date").reset_index(drop=True)
        entry_price = float(df.iloc[0]["close"])
        exit_price  = float(df.iloc[-1]["close"])
        actual_exit_date = df.iloc[-1]["trade_date"]

        if entry_price <= 0:
            print("入场价为0，跳过")
            results.append({**pos, "actual_return_63d": None, "entry_price": None,
                            "exit_price": None, "settlement_type": settlement_type,
                            "entry_date": as_of, "exit_date": exit_dt_str, "error": "zero_entry"})
            continue

        ret = exit_price / entry_price - 1
        print(f"entry={entry_price:.2f}  exit={exit_price:.2f}  ret={ret:+.2%}")

        results.append({
            **pos,
            "actual_return_63d": round(ret, 6),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_date": df.iloc[0]["trade_date"],
            "exit_date": actual_exit_date,
            "settlement_type": settlement_type,
            "error": None,
        })

    except Exception as e:
        print(f"失败: {e}")
        results.append({**pos, "actual_return_63d": None, "entry_price": None,
                        "exit_price": None, "settlement_type": settlement_type,
                        "entry_date": as_of, "exit_date": exit_dt_str, "error": str(e)})

# ─── 结算汇总 ─────────────────────────────────────────────────────────────────

df_result = pd.DataFrame(results)
valid = df_result[df_result["actual_return_63d"].notna()]
print(f"\n结算完成：{len(valid)}/{len(df_result)} 只有效")
print(f"平均收益: {valid['actual_return_63d'].mean():.2%}")
print(f"胜率(>0): {(valid['actual_return_63d'] > 0).mean():.2%}")
print("\n明细:")
print(df_result[["ticker", "as_of", "actual_return_63d", "settlement_type"]].to_string())

# ─── 保存结算结果 ─────────────────────────────────────────────────────────────

out_path = ROOT / "data" / "paper_trading" / "forward_settlement_2026Q2.parquet"
df_result.to_parquet(out_path, index=False)
print(f"\n结算文件已保存: {out_path}")

# ─── 追加到 realized_pnl.parquet ──────────────────────────────────────────────

pnl_path = ROOT / "data" / "feedback" / "realized_pnl.parquet"
pnl = pd.read_parquet(pnl_path)
print(f"\n追加前 realized_pnl: {pnl.shape}")

# 字段对齐
new_rows = []
for _, row in valid.iterrows():
    # 计算组内排名和超额（同 as_of 组）
    same_group = valid[valid["as_of"] == row["as_of"]]
    all_rets = same_group["actual_return_63d"].values
    sorted_rets = sorted(all_rets, reverse=True)
    median_ret = float(np.median(all_rets))
    actual_rank = list(sorted_rets).index(row["actual_return_63d"]) + 1

    new_rows.append({
        "as_of_date": pd.Timestamp(row["as_of"]),
        "ticker": row["ticker"],
        "entry_date": str(row["entry_date"]),
        "exit_date": str(row["exit_date"]),
        "holding_days": 63,
        "predicted_rank": int(row["predicted_rank"]),
        "predicted_score": float(row["predicted_score"]),
        "entry_price": float(row["entry_price"]),
        "exit_price": float(row["exit_price"]),
        "actual_return_63d": float(row["actual_return_63d"]),
        "key_factors": json.dumps({"predicted_score": float(row["predicted_score"]),
                                   "settlement_type": row["settlement_type"]}),
        "actual_rank": actual_rank,
        "pnl_vs_median": round(float(row["actual_return_63d"]) - median_ret, 6),
        "hit": actual_rank <= int(row["predicted_rank"]),
        "panel_return_63d": None,
        "return_diff": None,
    })

df_new = pd.DataFrame(new_rows)

# 去重：若 (as_of_date, ticker) 已存在则跳过
pnl["as_of_date"] = pd.to_datetime(pnl["as_of_date"])
existing_keys = set(zip(pnl["as_of_date"].astype(str), pnl["ticker"]))
df_new_dedup = df_new[
    ~df_new.apply(lambda r: (str(r["as_of_date"]), r["ticker"]) in existing_keys, axis=1)
]
print(f"新增行（去重后）: {len(df_new_dedup)}")

pnl_updated = pd.concat([pnl, df_new_dedup], ignore_index=True)
pnl_updated.to_parquet(pnl_path, index=False)
print(f"追加后 realized_pnl: {pnl_updated.shape}")
# ── DB 双写（增量 upsert，失败不中断）──────────────────────────────────────
try:
    from app.db.writers import get_writer
    get_writer().write_realized_pnl(pnl_updated, full_replace=True)
    print("DB 双写完成（realized_pnl）")
except Exception as _dw_e:
    print(f"realized_pnl DB 双写跳过: {_dw_e}")
print(f"\n✅ 完成！前向持仓结算结果已追加到 {pnl_path}")
