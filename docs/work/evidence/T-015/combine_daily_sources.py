"""合并两份日线来源（同一 Tushare pct_chg 定义）供 --price-panel 使用：完整 1374 票池（至 2026-10-09）优先，
旧 daily_prices_panel（508 只，至 2024-12-31）补充 1374 之外的票。重叠行核对一致后再去重。"""
import pandas as pd

Q = "/home/lenovo/projects/quantmind/data/raw"
cols = ["trade_date", "ts_code", "close", "pct_chg"]
a = pd.read_parquet(f"{Q}/alpha_prices_panel_20261009.parquet", columns=cols)
b = pd.read_parquet(f"{Q}/daily_prices_panel.parquet", columns=cols)
for d in (a, b):
    d["trade_date"] = pd.to_datetime(d["trade_date"])
m = a.merge(b, on=["trade_date", "ts_code"], suffixes=("_a", "_b"))
bad = ((m.pct_chg_a - m.pct_chg_b).abs() > 1e-6) | ((m.close_a - m.close_b).abs() > 1e-6)
print(f"overlap rows {len(m):,}; pct_chg/close mismatches {int(bad.sum())}")
assert not bad.any()
out = pd.concat([a, b], ignore_index=True).drop_duplicates(["trade_date", "ts_code"], keep="first")
out.to_parquet("/home/lenovo/c3_scratch/daily_sources_combined_20261009.parquet", index=False)
print(f"combined rows {len(out):,}, tickers {out.ts_code.nunique()}, b-only tickers {len(set(b.ts_code) - set(a.ts_code))}")
