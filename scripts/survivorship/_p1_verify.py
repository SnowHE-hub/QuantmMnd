"""P1 收尾验证：daily↔adj_factor 按 (ts_code,trade_date) 精确匹配 + 退市票专项."""
import pandas as pd

daily = pd.read_parquet("data/lake/daily.parquet", columns=["ts_code", "trade_date"])
adj = pd.read_parquet("data/lake/adj_factor.parquet",
                      columns=["ts_code", "trade_date", "adj_factor"])
m = daily.merge(adj, on=["ts_code", "trade_date"], how="left", indicator=True)
miss = int((m._merge == "left_only").sum())
print(f"daily rows={len(daily):,}  missing adj_factor={miss:,} ({miss/len(daily):.4%})")
if miss:
    mm = m[m._merge == "left_only"]
    print("missing tickers:", mm.ts_code.nunique())
    print(mm.head(5)[["ts_code", "trade_date"]].to_string(index=False))
for ts in ["000005.SZ", "000018.SZ", "000023.SZ"]:
    sub = m[m.ts_code == ts]
    nmiss = int((sub._merge == "left_only").sum())
    print(f"  {ts}: daily_days={len(sub)} missing_adj={nmiss}")
print("PASS" if miss == 0 else "GAPS_FOUND")
