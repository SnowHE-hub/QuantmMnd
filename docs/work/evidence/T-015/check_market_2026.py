import pandas as pd

Q = "/home/lenovo/projects/quantmind/data"
v6 = pd.read_parquet(f"{Q}/raw/alpha_prices_panel_v6_20261009.parquet",
                     columns=["ts_code", "trade_date", "close", "pct_chg", "adj_close", "amount"])
v6["trade_date"] = pd.to_datetime(v6["trade_date"])
s = v6[v6.trade_date >= "2026-05-11"]
ew = (1 + s.groupby("trade_date")["pct_chg"].mean() / 100).cumprod()
print("全市场等权（pct_chg 均值连乘）2026-05-11→2026-10-09:", round(ew.iloc[-1] - 1, 4))
w = s.pivot(index="trade_date", columns="ts_code", values="adj_close")
r = w.pct_change(fill_method=None)
print("同上（adj_close 日收益均值）:", round((1 + r.mean(axis=1).fillna(0)).cumprod().iloc[-1] - 1, 4))
print("每月等权收益:", (1 + s.groupby("trade_date")["pct_chg"].mean() / 100).groupby(pd.Grouper(freq="ME")).prod().sub(1).round(4).to_dict())
for t in ("600519.SH", "000001.SZ", "300750.SZ"):
    x = s[s.ts_code == t].sort_values("trade_date")
    print(t, x.close.iloc[0], "→", x.close.iloc[-1], "adj ratio", round(x.adj_close.iloc[-1] / x.adj_close.iloc[0], 4))
n = pd.read_csv("/home/lenovo/c3_scratch/t010/run_v4_1_to20261009/executable_nav_daily.csv", parse_dates=["date"]).set_index("date")
print(n.loc["2026-05-08":, ["nav_net", "nav_bench"]].iloc[[0, 20, 40, 60, 80, -1]].round(4).to_string())
