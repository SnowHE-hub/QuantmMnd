"""复权收盘价的日收益是否与 pct_chg 一致（只看相邻两天都有行的票）。重点：接缝日与补回的 18 天。"""
import pandas as pd

Q = "/home/lenovo/projects/quantmind/data"
v6 = pd.read_parquet(f"{Q}/raw/alpha_prices_panel_v6_20261009.parquet",
                     columns=["ts_code", "trade_date", "close", "pre_close", "pct_chg", "adj_factor", "adj_close"])
v6["trade_date"] = pd.to_datetime(v6["trade_date"])
v6 = v6.sort_values(["ts_code", "trade_date"])
cal = pd.DatetimeIndex(sorted(v6.trade_date.unique()))
prev_day = pd.Series(cal[:-1], index=cal[1:])
v6["prev_cal"] = v6["trade_date"].map(prev_day)
g = v6.groupby("ts_code")
v6["prev_date"] = g["trade_date"].shift(1)
v6["r_adj"] = (v6["adj_close"] / g["adj_close"].shift(1) - 1) * 100
cons = v6[v6["prev_date"] == v6["prev_cal"]].copy()   # 相邻交易日都有行
cons["bad"] = (cons["r_adj"] - cons["pct_chg"]).abs() > 0.05
by_day = cons.groupby("trade_date")["bad"].agg(["sum", "count"])
by_day["frac"] = by_day["sum"] / by_day["count"]
print("全期 bad 占比:", round(cons["bad"].mean(), 5), "bad 行:", int(cons["bad"].sum()))
gaps = pd.to_datetime(["2021-03-05", "2021-03-15", "2021-06-10", "2023-03-16", "2023-06-16", "2023-10-20",
                       "2026-03-03", "2026-03-04", "2026-05-12", "2026-05-13", "2026-06-01", "2026-08-03"])
print(by_day.reindex(gaps).to_string())
print("bad 占比最高的 8 天:")
print(by_day.sort_values("frac", ascending=False).head(8).to_string())
old_new = cons[cons.trade_date > "2026-05-11"]
print("2026-05-12 之后 bad 占比:", round(old_new["bad"].mean(), 5), "；之前:", round(cons[cons.trade_date <= "2026-05-11"]["bad"].mean(), 5))
