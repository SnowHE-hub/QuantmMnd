"""复权因子的"非事件"跳动：某日 adj_factor 变动、次日又变回（af[t-1]==af[t+1]!=af[t]）= 单日基准漂移。
统计幅度与天数；对比真实除权（变动后不回落）。"""
import numpy as np
import pandas as pd

Q = "/home/lenovo/projects/quantmind/data"
a = pd.read_parquet(f"{Q}/raw/alpha_prices_panel_v6_20261009.parquet", columns=["ts_code", "trade_date", "adj_factor", "pct_chg", "close", "pre_close"])
a["trade_date"] = pd.to_datetime(a["trade_date"])
a = a.sort_values(["ts_code", "trade_date"])
g = a.groupby("ts_code")["adj_factor"]
a["prev"], a["next"] = g.shift(1), g.shift(-1)
a["chg"] = a["adj_factor"] / a["prev"] - 1
blip = (a["prev"] == a["next"]) & (a["adj_factor"] != a["prev"])
b = a[blip]
print(f"单日回落型跳动：{len(b):,} 行，涉及 {b.trade_date.nunique()} 天；|变动| 中位 {b.chg.abs().median():.5f}，最大 {b.chg.abs().max():.4f}")
print(b.groupby("trade_date").size().sort_values(ascending=False).head(10).to_string())
# 真实除权：pre_close 与前一日 close 不同（交易所除权参考价）
a["prev_close"] = a.groupby("ts_code")["close"].shift(1)
ex = (a["pre_close"] - a["prev_close"]).abs() > 1e-6
step = a[(a["chg"].abs() > 1e-9) & ~blip]
print(f"非回落型变动：{len(step):,} 行；其中当天有除权参考价（pre_close≠前收）的占 {ex[step.index].mean():.1%}")
lvl = a[(a["chg"].abs() > 1e-9) & ~blip & ~ex]
print(f"无除权却出现的持续性变动：{len(lvl):,} 行，|变动| 中位 {lvl.chg.abs().median():.5f}，>1% 的 {int((lvl.chg.abs() > 0.01).sum())} 行")
print(lvl.groupby("trade_date").size().sort_values(ascending=False).head(8).to_string())
