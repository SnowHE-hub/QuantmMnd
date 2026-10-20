import pandas as pd
nb = pd.read_parquet("/home/lenovo/projects/quantmind/data/lake/north_bound.parquet")
nb["d"] = pd.to_datetime(nb["trade_date"].astype(str)); nb = nb.sort_values("d").set_index("d")
x = pd.to_numeric(nb["north_money"], errors="coerce")
w = x.loc["2024-07-15":"2024-09-15"]
print(w.to_string())
big = x[x.abs() > 60000]
print("first |x|>60000:", big.index.min().date() if len(big) else None, " last negative:", x[x < 0].index.max().date())
