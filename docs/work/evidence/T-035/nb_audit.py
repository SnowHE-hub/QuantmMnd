import pandas as pd, numpy as np
nb = pd.read_parquet("/home/lenovo/projects/quantmind/data/lake/north_bound.parquet")
nb["d"] = pd.to_datetime(nb["trade_date"].astype(str))
nb = nb.sort_values("d").set_index("d")
for c in ["north_money", "hgt", "sgt"]:
    nb[c] = pd.to_numeric(nb[c], errors="coerce")
nb["q"] = nb.index.to_period("Q")
g = nb.groupby("q")
out = pd.DataFrame({
    "n": g["north_money"].size(),
    "mean": g["north_money"].mean().round(0),
    "std": g["north_money"].std().round(0),
    "abs_med": g["north_money"].apply(lambda x: x.abs().median()).round(0),
    "frac_neg": g["north_money"].apply(lambda x: (x < 0).mean()).round(2),
    "id_ok": g.apply(lambda x: (np.isclose(x["north_money"], x["hgt"] + x["sgt"], rtol=1e-3, atol=1)).mean()).round(2),
})
print(out.loc["2023Q1":].to_string())
print(nb.loc["2024-05-01":"2024-05-31", ["hgt", "sgt", "north_money"]].head(15).to_string())
