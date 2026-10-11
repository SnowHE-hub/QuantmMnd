import pandas as pd

Q = "/home/lenovo/projects/quantmind/data"
a = pd.read_parquet(f"{Q}/paper_trading/positions.parquet")
b = pd.read_parquet(f"{Q}/paper_trading_20261009/positions.parquet")
print("columns:", list(a.columns)[:14])
key = [c for c in ("as_of", "as_of_date", "rec_date", "date") if c in a.columns][:1] + ["ticker"]
rets = [c for c in a.columns if c.startswith("ret_") or c.startswith("return_")]
print("key", key, "returns", rets)
m = a.merge(b, on=key, suffixes=("_raw", "_adj"))
print("same (date,ticker) rows:", len(m), "of", len(a))
for c in rets:
    x, y = m[f"{c}_raw"], m[f"{c}_adj"]
    ok = x.notna() & y.notna()
    d = (x[ok] - y[ok]).abs()
    print(f"{c}: n={int(ok.sum())} mean raw {x[ok].mean():+.4f} adj {y[ok].mean():+.4f} | |diff|>0.5% {int((d > 0.005).sum())} max {d.max():.4f}")
