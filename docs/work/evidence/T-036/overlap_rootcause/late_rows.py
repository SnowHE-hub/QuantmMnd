import gc, sys
import pandas as pd, numpy as np, pyarrow.dataset as ds
from pathlib import Path
L = Path("/home/lenovo/projects/quantmind/data/lake"); RAW = Path("/home/lenovo/projects/quantmind/data/raw")
LO, HI = "20260601", "20261009"
def rd(p, col, cols=None):
    df = pd.read_parquet(p, columns=cols)
    v = df[col].astype(str).str.replace("-", "").str.slice(0, 8)
    return df[(v >= LO) & (v <= HI)]
def h(df): return set(pd.util.hash_pandas_object(df.astype(str), index=False).to_numpy())
which = sys.argv[1]
if which == "prices":
    cols = ["ts_code", "trade_date"]
    a = rd(RAW / "alpha_prices_panel_v6_20261009.parquet", "trade_date", cols); b = rd(RAW / "alpha_prices_panel_v6_20261020.parquet", "trade_date", cols)
    ka, kb = h(a), pd.util.hash_pandas_object(b.astype(str), index=False).to_numpy()
    extra = b[~np.isin(kb, list(ka))]
    print("prices extra rows in [0601,1009]:", len(extra)); print(extra.groupby("ts_code")["trade_date"].agg(["min", "max", "count"]).to_string()[:2000])
elif which == "fund":
    for t in ("income", "balancesheet", "cashflow", "fina_indicator_full"):
        k = ["ts_code", "end_date", "ann_date"]
        o = pd.read_parquet(L / "_backup_20261019" / f"{t}.parquet", columns=k); n = pd.read_parquet(L / f"{t}.parquet", columns=k)
        n = n[n.ann_date.astype(str) <= HI]
        add = n[~pd.util.hash_pandas_object(n.astype(str), index=False).isin(h(o)).to_numpy()]
        print(t, "added rows ann<=1009:", len(add), "tickers", add.ts_code.nunique(), sorted(add.ts_code.unique())[:30], "ann", add.ann_date.min(), add.ann_date.max())
else:
    api, col = which.split(":")
    o = rd(L / f"events/{api}_20261009.parquet", col); n = rd(L / f"events/{api}_20261020.parquet", col)
    common = [c for c in o.columns if c in n.columns]
    ko, kn = h(o[common]), pd.util.hash_pandas_object(n[common].astype(str), index=False).to_numpy()
    add = n[~np.isin(kn, list(ko))]
    gone = len(ko - set(kn))
    print(f"{api}: rows[0601,1009] frozen {len(o)} new {len(n)}; added {len(add)} tickers {add.ts_code.nunique() if 'ts_code' in add else '-'} "
          f"{col} {add[col].min() if len(add) else ''}..{add[col].max() if len(add) else ''}; frozen rows not in new {gone}")
    if len(add) and len(add) < 400:
        print("   by date:", add[col].astype(str).value_counts().sort_index().to_dict())
