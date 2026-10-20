import pandas as pd, numpy as np
from pathlib import Path
R=Path("/home/lenovo/projects/quantmind/data/gpu_v1/panel"); S=Path("/home/lenovo/qm_fwd_root/20261020/main/data/gpu_v1/panel")
W0,W1=pd.Timestamp("2026-09-01"),pd.Timestamp("2026-10-09")
def rd(root,g,cols=None):
    df=pd.concat([pd.read_parquet(root/g/f"{m}.parquet",columns=cols) for m in ("2026-09","2026-10")])
    d=df.index.get_level_values("as_of"); return df[(d>=W0)&(d<=W1)].sort_index()
f,s=rd(R,"a158",["a158_KMID"]),rd(S,"a158",["a158_KMID"])
ex=s.index.difference(f.index); e=ex.to_frame(index=False)
print("a158 extra rows by ticker:"); print(e.groupby("ticker")["as_of"].agg(["min","max","count"]).to_string())
print("extra-row a158_KMID non-null:", int(s.loc[ex,"a158_KMID"].notna().sum()), "of", len(ex))
sus=pd.read_parquet("/home/lenovo/projects/quantmind/data/lake/events/suspend_d_20261020.parquet")
sus=sus[sus.ts_code.isin(e.ticker.unique()) & (sus.trade_date.astype(str)>="20260801")]
print("suspend_d rows (>=0801):"); print(sus.groupby(["ts_code","suspend_type"])["trade_date"].agg(["min","max","count"]).to_string())
px=pd.read_parquet("/home/lenovo/projects/quantmind/data/raw/alpha_prices_panel_v6_20261020.parquet",columns=["ts_code","trade_date","vol"])
px=px[px.ts_code.isin(e.ticker.unique()) & (pd.to_datetime(px.trade_date)>="2026-08-15")]
px["traded"]=px["vol"]>0
print("price rows since 08-15:"); print(px.groupby("ts_code").agg(first=("trade_date","min"),last=("trade_date","max"),n=("trade_date","count"),n_traded=("traded","sum")).to_string())
b=rd(R,"base",["momentum_1m"]); print("extra rows that are in base (PIT panel rows):", int(ex.isin(b.index).sum()), "of", len(ex))
mf=pd.read_parquet(R/"market/market_daily.parquet"); ms=pd.read_parquet(S/"market/market_daily.parquet")
w=lambda m: m[(m.index>=W0)&(m.index<=W1)]
mf,ms=w(mf),w(ms)
for c in ("disp_1","disp_20","amt_rel_20"):
    x,y=mf[c].to_numpy(float),ms[c].to_numpy(float); d=np.abs(x-y); print("market",c,"max abs",d.max(),"max rel",np.nanmax(d/np.abs(x)))
