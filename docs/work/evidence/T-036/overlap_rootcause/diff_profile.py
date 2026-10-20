import pandas as pd, numpy as np
from pathlib import Path
R=Path("/home/lenovo/projects/quantmind/data/gpu_v1/panel"); S=Path("/home/lenovo/qm_fwd_root/20261020/main/data/gpu_v1/panel")
W0,W1=pd.Timestamp("2026-09-01"),pd.Timestamp("2026-10-09")
sb9=pd.read_parquet("/home/lenovo/projects/quantmind/data/lake/stock_basic_full_merged_20261009.parquet")
sb0=pd.read_parquet("/home/lenovo/projects/quantmind/data/lake/stock_basic_full_merged_20261020.parquet")
new=set(sb0.ts_code)-set(sb9.ts_code)
print("new in merged_20261020:", len(new), sb0[sb0.ts_code.isin(new)][["ts_code","list_date","delist_date","list_status"]].sort_values("list_date").to_string()[:3000])
def rd(root,g):
    df=pd.concat([pd.read_parquet(root/g/f"{m}.parquet") for m in ("2026-09","2026-10")])
    d=df.index.get_level_values("as_of"); return df[(d>=W0)&(d<=W1)].sort_index()
for g in ("fnd","sh","a158","event"):
    f,s=rd(R,g),rd(S,g)
    ex=s.index.difference(f.index)
    print(f"\n== {g}: extra rows {len(ex)}; tickers {sorted(set(ex.get_level_values('ticker')))[:20]}; in new? {len(set(ex.get_level_values('ticker'))&new)}")
    idx=f.index.intersection(s.index)
    a,b=f.loc[idx],s.loc[idx,f.columns]
    for c in f.columns:
        if not pd.api.types.is_numeric_dtype(a[c]): continue
        x,y=a[c].to_numpy(float),b[c].to_numpy(float)
        m=~((x==y)|(np.isnan(x)&np.isnan(y)))
        if m.sum()==0: continue
        nanflip=(np.isnan(x)!=np.isnan(y))&m
        d=np.abs(x-y)[m&~nanflip]
        rel=(np.abs(x-y)/np.maximum(np.abs(x),1e-12))[m&~nanflip]
        dates=pd.Series(idx.get_level_values("as_of")[m]).value_counts().sort_index()
        print(f"  {c}: n={m.sum()} nanflip={nanflip.sum()} maxabs={d.max() if len(d) else 0:.3g} maxrel={rel.max() if len(rel) else 0:.3g} med_rel={np.median(rel) if len(rel) else 0:.3g} dates={dict(list(dates.items())[:6])} ndates={len(dates)}")
