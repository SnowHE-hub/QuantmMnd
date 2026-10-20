"""sector 的差异是否都落在"同日同申万一级行业里有 fnd 值变化的股票"的组里（sector = 行业内去均值等，依赖 fnd）。"""
import pandas as pd, numpy as np
from pathlib import Path
R=Path("/home/lenovo/projects/quantmind/data/gpu_v1/panel"); S=Path("/home/lenovo/qm_fwd_root/20261020/main/data/gpu_v1/panel")
W0,W1=pd.Timestamp("2026-09-01"),pd.Timestamp("2026-10-09")
def rd(root,g):
    df=pd.concat([pd.read_parquet(root/g/f"{m}.parquet") for m in ("2026-09","2026-10")])
    d=df.index.get_level_values("as_of"); return df[(d>=W0)&(d<=W1)].sort_index()
def changed(f,s):
    idx=f.index.intersection(s.index); a,b=f.loc[idx],s.loc[idx,f.columns]
    num=[c for c in f.columns if pd.api.types.is_numeric_dtype(f[c])]
    x,y=a[num].to_numpy(float),b[num].to_numpy(float)
    m=~((x==y)|(np.isnan(x)&np.isnan(y)))
    return pd.Series(m.any(axis=1),index=idx), pd.DataFrame(m,index=idx,columns=num)
fch,_=changed(rd(R,"fnd"),rd(S,"fnd"))
sf,ss=rd(R,"sector"),rd(S,"sector")
sch,cells=changed(sf,ss)
sw=ss["sw_l1"].reindex(sch.index)
g=pd.DataFrame({"f":fch.reindex(sch.index).fillna(False).to_numpy(),"sw":sw.to_numpy(),"d":sch.index.get_level_values("as_of")},index=sch.index)
hit=g.groupby(["d","sw"])["f"].transform("any").to_numpy()
ch=sch.to_numpy()
print("sector rows changed",int(ch.sum()),"in (as_of, sw_l1) groups with a fnd change",int((ch&hit).sum()),"unexplained",int((ch&~hit).sum()))
print("changed sector columns:", {c:int(v) for c,v in cells.sum().items() if v})
