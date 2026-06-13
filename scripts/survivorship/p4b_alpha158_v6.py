"""P4b：Alpha158 在 v6 全市场重提取（qlib_bakeoff env）.

= scripts/bakeoff/p3a_alpha158_asof.py 的 v6 版：provider_uri/qsym map/as_of 网格全指向 v6。
raw handler（infer_processors=[]）逐 instrument-batch → 只留 as_of 行 → qsym→ticker。
输出 data/bakeoff/alpha158_asof_v6.parquet（MultiIndex(as_of,ticker) × 158，列前缀 a158_）。
"""
import warnings, json, time
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158

REPO = Path("/home/lenovo/projects/quantmind")
qlib.init(provider_uri=str(REPO / "data/qlib_cn_daily_v6"), region="cn",
          expression_cache=None, dataset_cache=None)

qsym2ts = json.loads((REPO / "data/bakeoff/ticker_qsym_map_v6.json").read_text())
asof = sorted({d for (d, t) in pd.read_parquet(
    REPO / "data/panel/alpha_panel_weekly_v6.parquet", columns=["forward_return_12d"]).index})
asof_set = {pd.Timestamp(a).normalize() for a in asof}
START, END = str(min(asof).date()), str(max(asof).date())
insts = sorted(qsym2ts.keys())
print(f"[p4b] {len(insts)} instruments, {len(asof)} as_of, {START}->{END}", flush=True)

t0 = time.time(); parts = []
B = 250
for i in range(0, len(insts), B):
    batch = [s.upper() for s in insts[i:i + B]]
    h = Alpha158(instruments=batch, start_time=START, end_time=END,
                 fit_start_time=START, fit_end_time=START, infer_processors=[], learn_processors=[])
    f = h.fetch(col_set="feature").reset_index()
    f.columns = ["instrument" if str(c).lower() in ("instrument", "code") else
                 ("datetime" if str(c).lower() in ("datetime", "date") else c) for c in f.columns]
    f["datetime"] = pd.to_datetime(f["datetime"]).dt.normalize()
    f = f[f["datetime"].isin(asof_set)]
    parts.append(f)
    print(f"  batch {i//B}: {len(batch)} inst -> {len(f)} as_of rows ({time.time()-t0:.0f}s)", flush=True)

df = pd.concat(parts, ignore_index=True)
up2ts = {k.upper(): v for k, v in qsym2ts.items()}
df["ticker"] = df["instrument"].str.upper().map(up2ts)
df = df.dropna(subset=["ticker"])
df = df.rename(columns={"datetime": "as_of"}).drop(columns=["instrument"])
feat_cols = [c for c in df.columns if c not in ("as_of", "ticker")]
df = df.set_index(["as_of", "ticker"]).sort_index()
df.columns = [f"a158_{c}" for c in df.columns]
out = REPO / "data/bakeoff/alpha158_asof_v6.parquet"
df.to_parquet(out)
print(f"[p4b] saved {out} shape={df.shape} ({time.time()-t0:.0f}s)")
print(f"[p4b] nan frac: {round(float(df.isna().mean().mean()), 3)}  tickers={df.index.get_level_values('ticker').nunique()}")
