"""T-018 步骤 2：Alpha158 在 v6_1 网格上重提取（qlib_bakeoff env）。

= scripts/survivorship/p4b_alpha158_v6.py 的 v6_1 版：provider_uri → data/qlib_cn_daily_v6_1（完整价格表 dump），
as_of 网格 → asof_grid_v6_1.json（完整交易日历），qsym 映射 → ticker_qsym_map_v6_1.json；raw handler、分批、列前缀 a158_ 全部不变。
输出 data/bakeoff/alpha158_asof_v6_1.parquet（已存在则拒绝运行）。
用法：<qlib_bakeoff python> alpha158_v6_1.py <data_root> <asof_grid_v6_1.json>
"""
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
import qlib  # noqa: E402
from qlib.contrib.data.handler import Alpha158  # noqa: E402

root, grid_path = Path(sys.argv[1]), Path(sys.argv[2])
out = root / "data/bakeoff/alpha158_asof_v6_1.parquet"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
qlib.init(provider_uri=str(root / "data/qlib_cn_daily_v6_1"), region="cn", expression_cache=None, dataset_cache=None)
qsym2ts = json.loads((root / "data/bakeoff/ticker_qsym_map_v6_1.json").read_text())
asof = [pd.Timestamp(x) for x in json.loads(grid_path.read_text())["as_of"]]
asof_set = {a.normalize() for a in asof}
START, END = str(min(asof).date()), str(max(asof).date())
insts = sorted(qsym2ts.keys())
print(f"[a158 v6_1] {len(insts)} instruments, {len(asof)} as_of, {START}->{END}", flush=True)

t0 = time.time()
parts = []
B = 250
for i in range(0, len(insts), B):
    batch = [s.upper() for s in insts[i:i + B]]
    h = Alpha158(instruments=batch, start_time=START, end_time=END,
                 fit_start_time=START, fit_end_time=START, infer_processors=[], learn_processors=[])
    f = h.fetch(col_set="feature").reset_index()
    f.columns = ["instrument" if str(c).lower() in ("instrument", "code") else
                 ("datetime" if str(c).lower() in ("datetime", "date") else c) for c in f.columns]
    f["datetime"] = pd.to_datetime(f["datetime"]).dt.normalize()
    parts.append(f[f["datetime"].isin(asof_set)])
    print(f"  batch {i // B}: {len(batch)} inst -> {len(parts[-1])} as_of rows ({time.time() - t0:.0f}s)", flush=True)

df = pd.concat(parts, ignore_index=True)
up2ts = {k.upper(): v for k, v in qsym2ts.items()}
df["ticker"] = df["instrument"].str.upper().map(up2ts)
df = df.dropna(subset=["ticker"]).rename(columns={"datetime": "as_of"}).drop(columns=["instrument"])
df = df.set_index(["as_of", "ticker"]).sort_index()
df.columns = [f"a158_{c}" for c in df.columns]
df.to_parquet(out)
print(f"[a158 v6_1] saved {out} shape={df.shape} nan_frac={round(float(df.isna().mean().mean()), 3)} "
      f"tickers={df.index.get_level_values('ticker').nunique()} ({time.time() - t0:.0f}s)", flush=True)
