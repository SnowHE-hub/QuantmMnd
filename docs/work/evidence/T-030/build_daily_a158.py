"""T-030 P0：Alpha158 在日频网格上的取值（qlib_bakeoff env；= T-018 alpha158_v6_1.py 的逻辑，只换取样网格）.

provider_uri、raw handler（无 processor）、每批 250 只、列前缀 a158_、qsym → ticker 映射全部与 T-018 相同；
T-018 只保留周频 as_of 的行，这里保留完整交易日历（2019-01-02 → 2026-10-09）的每一天。
阶段 1：每批写 <data_root>/data/gpu_v1/panel/_a158_parts/batch_<k>.parquet（已存在跳过 = 断点续跑）；
阶段 2：按月重排为 <data_root>/data/gpu_v1/panel/a158/<YYYY-MM>.parquet（已存在跳过）。
用法：<qlib_bakeoff python> build_daily_a158.py <data_root>
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

warnings.filterwarnings("ignore")
import qlib  # noqa: E402
from qlib.contrib.data.handler import Alpha158  # noqa: E402

root = Path(sys.argv[1])
parts_dir = root / "data/gpu_v1/panel/_a158_parts"
out_dir = root / "data/gpu_v1/panel/a158"
parts_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
qlib.init(provider_uri=str(root / "data/qlib_cn_daily_v6_1"), region="cn", expression_cache=None, dataset_cache=None,
          kernels=2)          # 只限并行进程数（内存），不影响取值
qsym2ts = json.loads((root / "data/bakeoff/ticker_qsym_map_v6_1.json").read_text())
up2ts = {k.upper(): v for k, v in qsym2ts.items()}
cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(root / "data/lake/trade_cal_sse.parquet")["cal_date"])).sort_values()
px = pd.to_datetime(pd.read_parquet(root / "data/raw/alpha_prices_panel_v6_20261009.parquet", columns=["trade_date"])["trade_date"])
cal = cal[(cal >= px.min()) & (cal <= px.max())]
START, END = str(cal[0].date()), str(cal[-1].date())
insts = sorted(qsym2ts.keys())
t0 = time.time()
B = 250
for k, i in enumerate(range(0, len(insts), B)):
    out = parts_dir / f"batch_{k:03d}.parquet"
    if out.exists():
        continue
    batch = [s.upper() for s in insts[i:i + B]]
    h = Alpha158(instruments=batch, start_time=START, end_time=END,
                 fit_start_time=START, fit_end_time=START, infer_processors=[], learn_processors=[])
    f = h.fetch(col_set="feature").reset_index()
    f.columns = ["instrument" if str(c).lower() in ("instrument", "code") else
                 ("datetime" if str(c).lower() in ("datetime", "date") else c) for c in f.columns]
    f["datetime"] = pd.to_datetime(f["datetime"]).dt.normalize()
    f["ticker"] = f["instrument"].str.upper().map(up2ts)
    f = f.dropna(subset=["ticker"]).rename(columns={"datetime": "as_of"}).drop(columns=["instrument"])
    f.columns = [c if c in ("as_of", "ticker") else f"a158_{c}" for c in f.columns]
    f.to_parquet(parts_dir / f"batch_{k:03d}.tmp", index=False)
    os.replace(parts_dir / f"batch_{k:03d}.tmp", out)
    print(f"  batch {k}: {len(batch)} inst -> {len(f)} rows ({time.time() - t0:.0f}s)", flush=True)
dset = ds.dataset(sorted(str(p) for p in parts_dir.glob("batch_*.parquet")), format="parquet")
months = sorted({d.strftime("%Y-%m") for d in cal})
for g in range(0, len(months), 6):                      # 每次扫描读 6 个月（批文件按股票分块，按月过滤需全扫）
    todo = [ym for ym in months[g:g + 6] if not (out_dir / f"{ym}.parquet").exists()]
    if not todo:
        continue
    lo = pd.Timestamp(todo[0] + "-01")
    hi = pd.Timestamp(todo[-1] + "-01") + pd.offsets.MonthBegin(1)
    t = dset.to_table(filter=(ds.field("as_of") >= lo) & (ds.field("as_of") < hi)).to_pandas()
    ym_of = t["as_of"].dt.strftime("%Y-%m")
    for ym in todo:
        part = t[ym_of == ym].set_index(["as_of", "ticker"]).sort_index()
        part.to_parquet(out_dir / f"{ym}.tmp")
        os.replace(out_dir / f"{ym}.tmp", out_dir / f"{ym}.parquet")
    print(f"  months {todo[0]}..{todo[-1]} ({time.time() - t0:.0f}s)", flush=True)
    del t
print(f"[a158 daily] done ({time.time() - t0:.0f}s)", flush=True)
