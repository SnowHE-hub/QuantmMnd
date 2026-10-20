"""fnd 反事实重算：用冻结时刻的财报表（_backup_20261019，mtime 10-11）在窗口日上重跑冻结的 one_day，应逐值等于冻结面板；
用当前表重跑应逐值等于影子面板。"""
import sys, shutil, tempfile, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/home/lenovo/projects/qm-t035"); sys.path.insert(0, "/home/lenovo/projects/qm-t035/docs/work/evidence/T-030")
import build_daily_fnd as B
REAL = Path("/home/lenovo/projects/quantmind"); SH = Path("/home/lenovo/qm_fwd_root/20261020/main")
days = [pd.Timestamp(d) for d in ("2026-09-01", "2026-09-15", "2026-09-30", "2026-10-09")]
def eq(a, b):
    idx = a.index.union(b.index)
    a, b = a.reindex(idx), b.reindex(idx)[a.columns]
    x, y = a.to_numpy(float), b.to_numpy(float)
    m = ~((x == y) | (np.isnan(x) & np.isnan(y)))
    return int(m.sum()), len(a.index.difference(b.index)) if False else None
def frozen(root, d):
    df = pd.read_parquet(root / "data/gpu_v1/panel/fnd" / f"{d:%Y-%m}.parquet")
    return df.xs(d, level="as_of")
# 冻结时刻的输入：临时数据根，lake 里财报表链接到备份，其余链接到真实
tmp = Path(tempfile.mkdtemp(prefix="fndcf_", dir="/home/lenovo/t030_scratch"))
(tmp / "data/lake").mkdir(parents=True); (tmp / "data/raw").mkdir(parents=True)
bk = REAL / "data/lake/_backup_20261019"
for f in ("income", "balancesheet", "cashflow", "fina_indicator_full"):
    os.symlink(bk / f"{f}.parquet", tmp / f"data/lake/{f}.parquet")
for f in ("daily_basic.parquet", "trade_cal_sse.parquet"):
    os.symlink(REAL / "data/lake" / f, tmp / "data/lake" / f)
os.symlink(REAL / "data/raw/alpha_prices_panel_v6_20261009.parquet", tmp / "data/raw/alpha_prices_panel_v6_20261009.parquet")
for label, root, ref in (("frozen_inputs", tmp, REAL), ("current_inputs", SH, SH)):
    B.G.clear(); B.setup(root)
    for d in days:
        r = B.one_day(d); r["ticker"] = r["ticker"].astype(str); r = r.drop(columns="as_of").set_index("ticker").sort_index()
        f = frozen(ref, d).sort_index()
        idx = f.index.intersection(r.index)
        x, y = f.loc[idx].to_numpy(float), r.loc[idx, f.columns].to_numpy(float)
        m = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        print(label, d.date(), "rows ref", len(f), "recomputed", len(r), "only_ref", len(f.index.difference(r.index)),
              "only_recomp", len(r.index.difference(f.index)), "cells_diff", int(m.sum()), flush=True)
shutil.rmtree(tmp)
