#!/usr/bin/env python
"""序列路径去险：1 fold GRU on Alpha360（GPU），fit+predict+方向，小 epoch。"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"   # mlflow3.13 弃用 filestore，显式 opt-in
import warnings, json, sys, time
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np
import qlib
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha360
from qlib.contrib.model.pytorch_gru import GRU

REPO = Path("/home/lenovo/projects/quantmind"); sys.path.insert(0, str(REPO))
qlib.init(provider_uri=str(REPO / "data/qlib_cn_daily"), region="cn",
          expression_cache=None, dataset_cache=None)
q2t = {k.upper(): v for k, v in json.loads((REPO/"data/bakeoff/ticker_qsym_map.json").read_text()).items()}
insts = sorted(q2t.keys())[:400]   # 子集提速
H = 12
class A360H(Alpha360):
    def get_label_config(self):
        return ([f"Ref($close,-{H})/$close-1"], ["LABEL0"])
t0 = time.time()
h = A360H(instruments=insts, start_time="2021-01-01", end_time="2022-09-30",
          fit_start_time="2021-01-01", fit_end_time="2022-03-31")
ds = DatasetH(h, segments={"train": ("2021-01-01", "2022-03-31"),
                           "valid": ("2022-04-01", "2022-05-31"),
                           "test": ("2022-06-01", "2022-09-30")})
print(f"[smoke] dataset built ({time.time()-t0:.0f}s)")
from qlib.workflow import R
m = GRU(d_feat=6, hidden_size=64, num_layers=2, dropout=0.3,
        n_epochs=3, lr=1e-3, early_stop=5, batch_size=800, GPU=0, seed=0)
with R.start(experiment_name="bakeoff_smoke", uri="file:///tmp/qlib_mlruns"):
    m.fit(ds)
    print(f"[smoke] fit done ({time.time()-t0:.0f}s)")
    pr = m.predict(ds, segment="test")
print("PRED_TYPE", type(pr).__name__, "SHAPE", getattr(pr, "shape", None),
      "INDEX", list(pr.index.names))
print(pr.head(3).to_string())
# 方向：val IC
pv = m.predict(ds, segment="valid")
lab = ds.prepare("valid", col_set="label").iloc[:, 0]
j = pd.concat([pv.rename("p"), lab.rename("y")], axis=1).dropna()
from scipy import stats
ic = j.groupby(level=0).apply(lambda g: stats.spearmanr(g["p"], g["y"])[0] if len(g) > 5 else np.nan)
print("val mean IC =", round(float(np.nanmean(ic)), 4), "-> dir", 1 if np.nanmean(ic) >= 0 else -1)
print("SEQ_SMOKE_OK", f"{time.time()-t0:.0f}s")
