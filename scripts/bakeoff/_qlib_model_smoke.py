#!/usr/bin/env python
"""P3 API 去险：qlib Linear-on-Alpha158 单段 train/predict 跑通（自定义 12d label）。"""
import warnings, json; warnings.filterwarnings("ignore")
from pathlib import Path
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.linear import LinearModel

REPO = Path("/home/lenovo/projects/quantmind")
qlib.init(provider_uri=str(REPO / "data/qlib_cn_daily"), region="cn",
          expression_cache=None, dataset_cache=None)

all_txt = REPO / "data" / "qlib_cn_daily" / "instruments" / "all.txt"
insts = sorted({ln.split("\t")[0].strip().upper() for ln in all_txt.read_text().splitlines() if ln.strip()})[:300]


class H12(Alpha158):
    def get_label_config(self):
        return (["Ref($close,-12)/$close-1"], ["LABEL0"])


h = H12(instruments=insts, start_time="2020-01-01", end_time="2023-12-31",
        fit_start_time="2020-01-01", fit_end_time="2021-12-31",
        infer_processors=[{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                          {"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
        learn_processors=[{"class": "DropnaLabel"},
                          {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}])
ds = DatasetH(h, segments={"train": ("2020-01-01", "2021-12-31"),
                           "valid": ("2022-01-01", "2022-06-30"),
                           "test": ("2022-07-01", "2023-12-31")})
m = LinearModel(estimator="ridge", alpha=1.0)
m.fit(ds)
pred = m.predict(ds, segment="test")
print("PRED_TYPE", type(pred).__name__)
print("PRED_SHAPE", getattr(pred, "shape", None))
print("PRED_INDEX_NAMES", list(pred.index.names) if hasattr(pred, "index") else None)
print(pred.head(5).to_string())
print("SMOKE_OK")
