"""T-055：用冻结日（2026-10-09）之前的全部研究池样本拟合 HAR 系数，供前瞻观察臂 W 使用。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from x3_vol import HAR, build  # noqa: E402

from quantmind.research.gpu.folds import calendar_and_grid  # noqa: E402

root = Path(sys.argv[1])
cal, grid = calendar_and_grid(root)
df = build(root, list(grid))
tr = df[df["pool"] & (df.index.get_level_values("as_of") <= pd.Timestamp("2026-09-10"))].dropna(subset=["y", *HAR])
X = np.c_[np.ones(len(tr)), tr[HAR].to_numpy()]
beta = np.linalg.lstsq(X, tr["y"].to_numpy(), rcond=None)[0]
out = {"features": ["const", *HAR], "beta": beta.tolist(), "n": int(len(tr)), "fitted_through_as_of": "2026-09-10",
       "target": "log 年化已实现波动（t+1…t+20）", "note": "T-055 观察臂 W 使用；特征为过去 5/20/60 日日对数收益标准差 × √252 的对数"}
p = Path(sys.argv[2])
p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(out)
