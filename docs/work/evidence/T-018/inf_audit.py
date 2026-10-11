"""T-018：特征里的 ±inf 审计（只读）。用 train_line_b_v6_1.build 构造与训练完全相同的特征面板.

报告每个含 inf 的特征：inf 行数、涉及截面数、按板块分布、最早出现的截面。
用法：PYTHONPATH=. python inf_audit.py <data_root> <seed> <out.json>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.backtest.wf_costs import board_of

root, seed, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("trainmod", Path(__file__).with_name("train_line_b_v6_1.py"))
mod = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(mod)
panel, feats, _ = mod.build(root, seed)
num = [f for f in feats if f not in ("exposure_industry", "exposure_area")]
X = panel[num]
isinf = np.isinf(X.to_numpy(dtype=float))
cols = [c for c, n in zip(num, isinf.sum(axis=0), strict=True) if n > 0]
boards = pd.Series([board_of(t) for t in panel.index.get_level_values("ticker")], index=panel.index)
res = {"seed": seed, "n_rows": int(len(panel)), "n_features": len(num), "features_with_inf": {}}
for c in cols:
    m = np.isinf(X[c].to_numpy(dtype=float))
    sub = panel.index[m]
    res["features_with_inf"][c] = {
        "n_inf": int(m.sum()), "n_as_of": int(sub.get_level_values("as_of").nunique()),
        "first_as_of": str(sub.get_level_values("as_of").min().date()),
        "by_board": boards[m].value_counts().to_dict(),
        "sign": {"pos": int((X[c].to_numpy(dtype=float)[m] > 0).sum()), "neg": int((X[c].to_numpy(dtype=float)[m] < 0).sum())}}
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])
