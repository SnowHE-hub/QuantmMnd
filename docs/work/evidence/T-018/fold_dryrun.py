"""T-018 切折演练（不训练）：v6_1 网格 + 完整日历，打印各方案的折，检查覆盖与隔离。"""
import json
import sys

import pandas as pd

sys.path.insert(0, "/home/lenovo/projects/qm-t018")
sys.argv = ["x"]
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("tr", "/home/lenovo/projects/qm-t018/docs/work/evidence/T-018/train_line_b_v6_1.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
from quantmind.backtest.wf_split import PurgedWalkForwardSplit  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path("/home/lenovo/projects/quantmind")
cal = tr.calendar(root)
grid = [pd.Timestamp(x) for x in json.load(open("/home/lenovo/projects/qm-t018/docs/work/evidence/T-018/asof_grid_v6_1.json"))["as_of"]]
for seed in ("12d", "63d"):
    cfg = tr.SEEDS[seed]
    sp = PurgedWalkForwardSplit(cal, horizon=cfg["H"], embargo=cfg["E"], mode="rolling", rolling_lookback_td=756, n_val=2)
    for proto in ("R", "X", "C"):
        cuts = cfg["cuts"](cal, cfg["end_R"] if proto == "R" else cfg["end_X"])
        folds = tr.make_folds(sp, grid, cuts, proto)
        test = sorted({pd.Timestamp(d) for f in folds for d in f.test_dates})
        gaps = [(f.cutoff.date(), sp.idx(min(f.test_dates)) - sp.idx(f.cutoff)) for f in folds if f.test_dates]
        min_gap = min(g for _, g in gaps)
        trn_gap = min(sp.idx(f.cutoff) - sp.idx(max(f.train_dates)) for f in folds if f.train_dates)
        cover = len(test) / len([g for g in grid if test and test[0] <= g <= test[-1]])
        print(f"{seed} {proto}: cuts {len(cuts)} {cuts[0].date()}→{cuts[-1].date()}; test {test[0].date()}→{test[-1].date()} "
              f"n={len(test)} coverage_in_span={cover:.2f}; min(test−cut)={min_gap} td (E={cfg['E']}); "
              f"min(cut−last train)={trn_gap} td (H={cfg['H']})")
