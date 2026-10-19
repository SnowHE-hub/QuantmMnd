"""T-030 20d 主比较的事后敏感性（只报告，不改变已登记的判定"检测到增益"；本文件提交早于运行）.

起因：R2a 零分布校准中，移动块自助的 95% 百分位区间半宽 0.0033，比精确零分布（标准差 0.00193 → 半宽约 0.0038）窄约 12%。
20d 主比较的 97.5% 下界（块长 13：0.00668；块长 26：0.00722）相对半宽的余量约 14%，因此需要看换一种区间构造时结论是否仍成立。
做法：完全按 eval_primary.py 的口径重建逐日配对差 d_t（ENS 四族等权截面 rank 平均 − Ridge C，同一批股票与日期），
  1) 先用同一移动块自助（种子 0、10,000 次）复现已登记的均值与区间，作为重建正确性的核对；
  2) Newey–West（Bartlett 核）标准误，滞后 4 / 8 / 13 / 26 个周频期，正态 97.5% 双侧（z = 2.2414）；
  3) 以外层折为单位的 t 区间（19 折，df 18，97.5% 双侧）。
输出：docs/work/evidence/T-030/p6/sens_primary_20d.json
用法：PYTHONPATH=. python docs/work/evidence/T-030/sens_primary_20d.py <data_root>
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def newey_west_se(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    v = x @ x / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k] / n)
    return float(np.sqrt(v / n))


def main() -> None:
    root = Path(sys.argv[1])
    ep = _load("eval_primary")
    from quantmind.research.gpu.ensemble import rank_average_frames
    from quantmind.research.gpu.folds import calendar_and_grid, make_folds
    from quantmind.research.gpu.metrics import moving_block_bootstrap, two_sided_ci

    h, D = "20d", root / "data"
    cal, grid = calendar_and_grid(root)
    folds = make_folds(cal, grid, h)
    test = sorted({d for f in folds for d in f.test_asof})
    meta = pd.concat([pd.read_parquet(p, columns=["as_of", "ticker", "y20", "top1500"])
                      for p in sorted((D / "gpu_v1/model/v1").glob("*.meta.parquet"))], ignore_index=True)
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    meta = meta[meta["as_of"].isin(set(test))].set_index(["as_of", "ticker"]).rename(columns={"y20": "y"})
    ridge = pd.read_parquet(D / ep.BASE[h])
    ridge["as_of"] = pd.to_datetime(ridge["as_of"])
    ridge = ridge.set_index(["as_of", "ticker"])["score"]
    fams = {m: ep.load_family(root, m, h).reindex(meta.index) for m in ep.ENS_MEMBERS}
    ens = rank_average_frames(fams).reindex(meta.index)
    pool = meta[meta["top1500"]]
    both = pool.assign(e=ens.reindex(pool.index), r=ridge.reindex(pool.index)).dropna()
    d = (ep.ic_by_date(both, "e") - ep.ic_by_date(both, "r")).dropna().sort_index()
    x = d.to_numpy()
    rec = json.loads((HERE / "p6/primary_20d.json").read_text(encoding="utf-8"))["primary"]
    out = {"what": __doc__.split("\n")[0], "n_dates": int(len(x)), "n_rows": int(len(both)),
           "mean_diff": float(x.mean()), "recorded_mean_diff": rec["mean_diff"], "recorded_n_rows": rec["n_rows"]}
    out["mbb_reproduced"] = {}
    for blk in (13, 26):
        bt = moving_block_bootstrap(x, blk, 10_000, 0)
        ci = two_sided_ci(bt, 0.975)
        out["mbb_reproduced"][f"block{blk}"] = {"ci97_5": ci, "recorded": rec[f"block{blk}"]["ci97_5"],
                                                "matches": bool(np.allclose(ci, rec[f"block{blk}"]["ci97_5"]))}
    z = float(stats.norm.ppf(1 - 0.025 / 2))
    out["newey_west"] = {}
    for lag in (4, 8, 13, 26):
        se = newey_west_se(x, lag)
        out["newey_west"][f"lag{lag}"] = {"se": se, "ci97_5": [float(x.mean() - z * se), float(x.mean() + z * se)]}
    asof = d.index
    fold_means = []
    for f in folds:
        sel = asof.isin(pd.DatetimeIndex(pd.to_datetime(list(f.test_asof))))
        if sel.any():
            fold_means.append(float(x[sel].mean()))
    fm = np.array(fold_means)
    se_f = float(fm.std(ddof=1) / np.sqrt(len(fm)))
    tcrit = float(stats.t.ppf(1 - 0.025 / 2, len(fm) - 1))
    out["fold_t"] = {"n_folds": int(len(fm)), "fold_means": fold_means, "mean_of_fold_means": float(fm.mean()),
                     "se": se_f, "tcrit": tcrit,
                     "ci97_5": [float(fm.mean() - tcrit * se_f), float(fm.mean() + tcrit * se_f)]}
    out["iid_se"] = float(x.std(ddof=1) / np.sqrt(len(x)))
    out["note"] = "事后敏感性，只报告；已登记的判定以 primary_20d.json 为准，不因本文件改变。"
    (HERE / "p6/sens_primary_20d.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
