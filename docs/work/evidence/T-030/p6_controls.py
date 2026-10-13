"""T-030 P6：负对照（预登记 §7）。任何一项不通过 → 停下报告，不出主比较结果.

N1 标签日内打乱：读 run_trees.py --shuffle-labels 的折预测（preds/xgb_<h>_N1shuffle），周频测试 as_of、1500 池日均 RankIC，
   95% 区间（块长 13）含 0 为通过。
N2 未来特征金丝雀：折 0 内层训练日，特征加"t+1 日对数收益（F6 r_cc 通道）的当日 gauss-rank"，XGBoost 网格 c0 训练 500 轮；
   该列 gain 重要性第 1 为通过；并断言正式特征表里没有该列。
N3 日期错位：XGBoost 族各折的 5 个种子模型，对周频测试 as_of 的每只股票改用 t−1 日的特征（股票级与市场级都取 t−1）打分；
   1500 池日均 RankIC 低于同模型对齐特征的值为通过。
用法：PYTHONPATH=. python p6_controls.py <data_root> <20d|63d> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.research.gpu.configs import TREE_GRIDS
from quantmind.research.gpu.data import ModelData
from quantmind.research.gpu.dl import SeqStore
from quantmind.research.gpu.ensemble import rank_average
from quantmind.research.gpu.folds import calendar_and_grid, make_folds
from quantmind.research.gpu.metrics import daily_rank_ic, moving_block_bootstrap, two_sided_ci
from quantmind.research.gpu.runtime import start_ram_guard
from quantmind.research.gpu.transforms import gauss_rank_by_day
from quantmind.research.gpu.trees import XGB, TreeInput, dense_features, train_target

root, h, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
start_ram_guard()
ycol = "y20" if h == "20d" else "y63"
cal, grid = calendar_and_grid(root)
folds = make_folds(cal, grid, h)
md = ModelData(root)
res: dict = {"horizon": h}


def pool_ic(block, score) -> np.ndarray:
    return daily_rank_ic(np.asarray(score, dtype=np.float64), block.meta[ycol].to_numpy(dtype=np.float64),
                         block.meta["day"].to_numpy(), block.meta["top1500"].to_numpy())


# ── N1 ───────────────────────────────────────────────────────────────────
d1 = root / "data/gpu_v1/preds" / f"xgb_{h}_N1shuffle"
if len(list(d1.glob("fold_*.parquet"))) == len(folds):
    ics = []
    for k, f in enumerate(folds):
        p = pd.read_parquet(d1 / f"fold_{k}.parquet")
        p["as_of"] = pd.to_datetime(p["as_of"])
        b = md.load(f.test_asof)
        s = p.set_index(["as_of", "ticker"])["score"].reindex(pd.MultiIndex.from_frame(b.meta[["as_of", "ticker"]]))
        ics.append(pool_ic(b, s.to_numpy()))
    x = np.concatenate(ics)
    ci = two_sided_ci(moving_block_bootstrap(x, 13), 0.95)
    res["N1"] = {"ic_mean": float(x.mean()), "ci95_block13": ci, "n_dates": int(len(x)), "pass": bool(ci[0] <= 0 <= ci[1])}
else:
    res["N1"] = {"pass": None, "note": f"N1 predictions incomplete in {d1}"}

# ── N2 ───────────────────────────────────────────────────────────────────
assert not any("canary" in c or "fwd" in c for c in md.stock_features + md.market_features)
seq = SeqStore(root)
f0 = folds[0]
tr = md.load(f0.inner_train_days)
g = seq.day_index(tr.meta["as_of"].to_numpy())
r_next = seq.arr[tr.meta["seq_idx"].to_numpy(), np.minimum(g + 1, seq.arr.shape[1] - 1), seq.channels.index("r_cc")]
canary = gauss_rank_by_day(r_next.astype(np.float64), tr.meta["day"].to_numpy())
rows, y = train_target(tr, h, "regression")
X = TreeInput(md, tr, rows, extra=canary[:, None].astype(np.float32))
model = XGB().fit(TREE_GRIDS["xgb"][0], X, y, tr.meta["day"].to_numpy()[rows], 500, seed=0)
gain = model.get_score(importance_type="gain")
names = md.stock_features + md.market_features + ["CANARY_next_day_return"]
rank = sorted(gain, key=gain.get, reverse=True)
top = [names[int(k[1:])] for k in rank[:5]]
res["N2"] = {"top5_gain": top, "pass": top[0] == "CANARY_next_day_return"}
del X, model, tr

# ── N3 ───────────────────────────────────────────────────────────────────
import xgboost as xgb  # noqa: E402

d3 = root / "data/gpu_v1/runs" / f"xgb_{h}"
if all((d3 / f"k{k:02d}_s0.ubj").exists() for k in range(len(folds))):
    pos = {d: i for i, d in enumerate(cal)}
    al, st = [], []
    for k, f in enumerate(folds):
        b = md.load(f.test_asof)
        prev = md.load([cal[pos[d] - 1] for d in f.test_asof])
        key_now = pd.MultiIndex.from_arrays([b.meta["as_of"].map(lambda d: cal[pos[d] - 1]), b.meta["ticker"]])
        idx_prev = pd.Series(np.arange(len(prev.meta)),
                             index=pd.MultiIndex.from_frame(prev.meta[["as_of", "ticker"]])).reindex(key_now).to_numpy()
        Xa = dense_features(md, b)
        Xp_all = dense_features(md, prev)
        Xs = np.full_like(Xa, np.nan)
        ok = np.isfinite(idx_prev)
        Xs[ok] = Xp_all[idx_prev[ok].astype(int)]
        pa, ps = [], []
        for s in range(5):
            m = xgb.Booster()
            m.load_model(str(d3 / f"k{k:02d}_s{s}.ubj"))
            m.set_param({"device": "cuda"})
            pa.append(m.inplace_predict(Xa))
            ps.append(m.inplace_predict(Xs))
        day = b.meta["day"].to_numpy()
        al.append(pool_ic(b, rank_average(pa, day)))
        st.append(pool_ic(b, rank_average(ps, day)))
    a_, s_ = np.concatenate(al), np.concatenate(st)
    res["N3"] = {"ic_aligned": float(a_.mean()), "ic_stale_t_minus_1": float(s_.mean()), "diff": float(s_.mean() - a_.mean()),
                 "pass": bool(s_.mean() < a_.mean())}
else:
    res["N3"] = {"pass": None, "note": f"xgb models missing in {d3}"}

res["all_pass"] = all(res[k].get("pass") is True for k in ("N1", "N2", "N3"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
