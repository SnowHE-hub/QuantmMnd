"""T-030 P6 通道 B（预登记 §9）：未来 h 日复权对数收益的 Q10 / Q50 / Q90，对两条朴素基线.

模型：XGBoost GPU reg:quantileerror（三个分位一个模型），深度 6、eta 0.05、500 轮、种子 0，不调参；每个外层折在全训练窗上训练一次，
     预测该折的周频测试 as_of。目标 = log(1 + forward_return_h)。
基线：RW = 0 收益、σ60（过去 60 个交易日 F6 r_cc 的标准差，≥ 40 个观测）× √h × z_q；
     HIST = 该股过去 252 个交易日内（终点 ≤ as_of）滚动 h 日对数收益的经验分位数（≥ 126 个观测）。
评估：1500 池、标签非缺失、三种方法都有值的行；pinball（每个分位）、[Q10, Q90] 覆盖率（名义 80%）、平均宽度；分年。
输出 data/gpu_v1/preds/quantile_<h>/fold_<k>.parquet 与 <out.json>。
用法：PYTHONPATH=. python run_quantile.py <data_root> <20d|63d> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantmind.research.gpu.data import ModelData
from quantmind.research.gpu.dl import SeqStore
from quantmind.research.gpu.folds import calendar_and_grid, make_folds
from quantmind.research.gpu.metrics import pinball
from quantmind.research.gpu.runtime import start_ram_guard, wait_for_vram
from quantmind.research.gpu.trees import TreeInput, dense_features, xgb_dmatrix

Q = np.array([0.1, 0.5, 0.9])
root, h, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
H = int(h[:-1])
ycol = "y20" if h == "20d" else "y63"
cal, grid = calendar_and_grid(root)
folds = make_folds(cal, grid, h)
md = ModelData(root)
seq = SeqStore(root)
rcc = seq.channels.index("r_cc")
pred_dir = root / "data/gpu_v1/preds" / f"quantile_{h}"
pred_dir.mkdir(parents=True, exist_ok=True)


def baselines(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rw = np.full((len(meta), 3), np.nan)
    hist = np.full((len(meta), 3), np.nan)
    z = norm.ppf(Q)
    g = seq.day_index(meta["as_of"].to_numpy())
    si = meta["seq_idx"].to_numpy()
    for d in np.unique(g):
        rows = np.flatnonzero(g == d)
        lo = max(0, d - 252 - H + 1)
        w = np.asarray(seq.arr[si[rows], lo:d + 1, rcc], dtype=np.float64)            # [n, T]，只到 as_of
        last60 = w[:, -60:]
        ok60 = np.isfinite(last60).sum(1) >= 40
        sd = np.nanstd(last60, axis=1)
        rw[rows] = np.where(ok60[:, None], sd[:, None] * np.sqrt(H) * z[None, :], np.nan)
        c = np.concatenate([np.zeros((len(rows), 1)), np.cumsum(np.nan_to_num(w), axis=1)], axis=1)
        hret = c[:, H:] - c[:, :-H]                                                   # 终点 ≤ as_of 的 h 日对数收益
        valid = np.isfinite(w).astype(int)
        cv = np.concatenate([np.zeros((len(rows), 1)), np.cumsum(valid, axis=1)], axis=1)
        full = (cv[:, H:] - cv[:, :-H]) == H
        hret = np.where(full, hret, np.nan)[:, -252:]
        okh = np.isfinite(hret).sum(1) >= 126
        hq = np.nanquantile(hret, Q, axis=1).T if hret.shape[1] else np.full((len(rows), 3), np.nan)
        hist[rows] = np.where(okh[:, None], hq, np.nan)
    return rw, hist


def main() -> None:
    import xgboost as xgb
    start_ram_guard()
    rows_all = []
    for k, f in enumerate(folds):
        fp = pred_dir / f"fold_{k}.parquet"
        if not fp.exists():
            wait_for_vram(4000)
            tr = md.load(f.train_days)
            y = np.log1p(tr.meta[ycol].to_numpy(dtype=np.float64))
            r = np.flatnonzero(np.isfinite(y))
            X = TreeInput(md, tr, r)
            d = xgb_dmatrix(X, y[r])
            m = xgb.train({"objective": "reg:quantileerror", "quantile_alpha": Q.tolist(), "device": "cuda",
                           "tree_method": "hist", "max_depth": 6, "eta": 0.05, "seed": 0, "verbosity": 0}, d, 500)
            del d, X, tr
            te = md.load(f.test_asof)
            p = m.inplace_predict(dense_features(md, te))
            df = te.meta[["as_of", "ticker", "top1500", "seq_idx"]].copy()
            df["y"] = np.log1p(te.meta[ycol].to_numpy(dtype=np.float64))
            df[["m10", "m50", "m90"]] = np.sort(p, axis=1)                    # 防分位交叉
            df.to_parquet(fp.with_suffix(".tmp"), index=False)
            os.replace(fp.with_suffix(".tmp"), fp)
            print(f"[quantile_{h}] fold {k} done", flush=True)
        rows_all.append(pd.read_parquet(fp))
    df = pd.concat(rows_all, ignore_index=True)
    df["as_of"] = pd.to_datetime(df["as_of"])
    rw, hist = baselines(df)
    df[["rw10", "rw50", "rw90"]] = rw
    df[["hi10", "hi50", "hi90"]] = hist
    ev = df[df["top1500"]].dropna()
    res = {"horizon": h, "n_rows": int(len(ev)), "n_dates": int(ev["as_of"].nunique()), "methods": {}}

    def summ(e: pd.DataFrame) -> dict:
        o = {}
        for name, pre in (("model", "m"), ("random_walk", "rw"), ("historical", "hi")):
            q = e[[f"{pre}10", f"{pre}50", f"{pre}90"]].to_numpy()
            yv = e["y"].to_numpy()
            o[name] = {"pinball": {str(qq): pinball(yv, q[:, i], qq) for i, qq in enumerate(Q)},
                       "coverage_10_90": float(((yv >= q[:, 0]) & (yv <= q[:, 2])).mean()),
                       "mean_width_10_90": float((q[:, 2] - q[:, 0]).mean())}
        return o

    res["methods"] = summ(ev)
    res["by_year"] = {str(y): summ(g) for y, g in ev.groupby(ev["as_of"].dt.year)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["methods"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
