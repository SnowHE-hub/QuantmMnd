"""T-030 63d 根因清单 R3：已有 16 个打乱种子预测的因子暴露分解（只读已有预测，不训练、不追加种子，描述性）.

动机：R1 已逐位复现种子 0 的 T = +0.0238（两条代码路径、模型文件、逐股预测、逐日 IC 全部相同），而种子 1–15 均值 −0.0019。
用打乱标签训练的模型相当于特征的一个随机函数；若这个随机函数碰巧偏向某个本身有效的因子，就会在各折持续得到正 IC。
本脚本在冻结的 189 个可评估日期、1500 池上，对每个种子计算：
  - 分数与若干强因子（当日 gauss-rank）的日均截面 Spearman 相关（"暴露"）；
  - 分数与用真实标签训练的 XGBoost 63d 分数的日均截面相关（与真实信号方向的重合）；
  - 每个因子自身对 y63 的日均 RankIC。
不进入 N1-R，不改变任何判定；只用于判断种子 0 的异常是否来自"随机函数碰巧对齐了有效因子"。
输出：docs/work/evidence/T-030/p6/r3_factor_exposure.json
用法：PYTHONPATH=. python docs/work/evidence/T-030/rootcause_r3_factor_exposure.py <data_root>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
FACTORS = ["sh_resid_rev_10", "sh_resid_vol_10", "sh_rskew_21", "limit_up_cnt_20d", "log_circ_market_cap",
           "volatility_3m", "momentum_1m", "reversal_1w", "amihud_illiquidity", "distance_to_52w_high",
           "momentum_6m", "book_to_market", "earnings_yield"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 30:
        return np.nan
    return float(np.corrcoef(rankdata(a[ok]), rankdata(b[ok]))[0, 1])


def _load_scores(path: Path, key: pd.MultiIndex) -> np.ndarray:
    p = pd.read_parquet(path)
    p["as_of"] = pd.to_datetime(p["as_of"])
    return p.set_index(["as_of", "ticker"])["score"].reindex(key).to_numpy(dtype=np.float64)


def main() -> None:
    root = Path(sys.argv[1])
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.folds import calendar_and_grid, make_folds

    cal, wgrid = calendar_and_grid(root)
    folds = make_folds(cal, wgrid, "63d")
    md = ModelData(root)
    pos = {c: i for i, c in enumerate(md.stock_features)}
    cols = [pos[c] for c in FACTORS]
    preds = root / "data/gpu_v1/preds"
    seed_dirs = {0: "xgb_63d_N1shuffle", **{s: f"xgb_63d_N1s{s:02d}" for s in range(1, 16)}}
    names = list(seed_dirs) + ["real_xgb"]
    expo = {s: {f: [] for f in FACTORS + ["real_xgb_score"]} for s in seed_dirs}
    T = {s: [] for s in seed_dirs}
    fac_ic = {f: [] for f in FACTORS}
    for k, f in enumerate(folds):
        b = md.load(f.test_asof, cols)
        meta = b.meta
        y = meta["y63"].to_numpy(dtype=np.float64)
        if not np.isfinite(y).any():
            continue
        key = pd.MultiIndex.from_arrays([pd.to_datetime(meta["as_of"]), meta["ticker"]])
        scores = {s: _load_scores(preds / d / f"fold_{k}.parquet", key) for s, d in seed_dirs.items()}
        real = _load_scores(preds / "xgb_63d" / f"fold_{k}.parquet", key)
        X = b.X.astype(np.float64)
        top = meta["top1500"].to_numpy(dtype=bool)
        day = meta["day"].to_numpy()
        for d in np.unique(day):
            m = (day == d) & top & np.isfinite(y)
            if m.sum() < 30:
                continue
            for j, fname in enumerate(FACTORS):
                fac_ic[fname].append(_spearman(X[m, j], y[m]))
            for s, sc in scores.items():
                T[s].append(_spearman(sc[m], y[m]))
                for j, fname in enumerate(FACTORS):
                    expo[s][fname].append(_spearman(sc[m], X[m, j]))
                expo[s]["real_xgb_score"].append(_spearman(sc[m], real[m]))
        print(f"fold {k} done", flush=True)
    out = {"what": __doc__.split("\n")[0], "n_dates": len(T[0]), "names": names,
           "T_recomputed": {s: float(np.nanmean(v)) for s, v in T.items()},
           "factor_ic_vs_y63": {f: float(np.nanmean(v)) for f, v in fac_ic.items()},
           "exposure_mean": {s: {f: float(np.nanmean(v)) for f, v in e.items()} for s, e in expo.items()}}
    others = [s for s in seed_dirs if s != 0]
    z = {}
    for fname in FACTORS + ["real_xgb_score"]:
        vals = np.array([out["exposure_mean"][s][fname] for s in others])
        sd = vals.std(ddof=1)
        z[fname] = {"seed0": out["exposure_mean"][0][fname], "others_mean": float(vals.mean()),
                    "others_sd": float(sd), "z_seed0": float((out["exposure_mean"][0][fname] - vals.mean()) / sd)
                    if sd > 0 else None}
    out["seed0_vs_others"] = z
    exp_matrix = np.array([[out["exposure_mean"][s][f] for f in FACTORS] for s in seed_dirs])
    tvec = np.array([out["T_recomputed"][s] for s in seed_dirs])
    out["corr_T_vs_exposure_across_16_seeds"] = {f: float(np.corrcoef(exp_matrix[:, j], tvec)[0, 1])
                                                for j, f in enumerate(FACTORS)}
    out["corr_T_vs_real_xgb_alignment_across_16_seeds"] = float(np.corrcoef(
        [out["exposure_mean"][s]["real_xgb_score"] for s in seed_dirs], tvec)[0, 1])
    (HERE / "p6/r3_factor_exposure.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"T0": out["T_recomputed"][0], "seed0_vs_others": z,
                      "corr_T_vs_real_alignment": out["corr_T_vs_real_xgb_alignment_across_16_seeds"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
