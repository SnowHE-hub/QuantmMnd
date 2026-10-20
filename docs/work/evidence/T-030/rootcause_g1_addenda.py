"""T-030：按 G1 2026-10-19 收口裁定补报的两项描述性数字（不训练、不改任何判定）.

1. R2a 零分布校准的有限模拟 p 值：重算与 rootcause_r2a_null_calibration.py 完全相同的 500 个 T_j（同种子、同键），
   报告尾部计数 b 与 (b + 1) / (500 + 1)：双侧 |T_j| ≥ |T_obs|、上尾 T_j ≥ T_obs、下尾 T_j ≤ T_obs。
2. R3 跨种子相关的描述性敏感性：去掉种子 0 后（种子 1–15）重算 T 与各因子暴露、与真实模型重合度的相关。
输出：docs/work/evidence/T-030/p6/g1_addenda.json
用法：PYTHONPATH=. python docs/work/evidence/T-030/rootcause_g1_addenda.py <data_root>
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    root = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("cal", HERE / "rootcause_r2a_null_calibration.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    import pandas as pd
    from scipy.stats import rankdata

    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.folds import calendar_and_grid, make_folds

    r2 = json.loads((HERE / "p6/r2_synthetic_null.json").read_text(encoding="utf-8"))
    t_obs = float(r2["variants"]["R2a_synlabel"]["T"])
    c, g = calendar_and_grid(root)
    folds = make_folds(c, g, "63d")
    md = ModelData(root)
    pred_dir = root / "data/gpu_v1/preds/xgb_63d_R2a_synlabel"
    blocks = []
    for k, f in enumerate(folds):
        pb = md.load(f.pred_days, [0])
        saved = pd.read_parquet(pred_dir / f"fold_{k}.parquet")
        score = saved["score"].to_numpy(dtype=np.float64)
        y63 = pb.meta["y63"].to_numpy(dtype=np.float64)
        dpb = pb.meta["day"].to_numpy()
        test = pd.DatetimeIndex(pd.to_datetime(list(f.test_asof)))
        mask = (pb.meta["top1500"].to_numpy(dtype=bool) & pd.to_datetime(pb.meta["as_of"]).isin(test).to_numpy()
                & np.isfinite(y63) & np.isfinite(score))
        for d in np.unique(dpb[mask]):
            m = mask & (dpb == d)
            if m.sum() >= 30:
                blocks.append((k, np.flatnonzero(m), rankdata(score[m]), len(y63)))
    null_T = np.empty(cal.N_NULL)
    for j in range(cal.N_NULL):
        zf, ics = {}, []
        for k, idx, rs, n in blocks:
            if k not in zf:
                zf[k] = np.random.default_rng([cal.SEED, 100 + j, k]).standard_normal(n)
            ics.append(np.corrcoef(rs, rankdata(zf[k][idx]))[0, 1])
        null_T[j] = float(np.mean(ics))
    N = cal.N_NULL
    b2 = int(np.sum(np.abs(null_T) >= abs(t_obs)))
    bu = int(np.sum(null_T >= t_obs))
    bl = int(np.sum(null_T <= t_obs))
    r2a = {"t_obs": t_obs, "n_sim": N, "null_mean": float(null_T.mean()), "null_sd": float(null_T.std(ddof=1)),
           "two_sided": {"definition": "|T_j| >= |T_obs|", "b": b2, "p_plus1": (b2 + 1) / (N + 1), "p_raw": b2 / N},
           "upper": {"definition": "T_j >= T_obs", "b": bu, "p_plus1": (bu + 1) / (N + 1)},
           "lower": {"definition": "T_j <= T_obs", "b": bl, "p_plus1": (bl + 1) / (N + 1)},
           "noise_mechanism_match": ("评估噪声与原 R2a 评估标签同机制：逐行 iid 标准正态（numpy default_rng，按折取键；"
                                     "原键 (SEED, 5, k)，校准键 (SEED, 100 + j, k)）；缺失掩码与 y63 有限位置相同；"
                                     "1500 池与冻结测试日期掩码相同；日期等权；两者都没有时间依赖。")}
    r3 = json.loads((HERE / "p6/r3_factor_exposure.json").read_text(encoding="utf-8"))
    seeds = [s for s in r3["T_recomputed"] if s != "0"]
    tv = np.array([r3["T_recomputed"][s] for s in seeds])
    corr_wo0 = {f: float(np.corrcoef([r3["exposure_mean"][s][f] for s in seeds], tv)[0, 1])
                for f in r3["exposure_mean"]["1"]}
    out = {"what": __doc__.split("\n")[0], "r2a_finite_sim_p": r2a,
           "r3_corr_T_vs_exposure_seeds_1_15": corr_wo0,
           "r3_note": "探索性：只有 15 / 16 个种子；不作为放行条件，不据此删除特征或调整模型。"}
    (HERE / "p6/g1_addenda.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
