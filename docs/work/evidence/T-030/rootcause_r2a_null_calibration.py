"""T-030 根因清单 R2a 追查：评估链路的零分布校准（事先写定，本文件提交早于运行）.

R2a（真实特征、合成标签）的 T = +0.00345，块长 13 的 95% 区间 [+0.00037, +0.00703]，按事先规则判"链路异常，需排查"。
评估标签 Z 与模型、特征完全独立，理论上 E[T] = 0、标准误约 1/√(n−1)/√189 ≈ 0.0019（n ≈ 1500）。
本脚本不重训：保持 R2a 已写出的 9 折预测不变，换 500 组独立评估噪声（键 (SEED, 100 + j, k)，与原评估噪声 (SEED, 5, k)
不同），用与 R2 相同的评估代码各算一次 T，得到评估链路给出的零分布。
判读（事先写定）：零分布均值在 ±0.0005 内、标准差在 [0.0015, 0.0025] 内，且观测 T 的双侧经验 p > 0.01 →
  "评估链路符合理论零分布，R2a 的 T 属于抽样波动"；否则 → "评估链路异常，继续定位"。
R2a 原判读（"链路异常，需排查"）照原样保留在 r2_synthetic_null.json，本结果另存，不覆盖。
输出：docs/work/evidence/T-030/p6/r2a_null_calibration.json
用法：PYTHONPATH=. python docs/work/evidence/T-030/rootcause_r2a_null_calibration.py <data_root>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
SEED, N_NULL = 20261019, 500


def main() -> None:
    root = Path(sys.argv[1])
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.folds import calendar_and_grid, make_folds
    from quantmind.research.gpu.metrics import daily_rank_ic

    r2 = json.loads((HERE / "p6/r2_synthetic_null.json").read_text(encoding="utf-8"))
    t_obs = float(r2["variants"]["R2a_synlabel"]["T"])
    cal, wgrid = calendar_and_grid(root)
    folds = make_folds(cal, wgrid, "63d")
    md = ModelData(root)
    pred_dir = root / "data/gpu_v1/preds/xgb_63d_R2a_synlabel"
    blocks = []                                        # 每折：按日的 (score 秩, 有效行掩码)
    t_check_parts = []
    for k, f in enumerate(folds):
        pb = md.load(f.pred_days, [0])                 # 只要 meta；取一列特征减少内存
        saved = pd.read_parquet(pred_dir / f"fold_{k}.parquet")
        if len(saved) != len(pb.meta) or not (saved["ticker"].to_numpy() == pb.meta["ticker"].to_numpy()).all():
            raise RuntimeError(f"fold {k}: saved rows do not match the prediction block")
        score = saved["score"].to_numpy(dtype=np.float64)
        y63 = pb.meta["y63"].to_numpy(dtype=np.float64)
        dpb = pb.meta["day"].to_numpy()
        test = pd.DatetimeIndex(pd.to_datetime(list(f.test_asof)))
        mask = (pb.meta["top1500"].to_numpy(dtype=bool) & pd.to_datetime(pb.meta["as_of"]).isin(test).to_numpy()
                & np.isfinite(y63) & np.isfinite(score))
        # 复核：用原评估噪声重算，应等于 R2 记录的 R2a 逐日 IC
        z0 = np.where(np.isfinite(y63), np.random.default_rng([SEED, 5, k]).standard_normal(len(y63)), np.nan)
        t_check_parts.append(daily_rank_ic(score, z0, dpb, pb.meta["top1500"].to_numpy(dtype=bool)
                                           & pd.to_datetime(pb.meta["as_of"]).isin(test).to_numpy()))
        for d in np.unique(dpb[mask]):
            m = mask & (dpb == d)
            if m.sum() >= 30:
                blocks.append((k, np.flatnonzero(m), rankdata(score[m]), len(y63)))
        print(f"fold {k}: {sum(1 for b in blocks if b[0] == k)} dates", flush=True)
    t_check = float(np.concatenate([a for a in t_check_parts if len(a)]).mean())
    null_T = np.empty(N_NULL)
    for j in range(N_NULL):
        z_by_fold = {}
        ics = []
        for k, idx, rs, n in blocks:
            if k not in z_by_fold:
                z_by_fold[k] = np.random.default_rng([SEED, 100 + j, k]).standard_normal(n)
            rz = rankdata(z_by_fold[k][idx])
            ics.append(np.corrcoef(rs, rz)[0, 1])
        null_T[j] = float(np.mean(ics))
    mean, sd = float(null_T.mean()), float(null_T.std(ddof=1))
    p_two = float(np.mean(np.abs(null_T) >= abs(t_obs)))
    ok = abs(mean) <= 0.0005 and 0.0015 <= sd <= 0.0025 and p_two > 0.01
    out = {"what": __doc__.split("\n")[0], "n_null": N_NULL, "n_dates": len(blocks),
           "t_obs_from_r2": t_obs, "t_recomputed_with_original_eval_noise": t_check,
           "null_mean": mean, "null_sd": sd, "theory_sd_approx": float(np.mean([1 / np.sqrt(len(b[1]) - 1) for b in blocks])
                                                                       / np.sqrt(len(blocks))),
           "empirical_p_two_sided": p_two,
           "reading": "评估链路符合理论零分布，R2a 的 T 属于抽样波动" if ok else "评估链路异常，继续定位"}
    (HERE / "p6/r2a_null_calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
