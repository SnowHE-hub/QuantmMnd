"""T-036 X0-b：新打分模块在**冻结输入**上复现第 18 折 ENS 20d（2026-10-21 写定；在 X0 规则 2 因"干净日 = 0"无法检验之后补充）.

做法：quantmind.line_a.scoring.score_ens20 直接读冻结数据根（输入与研究期完全相同），在 X0 的同一窗口
（第 18 折预测日 ∩ [2026-09-01, 2026-10-09]）打分。
判定（运行前写定）：
1. 每族每种子的原始预测与冻结的 runs/<族>_20d/k18_s*.npy 逐行最大差 = 0；
2. ENS 与由冻结折预测（preds/<族>_20d/fold_18.parquet）合成的 ENS 逐行最大差 = 0。
它检验的是新代码路径（模型包加载、归一化、推理、种子与族的排名平均）本身，不涉及前瞻数据。
用法：python x0b_ens20_frozen_inputs.py <data_root> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
W0, W1 = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-09")
TAGS = {"xgb": "xgb_20d", "cat": "cat_20d", "tabm": "tabm_20d", "seqfusion": "seqfusion_20d"}


def main() -> None:
    from quantmind.line_a.scoring import FAMS, score_ens20
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.ensemble import rank_average_frames
    from quantmind.research.gpu.folds import calendar_and_grid, make_folds

    real, out = Path(sys.argv[1]), Path(sys.argv[2])
    cal, grid = calendar_and_grid(real)
    f = make_folds(cal, grid, "20d")[18]
    days = [d for d in f.pred_days if W0 <= pd.Timestamp(d) <= W1]
    new, raw = score_ens20(real, days, real / "data/gpu_v1/forward_models/20d_fold18")
    pb = ModelData(real).load(f.pred_days)
    kf = pd.MultiIndex.from_arrays([pd.to_datetime(pb.meta["as_of"]), pb.meta["ticker"]], names=["as_of", "ticker"])
    ks = new.index
    res: dict = {"days": len(days), "rows": int(len(ks)), "families": {}}
    for fm in FAMS:
        diffs = []
        for s, p in enumerate(raw[fm]):
            old = pd.Series(np.load(real / "data/gpu_v1/runs" / TAGS[fm] / f"k18_s{s}.npy"), index=kf).reindex(ks).to_numpy()
            d = np.abs(p.astype(np.float64) - old)
            diffs.append(float(np.nanmax(d)))
        res["families"][fm] = {"max_abs_diff_by_seed": diffs, "pass": all(x == 0 for x in diffs)}
    parts = {}
    for fm in FAMS:
        q = pd.read_parquet(real / "data/gpu_v1/preds" / f"{fm}_20d" / "fold_18.parquet")
        q["as_of"] = pd.to_datetime(q["as_of"])
        parts[fm] = q.set_index(["as_of", "ticker"])["score"].reindex(ks)
    old_ens = rank_average_frames(parts)
    d = (new["ENS"] - old_ens).abs()
    res["ens"] = {"max_abs_diff": float(d.max()), "rows_nan_mismatch": int((new["ENS"].isna() != old_ens.isna()).sum())}
    res["ens"]["pass"] = res["ens"]["max_abs_diff"] == 0 and res["ens"]["rows_nan_mismatch"] == 0
    res["pass"] = res["ens"]["pass"] and all(v["pass"] for v in res["families"].values())
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
