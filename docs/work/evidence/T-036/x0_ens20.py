"""T-036 X0：新管线（影子数据根 + 冻结模型包 + quantmind.line_a.scoring）在重叠期复现第 18 折的 ENS 20d（规则见 T-036.md）.

范围：第 18 折预测日 ∩ [2026-09-01, 2026-10-09]（这两个月在影子根里是重算的；更早的月份是冻结链接，
其 meta.seq_idx 按旧序列下标编排，不能与重算的序列数组混用，所以不在 X0 范围内）。
判定（运行前写定）：
1. 输入未变的行：每族每种子的原始预测，树模型与冻结的 runs/<族>_20d/k18_s*.npy 最大差 = 0；
   深度模型最大差 ≤ 1e-3 且相关 ≥ 0.9999（与 T-034 A 相同）；
2. 当天没有任何一行输入变化的日子：ENS 与冻结折预测合成的 ENS 逐行相同（最大差 = 0）；
3. 输入有变化的行和日子（重叠期检查中被解释的迟到财报差异）单独计数，报告相关，不计入 1、2。
用法：python x0_ens20.py <data_root> <fwd_main_root> <out.json>
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

    real, shadow, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    cal, grid = calendar_and_grid(real)
    f = make_folds(cal, grid, "20d")[18]
    days = [d for d in f.pred_days if W0 <= pd.Timestamp(d) <= W1]
    bundle = real / "data/gpu_v1/forward_models/20d_fold18"
    new, raw = score_ens20(shadow, days, bundle)

    # 冻结：第 18 折全部预测日的行顺序与 runs/*.npy 对齐
    md_f = ModelData(real)
    pb_f = md_f.load(f.pred_days)
    kf = pd.MultiIndex.from_arrays([pd.to_datetime(pb_f.meta["as_of"]), pb_f.meta["ticker"]], names=["as_of", "ticker"])
    md_s = ModelData(shadow)
    pb_s = md_s.load(days)
    ks = new.index
    # 输入是否变化：逐行比较 X（float16 原值）和当日市场特征
    xf = pd.DataFrame(pb_f.X.astype(np.float32), index=kf).reindex(ks)
    xs = pd.DataFrame(pb_s.X.astype(np.float32), index=ks)
    same_x = ((xf.to_numpy() == xs.to_numpy()) | (np.isnan(xf.to_numpy()) & np.isnan(xs.to_numpy()))).all(axis=1)
    mk_same = (md_f.market.reindex(days).fillna(-9e9).to_numpy() == md_s.market.reindex(days).fillna(-9e9).to_numpy()).all(axis=1)
    mk_ok = pd.Series(mk_same, index=pd.DatetimeIndex(days))
    same = same_x & mk_ok.reindex(ks.get_level_values("as_of")).to_numpy()
    changed_days = sorted({str(d.date()) for d in ks.get_level_values("as_of")[~same]})

    res = {"days": len(days), "rows": int(len(ks)), "rows_inputs_changed": int((~same).sum()),
           "days_with_changed_inputs": changed_days, "families": {}}
    for fm in FAMS:
        recs = []
        for s, p in enumerate(raw[fm]):
            old = pd.Series(np.load(real / "data/gpu_v1/runs" / TAGS[fm] / f"k18_s{s}.npy"), index=kf).reindex(ks).to_numpy()
            d = np.abs(p.astype(np.float64) - old)
            u = same & np.isfinite(d)
            recs.append({"seed": s, "max_abs_diff_unchanged": float(d[u].max()) if u.any() else None,
                         "pearson_unchanged": float(np.corrcoef(p[u], old[u])[0, 1]) if u.sum() > 2 else None,
                         "pearson_changed": float(np.corrcoef(p[~same], old[~same])[0, 1]) if (~same).sum() > 2 else None})
        tree = fm in ("xgb", "cat")
        ok = all((r["max_abs_diff_unchanged"] == 0) if tree else
                 (r["max_abs_diff_unchanged"] <= 1e-3 and r["pearson_unchanged"] >= 0.9999) for r in recs)
        res["families"][fm] = {"seeds": recs, "pass": bool(ok)}
    old_parts = {}
    for fm in FAMS:
        q = pd.read_parquet(real / "data/gpu_v1/preds" / f"{fm}_20d" / "fold_18.parquet")
        q["as_of"] = pd.to_datetime(q["as_of"])
        old_parts[fm] = q.set_index(["as_of", "ticker"])["score"].reindex(ks)
    old_ens = rank_average_frames(old_parts)
    clean_day = ~ks.get_level_values("as_of").isin(pd.to_datetime(changed_days))
    diff = (new["ENS"] - old_ens).abs()
    res["ens"] = {"rows_clean_days": int(clean_day.sum()),
                  "max_abs_diff_clean_days": float(diff[clean_day].max()) if clean_day.any() else None,
                  "spearman_all": float(new["ENS"].corr(old_ens, method="spearman")),
                  "max_abs_diff_all": float(diff.max())}
    res["ens"]["pass"] = res["ens"]["max_abs_diff_clean_days"] == 0
    res["pass"] = res["ens"]["pass"] and all(v["pass"] for v in res["families"].values())
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pass": res["pass"], "rows": res["rows"], "rows_inputs_changed": res["rows_inputs_changed"],
                      "families": {k: v["pass"] for k, v in res["families"].items()}, "ens": res["ens"]}))


if __name__ == "__main__":
    main()
