"""T-036 X0（Ridge C）：新管线复现参考臂 Ridge C 的最后一折（规则运行前写定，标准沿用 T-033 A0：最大差 ≤ 1e-6）.

对 20d（ridge_20d_v6_1_BASE）与 63d（ridge_63d_v6_1_C_infnan）各做：
1. 训练复现：quantmind.line_a.ridge_scoring.fit_reference 重建的最后一折测试预测 vs 已登记预测，逐行最大差 ≤ 1e-6；
2. 前瞻特征复现：在 [2026-09-01, 2026-10-09] 的最后一折测试日上，用影子数据根的日频面板拼出特征打分，
   与已登记预测比较。特征与"真实数据根的日频面板"不同的行（被重叠期检查解释的迟到财报）单独计数、不计入判定；
   其余行最大差 ≤ 1e-6。
3. 两项都通过才把重建的最后一折存档到 data/line_a_v2/refs/ridge_<期限>_lastfold.npz（日常打分只读存档），
   并核对读回存档后的打分与内存模型逐行相同（最大差 = 0）。
用法：python x0_ridge.py <data_root> <fwd_main_root> <out.json> [期限 …]
（2026-10-21 补充，判定不变：可只跑指定期限，两个期限在同一进程里会超出 6.5 GB 内存上限；
 另报告全部行的分数差与秩相关，只作参考；存档已存在时不重写，只核对读回后的打分与内存模型相同。）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
W0, W1 = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-09")
FROZEN = {"20d": "data/bakeoff/preds_v6_1_t028/ridge_20d_v6_1_BASE.parquet",
          "63d": "data/bakeoff/preds_v6_1/ridge_63d_v6_1_C_infnan.parquet"}


def main() -> None:
    from quantmind.line_a.ridge_scoring import (
        features_from_daily,
        fit_reference,
        load_ref,
        save_ref,
        score,
    )
    real, shadow, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    res = {}
    import gc
    for h in (sys.argv[4:] or ["20d", "63d"]):
        ref = fit_reference(real, h)
        fz = pd.read_parquet(real / FROZEN[h])
        fz["as_of"] = pd.to_datetime(fz["as_of"])
        fz = fz.set_index(["as_of", "ticker"])["score"]
        rt = ref.frozen_test.set_index(["as_of", "ticker"])["score"]
        j = pd.concat([rt.rename("re"), fz.rename("fz")], axis=1, join="inner")
        train_ok = {"rows": int(len(j)), "rows_rebuilt": int(len(rt)),
                    "rows_unmatched": int(len(rt) - len(j)), "max_abs_diff": float((j["re"] - j["fz"]).abs().max())}
        train_ok["pass"] = train_ok["rows_unmatched"] == 0 and train_ok["max_abs_diff"] <= 1e-6
        days = sorted(d for d in rt.index.get_level_values("as_of").unique() if W0 <= d <= W1)
        Xs = features_from_daily(shadow, days, ref.feats)
        Xr = features_from_daily(real, days, ref.feats).reindex(Xs.index)
        a, b = Xs.to_numpy(dtype=np.float64), Xr.to_numpy(dtype=np.float64)
        same = ((a == b) | (np.isnan(a) & np.isnan(b))).all(axis=1)
        s = score(ref, Xs)
        k = s.index.intersection(fz.index)
        d = (s.reindex(k) - fz.reindex(k)).abs().to_numpy()
        same_k = pd.Series(same, index=Xs.index).reindex(k).to_numpy()
        fwd = {"days": [str(x.date()) for x in days], "rows": int(len(k)), "rows_features_changed": int((~same_k).sum()),
               "rows_frozen_not_scored": int(len(fz[fz.index.get_level_values("as_of").isin(days)]) - len(k)),
               "max_abs_diff_unchanged": float(d[same_k].max()) if same_k.any() else None,
               "info_max_abs_diff_all": float(np.nanmax(d)) if len(d) else None,
               "info_spearman_all": float(s.reindex(k).corr(fz.reindex(k), method="spearman")) if len(k) else None}
        fwd["pass"] = (fwd["rows_frozen_not_scored"] == 0 and fwd["max_abs_diff_unchanged"] is not None
                       and fwd["max_abs_diff_unchanged"] <= 1e-6)
        saved = None
        if train_ok["pass"] and fwd["pass"]:
            ref_path = real / f"data/line_a_v2/refs/ridge_{h}_lastfold.npz"
            if ref_path.exists():
                from quantmind.line_a.atomic import sha256_file
                sha = sha256_file(ref_path)
            else:
                sha = save_ref(ref, ref_path)
            rt2 = score(load_ref(ref_path), Xs)
            saved = {"path": str(ref_path.relative_to(real)), "sha256": sha,
                     "roundtrip_max_abs_diff": float((rt2 - s).abs().max())}
            fwd["pass"] = saved["roundtrip_max_abs_diff"] == 0.0
        res[h] = {"saved_ref": saved, "cutoff": str(ref.cutoff.date()) if pd.notna(ref.cutoff) else None, "direction": ref.direction,
                  "n_features": len(ref.feats), "train_reproduction": train_ok, "forward_features": fwd,
                  "pass": bool(train_ok["pass"] and fwd["pass"])}
        print(h, json.dumps({"train": train_ok, "forward": {k2: v for k2, v in fwd.items() if k2 != "days"}}))
        del ref, Xs, Xr, a, b, s, fz, rt, j
        gc.collect()
    res["pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("OVERALL", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()
