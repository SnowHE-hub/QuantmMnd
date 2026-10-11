"""T-022：主线 B 63d 种子的两个行业方案（连续样本外 C 折；±inf 视为缺失，与 T-018 的 C_infnan 同口径）.

  IND   按申万一级行业分别训练 Ridge（每折每个行业一个模型；行业按 as_of 时点归属；无行业的票不打分）
  SECF  统一模型 + 9 个行业特征（sector_features_v6_1.parquet 的 ind_* / rel_*）
其余全部沿用 T-018 方案 C（特征、Ridge(α=10)、PurgedWalkForward、方向由验证集决定、半年切点）。
输出 <out_dir>/ridge_63d_v6_1_<IND|SECF>_infnan.parquet + .dirs.json；已存在则拒绝。
用法：PYTHONPATH=. python train_sector.py <data_root> <IND|SECF> <out_dir>
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.features.weekly_panel import merge_increment

_spec = importlib.util.spec_from_file_location(
    "t018train", Path(__file__).resolve().parents[1] / "T-018/train_line_b_v6_1.py")
t18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t18)
SECF_COLS = ["ind_mom_1m", "ind_mom_3m", "ind_mom_6m", "rel_mom_3m", "rel_mom_6m",
             "ind_dm_roe_ttm", "ind_dm_pb", "ind_dm_pe_ttm", "ind_dm_operating_profit_yoy"]


def main() -> None:
    root, mode, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    assert mode in ("IND", "SECF")
    cfg = t18.SEEDS["63d"]
    out = out_dir / f"ridge_63d_v6_1_{mode}_infnan.parquet"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    t0 = time.time()
    panel, feats, inputs = t18.build(root, "63d")
    sf_p = root / "data/panel/sector_features_v6_1.parquet"
    sf = pd.read_parquet(sf_p).reindex(panel.index)
    inputs["sector_features"] = {"path": str(sf_p), "sha256": t18.sha(sf_p)}
    if mode == "SECF":
        panel = merge_increment(panel, sf[SECF_COLS])
        feats = feats + SECF_COLS
    num = [f for f in feats if f not in ("exposure_industry", "exposure_area")]
    n_inf = int(np.isinf(panel[num].to_numpy(dtype=float)).sum())
    panel[num] = panel[num].replace([np.inf, -np.inf], np.nan)
    inputs["inf_to_nan"] = {"n_values_replaced": n_inf}
    cal = t18.calendar(root)
    as_of = sorted(panel.index.get_level_values("as_of").unique())
    sp = PurgedWalkForwardSplit(cal, horizon=cfg["H"], embargo=cfg["E"], mode="rolling",
                                rolling_lookback_td=756, n_val=2)
    cuts = cfg["cuts"](cal, cfg["end_X"])
    folds = t18.make_folds(sp, as_of, cuts, "C")
    l1 = sf["sw_l1"]
    print(f"[63d/{mode}] feats={len(feats)} folds={len(folds)} panel={panel.shape}", flush=True)
    rows, dirs, fold_log = [], {}, []
    for i, fold in enumerate(folds):
        if not fold.test_dates or not fold.train_dates:
            continue
        PurgedWalkForwardSplit.assert_no_leakage(fold)
        if mode == "IND":
            for g in sorted(l1.dropna().unique()):
                sub = panel[(l1 == g).values]
                r, d = t18.fit_score(sub, feats, cfg["label"], fold)
                rows += r
                dirs[f"{i}_{g}"] = d
        else:
            r, d = t18.fit_score(panel, feats, cfg["label"], fold)
            rows += r
            dirs[str(i)] = d
        fold_log.append({"cutoff": str(fold.cutoff.date()), "n_test": len(fold.test_dates)})
        print(f"  fold {i} cut={fold.cutoff.date()} ({time.time() - t0:.0f}s)", flush=True)
    preds = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    out.with_suffix(".dirs.json").write_text(json.dumps(
        {"seed": "63d", "mode": mode, "dirs": dirs, "n_flips": sum(1 for v in dirs.values() if v < 0),
         "n_feats": len(feats), "folds": fold_log, "inputs": inputs, "train_seconds": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[63d/{mode}] saved {out} {preds.shape}", flush=True)


if __name__ == "__main__":
    main()
