"""T-023：20d / 40d 两个新期限的主线 B 种子（连续样本外，方案 C）.

除标签、隔离带与切点外，全部沿用 T-018 方案 C（train_line_b_v6_1.py 的 build / make_folds / fit_score）：
  特征 = 35 + 16 短周期 + 16 基本面 + Alpha158（与 63d 种子相同的超集）；Ridge(α=10)；PurgedWalkForward(rolling 756, n_val=2)；
  方向只由验证集决定；±inf 处理与 T-018 方案 C 相同（不做 --inf-to-nan，保持与 12d / 63d C 同一管线）。
  20d：H=20、E=20、季度切点；40d：H=40、E=40、季度切点；切点 2022-01-01 → 2026-07-01。
输出 <out_dir>/ridge_<seed>_v6_1_C.parquet + .dirs.json；已存在则拒绝。
用法：PYTHONPATH=. python train_horizon.py <data_root> <seed: 20d|40d> <out_dir>
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from scripts.bakeoff.p3b_tabular import make_quarterly_cutoffs

_spec = importlib.util.spec_from_file_location(
    "t018train", Path(__file__).resolve().parents[1] / "T-018/train_line_b_v6_1.py")
t18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t18)

SEEDS = {"20d": {"label": "forward_return_20d", "H": 20, "E": 20},
         "40d": {"label": "forward_return_40d", "H": 40, "E": 40}}
CUT_START, CUT_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2026-07-01")


def main() -> None:
    root, seed, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    cfg = SEEDS[seed]
    out = out_dir / f"ridge_{seed}_v6_1_C.parquet"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    t0 = time.time()
    panel, feats, inputs = t18.build(root, "63d")                      # 63d 的特征超集（含基本面）
    extra_p = root / "data/panel/labels_extra_v6_1.parquet"
    extra = pd.read_parquet(extra_p).reindex(panel.index)
    panel[cfg["label"]] = extra[cfg["label"]]
    inputs["labels_extra"] = {"path": str(extra_p), "sha256": t18.sha(extra_p)}
    cal = t18.calendar(root)
    as_of = sorted(panel.index.get_level_values("as_of").unique())
    sp = PurgedWalkForwardSplit(cal, horizon=cfg["H"], embargo=cfg["E"], mode="rolling",
                                rolling_lookback_td=756, n_val=2)
    cuts = make_quarterly_cutoffs(cal, CUT_START, CUT_END)
    folds = t18.make_folds(sp, as_of, cuts, "C")
    print(f"[{seed}/C] feats={len(feats)} cutoffs={len(cuts)} folds={len(folds)} panel={panel.shape}", flush=True)
    rows, dirs, fold_log = [], {}, []
    for i, fold in enumerate(folds):
        if not fold.test_dates or not fold.train_dates:
            continue
        PurgedWalkForwardSplit.assert_no_leakage(fold)
        r, d = t18.fit_score(panel, feats, cfg["label"], fold)
        rows += r
        dirs[str(i)] = d
        fold_log.append({"cutoff": str(fold.cutoff.date()),
                         "train": [str(fold.train_dates[0].date()), str(fold.train_dates[-1].date())],
                         "test": [str(pd.Timestamp(fold.test_dates[0]).date()),
                                  str(pd.Timestamp(fold.test_dates[-1]).date())], "n_test": len(fold.test_dates)})
        print(f"  fold {i} cut={fold.cutoff.date()} test {fold_log[-1]['test']} ({time.time() - t0:.0f}s)", flush=True)
    preds = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    out.with_suffix(".dirs.json").write_text(json.dumps(
        {"seed": seed, "protocol": "C", "dirs": dirs, "n_flips": sum(1 for v in dirs.values() if v < 0),
         "n_feats": len(feats), "folds": fold_log, "inputs": inputs, "train_seconds": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{seed}/C] saved {out} {preds.shape}; {preds.as_of.nunique()} dates", flush=True)


if __name__ == "__main__":
    main()
