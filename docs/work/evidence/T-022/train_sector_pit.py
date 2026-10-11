"""T-022 第二轮修订：PIT 行业数据下的三个方案（连续样本外 C 折；±inf 视为缺失，与 T-018 的 C_infnan 同口径）.

行业与行业特征取 sector_features_v6_1_pit.parquet（体系发布日 2021-12-13 前一律缺失）。
  IND   按申万一级行业分别训练 Ridge：只能用发布后的训练行；
  CPOST 统一模型，训练行同样限于发布后（与 IND 的可用训练数据相同，用来隔离"分行业 vs 统一"）；
  SECF  统一模型 + 9 个行业特征（发布前为缺失 → 标准化后填 0），训练窗口与 C_infnan 相同。
IND / CPOST：某折训练窗口里发布后的截面少于 MIN_TRAIN_DATES（26 周）时，该折不训练、不打分（两者一致）；
IND 中某行业在该折训练行少于 MIN_ROWS（200）时该行业不打分（记录在 .dirs.json）。
运行时断言：IND / CPOST 的任何训练行 as_of ≥ 发布日。
其余全部沿用 T-018 方案 C（特征、Ridge(α=10)、PurgedWalkForward、方向由验证集决定、半年切点）。
输出 <out_dir>/ridge_63d_v6_1_<MODE>_pit_infnan.parquet + .dirs.json；已存在则拒绝。
用法：PYTHONPATH=. python train_sector_pit.py <data_root> <IND|CPOST|SECF> <out_dir>
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

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("t018train", _here.parent / "T-018/train_line_b_v6_1.py")
t18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t18)
_bspec = importlib.util.spec_from_file_location("t022pit", _here / "build_sector_features_pit.py")
bpit = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(bpit)

SECF_COLS = ["ind_mom_1m", "ind_mom_3m", "ind_mom_6m", "rel_mom_3m", "rel_mom_6m",
             "ind_dm_roe_ttm", "ind_dm_pb", "ind_dm_pe_ttm", "ind_dm_operating_profit_yoy"]
MIN_TRAIN_DATES, MIN_ROWS = 26, 200
PUBLIC = bpit.TAXONOMY_PUBLIC


def post_public(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel.index.get_level_values("as_of") >= PUBLIC]


def assert_train_pit(sub: pd.DataFrame, fold) -> None:
    tr = sub.index.get_level_values("as_of")
    tr = tr[tr.isin(set(fold.train_dates))]
    assert len(tr) == 0 or tr.min() >= PUBLIC, "training row before taxonomy publication"


def main() -> None:
    root, mode, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    assert mode in ("IND", "CPOST", "SECF")
    cfg = t18.SEEDS["63d"]
    out = out_dir / f"ridge_63d_v6_1_{mode}_pit_infnan.parquet"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    t0 = time.time()
    panel, feats, inputs = t18.build(root, "63d")
    sf_p = root / "data/panel/sector_features_v6_1_pit.parquet"
    sf = pd.read_parquet(sf_p).reindex(panel.index)
    inputs["sector_features"] = {"path": str(sf_p), "sha256": t18.sha(sf_p), "taxonomy_public": str(PUBLIC.date())}
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
    print(f"[63d/{mode}/pit] feats={len(feats)} folds={len(folds)} panel={panel.shape}", flush=True)
    rows, dirs, fold_log = [], {}, []
    for i, fold in enumerate(folds):
        if not fold.test_dates or not fold.train_dates:
            continue
        PurgedWalkForwardSplit.assert_no_leakage(fold)
        n_post = int(sum(pd.Timestamp(d) >= PUBLIC for d in fold.train_dates))
        rec = {"cutoff": str(fold.cutoff.date()), "n_test": len(fold.test_dates), "n_train_dates_post_public": n_post}
        if mode in ("IND", "CPOST") and n_post < MIN_TRAIN_DATES:
            rec["skipped"] = f"post-publication training dates {n_post} < {MIN_TRAIN_DATES}"
            fold_log.append(rec)
            print(f"  fold {i} cut={fold.cutoff.date()} skipped ({n_post} post-public train dates)", flush=True)
            continue
        if mode == "IND":
            post = post_public(panel)
            l1p = l1.reindex(post.index)
            skipped = []
            for g in sorted(l1p.dropna().unique()):
                sub = post[(l1p == g).values]
                n_tr = int(sub.index.get_level_values("as_of").isin(set(fold.train_dates)).sum())
                if n_tr < MIN_ROWS:
                    skipped.append({"industry": g, "train_rows": n_tr})
                    continue
                assert_train_pit(sub, fold)
                r, d = t18.fit_score(sub, feats, cfg["label"], fold)
                rows += r
                dirs[f"{i}_{g}"] = d
            rec["industries_skipped"] = skipped
        elif mode == "CPOST":
            sub = post_public(panel)
            assert_train_pit(sub, fold)
            r, d = t18.fit_score(sub, feats, cfg["label"], fold)
            rows += r
            dirs[str(i)] = d
        else:
            r, d = t18.fit_score(panel, feats, cfg["label"], fold)
            rows += r
            dirs[str(i)] = d
        fold_log.append(rec)
        print(f"  fold {i} cut={fold.cutoff.date()} post-public train dates={n_post} ({time.time() - t0:.0f}s)", flush=True)
    preds = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    out.with_suffix(".dirs.json").write_text(json.dumps(
        {"seed": "63d", "mode": mode, "pit": True, "min_train_dates": MIN_TRAIN_DATES, "min_rows": MIN_ROWS,
         "dirs": dirs, "n_flips": sum(1 for v in dirs.values() if v < 0), "n_feats": len(feats), "folds": fold_log,
         "inputs": inputs, "train_seconds": round(time.time() - t0, 1)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[63d/{mode}/pit] saved {out} {preds.shape}", flush=True)


if __name__ == "__main__":
    main()
