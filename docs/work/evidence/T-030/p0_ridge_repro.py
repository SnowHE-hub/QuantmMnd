"""T-030 P0：只用日频构建器的产物（data/gpu_v1/panel/*）在周频网格上重建 v6_1 面板，复刻 Ridge 方案 C，与基线逐值比较.

网格 = build_asof_grid(完整交易日历)（与 T-018 asof_grid_v6_1.py 相同），不读 v6_1 面板文件。
特征/折/拟合/打分逐字调用 T-018 `train_line_b_v6_1.py` 的 make_folds(protocol="C") 与 fit_score：
  63d：= T-018 `63d C --inf-to-nan` → 对 data/bakeoff/preds_v6_1/ridge_63d_v6_1_C_infnan.parquet
  20d：= T-028 `train_event_aug.py 20d BASE`（63d 特征超集 + 20d 标签）→ 对 data/bakeoff/preds_v6_1_t028/ridge_20d_v6_1_BASE.parquet
同时用 quantmind.research.gpu.folds.make_folds 生成本卡的折，核对周频测试 as_of 与 T-018 折逐折相同。
输出 <data_root>/data/gpu_v1/p0/ridge_repro_<seed>.parquet + <evidence>/p0/ridge_repro_<seed>.json；已存在则拒绝。
用法：PYTHONPATH=. python p0_ridge_repro.py <data_root> <63d|20d> <evidence_dir>
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.features.weekly_panel import WHITELIST_35, build_asof_grid, merge_increment
from quantmind.research.gpu.folds import make_folds as gpu_make_folds

_ev = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("t018train", _ev / "T-018/train_line_b_v6_1.py")
t18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t18)

BASELINE = {"63d": "bakeoff/preds_v6_1/ridge_63d_v6_1_C_infnan.parquet",
            "20d": "bakeoff/preds_v6_1_t028/ridge_20d_v6_1_BASE.parquet"}
LABEL = {"63d": "forward_return_63d", "20d": "forward_return_20d"}


def read_weekly(group_dir: Path, dates: list[pd.Timestamp], columns: list[str] | None = None) -> pd.DataFrame:
    files = sorted(str(p) for p in group_dir.glob("*.parquet"))
    assert files, f"no daily files in {group_dir}"
    dset = ds.dataset(files, format="parquet")
    cols = None if columns is None else ["as_of", "ticker", *columns]
    t = dset.to_table(filter=ds.field("as_of").isin(pd.DatetimeIndex(dates).as_unit("ns").to_numpy()),
                      columns=cols).to_pandas()
    if "as_of" not in t.columns:
        t = t.reset_index()
    return t.set_index(["as_of", "ticker"]).sort_index()


def build_from_daily(root: Path, grid: list[pd.Timestamp]) -> tuple[pd.DataFrame, list[str]]:
    D = root / "data"
    P = D / "gpu_v1/panel"
    base = read_weekly(P / "base", grid)
    sh_all = read_weekly(P / "sh", grid)
    surv_sh = json.loads((D / "loss_signals_v4/short_horizon_survivors.json").read_text())["survivors"]
    sh = [c for c in surv_sh if c in sh_all.columns]
    base = merge_increment(base, sh_all[sh].reindex(base.index))
    feats = list(WHITELIST_35) + sh
    surv = json.loads((D / "loss_signals_v4/fundamental_survivors_v6.json").read_text())["survivors"]
    fnd = read_weekly(P / "fnd", grid, surv).reindex(base.index)
    fnd.columns = [f"fnd_{c}" for c in fnd.columns]
    panel = merge_increment(base, fnd)
    feats += list(fnd.columns)
    a = read_weekly(P / "a158", grid).reindex(base.index)
    a = a.loc[:, a.notna().any()]
    panel = merge_increment(panel, a)
    feats += list(a.columns)
    return panel, feats


def main() -> None:
    root, seed, ev = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out = root / f"data/gpu_v1/p0/ridge_repro_{seed}.parquet"
    rep = ev / f"p0/ridge_repro_{seed}.json"
    if out.exists() or rep.exists():
        raise SystemExit(f"refusing to overwrite {out} / {rep}")
    t0 = time.time()
    cal = t18.calendar(root)
    grid = list(build_asof_grid(cal))
    panel, feats = build_from_daily(root, grid)
    label = LABEL[seed]
    n_inf = 0
    if seed == "63d":                                    # = T-018 --inf-to-nan
        num = [f for f in feats if f not in ("exposure_industry", "exposure_area")]
        n_inf = int(np.isinf(panel[num].to_numpy(dtype=float)).sum())
        panel[num] = panel[num].replace([np.inf, -np.inf], np.nan)
    as_of = sorted(panel.index.get_level_values("as_of").unique())
    cfg = t18.SEEDS["63d"] if seed == "63d" else {"H": 20, "E": 20}
    sp = PurgedWalkForwardSplit(cal, horizon=cfg["H"], embargo=cfg["E"], mode="rolling",
                                rolling_lookback_td=756, n_val=2)
    if seed == "63d":
        cuts = t18.SEEDS["63d"]["cuts"](cal, t18.SEEDS["63d"]["end_X"])
    else:
        cuts = t18.make_quarterly_cutoffs(cal, pd.Timestamp("2022-01-01"), pd.Timestamp("2026-07-01"))
    folds = t18.make_folds(sp, as_of, cuts, "C")
    gfolds = gpu_make_folds(cal, grid, seed)
    folds = [f for f in folds if f.test_dates and f.train_dates]
    assert len(folds) == len(gfolds)
    for f, g in zip(folds, gfolds, strict=True):
        assert [pd.Timestamp(x) for x in f.test_dates] == g.test_asof and f.cutoff == g.cutoff
    print(f"[{seed}] panel={panel.shape} feats={len(feats)} folds={len(folds)} inf={n_inf} ({time.time() - t0:.0f}s)",
          flush=True)
    rows, dirs = [], {}
    for i, fold in enumerate(folds):
        PurgedWalkForwardSplit.assert_no_leakage(fold)
        r, d = t18.fit_score(panel, feats, label, fold)
        rows += r
        dirs[str(i)] = d
        print(f"  fold {i} cut={fold.cutoff.date()} d={d} ({time.time() - t0:.0f}s)", flush=True)
    preds = pd.concat(rows, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)

    ref = pd.read_parquet(root / "data" / BASELINE[seed])
    ref["as_of"] = pd.to_datetime(ref["as_of"])
    a = preds.set_index(["as_of", "ticker"])["score"].sort_index()
    b = ref.set_index(["as_of", "ticker"])["score"].sort_index()
    same_rows = bool(a.index.equals(b.index))
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    diff = (j["a"] - j["b"]).abs()
    ref_dirs = json.loads((root / "data" / BASELINE[seed]).with_suffix(".dirs.json").read_text())["dirs"]
    res = {"seed": seed, "baseline": BASELINE[seed], "n_feats": len(feats), "feats": feats, "n_inf_replaced": n_inf,
           "panel_shape": list(panel.shape), "grid": [str(grid[0].date()), str(grid[-1].date()), len(grid)],
           "n_rows_repro": int(len(a)), "n_rows_baseline": int(len(b)), "row_index_equal": same_rows,
           "max_abs_diff": float(diff.max()), "n_exact": int((diff == 0).sum()), "n_compared": int(len(j)),
           "dirs": dirs, "dirs_equal_baseline": dirs == ref_dirs, "pred_file": str(out),
           "seconds": round(time.time() - t0, 1)}
    res["ok"] = bool(same_rows and res["max_abs_diff"] <= 1e-9 and res["dirs_equal_baseline"])
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: v for k, v in res.items() if k != "feats"}, flush=True)


if __name__ == "__main__":
    main()
