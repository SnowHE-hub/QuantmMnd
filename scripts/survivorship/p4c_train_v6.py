"""P4c：v6 上 Ridge(full) 12d 季度训练（quantmind env）.

完全复刻 batch-A（scripts/bakeoff/p3b_tabular.py），唯一变量 = universe/面板（v5→v6）：
  base = alpha_panel_weekly_v6.parquet（已含 WHITELIST_35 + 16 survivors，54 列）
  a158 = alpha158_asof_v6.parquet（158 列，前缀 a158_）
  feats = WHITELIST_35 + 16 survivors + a158（all-NaN 列丢弃）
配置逐字一致：RidgePredictor(alpha=10)，PurgedWalkForwardSplit(h=12,emb=20,rolling756,n_val=2)，
quarterly cutoffs 2022-01-01→2025-10-01，oos_start=2022-01-01，decide_direction(val-only)，score*=d。
输出 data/bakeoff/preds/ridge_full_12d_quarterly_v6.parquet。
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from quantmind.data.lake import load_trading_calendar
from quantmind.features.weekly_panel import WHITELIST_35, merge_increment
from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.backtest.wf_gate import decide_direction
from scripts.bakeoff.p3b_tabular import RidgePredictor, make_quarterly_cutoffs

V6_PANEL = REPO / "data/panel/alpha_panel_weekly_v6.parquet"
A158_V6 = REPO / "data/bakeoff/alpha158_asof_v6.parquet"
SURV_JSON = REPO / "data/loss_signals_v4/short_horizon_survivors.json"
PREDS = REPO / "data/bakeoff/preds"; PREDS.mkdir(parents=True, exist_ok=True)
LABEL, H, EMB = "forward_return_12d", 12, 20


def build_panel():
    base = pd.read_parquet(V6_PANEL)                      # 35 + labels + 16 surv
    surv = json.loads(SURV_JSON.read_text())["survivors"]
    feats = list(WHITELIST_35) + [s for s in surv if s in base.columns]
    a = pd.read_parquet(A158_V6).reindex(base.index)
    a = a.loc[:, a.notna().any()]                         # 丢全 NaN 列
    panel = merge_increment(base, a)
    feats += list(a.columns)
    return panel, feats


def main():
    t0 = time.time()
    panel, feats = build_panel()
    cal = load_trading_calendar()
    as_of = sorted(panel.index.get_level_values("as_of").unique())
    oos_start = pd.Timestamp("2022-01-01"); oos_end = as_of[-1]
    sp = PurgedWalkForwardSplit(cal, horizon=H, embargo=EMB, mode="rolling",
                                rolling_lookback_td=756, n_val=2)
    cut = make_quarterly_cutoffs(cal, oos_start, pd.Timestamp("2025-10-01"))
    folds = sp.make_folds(as_of, cut, oos_start=oos_start, oos_end=oos_end)
    print(f"[p4c ridge/full/12d/quarterly v6] feats={len(feats)} folds={len(folds)} "
          f"panel={panel.shape} ({time.time()-t0:.0f}s)", flush=True)
    rows = []; dirs = {}
    for i, fold in enumerate(folds):
        if not fold.test_dates or not fold.train_dates:
            continue
        pred = RidgePredictor(alpha=10.0)
        pred.fit(panel, fold.train_dates, fold.val_dates, feats, LABEL)
        d = decide_direction(pred, panel, fold.val_dates, feats, LABEL); dirs[i] = int(d)
        for dt in fold.test_dates:
            dt = pd.Timestamp(dt)
            try:
                xs = panel.xs(dt, level="as_of")
            except KeyError:
                continue
            s = pred.score(panel, dt, feats) * d
            rows.append(pd.DataFrame({"as_of": dt, "ticker": xs.index, "score": s.values})
                        .dropna(subset=["score"]))
        print(f"  fold {i} cut={fold.cutoff.date()} dir={d:+d} ({time.time()-t0:.0f}s)", flush=True)
    preds = pd.concat(rows, ignore_index=True)
    out = PREDS / "ridge_full_12d_quarterly_v6.parquet"
    preds.to_parquet(out)
    elapsed = time.time() - t0
    flips = sum(1 for v in dirs.values() if v < 0)
    (PREDS / "ridge_full_12d_quarterly_v6.dirs.json").write_text(json.dumps(
        {"dirs": dirs, "n_flips": flips, "n_feats": len(feats),
         "train_seconds": round(elapsed, 1)}, ensure_ascii=False, indent=2))
    print(f"[save] {out} shape={preds.shape} flips={flips}/{len(dirs)} train={elapsed:.0f}s", flush=True)
    from scripts.bakeoff.run_manifest import write_run_manifest
    write_run_manifest(PREDS, kind="train", data_version="v6", feature_set="full(35+16+158)",
                       label=LABEL, horizon="12d", artifacts=[out],
                       extra={"model": "ridge", "alpha": 10.0, "refit": "quarterly",
                              "n_feats": len(feats), "train_seconds": round(elapsed, 1),
                              "n_flips": flips, "n_folds": len(dirs)})


if __name__ == "__main__":
    main()
