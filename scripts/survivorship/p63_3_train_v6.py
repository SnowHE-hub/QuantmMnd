"""P63-3：v6 上 63d Ridge WF 判定训练（逐字复刻 P4c，只换 label/horizon/featset）.

3 个变体（同一 merged panel，只换 feats 列表）：
  full_fnd   = 35 v5 + 16 short_horizon surv + 16 基本面 surv + Alpha158   （B：主判定）
  full_nofnd = 35 v5 + 16 short_horizon surv + Alpha158                    （C：ablation 去基本面）
  fnd_only   = 16 基本面 surv                                              （D：纯基本面 standalone）
配置：Ridge(α=10)，PurgedWFS(h=63,emb=63,rolling756,n_val=2)，季度 cutoffs 2022Q1→2025Q3，
oos_start=2022-01-01，decide_direction val-only(auto_flip=False)，score*=d。不调参。
输出 data/bakeoff/preds/ridge_<variant>_63d_quarterly_v6.parquet。
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
FND = REPO / "data/panel/fundamental_factors_v6.parquet"
A158 = REPO / "data/bakeoff/alpha158_asof_v6.parquet"
SH_SURV = REPO / "data/loss_signals_v4/short_horizon_survivors.json"
FND_SURV = REPO / "data/loss_signals_v4/fundamental_survivors_v6.json"
PREDS = REPO / "data/bakeoff/preds"; PREDS.mkdir(parents=True, exist_ok=True)
LABEL, H, EMB = "forward_return_63d", 63, 63


def build():
    base = pd.read_parquet(V6_PANEL)
    sh = json.loads(SH_SURV.read_text())["survivors"]
    sh = [c for c in sh if c in base.columns]
    fnd_surv = json.loads(FND_SURV.read_text())["survivors"]
    fnd = pd.read_parquet(FND)[fnd_surv].reindex(base.index)
    fnd.columns = [f"fnd_{c}" for c in fnd.columns]
    panel = merge_increment(base, fnd)
    a = pd.read_parquet(A158).reindex(base.index)
    a = a.loc[:, a.notna().any()]
    panel = merge_increment(panel, a)
    feats = {
        "full_fnd": list(WHITELIST_35) + sh + list(fnd.columns) + list(a.columns),
        "full_nofnd": list(WHITELIST_35) + sh + list(a.columns),
        "fnd_only": list(fnd.columns),
    }
    return panel, feats


def run_variant(panel, feats, variant, cal, folds):
    t0 = time.time(); rows = []; dirs = {}
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
            rows.append(pd.DataFrame({"as_of": dt, "ticker": xs.index, "score": s.values}).dropna(subset=["score"]))
    preds = pd.concat(rows, ignore_index=True)
    out = PREDS / f"ridge_{variant}_63d_quarterly_v6.parquet"
    preds.to_parquet(out)
    flips = sum(1 for v in dirs.values() if v < 0)
    (PREDS / f"ridge_{variant}_63d_quarterly_v6.dirs.json").write_text(json.dumps(
        {"dirs": dirs, "n_flips": flips, "n_feats": len(feats), "train_seconds": round(time.time()-t0, 1)},
        ensure_ascii=False, indent=2))
    print(f"[{variant}] feats={len(feats)} folds={len(dirs)} flips={flips} "
          f"shape={preds.shape} ({time.time()-t0:.0f}s)", flush=True)


def main():
    panel, feats = build()
    cal = load_trading_calendar()
    as_of = sorted(panel.index.get_level_values("as_of").unique())
    oos_start = pd.Timestamp("2022-01-01"); oos_end = as_of[-1]
    sp = PurgedWalkForwardSplit(cal, horizon=H, embargo=EMB, mode="rolling",
                                rolling_lookback_td=756, n_val=2)
    cut = make_quarterly_cutoffs(cal, oos_start, pd.Timestamp("2025-07-01"))  # →2025Q3，splitter 自动收尾
    folds = sp.make_folds(as_of, cut, oos_start=oos_start, oos_end=oos_end)
    print(f"[p63-3] panel={panel.shape} cutoffs={len(cut)} folds={len(folds)} "
          f"oos={oos_start.date()}→{oos_end.date()}", flush=True)
    for variant in ("full_fnd", "full_nofnd", "fnd_only"):
        run_variant(panel, feats[variant], variant, cal, folds)
    from scripts.bakeoff.run_manifest import write_run_manifest
    write_run_manifest(PREDS, kind="train", data_version="v6", feature_set="full_v6_63d(B/C/D)",
                       label=LABEL, horizon="63d", extra={"model": "ridge", "variants": "full_fnd/full_nofnd/fnd_only"})


if __name__ == "__main__":
    main()
