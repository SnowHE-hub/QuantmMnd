#!/usr/bin/env python
"""P3b：tabular 模型 fold 驱动训练 + 导出预测（quantmind env，复用 v2 口径）.

模型：lgbm（= v2 的 LGBMRankerModel/LGBMPredictor，含 categorical）｜ ridge（诚实线性基线）。
特征集：v2_35 / plus_surv(35+16) / full(35+16+158 Alpha158)。
fold：PurgedWalkForwardSplit 季度 cutoff（与 v2/_diag_wf_v2 完全一致）。方向 H-A 仅用 val。
导出：data/bakeoff/preds/<model>_<featset>_<period>_<refit>.parquet  (as_of,ticker,score)
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from quantmind.data.lake import load_trading_calendar
from quantmind.features.weekly_panel import WHITELIST_35, OUT_PATH, merge_increment
from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.backtest.wf_gate import LGBMPredictor, decide_direction
from quantmind.backtest.wf_metrics import cross_sectional_ic

SURV_JSON = REPO / "data/loss_signals_v4/short_horizon_survivors.json"
SH_PANEL = REPO / "data/panel/short_horizon_factors.parquet"
A158 = REPO / "data/bakeoff/alpha158_asof.parquet"
PREDS = REPO / "data/bakeoff/preds"; PREDS.mkdir(parents=True, exist_ok=True)
PERIOD = {"12d": ("forward_return_12d", 12, 20), "63d": ("forward_return_63d", 63, 63)}


def make_quarterly_cutoffs(cal, start, end):
    cal_sorted = sorted(cal); out = []
    for qd in pd.date_range(start, end, freq="QS-JAN"):
        after = [d for d in cal_sorted if d >= qd]
        if after: out.append(after[0])
    return out


def make_monthly_cutoffs(cal, start, end):
    cal_sorted = sorted(cal); out = []
    for qd in pd.date_range(start, end, freq="MS"):
        after = [d for d in cal_sorted if d >= qd]
        if after: out.append(after[0])
    return out


class RidgePredictor:
    """诚实线性基线：数值特征标准化 + Ridge 回归 raw label。categorical 丢弃。"""
    def __init__(self, alpha=10.0):
        from sklearn.linear_model import Ridge
        self.alpha = alpha; self._m = None; self._mu = None; self._sd = None; self._num = None
    def fit(self, panel, train_dates, val_dates, feats, label):
        from sklearn.linear_model import Ridge
        self._num = [f for f in feats if f not in ("exposure_industry", "exposure_area")]
        tr = panel.loc[panel.index.get_level_values("as_of").isin(set(train_dates))]
        X = tr[self._num].astype(float); y = tr[label].astype(float)
        ok = y.notna()
        X = X[ok]; y = y[ok]
        self._mu = X.mean(); self._sd = X.std().replace(0, 1.0)
        Xz = ((X - self._mu) / self._sd).fillna(0.0).values
        self._m = Ridge(alpha=self.alpha).fit(Xz, y.values)
        return self
    def score(self, panel, date, feats):
        xs = panel.xs(pd.Timestamp(date), level="as_of")
        Xz = ((xs[self._num].astype(float) - self._mu) / self._sd).fillna(0.0).values
        return pd.Series(self._m.predict(Xz), index=xs.index, name="score")


def build_feature_panel(featset: str) -> tuple[pd.DataFrame, list[str]]:
    base = pd.read_parquet(OUT_PATH)                       # 35 + labels
    feats = list(WHITELIST_35)
    panel = base
    if featset in ("plus_surv", "full"):
        surv = json.loads(SURV_JSON.read_text())["survivors"]
        sh = pd.read_parquet(SH_PANEL)[surv].reindex(base.index)
        panel = merge_increment(panel, sh); feats += surv
    if featset == "full":
        a = pd.read_parquet(A158).reindex(base.index)
        # 丢弃整列全 NaN / 常量列
        a = a.loc[:, a.notna().any()]
        panel = merge_increment(panel, a); feats += list(a.columns)
    return panel, feats


def run(model: str, featset: str, period: str, refit: str):
    label, H, E = PERIOD[period]
    panel, feats = build_feature_panel(featset)
    cal = load_trading_calendar()
    as_of = sorted(panel.index.get_level_values("as_of").unique())
    oos_start = pd.Timestamp("2022-01-01"); oos_end = as_of[-1]
    sp = PurgedWalkForwardSplit(cal, horizon=H, embargo=E, mode="rolling",
                                rolling_lookback_td=756, n_val=2)
    cut = (make_quarterly_cutoffs if refit == "quarterly" else make_monthly_cutoffs)(
        cal, oos_start, pd.Timestamp("2025-10-01"))
    folds = sp.make_folds(as_of, cut, oos_start=oos_start, oos_end=oos_end)
    print(f"[{model}/{featset}/{period}/{refit}] feats={len(feats)} folds={len(folds)}")
    rows = []; dirs = {}; t0 = time.time()
    for i, fold in enumerate(folds):
        if not fold.test_dates or not fold.train_dates:
            continue
        pred = (LGBMPredictor(n_quantiles=5, n_estimators=400, learning_rate=0.05,
                              num_leaves=31, verbose=-1) if model == "lgbm"
                else RidgePredictor(alpha=10.0))
        pred.fit(panel, fold.train_dates, fold.val_dates, feats, label)
        d = decide_direction(pred, panel, fold.val_dates, feats, label); dirs[i] = int(d)
        for dt in fold.test_dates:
            dt = pd.Timestamp(dt)
            try:
                xs = panel.xs(dt, level="as_of")
            except KeyError:
                continue
            s = pred.score(panel, dt, feats) * d
            sub = pd.DataFrame({"as_of": dt, "ticker": xs.index, "score": s.values})
            rows.append(sub.dropna(subset=["score"]))
        print(f"  fold {i} cut={fold.cutoff.date()} dir={d:+d} ({time.time()-t0:.0f}s)")
    preds = pd.concat(rows, ignore_index=True)
    out = PREDS / f"{model}_{featset}_{period}_{refit}.parquet"
    preds.to_parquet(out)
    (PREDS / f"{model}_{featset}_{period}_{refit}.dirs.json").write_text(
        json.dumps({"dirs": dirs, "n_flips": sum(1 for v in dirs.values() if v < 0),
                    "n_feats": len(feats)}, ensure_ascii=False, indent=2))
    print(f"[save] {out} shape={preds.shape} flips={sum(1 for v in dirs.values() if v<0)}/{len(dirs)} "
          f"({time.time()-t0:.0f}s)")
    # F-13：run manifest（溯源）
    from scripts.bakeoff.run_manifest import write_run_manifest
    write_run_manifest(PREDS, kind="train", data_version="v5", feature_set=featset,
                       label=label, horizon=period,
                       extra={"model": model, "refit": refit, "n_feats": len(feats), "preds": out.name})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm", "ridge"], required=True)
    ap.add_argument("--featset", choices=["v2_35", "plus_surv", "full"], required=True)
    ap.add_argument("--period", choices=["12d", "63d"], default="12d")
    ap.add_argument("--refit", choices=["quarterly", "monthly"], default="quarterly")
    a = ap.parse_args()
    run(a.model, a.featset, a.period, a.refit)
