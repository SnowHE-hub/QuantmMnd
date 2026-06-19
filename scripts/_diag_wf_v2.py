#!/usr/bin/env python
"""scripts/_diag_wf_v2.py — WF v2 诊断（只诊断，不建模/不组合优化/不改 NAV）.

复刻 run_wf_full_v1.py 的 fold 切分与 LGBMPredictor 训练，但**捕获每个 OOS
截面的逐票预测**，用于：
  1. regime 条件 IC（market_momentum_60d 三档 牛/中性/熊；PIT 安全）
  2. 行业+规模中性化（pred ~ C(exposure_industry)+log_market_cap，残差重算 IC）
  3. 特征重要性（gain）+ 单变量 IC 归因

产物：
  data/loss_signals_v4/wf_v2_oos_predictions.parquet  逐票 OOS 预测
  data/loss_signals_v4/wf_v2_diag_summary.json        所有诊断结果（供报告引用）
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quantmind.data.lake import load_trading_calendar
from quantmind.features.weekly_panel import WHITELIST_35, OUT_PATH
from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.backtest.wf_gate import LGBMPredictor, decide_direction

# ---- 与 run_wf_full_v1.py 完全一致的参数 ----
HORIZON = 12
EMBARGO = 20
ROLLING_LOOKBACK = 756
N_VAL = 2
LABEL_COL = "forward_return_12d"
FEAT_COLS = list(WHITELIST_35)
OOS_START = pd.Timestamp("2022-01-01")
N_ESTIMATORS = 400
LEARNING_RATE = 0.05
NUM_LEAVES = 31

OUT_DIR = REPO_ROOT / "data" / "loss_signals_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_quarterly_cutoffs(cal, start, end):
    cal_sorted = sorted(cal)
    cutoffs = []
    for qd in pd.date_range(start, end, freq="QS-JAN"):
        after = [d for d in cal_sorted if d >= qd]
        if after:
            cutoffs.append(after[0])
    return cutoffs


def spearman(a, b):
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    rho, _ = stats.spearmanr(df["a"], df["b"])
    return float(rho)


def load_log_mktcap(as_of_grid, tickers):
    """从 data/lake/daily_basic.parquet 取 circ_mv（流通市值,万元），PIT 对齐到 as_of 当日.

    as_of 是交易日，circ_mv 在 as_of 收盘已知 → PIT 安全。返回 dict[as_of] -> Series(ticker->log_mktcap).
    """
    db = pd.read_parquet(
        REPO_ROOT / "data" / "lake" / "daily_basic.parquet",
        columns=["ticker", "trade_date", "circ_mv"],
    )
    db["trade_date"] = pd.to_datetime(db["trade_date"]).dt.normalize()
    as_of_set = {pd.Timestamp(d).normalize() for d in as_of_grid}
    db = db[db["trade_date"].isin(as_of_set)]
    db = db[db["circ_mv"] > 0]
    db["log_mktcap"] = np.log(db["circ_mv"].astype(float))
    out = {}
    for d, sub in db.groupby("trade_date"):
        out[pd.Timestamp(d)] = sub.set_index("ticker")["log_mktcap"]
    return out


def build_regime(panel, as_of_grid):
    """market_momentum_60d（每 as_of 常量, PIT 安全）三档：bear/neutral/bull（OOS 期内 tercile）.

    regime 标签输入是 trailing 60d 市场动量(PIT 安全)；tercile 切点为 OOS 期描述性分桶（诊断用,
    非可交易决策）。返回 Series(as_of -> regime)。
    """
    mm = panel.groupby(level="as_of")["market_momentum_60d"].first()
    mv = panel.groupby(level="as_of")["market_volatility_60d"].first()
    oos = [pd.Timestamp(d) for d in as_of_grid if pd.Timestamp(d) >= OOS_START]
    mm_oos = mm.reindex(oos).dropna()
    q33, q67 = mm_oos.quantile([1 / 3, 2 / 3]).tolist()
    def lab(v):
        if v != v:
            return "unknown"
        if v <= q33:
            return "bear"
        if v >= q67:
            return "bull"
        return "neutral"
    reg = mm.reindex(oos).map(lab)
    return reg, {"q33": float(q33), "q67": float(q67),
                 "mm": mm.reindex(oos), "mv": mv.reindex(oos)}


def main():
    t0 = time.perf_counter()
    print(f"[load] panel={OUT_PATH}")
    panel = pd.read_parquet(OUT_PATH)
    cal = load_trading_calendar()
    as_of_grid = sorted(panel.index.get_level_values("as_of").unique())
    print(f"[load] panel{panel.shape} as_of n={len(as_of_grid)} "
          f"{as_of_grid[0].date()}→{as_of_grid[-1].date()}")

    OOS_END = as_of_grid[-1]
    splitter = PurgedWalkForwardSplit(
        cal, horizon=HORIZON, embargo=EMBARGO,
        mode="rolling", rolling_lookback_td=ROLLING_LOOKBACK, n_val=N_VAL,
    )
    cutoffs = make_quarterly_cutoffs(cal, OOS_START, pd.Timestamp("2025-10-01"))
    folds = splitter.make_folds(as_of_grid, cutoffs, oos_start=OOS_START, oos_end=OOS_END)
    print(f"[wf] cutoffs={len(cutoffs)} folds={len(folds)}")

    # 规模 & regime
    tickers = panel.index.get_level_values("ticker").unique()
    log_mktcap = load_log_mktcap(as_of_grid, tickers)
    print(f"[size] log_mktcap as_of-days covered: {len(log_mktcap)}")
    regime, reg_meta = build_regime(panel, as_of_grid)
    print(f"[regime] tercile mm cut: q33={reg_meta['q33']:+.4f} q67={reg_meta['q67']:+.4f}")
    print(f"[regime] counts: {regime.value_counts().to_dict()}")

    rows = []                 # 逐票 OOS 预测
    fold_dir = {}             # fold -> direction
    fold_gain = {}            # fold -> {feat: gain}
    fold_meta = []

    for i, fold in enumerate(folds):
        if not fold.test_dates or not fold.train_dates:
            continue
        pred = LGBMPredictor(n_quantiles=5, n_estimators=N_ESTIMATORS,
                             learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES, verbose=-1)
        pred.fit(panel, fold.train_dates, fold.val_dates, FEAT_COLS, LABEL_COL)
        direction = decide_direction(pred, panel, fold.val_dates, FEAT_COLS, LABEL_COL)
        fold_dir[i] = int(direction)

        # gain 重要性（直接读 booster，importance_type=gain）
        try:
            booster = pred._model._model.booster_
            gains = booster.feature_importance(importance_type="gain")
            names = booster.feature_name()
            fold_gain[i] = {n: float(g) for n, g in zip(names, gains)}
        except Exception as e:
            print(f"  [fold {i}] gain err: {e}")

        n_test_used = 0
        for d in fold.test_dates:
            d = pd.Timestamp(d)
            try:
                xs = panel.xs(d, level="as_of")
            except KeyError:
                continue
            score = pred.score(panel, d, FEAT_COLS) * direction  # 有效预测(已乘方向)
            realized = xs[LABEL_COL]
            ind = xs["exposure_industry"]
            lmc = log_mktcap.get(d, pd.Series(dtype=float))
            sub = pd.DataFrame({
                "pred": score,
                "realized": realized,
                "industry": ind.astype(object),
                "log_mktcap": lmc.reindex(xs.index),
            })
            sub = sub.dropna(subset=["pred", "realized"])
            if len(sub) < 10:
                continue
            sub["as_of"] = d
            sub["fold"] = i
            sub["regime"] = regime.get(d, "unknown")
            sub.index.name = "ticker"
            rows.append(sub.reset_index())
            n_test_used += 1
        fold_meta.append({"fold": i, "cutoff": str(fold.cutoff.date()),
                          "direction": int(direction),
                          "n_train": len(fold.train_dates),
                          "n_test_dates": n_test_used,
                          "test_first": str(pd.Timestamp(fold.test_dates[0]).date()),
                          "test_last": str(pd.Timestamp(fold.test_dates[-1]).date())})
        print(f"  [fold {i}] cutoff={fold.cutoff.date()} dir={direction:+d} "
              f"test_dates={n_test_used}  ({time.perf_counter()-t0:.0f}s)")

    oos = pd.concat(rows, ignore_index=True)
    oos_path = OUT_DIR / "wf_v2_oos_predictions.parquet"
    oos.to_parquet(oos_path)
    print(f"[save] {oos_path}  shape={oos.shape}")

    # =====================================================================
    # 诊断 1：per-(as_of) IC，按 fold × regime 聚合
    # =====================================================================
    def xs_ic(g):
        return spearman(g["pred"], g["realized"])
    per_date = oos.groupby("as_of").apply(
        lambda g: pd.Series({"ic": xs_ic(g), "regime": g["regime"].iloc[0],
                             "fold": g["fold"].iloc[0], "n": len(g)})
    ).reset_index()
    per_date["ic"] = per_date["ic"].astype(float)

    # IC × regime（总体）
    reg_ic = per_date.groupby("regime")["ic"].agg(
        ic_mean="mean", ic_std="std", n_dates="count",
        pos_frac=lambda s: float((s > 0).mean())).reset_index()
    # IC × regime × fold
    rf_ic = per_date.pivot_table(index="fold", columns="regime", values="ic", aggfunc="mean")
    # fold 总 IC + 方向翻转
    fold_ic = per_date.groupby("fold")["ic"].mean()
    overall_ic = float(per_date["ic"].mean())
    overall_ir = float(per_date["ic"].mean() / per_date["ic"].std())
    n_flip = sum(1 for v in fold_dir.values() if v < 0)

    # =====================================================================
    # 诊断 2：行业+规模中性化（逐 as_of 截面回归取残差）
    # =====================================================================
    def neutralize_date(g):
        d = g.dropna(subset=["pred", "realized", "industry", "log_mktcap"]).copy()
        if d["industry"].nunique() < 2 or len(d) < 30:
            return None
        D = pd.get_dummies(d["industry"], drop_first=True).astype(float)
        z = (d["log_mktcap"] - d["log_mktcap"].mean()) / (d["log_mktcap"].std() + 1e-12)
        X = np.column_stack([np.ones(len(d)), z.values, D.values])
        y = d["pred"].values.astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        raw_ic = spearman(pd.Series(y, index=d.index), d["realized"])
        neu_ic = spearman(pd.Series(resid, index=d.index), d["realized"])
        return pd.Series({"raw_ic": raw_ic, "neu_ic": neu_ic,
                          "regime": d["regime"].iloc[0] if "regime" in d else "?",
                          "n": len(d)})
    neu = oos.groupby("as_of").apply(neutralize_date).dropna(how="all")
    neu = neu.reset_index()
    for c in ("raw_ic", "neu_ic", "n"):
        neu[c] = neu[c].astype(float)
    neu_overall = {
        "raw_ic_mean": float(neu["raw_ic"].mean()),
        "neu_ic_mean": float(neu["neu_ic"].mean()),
        "raw_ic_ir": float(neu["raw_ic"].mean() / neu["raw_ic"].std()),
        "neu_ic_ir": float(neu["neu_ic"].mean() / neu["neu_ic"].std()),
        "retention_frac": float(neu["neu_ic"].mean() / neu["raw_ic"].mean())
        if neu["raw_ic"].mean() != 0 else float("nan"),
        "n_dates": int(len(neu)),
    }
    neu_by_regime = neu.groupby("regime")[["raw_ic", "neu_ic"]].mean().reset_index()

    # =====================================================================
    # 诊断 3：特征重要性（gain，跨 fold 平均）+ 单变量 IC
    # =====================================================================
    gain_df = pd.DataFrame(fold_gain).T  # fold × feat
    gain_mean = gain_df.mean().sort_values(ascending=False)
    gain_norm = (gain_mean / gain_mean.sum())

    # 单变量 IC：每个数值特征 vs realized（OOS 全期截面 Spearman 平均）
    NUM_FEATS = [c for c in FEAT_COLS if c not in ("exposure_industry", "exposure_area")]
    uni_ic = {}
    # 仅在 OOS as_of 上算，复用 panel
    oos_dates = sorted(per_date["as_of"].unique())
    for f in NUM_FEATS:
        ics = []
        for d in oos_dates:
            try:
                xs = panel.xs(pd.Timestamp(d), level="as_of")
            except KeyError:
                continue
            ic = spearman(xs[f], xs[LABEL_COL])
            if ic == ic:
                ics.append(ic)
        uni_ic[f] = float(np.mean(ics)) if ics else float("nan")
    uni_ic_s = pd.Series(uni_ic).sort_values(key=lambda s: s.abs(), ascending=False)

    # =====================================================================
    # 汇总 JSON
    # =====================================================================
    summary = {
        "params": {"horizon": HORIZON, "embargo": EMBARGO, "rolling_lookback": ROLLING_LOOKBACK,
                   "label": LABEL_COL, "n_features": len(FEAT_COLS),
                   "oos_start": str(OOS_START.date()), "oos_end": str(OOS_END.date()),
                   "n_folds": len(fold_meta)},
        "overall_ic": overall_ic, "overall_ir": overall_ir,
        "n_oos_dates": int(len(per_date)),
        "n_direction_flips": int(n_flip),
        "fold_directions": fold_dir,
        "fold_meta": fold_meta,
        "regime_meta": {"q33": reg_meta["q33"], "q67": reg_meta["q67"],
                        "counts": regime.value_counts().to_dict()},
        "regime_ic": reg_ic.to_dict(orient="records"),
        "regime_fold_ic": rf_ic.round(4).to_dict(orient="index"),
        "fold_ic": {int(k): round(float(v), 4) for k, v in fold_ic.items()},
        "neutralization_overall": neu_overall,
        "neutralization_by_regime": neu_by_regime.round(4).to_dict(orient="records"),
        "gain_importance_norm": {k: round(float(v), 4) for k, v in gain_norm.items()},
        "univariate_ic": {k: round(float(v), 4) for k, v in uni_ic_s.items()},
    }
    sjson = OUT_DIR / "wf_v2_diag_summary.json"
    sjson.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # 也存中间表
    per_date.to_csv(OUT_DIR / "wf_v2_per_date_ic.csv", index=False)
    neu.to_csv(OUT_DIR / "wf_v2_neutralization.csv", index=False)
    rf_ic.to_csv(OUT_DIR / "wf_v2_regime_fold_ic.csv")

    print("\n" + "=" * 60)
    print(f"overall OOS IC={overall_ic:+.4f} IR={overall_ir:+.3f} flips={n_flip}/{len(fold_dir)}")
    print("regime IC:\n", reg_ic.to_string(index=False))
    print(f"neutralization: raw={neu_overall['raw_ic_mean']:+.4f} "
          f"neu={neu_overall['neu_ic_mean']:+.4f} retention={neu_overall['retention_frac']:.2f}")
    print("top gain:\n", gain_norm.head(12).round(4).to_string())
    print("top |univariate IC|:\n", uni_ic_s.head(12).round(4).to_string())
    print(f"[done] {time.perf_counter()-t0:.0f}s  -> {sjson}")


if __name__ == "__main__":
    main()
