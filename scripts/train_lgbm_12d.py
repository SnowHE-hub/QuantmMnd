#!/usr/bin/env python
"""scripts/train_lgbm_12d.py — 用 forward_return_12d 训练 LGBMRanker 参考模型.

特征：WHITELIST_35 全部 35 列（含 exposure_industry / exposure_area）
标签：forward_return_12d
切分：与 WF 一致（H=12, E=20, rolling lookback=756）
方向：auto_flip=True（按 val IC 自动定方向）
产物：models/lgbm_12d_v1.pkl（pickle 整个 LGBMRankerModel 实例）

训练方式：取 OOS 起始之前的最后一个 cutoff 训练一次，作为可复现的参考模型
（非 WF 每 fold 重训）。WF 评估中各 fold 独立重训（见 run_wf_full_v2.py）。

用法：python scripts/train_lgbm_12d.py
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quantmind.data.lake import load_trading_calendar
from quantmind.features.weekly_panel import WHITELIST_35, OUT_PATH
from quantmind.models.factor_model import (
    build_lgbm_arrays, detect_categorical_cols,
)
from quantmind.models.lgbm_ranker import LGBMRankerModel
from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.backtest.wf_metrics import cross_sectional_ic

# ============================================================================
# 参数（与 WF 评估一致）
# ============================================================================
HORIZON = 12
EMBARGO = 20
ROLLING_LOOKBACK = 756
N_VAL = 2
LABEL_COL = "forward_return_12d"
# OOS 起点：用 2022-01-04（与 run_wf_full_v1 第一个 cutoff 一致）
OOS_START = pd.Timestamp("2022-01-04")

MODELS_DIR = REPO_ROOT / "models"
OUT_PKL = MODELS_DIR / "lgbm_12d_v1.pkl"


def main() -> int:
    print("=" * 72)
    print("Train LGBMRanker on forward_return_12d (35 features incl. categorical)")
    print(f"H={HORIZON} E={EMBARGO} rolling_lookback={ROLLING_LOOKBACK} n_val={N_VAL}")
    print(f"OOS cutoff (train ends here): {OOS_START.date()}")
    print("=" * 72)

    # ---- 加载面板 ----
    print(f"[load] {OUT_PATH}")
    panel = pd.read_parquet(OUT_PATH)
    cal = load_trading_calendar()
    as_of_grid = sorted(panel.index.get_level_values("as_of").unique())
    print(f"[load] shape={panel.shape}  n_asof={len(as_of_grid)}")

    feat_cols = list(WHITELIST_35)  # 全部 35 列（含 categorical）
    cat_cols = detect_categorical_cols(panel, feat_cols)
    print(f"[load] features={len(feat_cols)}  categorical={cat_cols}")

    # ---- 切分（取 cutoff = OOS_START 一个 fold）----
    splitter = PurgedWalkForwardSplit(
        cal, horizon=HORIZON, embargo=EMBARGO,
        mode="rolling", rolling_lookback_td=ROLLING_LOOKBACK, n_val=N_VAL,
    )
    fold = splitter.split(as_of_grid, OOS_START)
    print(f"[fold] cutoff={fold.cutoff.date()}  n_train={len(fold.train_dates)}  "
          f"n_val={len(fold.val_dates)}")
    print(f"[fold] train: {fold.train_dates[0].date()} → {fold.train_dates[-1].date()}")
    print(f"[fold] val:   {fold.val_dates[0].date()} → {fold.val_dates[-1].date()}")

    # ---- 构造数组 ----
    train_core = [d for d in fold.train_dates if d not in set(fold.val_dates)]
    print(f"[arrays] building...  train_core={len(train_core)} dates  val={len(fold.val_dates)}")
    X, y, g = build_lgbm_arrays(panel, train_core, feat_cols, LABEL_COL, 5)
    Xv, yv, gv = build_lgbm_arrays(panel, fold.val_dates, feat_cols, LABEL_COL, 5)
    print(f"[arrays] X={X.shape}  y={y.shape}  groups={g.shape}")
    print(f"[arrays] Xv={Xv.shape}  yv={yv.shape}  groups_val={gv.shape}")

    # ---- 训练 ----
    print(f"[train] auto_flip=True  n_estimators=400  lr=0.05")
    model = LGBMRankerModel(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        auto_flip=True, verbose=-1,
    )
    model.fit(X, y, groups=g, X_val=Xv, y_val=yv, groups_val=gv,
              feature_names=feat_cols, categorical_feature=cat_cols)
    print(f"[train] best_iteration={model.best_iteration_}")

    # ---- val IC ----
    print(f"[val IC] 计算 val 段 IC 均值（用于 auto_flip 方向决策）...")
    val_ics = []
    for d in fold.val_dates:
        try:
            xs_full = panel.xs(d, level="as_of")
            from quantmind.models.factor_model import encode_categorical_codes
            xs_enc = encode_categorical_codes(panel, cat_cols).xs(d, level="as_of") if cat_cols else xs_full
            import numpy as np
            X_d = xs_enc[feat_cols].fillna(0.0).values.astype(np.float32)
            scores = model.predict(X_d)
            realized = xs_full[LABEL_COL]
            pred_ser = pd.Series(scores, index=xs_enc.index)
            ic = cross_sectional_ic(pred_ser, realized)
            if ic == ic:
                val_ics.append(ic)
        except Exception as e:
            print(f"  [warn] val IC {d}: {e}")
    val_ic_mean = sum(val_ics) / len(val_ics) if val_ics else float("nan")
    print(f"[val IC] mean={val_ic_mean:+.4f}  n={len(val_ics)}")
    # 应用 auto_flip
    model.set_direction(val_ic_mean)
    print(f"[direction] auto_flip → direction={model.direction:+d}")

    # ---- 特征重要性 ----
    imp = model.feature_importance("gain")
    top10 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n[top-10 feature importance by gain]")
    for n, v in top10:
        print(f"  {n:40s}  {v:.4f}")

    # ---- 保存 ----
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_cols": feat_cols,
            "categorical_cols": cat_cols,
            "label_col": LABEL_COL,
            "cutoff": str(fold.cutoff.date()),
            "n_train_dates": len(train_core),
            "n_val_dates": len(fold.val_dates),
            "val_ic_mean": val_ic_mean,
            "direction": model.direction,
            "best_iteration": model.best_iteration_,
            "params": {
                "horizon": HORIZON, "embargo": EMBARGO,
                "rolling_lookback_td": ROLLING_LOOKBACK,
                "n_estimators": 400, "lr": 0.05, "num_leaves": 31,
                "auto_flip": True,
            },
        }, f)
    print(f"\n💾 模型已保存: {OUT_PKL}")
    print(f"   尺寸: {OUT_PKL.stat().st_size/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
