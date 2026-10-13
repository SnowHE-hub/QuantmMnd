"""T-030 预登记的配置网格（每族上限见设计稿 §4；本文件在看任何外层结果前随预登记一起提交，之后不得修改）."""
from __future__ import annotations

from itertools import product

_XGB_COMMON = {"eta": 0.05, "subsample": 0.8, "min_child_weight": 200.0, "reg_lambda": 10.0}
# 学习率与网格规模按 P0 实测耗时定（docs/work/evidence/T-030/p0/resource_probe_v3.json），没有看任何 IC
_CAT_COMMON = {"learning_rate": 0.08, "l2_leaf_reg": 3.0, "bootstrap_type": "Bernoulli", "subsample": 0.8}
_LGB_COMMON = {"learning_rate": 0.1, "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
               "min_data_in_leaf": 500, "lambda_l2": 10.0}

TREE_GRIDS: dict[str, list[dict]] = {
    "xgb": [{"objective": o, "max_depth": d, "colsample_bytree": c, **_XGB_COMMON}
            for o, d, c in product(("regression", "rank"), (6, 8), (0.5, 0.8))],                       # 8 ≤ 16
    "cat": [{"objective": o, "depth": d, **_CAT_COMMON}
            for o, d in product(("regression", "yetirank"), (6, 8))],                                   # 4 ≤ 16
    "lgbm": [{"objective": o, "num_leaves": n, **_LGB_COMMON}
             for o, n in product(("regression", "rank"), (63, 255))],                                   # 4 ≤ 8
}

_DL_COMMON = {"days_per_batch": 2, "weight_decay": 1e-4, "aux_mse": 0.1, "patience": 3, "day_stride": 2}

DL_GRIDS: dict[str, list[dict]] = {
    "tabm": [{"k": 32, "d": 256, "n_blocks": 2, "dropout": 0.1, "lr": 1e-3, "max_epochs": 12, **_DL_COMMON},
             {"k": 32, "d": 512, "n_blocks": 3, "dropout": 0.25, "lr": 5e-4, "max_epochs": 12, **_DL_COMMON},
             {"k": 32, "d": 256, "n_blocks": 3, "dropout": 0.25, "lr": 1e-3, "max_epochs": 12, **_DL_COMMON}],  # 3 ≤ 16
             # TabM 的 k 个成员共用一个 batch，显存约为 MLP 的 k 倍 → 每 batch 1 天（见下方覆盖）
    "mlp": [{"d": 512, "n_blocks": 3, "dropout": 0.25, "lr": 5e-4, "max_epochs": 12, **_DL_COMMON}],              # 1（TabM 对照）
    "seqfusion": [{"L": 64, "d": 64, "n_layers": 6, "d_snap": 64, "dropout": 0.1, "lr": 1e-3, "max_epochs": 10, **_DL_COMMON},
                  {"L": 128, "d": 64, "n_layers": 7, "d_snap": 64, "dropout": 0.1, "lr": 1e-3, "max_epochs": 10, **_DL_COMMON},
                  {"L": 64, "d": 128, "n_layers": 6, "d_snap": 128, "dropout": 0.2, "lr": 5e-4, "max_epochs": 10,
                   **_DL_COMMON}],                                                                                   # 3 ≤ 16
    "gru": [{"L": 64, "d": 64, "n_layers": 2, "dropout": 0.1, "lr": 1e-3, "max_epochs": 10, **_DL_COMMON}],         # GRU+ALSTM 2 ≤ 8
    "alstm": [{"L": 64, "d": 64, "n_layers": 2, "dropout": 0.1, "lr": 1e-3, "max_epochs": 10, **_DL_COMMON}],
    "master": [{"L": 64, "d": 64, "heads": 4, "dropout": 0.1, "lr": 5e-4, "max_epochs": 10, **_DL_COMMON},
               {"L": 64, "d": 128, "heads": 4, "dropout": 0.2, "lr": 3e-4, "max_epochs": 10, **_DL_COMMON}],      # 2 ≤ 8
}
for _c in DL_GRIDS["tabm"]:
    _c["days_per_batch"] = 1
DL_SEEDS = 5
DL_MIN_FREE_VRAM_MB = 4500

TREE_SETTINGS: dict[str, dict] = {
    "xgb": {"max_rounds": 1500, "eval_every": 50, "seeds": 5, "gpu": True, "min_free_vram_mb": 4000},
    "cat": {"max_rounds": 1500, "eval_every": 100, "seeds": 5, "gpu": True, "min_free_vram_mb": 4500, "day_stride": 2},
    "lgbm": {"max_rounds": 600, "eval_every": 50, "seeds": 3, "gpu": False, "min_free_vram_mb": 0, "day_stride": 2},
}
