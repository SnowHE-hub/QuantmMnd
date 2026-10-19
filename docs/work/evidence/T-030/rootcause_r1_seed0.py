"""T-030 63d 根因清单 R1：固定置换种子 0 的第二代码路径复现（G1 2026-10-19 确认，允许执行一次）.

不是追加置换种子，也不是重新检验 N1-R：结果写独立目录，不替换原种子 0 的 T，不与之平均，不进入种子 1–15 的区间，
不授予 63d 主比较解封资格。原运行：`run_trees.py --shuffle-labels`（2026-10-17 13:12–13:36，preds/xgb_63d_N1shuffle）。
本次：`controls_diag.py n1-seed` 的训练 / 预测写法（它拒绝种子 0，所以这里照抄其循环体，置换种子固定为 0）。

逐层比较（G1 要求的四层）：
  数据与置换  训练行键的 sha256；两条路径的置换函数各算一遍、逐位比较；置换后标签的 sha256
  模型输入    特征列顺序 sha256；训练 / 预测矩阵 float32 字节的 sha256；参数
  训练与预测  复现模型文件与原模型文件的 sha256；原模型重新预测与原 .npy 的差；复现预测与原预测的差（逐股票）
  结果        两边按同一函数（controls_diag._meta + metrics.daily_rank_ic）算逐日 IC、各折均值、日期等权 T
输出：data/gpu_v1/preds/xgb_63d_N1s00_repro/、data/gpu_v1/runs/xgb_63d_N1s00_repro/、
      docs/work/evidence/T-030/p6/r1_seed0_repro.json
用法：PYTHONPATH=. python docs/work/evidence/T-030/rootcause_r1_seed0.py <data_root>
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAG_ORIG, TAG_REPRO = "xgb_63d_N1shuffle", "xgb_63d_N1s00_repro"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _matrix_sha(src) -> str:
    h = hashlib.sha256()
    for c in src.chunks():
        h.update(np.ascontiguousarray(c, dtype=np.float32).tobytes())
    return h.hexdigest()


def _keys_sha(meta: pd.DataFrame, rows: np.ndarray) -> str:
    m = meta.iloc[rows]
    return _sha(pd.to_datetime(m["as_of"]).to_numpy().astype("datetime64[ns]").astype(np.int64),
                np.array(m["ticker"].astype(str).tolist(), dtype="U12"))


def _perm_index(day: np.ndarray, seed: int, day_bounds) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(day))
    b = day_bounds(day)
    for s, e in zip(b[:-1], b[1:], strict=True):
        idx[s:e] = s + rng.permutation(e - s)
    return idx


def _daily_ics(root: Path, cd, folds, pred_dir: Path) -> list[np.ndarray]:
    from quantmind.research.gpu.metrics import daily_rank_ic
    out = []
    for k, f in enumerate(folds):
        meta = cd._meta(root, f.test_asof, ["as_of", "ticker", "y63", "top1500"])
        p = pd.read_parquet(pred_dir / f"fold_{k}.parquet")
        p["as_of"] = pd.to_datetime(p["as_of"])
        score = p.set_index(["as_of", "ticker"])["score"].reindex(
            pd.MultiIndex.from_frame(meta[["as_of", "ticker"]])).to_numpy()
        out.append(daily_rank_ic(score, meta["y63"].to_numpy(dtype=np.float64), meta["day"].to_numpy(),
                                 meta["top1500"].to_numpy()))
    return out


def main() -> None:
    root = Path(sys.argv[1])
    os.environ.setdefault("WRITE_MODE", "parquet_only")
    import xgboost as xgb

    from quantmind.research.gpu.configs import TREE_GRIDS
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.ensemble import rank_average
    from quantmind.research.gpu.folds import calendar_and_grid, check_folds, make_folds
    from quantmind.research.gpu.runtime import start_ram_guard, wait_for_vram
    from quantmind.research.gpu.transforms import day_bounds
    from quantmind.research.gpu.trees import XGB, TreeInput, train_target

    rt, cd = _load("run_trees"), _load("controls_diag")
    start_ram_guard()
    run_orig, run_rep = root / "data/gpu_v1/runs" / TAG_ORIG, root / "data/gpu_v1/runs" / TAG_REPRO
    pred_orig, pred_rep = root / "data/gpu_v1/preds" / TAG_ORIG, root / "data/gpu_v1/preds" / TAG_REPRO
    # G1 只允许跑一次：已写出的折一律不重算（进程被杀时只续跑缺的折，并记进报告），不按结果重跑
    run_rep.mkdir(parents=True, exist_ok=True)
    pred_rep.mkdir(parents=True, exist_ok=True)
    md_dir = root / "data/gpu_v1/model/v1"
    env = {
        "utc_start": datetime.now(UTC).isoformat(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__,
        "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip(),
        "features_json_sha256": _file_sha(md_dir / "features.json"),
        "market_parquet_sha256": _file_sha(md_dir / "market.parquet"),
        "model_v1_latest_mtime_utc": datetime.fromtimestamp(max(p.stat().st_mtime for p in md_dir.iterdir()),
                                                            UTC).isoformat(),
        "original_run_window_aedt": "2026-10-17 13:12-13:36 (queue item 09)",
    }
    cal, wgrid = calendar_and_grid(root)
    folds = make_folds(cal, wgrid, "63d")
    check_folds(folds, cal)
    md = ModelData(root)
    params = TREE_GRIDS["xgb"][0]
    feat_names = list(md.stock_features) + list(md.market_features)
    env["n_features"] = len(feat_names)
    env["feature_order_sha256"] = hashlib.sha256("\n".join(feat_names).encode()).hexdigest()
    per_fold, resumed = [], []
    for k, f in enumerate(folds):
        rec_path = run_rep / f"k{k:02d}_record.json"
        if (pred_rep / f"fold_{k}.parquet").exists() and rec_path.exists():
            per_fold.append(json.loads(rec_path.read_text(encoding="utf-8")))
            resumed.append(k)
            continue
        wait_for_vram(4000, log=lambda m: print(m, flush=True))
        full = md.load(f.train_days)
        rows, y = train_target(full, "63d", params["objective"])
        day = full.meta["day"].to_numpy()[rows]
        y_a = rt.shuffle_within_day(y, day, seed=0)              # 原路径的置换函数
        y_b = cd._shuffle_within_day(y, day, 0)                  # 第二路径的置换函数
        idx = _perm_index(day, 0, day_bounds)
        rec = {"fold": k, "n_train_rows": int(len(rows)), "n_train_days": int(len(np.unique(day))),
               "train_keys_sha256": _keys_sha(full.meta, rows),
               "perm_paths_identical": bool(np.array_equal(y_a, y_b, equal_nan=True)),
               "perm_index_reproduces_labels": bool(np.array_equal(y[idx], y_b, equal_nan=True)),
               "perm_index_sha256": _sha(idx), "y_shuffled_sha256": _sha(y_b)}
        np.save(run_rep / f"k{k:02d}_perm_idx.npy", idx.astype(np.int32))   # 每折每日的实际置换索引（G1 要求留存）
        X = TreeInput(md, full, rows)
        rec["train_matrix_sha256"] = _matrix_sha(X)
        del full
        gc.collect()
        model = XGB().fit(params, X, y_b, day, 500, seed=0)
        model.save_model(str(run_rep / f"k{k:02d}_s0.ubj"))
        del X
        gc.collect()
        pb = md.load(f.pred_days)
        Xp = TreeInput(md, pb)
        rec["pred_matrix_sha256"] = _matrix_sha(Xp)
        p_rep = XGB().predict(model, Xp, 500)
        orig_model = xgb.Booster(model_file=str(run_orig / f"k{k:02d}_s0.ubj"))
        p_orig_model = XGB().predict(orig_model, Xp, 500)
        p_orig = np.load(run_orig / f"k{k:02d}_s0.npy").astype(np.float64)
        np.save(run_rep / f"k{k:02d}_s0.npy", p_rep.astype(np.float32))
        rec.update({
            "model_file_sha256_repro": _file_sha(run_rep / f"k{k:02d}_s0.ubj"),
            "model_file_sha256_orig": _file_sha(run_orig / f"k{k:02d}_s0.ubj"),
            "orig_model_repredict_max_abs_diff_vs_orig_npy": float(np.max(np.abs(p_orig_model.astype(np.float32) - p_orig))),
            "repro_vs_orig_pred_max_abs_diff": float(np.max(np.abs(p_rep.astype(np.float32) - p_orig))),
            "repro_vs_orig_pred_exact_share": float(np.mean(p_rep.astype(np.float32) == p_orig.astype(np.float32))),
            "repro_vs_orig_pred_pearson": float(np.corrcoef(p_rep, p_orig)[0, 1]),
            "n_pred_rows": int(len(p_rep)),
        })
        rec["model_files_identical"] = rec["model_file_sha256_repro"] == rec["model_file_sha256_orig"]
        score = rank_average([p_rep], pb.meta["day"].to_numpy())
        frame = pd.DataFrame({"as_of": pb.meta["as_of"], "ticker": pb.meta["ticker"], "score": score,
                              "top1500": pb.meta["top1500"]})
        tmp = pred_rep / f"fold_{k}.tmp"
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, pred_rep / f"fold_{k}.parquet")
        rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        per_fold.append(rec)
        print(json.dumps({x: rec[x] for x in ("fold", "perm_paths_identical", "model_files_identical",
                                                 "repro_vs_orig_pred_max_abs_diff")}), flush=True)
        del model, orig_model, pb, Xp, frame
        gc.collect()

    ic_o, ic_r = _daily_ics(root, cd, folds, pred_orig), _daily_ics(root, cd, folds, pred_rep)
    flat_o, flat_r = np.concatenate([a for a in ic_o if len(a)]), np.concatenate([a for a in ic_r if len(a)])
    report = {
        "what": "G1 2026-10-19 授权的一次性根因排查：固定置换种子 0 的第二代码路径复现。不是追加种子，不进入 N1-R，"
                "不替换原种子 0 的 T，不与之平均，不授予 63d 主比较解封资格。",
        "env": env, "resumed_folds_from_earlier_process": resumed, "per_fold": per_fold,
        "results": {
            "T_orig": float(flat_o.mean()), "T_repro": float(flat_r.mean()), "n_dates": int(len(flat_o)),
            "daily_ic_max_abs_diff": float(np.max(np.abs(flat_o - flat_r))) if len(flat_o) == len(flat_r) else None,
            "fold_means_orig": [float(a.mean()) if len(a) else None for a in ic_o],
            "fold_means_repro": [float(a.mean()) if len(a) else None for a in ic_r],
        },
        "utc_end": datetime.now(UTC).isoformat(),
    }
    out = HERE / "p6/r1_seed0_repro.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["results"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
