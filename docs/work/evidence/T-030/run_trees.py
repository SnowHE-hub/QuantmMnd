"""T-030 P2 / P6：树模型族的嵌套训练（每折：内层选 (配置, 轮数) → 全训练窗按选定配置重训 S 个种子 → 预测该折全部 pred_days）.

配置网格与种子数在 quantmind/research/gpu/configs.py（预登记）。外层样本外不参与任何选择。
CatBoost / LightGBM 隔 day_stride 个交易日取 1 天训练（内存与耗时减半；最终种子的偏移 = 种子 % day_stride，合起来覆盖全部交易日）。
台账：data/gpu_v1/runs/<tag>/ledger/*.json（每个内层配置、每个最终种子一个单元；存在即跳过 → 断点续跑）；
失败的配置也写进台账（status=failed）。种子预测：runs/<tag>/k<k>_s<s>.npy（XGBoost 另存模型 .ubj，供负对照 N3）；
折预测：data/gpu_v1/preds/<tag>/fold_<k>.parquet（as_of, ticker, score = 种子当日 gauss-rank 均值, top1500）。
消融 / 负对照（预登记 §7–§8）：
  --groups F1,F2      只用这些股票级特征组（市场级特征保留）；tag 加 _g<组>
  --weekly-train      训练行只取 v6_1 周频网格日（内层验证不变）；tag 加 _weekly
  --shuffle-labels    N1：网格 c0、500 轮、种子 0、不做内层选择，训练标签在每个交易日内随机置换；tag 加 _N1shuffle
用法：PYTHONPATH=. python run_trees.py <data_root> <xgb|cat|lgbm> <20d|63d> [--folds 0,1] [--inner-only] [选项]
"""
from __future__ import annotations

import argparse
import gc
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.research.gpu.configs import TREE_GRIDS, TREE_SETTINGS
from quantmind.research.gpu.data import Block, ModelData
from quantmind.research.gpu.ensemble import rank_average
from quantmind.research.gpu.folds import calendar_and_grid, check_folds, make_folds
from quantmind.research.gpu.runtime import UnitLedger, start_ram_guard, wait_for_vram
from quantmind.research.gpu.transforms import day_bounds
from quantmind.research.gpu.trees import FAMILIES, TreeInput, inner_curve, train_target

N1_ROUNDS = 500


def select(records: list[dict]) -> tuple[int, int, float]:
    """日均 RankIC 最高的 (配置, 轮数)；并列取轮数少、配置序号小的。"""
    best = None
    for r in records:
        if r.get("status") != "ok":
            continue
        for pt in r["curve"]:
            key = (pt["ic_mean"], -pt["rounds"], -r["config"])
            if best is None or key > best[0]:
                best = (key, r["config"], pt["rounds"], pt["ic_mean"])
    if best is None:
        raise RuntimeError("no successful inner config")
    return best[1], best[2], best[3]


def shuffle_within_day(y: np.ndarray, day: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = y.copy()
    b = day_bounds(day)
    for s, e in zip(b[:-1], b[1:], strict=True):
        out[s:e] = rng.permutation(out[s:e])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("horizon", choices=["20d", "63d"])
    ap.add_argument("--folds", default=None)
    ap.add_argument("--inner-only", action="store_true")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--weekly-train", action="store_true")
    ap.add_argument("--shuffle-labels", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("WRITE_MODE", "parquet_only")
    start_ram_guard()
    fam, grid, st = FAMILIES[a.family](), TREE_GRIDS[a.family], dict(TREE_SETTINGS[a.family])
    tag = f"{a.family}_{a.horizon}"
    if a.groups:
        tag += "_g" + a.groups.replace(",", "")
    if a.weekly_train:
        tag += "_weekly"
    if a.shuffle_labels:
        tag += "_N1shuffle"
        grid, st["seeds"] = grid[:1], 1
    run_dir = a.root / "data/gpu_v1/runs" / tag
    pred_dir = a.root / "data/gpu_v1/preds" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    led = UnitLedger(run_dir / "ledger")
    cal, wgrid = calendar_and_grid(a.root)
    folds = make_folds(cal, wgrid, a.horizon)
    check_folds(folds, cal)
    md = ModelData(a.root)
    cols = md.group_columns(a.groups.split(",")) if a.groups else None
    weekly = set(wgrid)
    keep = (lambda days: [d for d in days if d in weekly]) if a.weekly_train else (lambda days: days)  # noqa: E731
    todo = [int(x) for x in a.folds.split(",")] if a.folds else range(len(folds))
    stride = st.get("day_stride", 1)
    t0 = time.time()
    log = lambda m: print(f"[{tag}] {m} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731

    def load(days) -> Block:
        b = md.load(days, cols)
        return b

    for k in todo:
        f = folds[k]
        if (pred_dir / f"fold_{k}.parquet").exists():
            continue
        if a.shuffle_labels:
            ci, rounds, ic = 0, N1_ROUNDS, float("nan")
        else:
            pending = [ci for ci in range(len(grid)) if not led.done(f"k{k:02d}_inner_c{ci:02d}")]
            if pending:
                tr, va = load(keep(f.inner_train_days)[::stride]), load(f.inner_val_days)
                for ci in pending:
                    if st["gpu"]:
                        wait_for_vram(st["min_free_vram_mb"], log=log)
                    t1 = time.time()
                    try:
                        curve = inner_curve(fam, grid[ci], md, tr, va, a.horizon, st["max_rounds"], st["eval_every"])
                        rec = {"status": "ok", "curve": curve}
                    except Exception as e:  # noqa: BLE001
                        rec = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-2000:]}
                    rec.update(config=ci, params=grid[ci], fold=k, n_train_days=len(tr.days),
                               seconds=round(time.time() - t1, 1))
                    led.mark(f"k{k:02d}_inner_c{ci:02d}", rec)
                    best = max((p["ic_mean"] for p in rec.get("curve", [])), default=float("nan"))
                    log(f"fold {k} inner c{ci} {rec['status']} best_ic={best:.4f} {rec['seconds']}s")
                    gc.collect()
                del tr, va
                gc.collect()
            if a.inner_only:
                continue
            ci, rounds, ic = select([led.get(f"k{k:02d}_inner_c{c:02d}") for c in range(len(grid))])
        led.mark(f"k{k:02d}_selected", {"config": ci, "rounds": rounds, "inner_ic": ic, "params": grid[ci]})
        log(f"fold {k} selected c{ci} rounds={rounds} inner_ic={ic:.4f}")
        seeds = [s for s in range(st["seeds"]) if not led.done(f"k{k:02d}_final_s{s}")]
        pb = load(f.pred_days)
        Xp = TreeInput(md, pb)
        for par in sorted({s % stride for s in seeds}):        # 隔 stride 天取一天，偏移 = 种子 % stride
            full = load(keep(f.train_days)[par::stride])
            rows, y = train_target(full, a.horizon, grid[ci]["objective"])
            day = full.meta["day"].to_numpy()[rows]
            if a.shuffle_labels:
                y = shuffle_within_day(y, day, seed=0)
            X = TreeInput(md, full, rows)
            n_days = len(full.days)
            del full
            for s in [s for s in seeds if s % stride == par]:
                if st["gpu"]:
                    wait_for_vram(st["min_free_vram_mb"], log=log)
                t1 = time.time()
                model = fam.fit(grid[ci], X, y, day, rounds, seed=s)
                p = fam.predict(model, Xp, rounds).astype(np.float32)
                np.save(run_dir / f"k{k:02d}_s{s}.npy", p)
                if a.family == "xgb":
                    model.save_model(str(run_dir / f"k{k:02d}_s{s}.ubj"))
                led.mark(f"k{k:02d}_final_s{s}", {"config": ci, "rounds": rounds, "seed": s, "n_train": int(len(y)),
                                                  "n_train_days": n_days, "day_stride": stride, "day_offset": par,
                                                  "seconds": round(time.time() - t1, 1)})
                log(f"fold {k} final seed {s} {time.time() - t1:.0f}s")
                del model
                gc.collect()
            del X
            gc.collect()
        del Xp
        preds = [np.load(run_dir / f"k{k:02d}_s{s}.npy") for s in range(st["seeds"])]
        score = rank_average(preds, pb.meta["day"].to_numpy())
        out = pd.DataFrame({"as_of": pb.meta["as_of"], "ticker": pb.meta["ticker"], "score": score,
                            "top1500": pb.meta["top1500"]})
        tmp = pred_dir / f"fold_{k}.tmp"
        out.to_parquet(tmp, index=False)
        os.replace(tmp, pred_dir / f"fold_{k}.parquet")
        log(f"fold {k} written {out.shape}")
        del pb
        gc.collect()


if __name__ == "__main__":
    main()
