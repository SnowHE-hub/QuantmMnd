"""T-030 P3–P5：深度模型族的嵌套训练（每折：内层选 (配置, epoch 数) → 全训练窗重训 5 个种子 → 预测该折全部 pred_days）.

网格在 quantmind/research/gpu/configs.py::DL_GRIDS（预登记）。归一化：内层用内层训练日拟合，最终用全训练窗拟合。
台账 data/gpu_v1/runs/<arch>_<h>/ledger/；每个 epoch 的 checkpoint 在 runs/<arch>_<h>/ckpt/（进程被杀从断点续跑）；
折预测 data/gpu_v1/preds/<arch>_<h>/fold_<k>.parquet（as_of, ticker, score = 种子当日 gauss-rank 均值, top1500）。
显存：启动每个单元前等空闲显存 ≥ DL_MIN_FREE_VRAM_MB（日更 Step7a 的 ollama 占着时等待，不杀进程）；进程内上限 6500 MiB；
CUDA OOM → 等显存后从 checkpoint 续跑，再 OOM 则 days_per_batch 减半（台账记 oom_retries / days_per_batch_used）。
用法：PYTHONPATH=. python run_dl.py <data_root> <arch> <20d|63d> [--folds 0,1] [--inner-only]
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
import torch

from quantmind.research.gpu import dl
from quantmind.research.gpu.configs import DL_GRIDS, DL_MIN_FREE_VRAM_MB, DL_SEEDS
from quantmind.research.gpu.data import ModelData
from quantmind.research.gpu.ensemble import rank_average
from quantmind.research.gpu.folds import calendar_and_grid, check_folds, make_folds
from quantmind.research.gpu.nets import INPUTS
from quantmind.research.gpu.runtime import (
    UnitLedger,
    cap_torch_memory,
    start_ram_guard,
    wait_for_vram,
)


def select(records: list[dict]) -> tuple[int, int, float]:
    best = None
    for r in records:
        if r.get("status") != "ok":
            continue
        for pt in r["curve"]:
            key = (pt["ic_mean"], -pt["epoch"], -r["config"])
            if best is None or key > best[0]:
                best = (key, r["config"], pt["epoch"], pt["ic_mean"])
    if best is None:
        raise RuntimeError("no successful inner config")
    return best[1], best[2], best[3]


def train_retry(arch, cfg, md, tr, va, seq_t, seq, seed, ckpt, log, epochs=None):
    """CUDA OOM（多半是别的进程占了显存）→ 释放、等空闲显存、从上一个 epoch 的 checkpoint 续跑；
    第二次起 days_per_batch 减半（最小 1）。返回 (model, curve, 实际用的 days_per_batch, OOM 次数)。"""
    c = dict(cfg)
    for attempt in range(4):
        try:
            model, curve = dl.train(arch, c, md, tr, va, seq_t, seq, seed=seed, ckpt=ckpt, epochs=epochs, log=log)
            return model, curve, c["days_per_batch"], attempt
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if attempt == 3:
                raise
            if attempt >= 1:
                c["days_per_batch"] = max(1, c["days_per_batch"] // 2)
            log(f"    CUDA OOM (attempt {attempt + 1}); days_per_batch={c['days_per_batch']}, waiting for VRAM")
            wait_for_vram(DL_MIN_FREE_VRAM_MB, log=log)
    raise AssertionError("unreachable")


def phase(md, seq, arch, cfg, horizon, train_days, other: dict):
    """该阶段的归一化（只用 train_days）、F6 张量与各块输入。"""
    L = cfg.get("L", 1)
    trb = md.load(train_days)
    norm = dl.fit_norm(md, trb, seq if "seq" in INPUTS[arch] else None, L)
    seq_t = dl.seq_tensor(seq, norm) if "seq" in INPUTS[arch] else None
    ins = {"train": dl.Inputs(md, trb, norm, arch, horizon, seq)}
    blocks = {"train": trb}
    for k, days in other.items():
        b = md.load(days)
        ins[k] = dl.Inputs(md, b, norm, arch, horizon, seq)
        blocks[k] = b
    return ins, blocks, seq_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("arch", choices=list(DL_GRIDS))
    ap.add_argument("horizon", choices=["20d", "63d"])
    ap.add_argument("--folds", default=None)
    ap.add_argument("--inner-only", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("WRITE_MODE", "parquet_only")
    start_ram_guard()
    cap_torch_memory()
    grid = DL_GRIDS[a.arch]
    tag = f"{a.arch}_{a.horizon}"
    run_dir = a.root / "data/gpu_v1/runs" / tag
    pred_dir = a.root / "data/gpu_v1/preds" / tag
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    led = UnitLedger(run_dir / "ledger")
    cal, wgrid = calendar_and_grid(a.root)
    folds = make_folds(cal, wgrid, a.horizon)
    check_folds(folds, cal)
    md = ModelData(a.root)
    seq = dl.SeqStore(a.root) if "seq" in INPUTS[a.arch] else None
    todo = [int(x) for x in a.folds.split(",")] if a.folds else range(len(folds))
    t0 = time.time()
    log = lambda m: print(f"[{tag}] {m} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    for k in todo:
        f = folds[k]
        if (pred_dir / f"fold_{k}.parquet").exists():
            continue
        pending = [ci for ci in range(len(grid)) if not led.done(f"k{k:02d}_inner_c{ci:02d}")]
        for ci in pending:
            wait_for_vram(DL_MIN_FREE_VRAM_MB, log=log)
            t1 = time.time()
            try:
                ins, _, seq_t = phase(md, seq, a.arch, grid[ci], a.horizon, f.inner_train_days, {"val": f.inner_val_days})
                _, curve, dpb_used, n_oom = train_retry(a.arch, grid[ci], md, ins["train"], ins["val"], seq_t, seq, 0,
                                                        run_dir / "ckpt" / f"k{k:02d}_inner_c{ci:02d}.pt", log)
                rec = {"status": "ok", "curve": curve, "days_per_batch_used": dpb_used, "oom_retries": n_oom}
            except Exception as e:  # noqa: BLE001
                rec = {"status": "failed", "error": repr(e), "trace": traceback.format_exc()[-2000:]}
            rec.update(config=ci, params=grid[ci], fold=k, seconds=round(time.time() - t1, 1),
                       peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20))
            led.mark(f"k{k:02d}_inner_c{ci:02d}", rec)
            best = max((p.get("ic_mean", np.nan) for p in rec.get("curve", [])), default=float("nan"))
            log(f"fold {k} inner c{ci} {rec['status']} best_ic={best:.4f} {rec['seconds']}s")
            ins = seq_t = None
            gc.collect()
            torch.cuda.empty_cache()
        if a.inner_only:
            continue
        ci, epochs, ic = select([led.get(f"k{k:02d}_inner_c{c:02d}") for c in range(len(grid))])
        led.mark(f"k{k:02d}_selected", {"config": ci, "epochs": epochs, "inner_ic": ic, "params": grid[ci]})
        log(f"fold {k} selected c{ci} epochs={epochs} inner_ic={ic:.4f}")
        seeds = [s for s in range(DL_SEEDS) if not led.done(f"k{k:02d}_final_s{s}")]
        ins, blocks, seq_t = phase(md, seq, a.arch, grid[ci], a.horizon, f.train_days, {"pred": f.pred_days})
        L, dpb = grid[ci].get("L", 1), grid[ci]["days_per_batch"]
        for s in seeds:
            wait_for_vram(1000, log=log)
            t1 = time.time()
            model, curve, dpb_used, n_oom = train_retry(a.arch, grid[ci], md, ins["train"], None, seq_t, seq, s,
                                                        run_dir / "ckpt" / f"k{k:02d}_final_s{s}.pt", log, epochs=epochs)
            p = dl.predict(model, ins["pred"], seq_t, L, dpb)
            np.save(run_dir / f"k{k:02d}_s{s}.npy", p)
            led.mark(f"k{k:02d}_final_s{s}", {"config": ci, "epochs": epochs, "seed": s, "curve": curve,
                                              "days_per_batch_used": dpb_used, "oom_retries": n_oom,
                                              "seconds": round(time.time() - t1, 1),
                                              "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)})
            log(f"fold {k} final seed {s} {time.time() - t1:.0f}s")
            del model
            torch.cuda.empty_cache()
        pb = blocks["pred"]
        preds = [np.load(run_dir / f"k{k:02d}_s{s}.npy") for s in range(DL_SEEDS)]
        score = rank_average(preds, pb.meta["day"].to_numpy())
        out = pd.DataFrame({"as_of": pb.meta["as_of"], "ticker": pb.meta["ticker"], "score": score,
                            "top1500": pb.meta["top1500"]})
        tmp = pred_dir / f"fold_{k}.tmp"
        out.to_parquet(tmp, index=False)
        os.replace(tmp, pred_dir / f"fold_{k}.parquet")
        log(f"fold {k} written {out.shape}")
        ins = blocks = seq_t = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
