"""T-030 P0：真实数据上的内存 / 显存 / 耗时实测（用于定预登记的算力预算）.

只用 20d 最后一折（股票最多）的**训练窗**；不预测任何外层日期、不计算任何 IC / 收益指标。
每个族在单独子进程里测（同一进程先建 torch CUDA 上下文会干扰 XGBoost / CatBoost 的显存统计）。
记录：数据装载耗时与 RSS、每轮 / 每 epoch 耗时、峰值显存（nvidia-smi 0.5 s 轮询；torch 另报 max_memory_allocated）。
用法：PYTHONPATH=. python p0_resource_probe.py <data_root> <out.json> [unit1,unit2,...]
"""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

UNITS = ["load", "xgb", "cat_rmse", "cat_yetirank", "lgbm", "tabm", "mlp", "seqfusion", "gru", "alstm", "master"]


def rss_peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class VramPoll:
    def __init__(self):
        self.peak, self.base, self._stop = 0, None, False

    def _read(self) -> int | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            return int(out.splitlines()[0])
        except Exception:  # noqa: BLE001
            return None

    def __enter__(self):
        self.base = self._read()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def _loop(self):
        while not self._stop:
            v = self._read()
            if v is not None:
                self.peak = max(self.peak, v)
            time.sleep(0.5)

    def __exit__(self, *a):
        self._stop = True
        self.t.join(timeout=5)


def fold_and_data(root: Path):
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.folds import calendar_and_grid, make_folds
    cal, grid = calendar_and_grid(root)
    f = make_folds(cal, grid, "20d")[-1]
    return f, ModelData(root)


def u_load(root: Path) -> dict:
    t0 = time.time()
    f, md = fold_and_data(root)
    b = md.load(f.train_days)
    t1 = time.time()
    y = b.label_rank("20d")
    t2 = time.time()
    return {"fold": f.k, "cutoff": str(f.cutoff.date()), "n_train_days": len(f.train_days), "rows": int(len(b.meta)),
            "stock_features": int(b.X.shape[1]), "market_features": len(md.market_features),
            "X_float16_mb": round(b.X.nbytes / 2**20), "load_s": round(t1 - t0, 1), "label_rank_s": round(t2 - t1, 1),
            "label_nonnull": int(np.isfinite(y).sum()), "rows_per_day": round(len(b.meta) / len(b.days))}


def tree_unit(root: Path, fam_name: str, params: dict, rounds: int, stride: int = 1) -> dict:
    from quantmind.research.gpu.trees import FAMILIES, TreeInput, train_target
    f, md = fold_and_data(root)
    b = md.load(f.train_days[::stride])
    rows, y = train_target(b, "20d", params["objective"])
    day = b.meta["day"].to_numpy()[rows]
    X = TreeInput(md, b, rows)
    rss_loaded = rss_peak_mb()
    fam = FAMILIES[fam_name]()
    with VramPoll() as vp:
        t1 = time.time()
        model = fam.fit(params, X, y, day, rounds, seed=0)
        fit_s = time.time() - t1
    t2 = time.time()
    fam.predict(model, TreeInput(md, b, rows[:200_000]), rounds)
    pred_s = time.time() - t2
    return {"rows": int(len(y)), "cols": int(X.n_cols), "day_stride": stride, "rss_after_load_mb": round(rss_loaded),
            "rounds": rounds, "fit_s": round(fit_s, 1),
            "s_per_round": round(fit_s / rounds, 3), "predict_200k_s": round(pred_s, 2),
            "vram_base_mb": vp.base, "vram_peak_mb": vp.peak}


def dl_unit(root: Path, arch: str) -> dict:
    import torch

    from quantmind.research.gpu import dl
    from quantmind.research.gpu.configs import DL_GRIDS
    from quantmind.research.gpu.nets import INPUTS
    from quantmind.research.gpu.runtime import cap_torch_memory
    cap_torch_memory()
    f, md = fold_and_data(root)
    cfg = dict(max(DL_GRIDS[arch], key=lambda c: (c.get("L", 1), c.get("d", 0), c.get("n_layers", c.get("n_blocks", 0)))))
    L = cfg.get("L", 1)
    seq = dl.SeqStore(root) if "seq" in INPUTS[arch] else None
    t0 = time.time()
    b = md.load(f.train_days)
    norm = dl.fit_norm(md, b, seq, L)
    seq_t = dl.seq_tensor(seq, norm) if seq is not None else None
    inp = dl.Inputs(md, b, norm, arch, "20d", seq)
    prep_s = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    cfg["max_epochs"] = 1
    with VramPoll() as vp:
        t1 = time.time()
        dl.train(arch, cfg, md, inp, None, seq_t, seq, seed=0,
                 ckpt=Path("/tmp") / f"t030_probe_{arch}.pt", epochs=1, log=lambda m: None)
        ep_s = time.time() - t1
    Path(f"/tmp/t030_probe_{arch}.pt").unlink(missing_ok=True)
    return {"config": cfg, "rows": int(inp.n), "prep_s": round(prep_s, 1), "epoch_s": round(ep_s, 1),
            "torch_peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "vram_base_mb": vp.base, "vram_peak_mb": vp.peak}


XGB_P = {"objective": "regression", "max_depth": 8, "colsample_bytree": 0.8, "eta": 0.05, "subsample": 0.8,
         "min_child_weight": 200.0, "reg_lambda": 10.0}
CAT_P = {"depth": 8, "l2_leaf_reg": 3.0, "learning_rate": 0.05, "bootstrap_type": "Bernoulli", "subsample": 0.8}
LGB_P = {"objective": "regression", "num_leaves": 255, "learning_rate": 0.05, "feature_fraction": 0.7,
         "bagging_fraction": 0.7, "bagging_freq": 1, "min_data_in_leaf": 500, "lambda_l2": 10.0}
FNS = {
    "load": u_load,
    "xgb": lambda r: tree_unit(r, "xgb", XGB_P, 100),
    "cat_rmse": lambda r: tree_unit(r, "cat", {"objective": "regression", **CAT_P}, 100),
    "cat_yetirank": lambda r: tree_unit(r, "cat", {"objective": "yetirank", **CAT_P}, 50),
    "lgbm": lambda r: tree_unit(r, "lgbm", LGB_P, 50),
    "cat_rmse_s2": lambda r: tree_unit(r, "cat", {"objective": "regression", **CAT_P}, 100, stride=2),
    "cat_yetirank_s2": lambda r: tree_unit(r, "cat", {"objective": "yetirank", **CAT_P}, 50, stride=2),
    "lgbm_s2": lambda r: tree_unit(r, "lgbm", LGB_P, 50, stride=2),
    "xgb_rank": lambda r: tree_unit(r, "xgb", {**XGB_P, "objective": "rank"}, 100),
    "xgb_300": lambda r: tree_unit(r, "xgb", XGB_P, 300),
    "xgb_rank_300": lambda r: tree_unit(r, "xgb", {**XGB_P, "objective": "rank"}, 300),
    **{a: (lambda r, a=a: dl_unit(r, a)) for a in ("tabm", "mlp", "seqfusion", "gru", "alstm", "master")},
}


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    if len(sys.argv) > 3 and sys.argv[3] in FNS:
        from quantmind.research.gpu.runtime import start_ram_guard
        start_ram_guard()
        name = sys.argv[3]
        t0 = time.time()
        try:
            r = FNS[name](root)
            r["ok"] = True
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-2000:]}
        r["seconds"] = round(time.time() - t0, 1)
        r["rss_peak_mb"] = round(rss_peak_mb())
        out.write_text(json.dumps(r, default=str), encoding="utf-8")
        return
    res = json.loads(out.read_text()) if out.exists() else {}
    units = sys.argv[3].split(",") if len(sys.argv) > 3 else UNITS
    for name in units:
        if res.get(name, {}).get("ok"):
            continue
        part = out.with_suffix(f".{name}.part")
        proc = subprocess.run([sys.executable, __file__, str(root), str(part), name], check=False)
        res[name] = json.loads(part.read_text()) if part.exists() else {"ok": False, "error": "subprocess died"}
        res[name]["returncode"] = proc.returncode
        part.unlink(missing_ok=True)
        print(name, json.dumps({k: v for k, v in res[name].items() if k not in ("trace", "config")}, default=str),
              flush=True)
        out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
