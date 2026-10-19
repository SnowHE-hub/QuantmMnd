"""T-034 A：20d 第 18 折（最后一折，切点 2026-07-01）的 ENS 四族模型包：重建、核对、存盘（设计见 docs/work/tasks/T-034.md）.

  XGBoost   复制已存的 k18_s{0..4}.ubj；对第 18 折预测日重新预测，与 k18_s*.npy 比较（事先写定：最大绝对差 = 0）
  TabM      从 ckpt/k18_final_s{0..4}.pt 加载；按 dl.fit_norm 用该折训练日重算归一化统计；重新预测并比较（≤ 1e-3 且相关 ≥ 0.9999）
  序列融合   同 TabM
  CatBoost  原运行没存模型：按 ledger/k18_selected.json 的配置、种子 0–4、隔日偏移（种子 % 2）、选定轮数重训并存 .cbm；
            CatBoost GPU 不保证逐位确定 → 报告与 k18_s*.npy 的相关与最大差（相关 < 0.99 单独标注）
最后用四族重建分数组成 ENS（与 eval_primary 相同的等权截面 rank 平均），与已存折预测合成的 ENS 比较。
输出：data/gpu_v1/forward_models/20d_fold18/<族>/…、manifest.json（文件 sha256、配置、核对结果、环境）。
用法：PYTHONPATH=. python docs/work/evidence/T-034/forward_bundle_fold18.py <data_root>
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

K, H = 18, "20d"
FAMS = ("xgb", "cat", "tabm", "seqfusion")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _cmp(new: np.ndarray, old: np.ndarray) -> dict:
    new, old = np.asarray(new, dtype=np.float64), np.asarray(old, dtype=np.float64)
    return {"n": int(len(new)), "max_abs_diff": float(np.max(np.abs(new - old))),
            "pearson": float(np.corrcoef(new, old)[0, 1]),
            "exact_share_f32": float(np.mean(new.astype(np.float32) == old.astype(np.float32)))}


def main() -> None:
    root = Path(sys.argv[1])
    os.environ.setdefault("WRITE_MODE", "parquet_only")
    import torch
    import xgboost as xgb

    from quantmind.research.gpu import dl
    from quantmind.research.gpu.configs import DL_SEEDS, TREE_SETTINGS
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.ensemble import rank_average, rank_average_frames
    from quantmind.research.gpu.folds import calendar_and_grid, make_folds
    from quantmind.research.gpu.nets import ARCHS
    from quantmind.research.gpu.runtime import start_ram_guard, wait_for_vram
    from quantmind.research.gpu.trees import CAT, XGB, TreeInput, train_target

    start_ram_guard()
    runs, preds = root / "data/gpu_v1/runs", root / "data/gpu_v1/preds"
    out = root / "data/gpu_v1/forward_models" / f"{H}_fold{K}"
    out.mkdir(parents=True, exist_ok=True)
    cal, grid = calendar_and_grid(root)
    f = make_folds(cal, grid, H)[K]
    md = ModelData(root)
    pb = md.load(f.pred_days)
    day = pb.meta["day"].to_numpy()
    man = {"what": __doc__.split("\n")[0], "fold": K, "horizon": H, "cutoff": str(f.cutoff) if hasattr(f, "cutoff") else None,
           "pred_days": [str(pd.Timestamp(min(f.pred_days)).date()), str(pd.Timestamp(max(f.pred_days)).date())],
           "n_pred_rows": int(len(pb.meta)), "utc_start": datetime.now(UTC).isoformat(),
           "env": {"xgboost": xgb.__version__, "torch": torch.__version__,
                   "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip()},
           "families": {}}
    scores = {}

    def selected(tag: str) -> dict:
        return json.loads((runs / tag / "ledger" / f"k{K:02d}_selected.json").read_text(encoding="utf-8"))

    # ---------------- XGBoost：复制模型，重新预测核对 ----------------
    sel = selected("xgb_20d")
    d = out / "xgb"
    d.mkdir(exist_ok=True)
    Xp = TreeInput(md, pb)
    seeds_pred, rec = [], {"selected": sel, "seeds": {}}
    for s in range(TREE_SETTINGS["xgb"]["seeds"]):
        src = runs / "xgb_20d" / f"k{K:02d}_s{s}.ubj"
        shutil.copy2(src, d / src.name)
        m = xgb.Booster(model_file=str(d / src.name))
        p = XGB().predict(m, Xp, sel["rounds"]).astype(np.float32)
        rec["seeds"][s] = {"file": src.name, "sha256": _sha(d / src.name),
                           **_cmp(p, np.load(runs / "xgb_20d" / f"k{K:02d}_s{s}.npy"))}
        seeds_pred.append(p)
    rec["check_pass"] = all(v["max_abs_diff"] == 0 for v in rec["seeds"].values())
    scores["xgb"] = rank_average(seeds_pred, day)
    man["families"]["xgb"] = rec
    print(json.dumps({"xgb": rec["check_pass"]}), flush=True)

    # ---------------- 深度模型：checkpoint 加载 + 重算归一化 ----------------
    for arch in ("tabm", "seqfusion"):
        tag = f"{arch}_{H}"
        sel = selected(tag)
        cfg = dict(sel["params"])
        L, dpb = cfg.get("L", 1), cfg["days_per_batch"]
        seq = dl.SeqStore(root) if "seq" in dl.INPUTS[arch] else None
        trb = md.load(f.train_days)
        norm = dl.fit_norm(md, trb, seq, L)
        del trb
        gc.collect()
        seq_t = dl.seq_tensor(seq, norm) if seq is not None else None
        inp = dl.Inputs(md, pb, norm, arch, H, seq)
        net_kw = {k: v for k, v in cfg.items() if k not in ("L", "days_per_batch", "lr", "weight_decay", "max_epochs",
                                                            "patience", "aux_mse", "day_stride")}
        d = out / arch
        d.mkdir(exist_ok=True)
        norm_file = d / "norm.npz"
        np.savez(norm_file, mkt_mu=norm.mkt.mu, mkt_sd=norm.mkt.sd,
                 **({"seq_mu": norm.seq_mu, "seq_sd": norm.seq_sd, "seq_lo": norm.seq_lo, "seq_hi": norm.seq_hi}
                    if norm.seq_mu is not None else {}))
        rec = {"selected": sel, "norm_file": norm_file.name, "norm_sha256": _sha(norm_file), "seeds": {}}
        seeds_pred = []
        for s in range(DL_SEEDS):
            wait_for_vram(1500, log=lambda m: print(m, flush=True))
            state = torch.load(runs / tag / "ckpt" / f"k{K:02d}_final_s{s}.pt", map_location="cpu", weights_only=False)
            model = ARCHS[arch](**dl.dims(md, arch, seq), **net_kw).cuda()
            model.load_state_dict(state["model"])
            p = dl.predict(model, inp, seq_t, L, dpb).astype(np.float32)
            wfile = d / f"k{K:02d}_s{s}_state.pt"
            torch.save(model.state_dict(), wfile)
            rec["seeds"][s] = {"file": wfile.name, "sha256": _sha(wfile), "ckpt_epoch": int(state["epoch"]),
                               **_cmp(p, np.load(runs / tag / f"k{K:02d}_s{s}.npy"))}
            seeds_pred.append(p)
            del model, state
            torch.cuda.empty_cache()
        rec["check_pass"] = all(v["max_abs_diff"] <= 1e-3 and v["pearson"] >= 0.9999 for v in rec["seeds"].values())
        scores[arch] = rank_average(seeds_pred, day)
        man["families"][arch] = rec
        print(json.dumps({arch: rec["check_pass"]}), flush=True)
        del inp, seq_t
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------- CatBoost：按选定配置重训并存盘 ----------------
    sel = selected("cat_20d")
    st = TREE_SETTINGS["cat"]
    stride = st.get("day_stride", 1)
    d = out / "cat"
    d.mkdir(exist_ok=True)
    rec = {"selected": sel, "seeds": {}, "note": "CatBoost GPU 不保证逐位确定；与原折预测比较相关与最大差"}
    seeds_pred = [None] * st["seeds"]
    Xp = TreeInput(md, pb)
    for par in sorted({s % stride for s in range(st["seeds"])}):
        full = md.load(list(f.train_days)[par::stride])
        rows, y = train_target(full, H, sel["params"]["objective"])
        dday = full.meta["day"].to_numpy()[rows]
        X = TreeInput(md, full, rows)
        del full
        gc.collect()
        for s in [s for s in range(st["seeds"]) if s % stride == par]:
            wait_for_vram(st["min_free_vram_mb"], log=lambda m: print(m, flush=True))
            m = CAT().fit(sel["params"], X, y, dday, sel["rounds"], seed=s)
            mfile = d / f"k{K:02d}_s{s}.cbm"
            m.save_model(str(mfile))
            p = CAT().predict(m, Xp, sel["rounds"]).astype(np.float32)
            rec["seeds"][s] = {"file": mfile.name, "sha256": _sha(mfile),
                               **_cmp(p, np.load(runs / "cat_20d" / f"k{K:02d}_s{s}.npy"))}
            seeds_pred[s] = p
            print(json.dumps({"cat_seed": s, "pearson": rec["seeds"][s]["pearson"]}), flush=True)
            del m
            gc.collect()
        del X
        gc.collect()
    rec["flag_low_corr"] = any(v["pearson"] < 0.99 for v in rec["seeds"].values())
    scores["cat"] = rank_average(seeds_pred, day)
    man["families"]["cat"] = rec

    # ---------------- ENS：重建 vs 已存折预测 ----------------
    key = pd.MultiIndex.from_arrays([pd.to_datetime(pb.meta["as_of"]), pb.meta["ticker"]])
    new = rank_average_frames({fm: pd.Series(scores[fm], index=key) for fm in FAMS})
    old_parts = {}
    for fm in FAMS:
        q = pd.read_parquet(preds / f"{fm}_{H}" / f"fold_{K}.parquet")
        q["as_of"] = pd.to_datetime(q["as_of"])
        old_parts[fm] = q.set_index(["as_of", "ticker"])["score"].reindex(key)
    old = rank_average_frames(old_parts)
    both = pd.concat([new.rename("new"), old.rename("old")], axis=1).dropna()
    man["ens_rebuilt_vs_stored"] = {"n": int(len(both)), "pearson": float(both["new"].corr(both["old"])),
                                    "spearman": float(both["new"].corr(both["old"], method="spearman")),
                                    "max_abs_diff": float((both["new"] - both["old"]).abs().max())}
    man["features_json_sha256"] = _sha(root / "data/gpu_v1/model/v1/features.json")
    man["utc_end"] = datetime.now(UTC).isoformat()
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"ens": man["ens_rebuilt_vs_stored"], "checks": {k: v.get("check_pass", None)
                                                                      for k, v in man["families"].items()},
                      "cat_low_corr": man["families"]["cat"]["flag_low_corr"]}), flush=True)


if __name__ == "__main__":
    main()
