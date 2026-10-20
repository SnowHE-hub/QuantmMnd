"""quantmind.line_a.scoring — 冻结模型包的前瞻打分（F3；03_final_plan_v2 第 5 节）.

ENS 20d：XGBoost、CatBoost、TabM、序列融合各 5 个种子，先按种子做当日截面排名平均，再四族等权排名平均
（与 T-030 的 ensemble.rank_average / rank_average_frames 相同）。模型、归一化统计和选定配置全部来自
T-034 冻结的模型包（data/gpu_v1/forward_models/20d_fold18/），不训练、不重算归一化。
GPU 部分在 GpuLease 内运行。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FAMS = ("xgb", "cat", "tabm", "seqfusion")
NON_NET_KEYS = ("L", "days_per_batch", "lr", "weight_decay", "max_epochs", "patience", "aux_mse", "day_stride")


def load_norm(npz_path: Path):
    from quantmind.research.gpu.dl import Norm
    from quantmind.research.gpu.transforms import TrainZScore
    z = np.load(npz_path)
    mkt = TrainZScore()
    mkt.mu, mkt.sd = z["mkt_mu"], z["mkt_sd"]
    norm = Norm(mkt=mkt)
    if "seq_mu" in z:
        norm.seq_mu, norm.seq_sd, norm.seq_lo, norm.seq_hi = z["seq_mu"], z["seq_sd"], z["seq_lo"], z["seq_hi"]
    return norm


def score_ens20(root: Path, days, bundle: Path, *, horizon: str = "20d", fold: int = 18,
                lease_mib: int = 3000) -> tuple[pd.DataFrame, dict]:
    """返回 (逐行分数表, 每族每种子的原始预测)。分数表索引 (as_of, ticker)，列为四族与 ENS。"""
    import torch
    import xgboost as xgb
    from catboost import CatBoost

    from quantmind.line_a.gpu_lease import GpuLease
    from quantmind.research.gpu import dl
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.ensemble import rank_average, rank_average_frames
    from quantmind.research.gpu.nets import ARCHS
    from quantmind.research.gpu.trees import CAT, XGB, TreeInput

    man = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    md = ModelData(root)
    pb = md.load(list(days))
    day = pb.meta["day"].to_numpy()
    key = pd.MultiIndex.from_arrays([pd.to_datetime(pb.meta["as_of"]), pb.meta["ticker"]], names=["as_of", "ticker"])
    raw: dict[str, list[np.ndarray]] = {}
    with GpuLease(holder=f"line_a F3 ens{horizon} fold{fold}", need_mib=lease_mib):
        Xp = TreeInput(md, pb)
        sel = man["families"]["xgb"]["selected"]
        raw["xgb"] = []
        for s in sorted(man["families"]["xgb"]["seeds"], key=int):
            m = xgb.Booster(model_file=str(bundle / "xgb" / man["families"]["xgb"]["seeds"][s]["file"]))
            raw["xgb"].append(XGB().predict(m, Xp, sel["rounds"]).astype(np.float32))
        sel = man["families"]["cat"]["selected"]
        raw["cat"] = []
        for s in sorted(man["families"]["cat"]["seeds"], key=int):
            m = CatBoost()
            m.load_model(str(bundle / "cat" / man["families"]["cat"]["seeds"][s]["file"]))
            raw["cat"].append(CAT().predict(m, Xp, sel["rounds"]).astype(np.float32))
        del Xp
        for arch in ("tabm", "seqfusion"):
            rec = man["families"][arch]
            cfg = dict(rec["selected"]["params"])
            L, dpb = cfg.get("L", 1), cfg["days_per_batch"]
            seq = dl.SeqStore(root) if "seq" in dl.INPUTS[arch] else None
            norm = load_norm(bundle / arch / rec["norm_file"])
            seq_t = dl.seq_tensor(seq, norm) if seq is not None else None
            inp = dl.Inputs(md, pb, norm, arch, horizon, seq)
            net_kw = {k: v for k, v in cfg.items() if k not in NON_NET_KEYS}
            raw[arch] = []
            for s in sorted(rec["seeds"], key=int):
                model = ARCHS[arch](**dl.dims(md, arch, seq), **net_kw).cuda()
                model.load_state_dict(torch.load(bundle / arch / rec["seeds"][s]["file"], map_location="cuda"))
                raw[arch].append(dl.predict(model, inp, seq_t, L, dpb).astype(np.float32))
                del model
                torch.cuda.empty_cache()
            del inp, seq_t
            torch.cuda.empty_cache()
    fam = {f: pd.Series(rank_average(raw[f], day), index=key) for f in FAMS}
    out = pd.DataFrame(fam)
    out["ENS"] = rank_average_frames(fam)
    return out, raw
