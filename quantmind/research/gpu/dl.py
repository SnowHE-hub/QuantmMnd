"""深度模型的数据组装与训练循环（GPU）.

- 归一化只用该折训练日：市场特征按日等权 z-score；F6 每个通道用训练窗内（含 L 天预热）的 0.5%/99.5% 截断后均值 / 标准差。
- 一个 batch = 若干个完整交易日；损失 = 当日 IC 损失 + aux_mse × MSE（目标为当日 gauss-rank 标签）；TabM 的 k 个成员各算损失后平均。
- BF16 autocast、AdamW、按 max_epochs 定义的余弦学习率、梯度裁剪 1.0；每个 epoch 存 checkpoint（进程被杀可续跑）。
- 内层：每个 epoch 在内层验证日（前 1500 池）上算日均 RankIC，patience 个 epoch 不升即停；最终训练跑到内层选定的 epoch 数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from quantmind.research.gpu.batching import day_batches
from quantmind.research.gpu.data import Block, ModelData
from quantmind.research.gpu.losses import daily_ic_loss
from quantmind.research.gpu.metrics import daily_rank_ic
from quantmind.research.gpu.nets import ARCHS, INPUTS
from quantmind.research.gpu.runtime import load_checkpoint, save_checkpoint
from quantmind.research.gpu.transforms import TrainZScore

GROUPS = ("F1", "F2", "F3", "F4", "F5")
CHUNK = 200_000


@dataclass
class Norm:
    mkt: TrainZScore
    seq_mu: np.ndarray | None = None
    seq_sd: np.ndarray | None = None
    seq_lo: np.ndarray | None = None
    seq_hi: np.ndarray | None = None


class SeqStore:
    """F6 稠密数组（data/gpu_v1/seq/seq_raw_f32.npy，memmap）+ 交易日下标。"""

    def __init__(self, root: Path):
        import json
        d = Path(root) / "data/gpu_v1/seq"
        self.arr = np.load(d / "seq_raw_f32.npy", mmap_mode="r")
        meta = json.loads((d / "meta.json").read_text())
        self.days = np.array(meta["days"], dtype="datetime64[D]")
        self.channels = meta["channels"]

    def day_index(self, as_of) -> np.ndarray:
        return np.searchsorted(self.days, np.asarray(as_of, dtype="datetime64[D]"))


def fit_norm(md: ModelData, train: Block, seq: SeqStore | None, L: int) -> Norm:
    mk = md.market.reindex(train.days).to_numpy(dtype=np.float64)
    norm = Norm(mkt=TrainZScore().fit(mk))
    if seq is not None:
        g = seq.day_index(train.days)
        lo_d, hi_d = max(0, int(g.min()) - L + 1), int(g.max()) + 1
        rng = np.random.default_rng(0)
        stocks = np.unique(train.meta["seq_idx"].to_numpy())
        days = np.arange(lo_d, hi_d)
        samp = seq.arr[np.sort(rng.choice(stocks, size=min(len(stocks), 1500), replace=False))][:, days]
        flat = samp.reshape(-1, samp.shape[-1]).astype(np.float64)
        lo, hi = np.nanpercentile(flat, 0.5, axis=0), np.nanpercentile(flat, 99.5, axis=0)
        c = np.clip(flat, lo, hi)
        mu, sd = np.nanmean(c, axis=0), np.nanstd(c, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        norm.seq_mu, norm.seq_sd, norm.seq_lo, norm.seq_hi = mu, sd, lo, hi
    return norm


def seq_tensor(seq: SeqStore, norm: Norm, device: str = "cuda") -> torch.Tensor:
    """整个 F6 数组按该折统计量标准化后放上 GPU（float16，NaN→0，截断 ±5）。"""
    out = torch.empty(seq.arr.shape, dtype=torch.float16, device=device)
    for s in range(0, seq.arr.shape[0], 500):
        x = np.clip(np.asarray(seq.arr[s:s + 500], dtype=np.float32), norm.seq_lo, norm.seq_hi)
        x = np.clip((x - norm.seq_mu) / norm.seq_sd, -5, 5)
        out[s:s + 500] = torch.from_numpy(np.nan_to_num(x, nan=0.0).astype(np.float16)).to(device)
    return out


class Inputs:
    """一个 Block 在 GPU 上的输入张量。"""

    def __init__(self, md: ModelData, block: Block, norm: Norm, arch: str, horizon: str, seq: SeqStore | None,
                 device: str = "cuda"):
        kind = INPUTS[arch]
        self.n = len(block.meta)
        self.day = block.meta["day"].to_numpy()
        self.y = torch.from_numpy(block.label_rank(horizon).astype(np.float32)).to(device)
        self.y_raw = block.meta["y20" if horizon == "20d" else "y63"].to_numpy(dtype=np.float64)
        self.top = block.meta["top1500"].to_numpy()
        self.day_t = torch.from_numpy(self.day.astype(np.int64)).to(device)
        mk = norm.mkt.transform(md.market.reindex(block.days).to_numpy(dtype=np.float64))
        mk_t = torch.from_numpy(np.nan_to_num(mk, nan=0.0).astype(np.float16)).to(device)
        self.mkt = mk_t[self.day_t]                                     # [n, F_mkt]
        self.tab = self.snap = None
        n_mk = self.mkt.shape[1]
        if kind == "tab":                     # 按块直接写进显存张量：不在主机上整块复制 float16 矩阵
            gcols = [md.group_columns([g]) for g in GROUPS]
            F = block.X.shape[1]
            self.tab = torch.empty((self.n, F + len(GROUPS) + n_mk), dtype=torch.float16, device=device)
            for s in range(0, self.n, CHUNK):
                xb = block.X[s:s + CHUNK]
                flags = np.stack([np.isnan(xb[:, c]).all(axis=1) for c in gcols], axis=1).astype(np.float16)
                self.tab[s:s + len(xb), :F] = torch.from_numpy(np.nan_to_num(xb, nan=0.0)).to(device)
                self.tab[s:s + len(xb), F:F + len(GROUPS)] = torch.from_numpy(flags).to(device)
            self.tab[:, F + len(GROUPS):] = self.mkt
        if kind == "seq+snap":
            cols = md.group_columns(["F2", "F4"])
            self.snap = torch.empty((self.n, len(cols) + n_mk), dtype=torch.float16, device=device)
            for s in range(0, self.n, CHUNK):
                xb = block.X[s:s + CHUNK][:, cols]
                self.snap[s:s + len(xb), :len(cols)] = torch.from_numpy(np.nan_to_num(xb, nan=0.0)).to(device)
            self.snap[:, len(cols):] = self.mkt
        if "seq" in kind:
            self.sidx = torch.from_numpy(block.meta["seq_idx"].to_numpy().astype(np.int64)).to(device)
            self.gday = torch.from_numpy(seq.day_index(block.meta["as_of"].to_numpy()).astype(np.int64)).to(device)

    def batch(self, rows: torch.Tensor, seq_t: torch.Tensor | None, L: int) -> dict:
        b = {"day": self.day_t[rows], "mkt": self.mkt[rows].float()}
        if self.tab is not None:
            b["tab"] = self.tab[rows].float()
        if self.snap is not None:
            b["snap"] = self.snap[rows].float()
        if seq_t is not None:
            t = self.gday[rows, None] - torch.arange(L - 1, -1, -1, device=rows.device)[None, :]
            valid = (t >= 0)[..., None]
            w = seq_t[self.sidx[rows, None], t.clamp(min=0)]
            b["seq"] = (w * valid).float()
        return b


def dims(md: ModelData, arch: str, seq: SeqStore | None) -> dict:
    n_stock, n_mkt = len(md.stock_features), len(md.market_features)
    kind = INPUTS[arch]
    if kind == "tab":
        return {"d_in": n_stock + len(GROUPS) + n_mkt}
    c = len(seq.channels)
    if kind == "seq+snap":
        return {"c_in": c, "d_snap_in": len(md.group_columns(["F2", "F4"])) + n_mkt}
    if kind == "seq+mkt":
        return {"c_in": c, "d_mkt": n_mkt}
    return {"c_in": c}


def predict(model, inp: Inputs, seq_t, L: int, days_per_batch: int) -> np.ndarray:
    model.eval()
    out = np.empty(inp.n, dtype=np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for rows in day_batches(inp.day, days_per_batch):
            r = torch.from_numpy(rows).to(inp.day_t.device)
            p = model(**inp.batch(r, seq_t, L)).float()
            out[rows] = (p.mean(-1) if p.ndim == 2 else p).cpu().numpy()
    model.train()
    return out


def train(arch: str, cfg: dict, md: ModelData, tr: Inputs, va: Inputs | None, seq_t, seq: SeqStore | None,
          seed: int, ckpt: Path, epochs: int | None = None, log=print) -> tuple[torch.nn.Module, list[dict]]:
    """va 给出 → 内层模式（早停，返回每 epoch 的验证 RankIC）；否则训练 epochs 个 epoch。"""
    c = dict(cfg)
    L, dpb, lr, wd = c.pop("L", 1), c.pop("days_per_batch"), c.pop("lr"), c.pop("weight_decay")
    max_epochs, patience, aux = c.pop("max_epochs"), c.pop("patience"), c.pop("aux_mse")
    stride = c.pop("day_stride", 1)
    n_ep = max_epochs if epochs is None else epochs
    torch.manual_seed(seed)
    model = ARCHS[arch](**dims(md, arch, seq), **c).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ok = np.isfinite(tr.y.cpu().numpy())
    if stride > 1:                                    # 隔 stride 天取一天，偏移随种子变（种子平均覆盖全部交易日）
        ok &= (tr.day + seed) % stride == 0
    ok_rows = np.flatnonzero(ok)
    day_ok = tr.day[ok_rows]
    steps = math.ceil(len(np.unique(day_ok)) / dpb)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / (max_epochs * steps)))))
    state = load_checkpoint(ckpt)
    start, curve, best = 0, [], (-np.inf, -1, None)
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, curve = state["epoch"], state["curve"]
        best = state["best"]
    rng = np.random.default_rng(seed * 1000 + start)
    for ep in range(start, n_ep):
        for rows in day_batches(day_ok, dpb, rng):
            r = torch.from_numpy(ok_rows[rows]).to(tr.day_t.device)
            b = tr.batch(r, seq_t, L)
            y = tr.y[r]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(**b).float()
            yt = y[:, None] if p.ndim == 2 else y
            loss = daily_ic_loss(p, y, b["day"]) + aux * ((p - yt) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        rec = {"epoch": ep + 1, "loss": loss.item()}
        if va is not None:
            ic = daily_rank_ic(predict(model, va, seq_t, L, dpb).astype(np.float64), va.y_raw, va.day, va.top)
            rec["ic_mean"] = float(np.mean(ic))
            if rec["ic_mean"] > best[0]:
                best = (rec["ic_mean"], ep + 1, None)
        curve.append(rec)
        log(f"    epoch {ep + 1}/{n_ep} {rec}")
        save_checkpoint(ckpt, {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                               "epoch": ep + 1, "curve": curve, "best": best})
        if va is not None and ep + 1 - best[1] >= patience:
            break
    return model, curve
