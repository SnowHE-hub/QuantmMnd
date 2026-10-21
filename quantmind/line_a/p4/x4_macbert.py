"""quantmind.line_a.p4.x4_macbert — X4 的 MacBERT 监督分类基线（03_final_plan_v2 第 9.1 节"便宜的候选"；本机 GPU）.

与 VT 用同一份题包、同一种题目形式，结果能用 x4_eval 直接对比：
- 输入：句对（类型说明，公告标题），标题从题目文本的"公告标题："之后取；
- 模型：hfl/chinese-macbert-base + 二分类头；
- 训练：只用 train.jsonl；AdamW，学习率 2e-5，2 个 epoch，批大小 32，6% 预热，bf16 自动混合精度，种子 20261021；
  不用校准集挑 epoch（校准集留给 x4_eval 选阈值）；
- 打分：对 cells.jsonl 每题输出 p_yes，写成与 Colab 相同的行格式（cell、sha、p_yes、status、mode、model_key）。
用法：python -m quantmind.line_a.p4.x4_macbert <pack_dir> <out_dir> [--smoke]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from quantmind.line_a.p4.x4_dataset import TYPES

MODEL = "hfl/chinese-macbert-base"
KEY = "macbert-base"
CFG = {"lr": 2e-5, "epochs": 2, "batch": 32, "warmup": 0.06, "max_len": 128, "seed": 20261021}


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _verify(pack: Path) -> dict:
    man = json.loads((Path(pack) / "manifest.json").read_text(encoding="utf-8"))
    for n, v in man["files"].items():
        if hashlib.sha256((Path(pack) / n).read_bytes()).hexdigest() != v["sha256"]:
            raise SystemExit(f"pack file {n} does not match the manifest")
    return man


def _pair(r: dict) -> tuple[str, str]:
    return TYPES[r["type"]][0], r["user"].split("公告标题：", 1)[1]


def run(pack: Path, out_dir: Path, *, smoke: bool = False) -> Path:
    import numpy as np
    import torch
    import transformers
    man = _verify(pack)
    torch.manual_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    train, cells = _rows(Path(pack) / "train.jsonl"), _rows(Path(pack) / "cells.jsonl")
    if smoke:
        train, cells = train[:256], cells[:128]
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).cuda()

    def batch(rs):
        a, b = zip(*(_pair(r) for r in rs), strict=True)
        return tok(list(a), list(b), truncation="only_second", max_length=CFG["max_len"], padding=True,
                   return_tensors="pt").to("cuda")
    steps = CFG["epochs"] * ((len(train) + CFG["batch"] - 1) // CFG["batch"])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    sch = transformers.get_linear_schedule_with_warmup(opt, int(CFG["warmup"] * steps), steps)
    run_dir = Path(out_dir) / f"ft__{KEY}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "train_log.jsonl").open("w", encoding="utf-8")
    t0, step = time.time(), 0
    model.train()
    for ep in range(CFG["epochs"]):
        order = rng.permutation(len(train))
        for i in range(0, len(order), CFG["batch"]):
            rs = [train[j] for j in order[i:i + CFG["batch"]]]
            x = batch(rs)
            y = torch.tensor([r["label"] for r in rs], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(**x).logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0 or step == steps:
                log.write(json.dumps({"epoch": ep, "step": step, "loss": float(loss), "s": round(time.time() - t0, 1)}) + "\n")
                log.flush()
                print(f"  step {step}/{steps} loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    log.close()
    model.eval()
    path = run_dir / ("scores_smoke.jsonl" if smoke else "scores.jsonl")
    with path.open("w", encoding="utf-8") as fh, torch.inference_mode():
        for i in range(0, len(cells), 128):
            rs = cells[i:i + 128]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.softmax(model(**batch(rs)).logits.float(), dim=-1)[:, 1].tolist()
            for r, py in zip(rs, p, strict=True):
                fh.write(json.dumps({"cell": r["cell"], "sha": r["sha"], "p_yes": py, "status": "ok", "mode": "ft",
                                     "model_key": KEY}) + "\n")
    (run_dir / "config.json").write_text(json.dumps({**CFG, "model": MODEL, "protocol": man["protocol"],
                                                     "pack_files": man["files"], "smoke": smoke,
                                                     "torch": torch.__version__, "transformers": transformers.__version__,
                                                     "train_seconds": round(time.time() - t0, 1)}, indent=1))
    return path


if __name__ == "__main__":
    print(run(Path(sys.argv[1]), Path(sys.argv[2]), smoke="--smoke" in sys.argv))
