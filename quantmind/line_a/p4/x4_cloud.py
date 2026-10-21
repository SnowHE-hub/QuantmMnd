"""X4 on Colab: zero-shot scoring and LoRA fine-tuning of announcement-type questions under the VT protocol.

Adapted from the user's 5703 project (mvp/cloud/cs1_cloud.py): same protocol and engineering pattern.
- Self-contained: imports torch / transformers (v5) / peft only, never the quantmind package.
- The pack (cells.jsonl = calib+test questions, train.jsonl, manifest.json) is produced locally by
  quantmind/line_a/p4/x4_pack.py. Messages are pre-rendered and sha-checked here, never re-rendered.
- Score = p(Yes) / (p(Yes) + p(No)) at the first assistant position, thinking disabled.
- Every output row carries the message sha so the local importer can verify it.
- Everything is resumable: re-running a function skips rows / steps already on disk.

    import x4_cloud as xc
    xc.verify_pack(PACK); xc.smoke(PACK, OUT); xc.run_queue(JOBS, PACK, OUT); xc.status(OUT)
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODELS = {"q35-0.8b": "Qwen/Qwen3.5-0.8B", "q35-2b": "Qwen/Qwen3.5-2B", "q35-4b": "Qwen/Qwen3.5-4B",
          "q35-9b": "Qwen/Qwen3.5-9B"}
RUNTIME_ONLY = {"grad_ckpt"}
FT_DEFAULTS = {"lr": 1e-4, "r": 16, "alpha": 32, "dropout": 0.05, "accum": 16, "epochs": 1, "warmup": 0.03,
               "seed": 42, "max_len": 512, "max_train": 24000, "dev_cells": 1500, "eval_every": 100,
               "ckpt_every": 50, "load_in_4bit": False, "grad_ckpt": "auto"}


# ---------------------------------------------------------------------------- io
def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(p: Path) -> list[dict]:
    out = []
    p = Path(p)
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):   # half-written last line after a disconnect
                    out.append(json.loads(line))
    return out


def verify_pack(pack: Path) -> dict:
    man = json.loads((Path(pack) / "manifest.json").read_text(encoding="utf-8"))
    for name, meta in man["files"].items():
        got = _sha_file(Path(pack) / name)
        if got != meta["sha256"]:
            raise SystemExit(f"pack file {name} sha mismatch: {got} != {meta['sha256']}")
    print(f"pack ok: protocol {man['protocol']}, files {list(man['files'])}, rows {man.get('rows')}")
    return man


def _check_rows(rows: list[dict]) -> None:
    for r in rows:
        if hashlib.sha256(r["user"].encode("utf-8")).hexdigest() != r["sha"]:
            raise SystemExit(f"message sha mismatch at {r['cell']}")


def env_identity(model=None, model_key: str | None = None) -> dict:
    import torch
    import transformers
    try:
        import peft
        peft_v = peft.__version__
    except Exception:                                   # noqa: BLE001
        peft_v = None
    d = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_v,
         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() else "cpu",
         "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if model_key:
        d["model_key"], d["hf_id"] = model_key, MODELS[model_key]
    if model is not None:
        cfg = getattr(model, "config", None)
        d["revision"] = getattr(cfg, "_commit_hash", None)
    return d


# ------------------------------------------------------------------------- model
def free_gpu() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                   # noqa: BLE001
        pass


def gpu_gib() -> float:
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0


def load_model(model_key: str, *, device: str | None = None, load_in_4bit: bool = False, local_path: str | None = None):
    import torch
    import transformers
    hf = local_path or MODELS[model_key]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = transformers.AutoTokenizer.from_pretrained(hf)
    cfg = transformers.AutoConfig.from_pretrained(hf)
    arch = (cfg.architectures or ["AutoModelForCausalLM"])[0]
    loader = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
    kw = {"dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    if load_in_4bit:
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    free_gpu()
    for attn in ("sdpa", "eager"):
        try:
            model = loader.from_pretrained(hf, attn_implementation=attn,
                                           device_map={"": 0} if device == "cuda" else None, **kw)
            break
        except Exception as e:                          # noqa: BLE001
            if attn == "eager" or "out of memory" in str(e).lower():
                raise
            print(f"  sdpa unavailable ({type(e).__name__}: {e}); falling back to eager")
            free_gpu()
    if device != "cuda":
        model.to(device)
    model.eval()
    print(f"loaded {hf} as {type(model).__name__} on {device}")
    return model, tok


def answer_ids(tok) -> tuple[int, int]:
    ids = []
    for w in ("Yes", "No"):
        t = tok.encode(w, add_special_tokens=False)
        if len(t) != 1:
            raise SystemExit(f"answer word {w!r} is {len(t)} tokens for this tokenizer; VT protocol needs 1")
        ids.append(t[0])
    return ids[0], ids[1]


def encode(tok, user: str) -> list[int]:
    """Chat template with the generation prompt and thinking disabled (the next token is the answer word)."""
    msgs = [{"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)


def template_tail(tok) -> str:
    s = tok.decode(encode(tok, "PING"))
    return s[s.index("PING") + 4:]


def _last_logits(model, ids, grad: bool = False):
    import torch
    x = torch.tensor([ids], device=model.device)
    with (torch.enable_grad() if grad else torch.inference_mode()):
        try:
            out = model(input_ids=x, logits_to_keep=1, use_cache=False)
        except TypeError:
            out = model(input_ids=x, use_cache=False)
        return out.logits[0, -1]


# ------------------------------------------------------------------------ scoring
def _score_one(model, tok, ids, yes, no) -> dict:
    import torch
    lp = torch.log_softmax(_last_logits(model, ids).float(), dim=-1)
    ly, ln = float(lp[yes]), float(lp[no])
    m = max(ly, ln)
    p = math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m))
    return {"p_yes": p, "lp_yes": ly, "lp_no": ln, "mass": math.exp(ly) + math.exp(ln)}


def score_rows(model, tok, rows: list[dict], out_path: Path, tag: dict, *, flush_every: int = 200,
               max_len: int = 1024) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["cell"] for r in _read_jsonl(out_path)}
    todo = [r for r in rows if r["cell"] not in done]
    print(f"  scoring -> {out_path.name}: {len(rows)} target, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    yes, no = answer_ids(tok)
    t0, buf = time.time(), []
    with out_path.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            ids = encode(tok, r["user"])
            row = {"cell": r["cell"], "sha": r["sha"], "n_tokens": len(ids), **tag}
            if len(ids) > max_len:
                row["status"] = "too_long"
            else:
                row.update(_score_one(model, tok, ids, yes, no))
                row["status"] = "ok"
            buf.append(json.dumps(row, ensure_ascii=False))
            if len(buf) >= flush_every or i == len(todo):
                fh.write("\n".join(buf) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                buf = []
                el = time.time() - t0
                print(f"    [{i}/{len(todo)}] {el / 60:.1f} min, eta {(len(todo) - i) * el / i / 60:.0f} min", flush=True)


def _auc(y, s) -> float:
    pairs = sorted(zip(s, y, strict=True))
    n1 = sum(y)
    n0 = len(y) - n1
    if not n1 or not n0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2.0 * sum(1 for k in range(i, j) if pairs[k][1] == 1)
        i = j
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def run_zeroshot(model_key: str, pack: Path, out: Path, *, limit: int | None = None, load_in_4bit: bool = False) -> Path:
    rows = _read_jsonl(Path(pack) / "cells.jsonl")
    _check_rows(rows)
    rows = rows[:limit] if limit else rows
    model, tok = load_model(model_key, load_in_4bit=load_in_4bit)
    path = Path(out) / f"zs__{model_key}.jsonl"
    score_rows(model, tok, rows, path, {"model_key": model_key, "mode": "zs", "precision": "nf4" if load_in_4bit else "bf16"})
    del model
    free_gpu()
    return path


# ------------------------------------------------------------------------ training
def _lora_targets(model) -> list[str]:
    import torch
    names = []
    for name, mod in model.named_modules():
        low = name.lower()
        if any(k in low for k in ("vision", "visual", "image", "lm_head", "embed", "merger")):
            continue
        if isinstance(mod, torch.nn.Linear) or type(mod).__name__ in ("Linear4bit", "Linear8bitLt"):
            names.append(name)
    if not names:
        raise SystemExit("no Linear modules found for LoRA")
    return sorted(names)


def _open_run(run: Path, cfg: dict) -> None:
    run.mkdir(parents=True, exist_ok=True)
    p = run / "config.json"
    if p.exists():
        old = json.loads(p.read_text())
        diff = {k: (old.get(k), cfg[k]) for k in cfg if old.get(k) != cfg[k] and k not in RUNTIME_ONLY}
        if diff:
            raise SystemExit(f"{run.name}: config changed since the run started {diff}; use a new out dir")
    p.write_text(json.dumps(cfg, indent=2))


def run_finetune(model_key: str, pack: Path, out: Path, cfg: dict | None = None, *, max_steps: int | None = None) -> Path:
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    cfg = {**FT_DEFAULTS, **(cfg or {})}
    run = Path(out) / f"ft__{model_key}"
    _open_run(run, cfg)
    train = _read_jsonl(Path(pack) / "train.jsonl")
    _check_rows(train)
    rng0 = random.Random(cfg["seed"])
    if len(train) > cfg["max_train"]:
        train = rng0.sample(train, cfg["max_train"])
    cells = _read_jsonl(Path(pack) / "cells.jsonl")
    dev = [r for r in cells if r["split"] == "calib"][: cfg["dev_cells"]]
    assert not ({r["ann_id"] for r in train} & {r["ann_id"] for r in cells}), "train / held-out announcement overlap"
    model, tok = load_model(model_key, load_in_4bit=cfg["load_in_4bit"])
    yes, no = answer_ids(tok)
    enc = [(encode(tok, r["user"]), int(r["label"])) for r in train]
    if any(len(i) > cfg["max_len"] for i, _ in enc):
        raise SystemExit("training prompt exceeds max_len; refuse to truncate")
    dev_enc = [(encode(tok, r["user"]), int(r["label"])) for r in dev]
    if cfg["load_in_4bit"]:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                                             target_modules=_lora_targets(model), bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    gc_mode = cfg["grad_ckpt"]
    if gc_mode == "auto":
        gc_mode = "reentrant" if model_key == "q35-9b" or cfg["load_in_4bit"] else "off"
    if gc_mode != "off" and torch.cuda.is_available():
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": gc_mode == "reentrant"})
        model.enable_input_require_grads()
    n1 = sum(y for _, y in enc)
    w = torch.tensor([len(enc) / (2.0 * max(1, len(enc) - n1)), len(enc) / (2.0 * max(1, n1))], device=model.device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0)
    total = math.ceil(len(enc) * cfg["epochs"] / cfg["accum"])
    total = min(total, max_steps) if max_steps else total
    warm = max(1, int(total * cfg["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    rng = random.Random(cfg["seed"])
    order = []
    for _ in range(cfg["epochs"]):
        o = list(range(len(enc)))
        rng.shuffle(o)
        order += o
    order = order[: total * cfg["accum"]]
    step, best = 0, {"auc": -1.0, "step": None}
    ck = run / "ckpt"
    if (ck / "state.pt").exists():
        from safetensors.torch import load_file
        st = torch.load(ck / "state.pt", map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, load_file(str(ck / "adapter.safetensors")))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, best = st["step"], st["best"]
        print(f"  resumed at optimizer step {step}/{total}; best so far {best}")

    def save_ckpt():
        from safetensors.torch import save_file
        tmp = run / "ckpt.tmp"
        tmp.mkdir(exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()},
                  str(tmp / "adapter.safetensors"))
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "best": best}, tmp / "state.pt")
        if ck.exists():
            for f in ck.iterdir():
                f.unlink()
            ck.rmdir()
        tmp.rename(ck)

    def dev_auc():
        model.eval()
        s = [float((lambda lg: lg[yes] - lg[no])(_last_logits(model, ids).float())) for ids, _ in dev_enc]
        model.train()
        return _auc([y for _, y in dev_enc], s)

    log = (run / "train_log.jsonl").open("a", encoding="utf-8")
    model.train()
    t0, start = time.time(), step
    for s in range(step, total):
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        for j in order[s * cfg["accum"]:(s + 1) * cfg["accum"]]:
            ids, y = enc[j]
            lg = _last_logits(model, ids, grad=True)
            two = torch.stack([lg[no], lg[yes]]).float().unsqueeze(0)
            loss = torch.nn.functional.cross_entropy(two, torch.tensor([y], device=two.device), weight=w)
            (loss / cfg["accum"]).backward()
            run_loss += float(loss)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        step = s + 1
        rec = {"step": step, "loss": run_loss / cfg["accum"], "lr": sched.get_last_lr()[0],
               "elapsed_min": round((time.time() - t0) / 60, 2)}
        if step % cfg["eval_every"] == 0 or step == total:
            rec["dev_auc"] = dev_auc()
            if rec["dev_auc"] > best["auc"]:
                best = {"auc": rec["dev_auc"], "step": step}
                model.save_pretrained(run / "adapter_best")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 20 == 0 or "dev_auc" in rec:
            eta = (time.time() - t0) / max(1, step - start) * (total - step) / 60
            print(f"    step {step}/{total} loss {rec['loss']:.4f}"
                  + (f" dev_auc {rec['dev_auc']:.4f} (best {best['auc']:.4f} @ {best['step']})" if "dev_auc" in rec else "")
                  + f"  eta {eta:.0f} min", flush=True)
        if step % cfg["ckpt_every"] == 0 or step == total:
            save_ckpt()
    log.close()
    (run / "best.json").write_text(json.dumps({**best, "total_steps": total, "n_train": len(enc),
                                               "env": env_identity(model, model_key)}, indent=2))
    print(f"  done: best dev AUC {best['auc']:.4f} at step {best['step']}")
    model = opt = None          # 释放显存（不用 del：内部函数还引用这两个名字）
    free_gpu()
    return run


def score_finetuned(model_key: str, pack: Path, out: Path, *, limit: int | None = None) -> Path:
    from peft import PeftModel
    run = Path(out) / f"ft__{model_key}"
    best = json.loads((run / "best.json").read_text())
    q4 = json.loads((run / "config.json").read_text()).get("load_in_4bit", False)
    base, tok = load_model(model_key, load_in_4bit=q4)
    model = PeftModel.from_pretrained(base, run / "adapter_best")
    model.eval()
    rows = _read_jsonl(Path(pack) / "cells.jsonl")
    _check_rows(rows)
    rows = rows[:limit] if limit else rows
    path = run / "scores.jsonl"
    score_rows(model, tok, rows, path, {"model_key": model_key, "mode": "ft", "adapter_step": best["step"],
                                        "precision": "nf4" if q4 else "bf16"})
    del model, base
    free_gpu()
    return path


# ------------------------------------------------------------------------ queue
def run_queue(jobs: list[tuple], pack: Path, out: Path) -> None:
    """("zs", model) | ("ft", model). Each finished job leaves a marker so a new session skips it."""
    marks = Path(out) / "_done"
    marks.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        name = "__".join(job)
        if (marks / name).exists():
            print(f"[skip] {name} (done)")
            continue
        print(f"\n===== {name}  ({time.strftime('%H:%M:%S')}) =====", flush=True)
        t0 = time.time()
        q4 = job[1] == "q35-9b" and gpu_gib() < 30
        if job[0] == "zs":
            run_zeroshot(job[1], pack, out, load_in_4bit=q4)
        elif job[0] == "ft":
            run_finetune(job[1], pack, out, {"load_in_4bit": q4})
            score_finetuned(job[1], pack, out)
        else:
            raise SystemExit(f"unknown job {job}")
        (marks / name).write_text(json.dumps({"minutes": round((time.time() - t0) / 60, 1), "env": env_identity()}))
        print(f"[done] {name} in {(time.time() - t0) / 60:.1f} min", flush=True)


def smoke(pack: Path, out: Path, model_key: str = "q35-2b") -> None:
    """~5 minutes: the whole path on a few rows, written under out/_smoke (never mixed with real results)."""
    import torch
    sm = Path(out) / "_smoke"
    sm.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(model_key)
    print("answer ids", answer_ids(tok), " template tail", repr(template_tail(tok)))
    rows = _read_jsonl(Path(pack) / "cells.jsonl")
    pick = [r for r in rows if r["label"] == 1][:3] + [r for r in rows if r["label"] == 0][:3]
    score_rows(model, tok, pick, sm / f"zs__{model_key}.jsonl", {"model_key": model_key, "mode": "zs"})
    for r in _read_jsonl(sm / f"zs__{model_key}.jsonl"):
        print(f"  {r['cell']} p_yes={r['p_yes']:.3f} mass={r['mass']:.3f}")
    del model
    free_gpu()
    run_finetune(model_key, pack, sm, {"max_train": 64, "accum": 4, "dev_cells": 16, "eval_every": 2, "ckpt_every": 2},
                 max_steps=4)
    score_finetuned(model_key, pack, sm, limit=4)
    print("SMOKE OK - peak GPU memory %.1f GiB" % (torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0))


def status(out: Path) -> None:
    out = Path(out)
    print("finished jobs:", sorted(p.name for p in (out / "_done").glob("*")) if (out / "_done").exists() else [])
    for p in sorted(out.rglob("*.jsonl")):
        print(f"  {p.relative_to(out)}: {sum(1 for _ in p.open(encoding='utf-8'))} rows")
    for p in sorted(out.glob("ft__*/best.json")):
        b = json.loads(p.read_text())
        print(f"  {p.parent.name}: best dev AUC {b['auc']:.4f} at step {b['step']}/{b['total_steps']}")
