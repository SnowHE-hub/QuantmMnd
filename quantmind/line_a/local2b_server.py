"""quantmind.line_a.local2b_server — 同一个 2B 底座、多个 LoRA 适配器的本机服务（T-054；借鉴用户 capstone L2 的做法）.

一份 Qwen3.5-2B 底座权重常驻显存，按请求切换适配器：
- `a1`：A1 财务事实的固定格式 JSON 生成（蒸馏自 DeepSeek 核实过的输出）；
- `vt`：公告标题分类的 VT 判别（p(Yes) / (p(Yes) + p(No))）。
接口：
- POST /v1/chat/completions：与 llama-server / OpenAI 兼容（model 字段里带 "a1" 用 A1 适配器；response_format 只用于记录，
  不做约束解码，由 Harness 的结构化输出层修复和重问）；
- POST /vt/score：{"prompts": [...]} → {"p_yes": [...]}（VT 适配器）；
- GET /health。
启动：PYTHONPATH=. python -m quantmind.line_a.local2b_server --a1 <adapter_dir> [--vt <adapter_dir>] --port 8092
"""
from __future__ import annotations

import argparse
import math
import threading
import time

from fastapi import FastAPI
from pydantic import BaseModel

BASE = "/mnt/d/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc"
app = FastAPI(title="local 2B multi-adapter")
STATE: dict = {}
LOCK = threading.Lock()


class ChatReq(BaseModel):
    model: str = "qwen3.5-2b-a1"
    messages: list[dict]
    max_tokens: int = 8192
    temperature: float = 0.0
    response_format: dict | None = None
    chat_template_kwargs: dict | None = None


class VTReq(BaseModel):
    prompts: list[str]


def load(a1: str | None, vt: str | None, in4bit: bool) -> None:
    from peft import PeftModel

    from quantmind.line_a.p4 import x4_cloud as xc
    xc.MODELS["q35-2b"] = BASE
    base, tok = xc.load_model("q35-2b", load_in_4bit=in4bit)
    model = None
    for name, path in (("a1", a1), ("vt", vt)):
        if not path:
            continue
        if model is None:
            model = PeftModel.from_pretrained(base, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
    model.eval()
    STATE.update(model=model, tok=tok, adapters=[n for n, p in (("a1", a1), ("vt", vt)) if p])


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if STATE else "loading", "adapters": STATE.get("adapters")}


@app.post("/v1/chat/completions")
def chat(r: ChatReq) -> dict:
    import torch
    model, tok = STATE["model"], STATE["tok"]
    t0 = time.time()
    with LOCK:
        model.set_adapter("a1")
        text = tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=r.max_tokens, do_sample=False, repetition_penalty=1.05,
                                 pad_token_id=tok.eos_token_id)
        gen = out[0, ids["input_ids"].shape[1]:]
        content = tok.decode(gen, skip_special_tokens=True)
    n_in, n_out = int(ids["input_ids"].shape[1]), int(gen.shape[0])
    return {"id": f"local2b-{int(t0)}", "object": "chat.completion", "model": r.model,
            "choices": [{"index": 0, "finish_reason": "length" if n_out >= r.max_tokens else "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out}, "timings": {"secs": time.time() - t0}}


@app.post("/vt/score")
def vt_score(r: VTReq) -> dict:
    from quantmind.line_a.p4 import x4_cloud as xc
    model, tok = STATE["model"], STATE["tok"]
    yes, no = xc.answer_ids(tok)
    ps = []
    with LOCK:
        model.set_adapter("vt")
        for p in r.prompts:
            lg = xc._last_logits(model, xc.encode(tok, p)).float()
            ly, ln = float(lg[yes]), float(lg[no])
            m = max(ly, ln)
            ps.append(math.exp(ly - m) / (math.exp(ly - m) + math.exp(ln - m)))
    return {"p_yes": ps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1")
    ap.add_argument("--vt")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--bf16", action="store_true", help="底座用 bf16（默认 4-bit，省显存）")
    a = ap.parse_args()
    load(a.a1, a.vt, not a.bf16)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
