"""scripts/ablation_run.py — T-007 执行端：把冻结的 requests.jsonl 逐条发给本地 llama-server（C3，需要 GPU）.

只做"发请求、存原始输出"，不做任何判读（判读由 ablation_analyze.py 完成，规则已预注册）。

- 发送的就是 requests.jsonl 里冻结的 ``payload``（只加 ``model`` 字段）；私有映射不在 payload 里
- ``run_id`` = sha256(requests.jsonl 内容哈希 + 模型文件内容哈希 + 模型名)。每条响应带 run_id、request_id；
  断点续跑只跳过**同一 run_id** 下已成功的 request_id——换模型或换请求不会混入旧响应
- 传输失败（HTTP 错误、超时、连接）记一行 status=error；同一 run 内每条请求最多 MAX_TRANSPORT_ATTEMPTS 次，
  用尽后不再重试（报告里列出）。模型输出不合法**不重试**——是否合法由分析脚本判定，全部原始响应保留
- 不允许通过重跑挑选：已成功（status=ok）的请求永不重发
- 设计 v3：manifest 里有 ``lock`` 时，GGUF 哈希、llama.cpp build_info、聊天模板哈希必须与锁定值一致，否则拒绝运行
- 运行信息（模型文件哈希、服务端 /props）写入 ``run_<run_id[:12]>.json``

用法：
  python scripts/ablation_run.py --dir <冻结目录> --base-url http://<llama-server>:8090 --model bonsai-27b \
      --model-file /mnt/d/LocalLLM/models/<gguf>
  （WSL 里的 localhost 不一定指向 Windows 侧服务，地址必须显式给出，见 T-006 接收记录）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import requests

MAX_TRANSPORT_ATTEMPTS = 3


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def payload_sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def make_run_id(requests_sha: str, model_sha: str, model: str) -> str:
    return hashlib.sha256(f"{requests_sha}\n{model_sha}\n{model}".encode()).hexdigest()


def _records(path: Path, run_id: str):
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("run_id") == run_id:
                    yield r


def done_ids(path: Path, run_id: str) -> set[str]:
    return {r["request_id"] for r in _records(path, run_id) if r.get("status") == "ok"}


def error_counts(path: Path, run_id: str) -> Counter:
    return Counter(r["request_id"] for r in _records(path, run_id) if r.get("status") == "error")


def lock_mismatches(lock: dict, model_sha: str, props: dict) -> list[str]:
    bad = []
    if lock.get("gguf_sha256") and model_sha != lock["gguf_sha256"]:
        bad.append("gguf_sha256")
    if lock.get("llama_cpp_build_info") and props.get("build_info") != lock["llama_cpp_build_info"]:
        bad.append("llama_cpp_build_info")
    if lock.get("chat_template_sha256"):
        tpl = hashlib.sha256(str(props.get("chat_template", "")).encode("utf-8")).hexdigest()
        if tpl != lock["chat_template_sha256"]:
            bad.append("chat_template_sha256")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="写进每条结果，用于 EvalContract 的 llm.model 字段")
    ap.add_argument("--model-file", type=Path, help="GGUF 文件；给出时其内容哈希进入 run_id")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0, help="只发前 N 条（试跑）")
    ap.add_argument("--retry-wait", type=float, default=5.0, help="传输失败后的等待基数（秒）")
    a = ap.parse_args(argv)

    req_path = a.dir / "requests.jsonl"
    reqs = [json.loads(line) for line in req_path.read_text(encoding="utf-8").splitlines() if line]
    for r in reqs:
        if r["request_id"] != payload_sha(r["payload"]):
            raise SystemExit(f"request {r['as_of']}/{r['group']}: payload 与 request_id 不符（冻结文件被改过）")
    model_sha = file_sha(a.model_file) if a.model_file else "unhashed"
    run_id = make_run_id(file_sha(req_path), model_sha, a.model)
    base = a.base_url.rstrip("/")
    try:
        props = requests.get(base + "/props", timeout=10).json()
    except Exception as e:  # noqa: BLE001
        props = {"error": repr(e)[:200]}
    man_path = a.dir / "manifest.json"
    lock = json.loads(man_path.read_text(encoding="utf-8")).get("lock", {}) if man_path.exists() else {}
    if lock:
        bad = lock_mismatches(lock, model_sha, props)
        if bad:
            raise SystemExit(f"锁定值不一致，拒绝运行：{bad}")
    (a.dir / f"run_{run_id[:12]}.json").write_text(json.dumps(
        {"run_id": run_id, "model": a.model, "model_file": str(a.model_file) if a.model_file else None,
         "model_sha256": model_sha, "requests_sha256": file_sha(req_path), "server_props": props,
         "lock_checked": bool(lock), "started": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=False, indent=2), encoding="utf-8")

    out = a.dir / "responses.jsonl"
    seen = done_ids(out, run_id)
    errs = error_counts(out, run_id)
    exhausted = {r["request_id"] for r in reqs
                 if r["request_id"] not in seen and errs[r["request_id"]] >= MAX_TRANSPORT_ATTEMPTS}
    todo = [r for r in reqs if r["request_id"] not in seen and r["request_id"] not in exhausted]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[ablation_run] run {run_id[:12]}: {len(reqs)} requests, {len(seen)} done, {len(exhausted)} exhausted, "
          f"{len(todo)} to send", flush=True)
    url = base + "/v1/chat/completions"
    n_fail = 0
    with open(out, "a", encoding="utf-8") as fh:
        for i, req in enumerate(todo, 1):
            ok = False
            for attempt in range(errs[req["request_id"]], MAX_TRANSPORT_ATTEMPTS):
                t0 = time.perf_counter()
                rec = {"run_id": run_id, "request_id": req["request_id"], "as_of": req["as_of"],
                       "group": req["group"], "repeat": req.get("repeat"), "model": a.model, "attempt": attempt + 1}
                try:
                    resp = requests.post(url, json={"model": a.model, **req["payload"]}, timeout=a.timeout)
                    resp.raise_for_status()
                    body = resp.json()
                    msg = body["choices"][0]["message"]
                    rec |= {"status": "ok", "content": msg.get("content", ""),
                            "reasoning_content": msg.get("reasoning_content") or "",
                            "finish_reason": body["choices"][0].get("finish_reason"), "usage": body.get("usage", {})}
                except Exception as e:  # noqa: BLE001
                    rec |= {"status": "error", "error": repr(e)[:300]}
                rec["latency_s"] = round(time.perf_counter() - t0, 2)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {i}/{len(todo)} {req['as_of']} {req['group']} r{req.get('repeat')} {rec['status']} "
                      f"{rec['latency_s']}s", flush=True)
                if rec["status"] == "ok":
                    ok = True
                    break
                time.sleep(a.retry_wait * (attempt + 1))
            n_fail += not ok
    if exhausted:
        print(f"[ablation_run] {len(exhausted)} requests had already used {MAX_TRANSPORT_ATTEMPTS} transport attempts")
    return 0 if n_fail == 0 and not exhausted else 2


if __name__ == "__main__":
    raise SystemExit(main())
