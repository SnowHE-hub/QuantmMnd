# -*- coding: utf-8 -*-
"""QuantMind 场景化本地推理实测：#4 并发槽位显存 / #5 推测解码 / #6 冷启动 / #1 知识截止探针。
只读仓库；只写 D:\\LocalLLM\\results\\qm_bench_*.json 与本脚本目录。"""
import json, os, random, subprocess, sys, threading, time, urllib.request
from pathlib import Path

BIN = r"D:\LocalLLM\src\llama.cpp-prism\build\bin"
MODEL = r"D:\LocalLLM\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
CUDA = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
CUBLAS = r"D:\LocalLLM\downloads\cublas_extracted\nvidia\cu13\bin\x86_64"
PORT = 8095
URL = f"http://127.0.0.1:{PORT}"
OUT = Path(r"D:\LocalLLM\results")
LOGDIR = Path(__file__).parent

def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    return int(o.splitlines()[0])

def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def health():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False

def start(tag, extra):
    env = os.environ.copy()
    env["PATH"] = f"{CUDA}\\bin;{CUBLAS};{BIN};" + env["PATH"]
    args = [os.path.join(BIN, "llama-server.exe"), "-m", MODEL, "-ngl", "99",
            "--port", str(PORT), "--host", "127.0.0.1", "--jinja"] + extra
    log = open(LOGDIR / f"srv_{tag}.log", "w", encoding="utf-8", errors="replace")
    t0 = time.perf_counter()
    p = subprocess.Popen(args, cwd=BIN, env=env, stdout=log, stderr=subprocess.STDOUT)
    while time.perf_counter() - t0 < 300:
        if p.poll() is not None:
            return p, None, log
        if health():
            return p, time.perf_counter() - t0, log
        time.sleep(0.5)
    return p, None, log

def stop(p, log):
    p.terminate()
    try: p.wait(20)
    except Exception: p.kill()
    log.close(); time.sleep(3)

# ---------- QuantMind 风格 rerank 任务（原始量纲，匿名代码）----------
rng = random.Random(42)
CANDS = []
for i in range(30):
    CANDS.append(dict(t=f"S{i:02d}", pe=round(rng.uniform(8, 60), 1), pb=round(rng.uniform(0.8, 8), 2),
                      roe=round(rng.uniform(0.03, 0.35), 4), acc=round(rng.gauss(-0.02, 0.06), 4),
                      d52=round(rng.uniform(-0.3, 0.02), 4), m6=round(rng.gauss(0.05, 0.2), 4),
                      v3=round(rng.uniform(0.12, 0.45), 4)))
rows = "\n".join(f"| {i+1} | {c['t']} | {c['pe']} | {c['pb']} | {c['roe']*100:.1f}% | {c['acc']:.4f} | "
                 f"{c['d52']*100:.1f}% | {c['m6']*100:.1f}% | {c['v3']*100:.1f}% |" for i, c in enumerate(CANDS))
SYS = ("你是一名A股量化研究员。只能依据表格中的因子值排序，不得使用任何外部知识。"
       "评估维度依次为：盈利质量（ROE高、应计低）、估值、动量、风险。")
USER = ("请从下列30只候选中选出并排序前10名。\n"
        "| LGBM排名 | 代码 | PE_TTM | PB | ROE_TTM | Accruals | 距52W高 | 动量6M | 波动3M |\n"
        "|---|---|---|---|---|---|---|---|---|\n" + rows +
        "\n输出JSON：rankings(10项,每项ticker与30字内reason)、portfolio_thesis(100字内)、risk_warnings(3项)。")
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {
              "rankings": {"type": "array", "minItems": 10, "maxItems": 10,
                           "items": {"type": "object", "additionalProperties": False,
                                     "properties": {"ticker": {"type": "string", "enum": [c["t"] for c in CANDS]},
                                                    "reason": {"type": "string", "maxLength": 60}},
                                     "required": ["ticker", "reason"]}},
              "portfolio_thesis": {"type": "string", "maxLength": 200},
              "risk_warnings": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string", "maxLength": 60}}},
          "required": ["rankings", "portfolio_thesis", "risk_warnings"]}

def rerank_body(schema=True, think=False, max_tokens=1500):
    b = {"messages": [{"role": "system", "content": SYS}, {"role": "user", "content": USER}],
         "max_tokens": max_tokens, "temperature": 0.05, "seed": 42, "cache_prompt": False,
         "chat_template_kwargs": {"enable_thinking": think}}
    if schema:
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": "rerank", "schema": SCHEMA}}
    return b

def validate(txt):
    try:
        d = json.loads(txt)
    except Exception as e:
        return {"json_ok": False, "err": str(e)[:80]}
    ids = [r.get("ticker") for r in d.get("rankings", [])] if isinstance(d, dict) else []
    valid = set(c["t"] for c in CANDS)
    return {"json_ok": True, "n_rank": len(ids), "unique": len(set(ids)) == len(ids),
            "all_valid_ids": all(i in valid for i in ids)}

def run_one(body):
    t0 = time.perf_counter()
    r = post("/v1/chat/completions", body)
    wall = time.perf_counter() - t0
    m = r["choices"][0]["message"]; tm = r.get("timings", {})
    return {"wall_s": round(wall, 2), "finish": r["choices"][0]["finish_reason"],
            "prompt_n": tm.get("prompt_n"), "prompt_tps": round(tm.get("prompt_per_second", 0), 1),
            "pred_n": tm.get("predicted_n"), "pred_tps": round(tm.get("predicted_per_second", 0), 2),
            "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted"),
            "reasoning_chars": len(m.get("reasoning_content") or ""),
            **validate(m.get("content") or "")}

PROBES = [
    ("2024-11 美国大选", "2024年美国总统大选的获胜者是谁？只回答人名。"),
    ("2024 沪深300全年", "沪深300指数2024年全年涨跌幅大约是多少？只回答一个百分数，不知道就回答'不知道'。"),
    ("2024-09-24 政策", "2024年9月24日中国人民银行等部门宣布了什么？一句话回答，不知道就回答'不知道'。"),
    ("2025-01 DeepSeek", "DeepSeek-R1 是什么时候发布的？只回答年月，不知道就回答'不知道'。"),
    ("2025-04 关税", "2025年4月2日美国宣布了什么经济政策？一句话回答，不知道就回答'不知道'。"),
    ("2025 沪深300全年", "沪深300指数2025年全年涨跌幅大约是多少？只回答一个百分数，不知道就回答'不知道'。"),
    ("茅台 2024 股价", "贵州茅台（600519）2024年12月31日的收盘价大约是多少元？只回答数字，不知道就回答'不知道'。"),
    ("自述截止", "你的训练数据截止到哪一年哪一月？只回答年月。"),
]

def probe():
    out = []
    for tag, q in PROBES:
        b = {"messages": [{"role": "user", "content": q}], "max_tokens": 80, "temperature": 0,
             "seed": 1, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = post("/v1/chat/completions", b, timeout=120)
            out.append({"probe": tag, "q": q, "a": (r["choices"][0]["message"].get("content") or "").strip()[:160]})
        except Exception as e:
            out.append({"probe": tag, "q": q, "a": f"ERR {e}"})
    return out

def concurrent(n):
    res = [None] * n
    def w(i): res[i] = run_one(rerank_body(schema=True))
    th = [threading.Thread(target=w, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.perf_counter() - t0
    tot = sum((r or {}).get("pred_n") or 0 for r in res)
    return {"n": n, "wall_s": round(wall, 2), "total_pred_tokens": tot,
            "aggregate_tps": round(tot / wall, 2), "per_req": res}

CONFIGS = [
    ("C0_base_np1", ["-c", "4096", "-np", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on"], "base"),
    ("C1_ngramcache", ["-c", "4096", "-np", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on", "--spec-type", "ngram-cache"], "spec"),
    ("C2_ngrammod", ["-c", "4096", "-np", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on", "--spec-type", "ngram-mod"], "spec"),
    ("C3_mtp", ["-c", "4096", "-np", "1", "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on", "--spec-type", "draft-mtp"], "spec"),
    ("C4_np4_c16k", ["-c", "16384", "-np", "4", "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on"], "conc"),
    ("C5_np2_c8k", ["-c", "8192", "-np", "2", "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on"], "conc"),
]

def main():
    only = sys.argv[1:] or [c[0] for c in CONFIGS]
    for tag, extra, kind in CONFIGS:
        if tag not in only: continue
        print(f"=== {tag} ===", flush=True)
        v0 = vram()
        p, load_s, log = start(tag, extra)
        rec = {"tag": tag, "args": extra, "vram_before": v0}
        if load_s is None:
            rec["error"] = "server failed to start (see log)"; print(rec, flush=True)
            stop(p, log); json.dump(rec, open(OUT / f"qm_bench_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            continue
        rec["load_s"] = round(load_s, 2); rec["vram_loaded"] = vram()
        try:
            run_one(rerank_body(schema=False, max_tokens=64))  # warmup
            if kind in ("base", "spec"):
                rec["rerank_schema"] = run_one(rerank_body(schema=True))
                rec["rerank_noschema"] = run_one(rerank_body(schema=False))
            if tag == "C0_base_np1":
                rec["rerank_think"] = run_one(rerank_body(schema=False, think=True, max_tokens=2500))
                rec["probe"] = probe()
            if kind == "conc":
                n = int(extra[extra.index("-np") + 1])
                rec["single"] = run_one(rerank_body(schema=True))
                rec["concurrent"] = concurrent(n)
            rec["vram_peak"] = vram()
        except Exception as e:
            rec["error"] = repr(e)[:300]
        stop(p, log)
        json.dump(rec, open(OUT / f"qm_bench_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("probe", "concurrent")}, ensure_ascii=False)[:900], flush=True)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
