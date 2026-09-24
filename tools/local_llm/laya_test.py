# -*- coding: utf-8 -*-
"""#3 Laya 在 QuantMind 中文因子卡片上的零样本实测。真值由合成规则给出。"""
import json, os, random, statistics, sys, time
S = "/mnt/c/Users/lenovo/AppData/Local/Temp/claude/D--MCS-26S2----/168c14f6-f02a-41c9-8141-fef8d2fce422/scratchpad/laya_eval"
sys.path.insert(0, f"{S}/pkg")
os.environ["HF_HOME"] = f"{S}/hf"; os.environ["HF_HUB_OFFLINE"] = "1"
import torch, laya
from scipy.stats import spearmanr

rng = random.Random(7)
cards = []
for i in range(40):
    good = i < 20
    if good:
        roe, pe = rng.uniform(0.20, 0.35), rng.uniform(8, 20)
        acc, m6, vol = rng.uniform(-0.05, 0.0), rng.uniform(0.05, 0.30), rng.uniform(0.12, 0.25)
    else:
        if i % 2:
            roe, pe = rng.uniform(-0.10, 0.03), rng.uniform(40, 90)
        else:
            roe, pe = rng.uniform(0.02, 0.08), rng.uniform(150, 400)
        acc, m6, vol = rng.uniform(0.06, 0.15), rng.uniform(-0.40, -0.10), rng.uniform(0.40, 0.70)
    txt = (f"股票代码 S{i:02d}。市盈率(PE_TTM) {pe:.1f} 倍；净资产收益率(ROE_TTM) {roe*100:.1f}%；"
           f"应计利润率 {acc:.3f}；近6个月涨跌幅 {m6*100:.1f}%；近3个月年化波动率 {vol*100:.1f}%。")
    cards.append({"state": txt, "good": good, "roe": roe, "roe_gt15": roe > 0.15})

Q = {
    "verdict": {"type": "choice", "instructions": "根据因子卡片，判断该股票是否通过基本面筛选。",
                "criteria": {"pass": "盈利能力强、估值合理、应计利润低，通过筛选",
                             "reject_valuation": "市盈率过高或为负，估值不合理，拒绝",
                             "reject_quality": "净资产收益率低或应计利润高，盈利质量差，拒绝"}},
    "roe_high": {"type": "noul", "instructions": "卡片中的净资产收益率(ROE_TTM)是否高于15%？",
                 "criteria": {"false": "ROE 不高于 15%", "true": "ROE 高于 15%"}},
    "quality": {"type": "score", "instructions": "评价该股票的盈利质量。",
                "criteria": ["很差", "较差", "一般", "较好", "很好"]},
}

def run(device):
    t0 = time.perf_counter()
    agent = laya.load(f"{S}/hf/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
                      device=device, subfolder="multilingual")
    load_s = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    agent.predict(cards[0]["state"], Q)  # warmup
    lat = []
    res = []
    for c in cards:
        t = time.perf_counter(); r = agent.predict(c["state"], Q); lat.append((time.perf_counter() - t) * 1000)
        res.append(r)
    tb = time.perf_counter(); agent.predict_batch([c["state"] for c in cards], Q); batch_ms = (time.perf_counter() - tb) * 1000
    return agent, load_s, vram, lat, batch_ms, res

def ans(r, k): return r["answers"][k]

out = {}
for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
    agent, load_s, vram, lat, batch_ms, res = run(dev)
    rec = {"load_s": round(load_s, 2), "vram_alloc_mib": round(vram, 1) if vram else None,
           "latency_ms_single_p50": round(statistics.median(lat), 1), "latency_ms_single_p90": round(sorted(lat)[int(.9*len(lat))], 1),
           "batch40_ms": round(batch_ms, 1), "batch40_per_card_ms": round(batch_ms / 40, 1)}
    if dev == "cuda":
        # 选择题：pass vs 两类 reject
        pred_pass = [ans(r, "verdict")["choice"] == "pass" for r in res]
        acc_verdict = sum(p == c["good"] for p, c in zip(pred_pass, cards)) / len(cards)
        # 数值读取题
        noul = [ans(r, "roe_high")["noul"] for r in res]
        acc_roe = sum((p > 0.5) == c["roe_gt15"] for p, c in zip(noul, cards)) / len(cards)
        # 打分题
        sc = [ans(r, "quality")["score"] for r in res]
        rho = spearmanr(sc, [c["roe"] for c in cards]).correlation
        # 校准：选择题最大概率的均值 vs 实际准确率
        conf = []
        for r in res:
            a = ans(r, "verdict"); probs = a.get("probs") or a.get("probabilities") or a.get("distribution") or {}
            conf.append(max(probs.values()) if isinstance(probs, dict) and probs else None)
        conf = [x for x in conf if x is not None]
        rec.update({"acc_verdict_pass_vs_reject": round(acc_verdict, 3), "acc_roe_gt15_numeric_read": round(acc_roe, 3),
                    "spearman_quality_score_vs_roe": round(rho, 3),
                    "mean_top_prob": round(statistics.mean(conf), 3) if conf else "n/a",
                    "choice_distribution": {k: sum(ans(r, "verdict")["choice"] == k for r in res) for k in Q["verdict"]["criteria"]},
                    "sample_answer": res[0]["answers"]})
    out[dev] = rec
    print(dev, json.dumps({k: v for k, v in rec.items() if k != "sample_answer"}, ensure_ascii=False), flush=True)
    del agent; torch.cuda.empty_cache() if dev == "cuda" else None
print("SAMPLE:", json.dumps(out.get("cuda", {}).get("sample_answer"), ensure_ascii=False)[:800])
json.dump(out, open("/mnt/d/LocalLLM/results/qm_bench_laya_zeroshot.json", "w"), ensure_ascii=False, indent=2, default=str)
