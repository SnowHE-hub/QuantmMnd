"""scripts/line_a/run_x11.py — X11 转写绑定核对的评测（T-044；规则在运行前登记）.

1. 扰动集：从 T-040 试跑各案例的骨架里取有数值的事实和推断，先生成"正确分句"（数字换成占位符），
   再按固定规则造四类错误（方向、主体、阶段、期间），比较 v1（check_rewrite）与 v2（check_bindings）的检出率；
2. 真实转写集：对同一批案例重放转写调用（同一请求命中缓存，不重复付费），在通过 v1 的模型转写上跑 v2，
   把 v2 报出的条目写成表，供逐条查看是否误报。
用法：python scripts/line_a/run_x11.py --day-dir <research/daily/2026-10-22> --out <dir> [--seed 20261023]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from quantmind.line_a.harness import rewrite as rw  # noqa: E402

SWAP = [("增长", "下降"), ("增加", "减少"), ("上升", "下降"), ("提高", "降低"), ("扩大", "收窄"), ("盈利", "亏损"),
        ("扭亏", "首亏"), ("回升", "回落"), ("上涨", "下跌"), ("增至", "降至"), ("增持", "减持")]
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def correct_clause(item: dict, ref: str) -> str:
    """声明原文 → 正确分句：日期保留，其余数字换成该声明的占位符，去掉会切断分句的标点。"""
    t = re.sub(r"[，,；;。\n]", " ", item["text"])
    dates = []

    def keep(m):                                   # 标记里不能有数字，否则会被下一步当成数字替换掉（v1 评测脚本的错误）
        dates.append(m.group(0))
        return "\x00" + chr(0xE000 + len(dates) - 1) + "\x00"
    t = rw._DATE.sub(keep, t)
    t = NUM.sub("{{c:" + ref + "}}", t)
    t = re.sub("\x00(.)\x00", lambda m: dates[ord(m.group(1)) - 0xE000], t)
    if "{{c:" not in t:
        t += "（{{c:" + ref + "}}）"
    return t


def perturb(kind: str, s: str) -> str | None:
    if kind == "direction":
        for a, b in SWAP:
            if a in s:
                return s.replace(a, b)
            if b in s:
                return s.replace(b, a)
        return None
    if kind == "subject":
        for a, b in (("子公司", "公司"), ("控股股东", "公司"), ("实际控制人", "公司")):
            if a in s:
                return s.replace(a, b, 1)
        return s.replace("公司", "子公司", 1) if "公司" in s else None
    if kind == "stage":
        if "拟" in s:
            return s.replace("拟", "已", 1)
        m = rw.DONE.search(s)
        return s[:m.start()] + "拟" + s[m.end():] if m else None
    if kind == "period":
        m = rw.YEAR.search(s)
        return s[:m.start()] + str(int(m.group(0)) + 1) + s[m.end():] if m else None
    raise ValueError(kind)


def load_cases(day: Path) -> list[tuple[str, dict]]:
    cases = pd.read_parquet(day / "f6_cases.parquet")
    out = []
    for r in cases[cases["status"] == "ok"].itertuples():
        paths = json.loads(r.role_paths)
        outs = {k: json.loads(Path(p).read_text(encoding="utf-8")) for k, p in paths.items()
                if k != "A6" and Path(p).exists()}
        review = json.loads(Path(paths["A6"]).read_text(encoding="utf-8")).get("output") if "A6" in paths else None
        out.append((r.ticker, rw.skeleton(outs, review)))
    return out


def main() -> None:
    from quantmind.line_a.harness.llm import LLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--cache", type=Path, required=True, help="与每日研究相同的模型缓存目录")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(a.day_dir)
    rng = random.Random(a.seed)
    items = [(t, ref, it) for t, sk in cases for col in ("facts", "inferences") for ref in sk["columns"][col]
             if (it := sk["claims"][ref])["value"] is not None]
    res: dict = {"n_cases": len(cases), "n_items": len(items), "perturbation": {}}
    rows = []
    for kind in ("direction", "subject", "stage", "period"):
        pool = [(t, ref, it, p) for t, ref, it in items if (p := perturb(kind, correct_clause(it, ref))) is not None]
        rng.shuffle(pool)
        pool = pool[:100]
        hit1 = hit2 = fp2 = 0
        for t, ref, it, p in pool:
            mini = {"columns": {"facts": [ref], "inferences": [], "counter_and_limits": [], "to_verify": []},
                    "claims": {ref: it}}
            ok = correct_clause(it, ref)
            e1, e2 = rw.check_rewrite(mini, {"facts": p}), rw.check_bindings(mini, {"facts": p})
            hit1 += bool(e1)
            hit2 += bool(e2)
            fp2 += bool(rw.check_bindings(mini, {"facts": ok}))
            rows.append({"kind": kind, "ticker": t, "ref": ref, "orig": it["text"], "correct": ok, "perturbed": p,
                         "v1_errors": "; ".join(e1), "v2_errors": "; ".join(e2)})
        n = len(pool)
        res["perturbation"][kind] = {"n": n, "v1_detect": hit1 / max(1, n), "v2_detect": hit2 / max(1, n),
                                     "v2_flags_on_correct": fp2 / max(1, n)}
    pd.DataFrame(rows).to_csv(a.out / "perturbation_rows.csv", index=False, encoding="utf-8-sig")
    llm = LLM(ledger=a.out / "llm_ledger.jsonl", cache_dir=a.cache, budget_rmb=2.0)
    real = []
    n_model = n_v1_fail = 0
    for t, sk in cases:
        from quantmind.line_a.harness.structured import chat_structured
        try:
            text, rec = chat_structured(llm, rw.rewrite_messages(sk), rw.RewriteOut, thinking=None, max_tokens=4000,
                                        meta={"stage": "x11_replay", "ticker": t})
        except Exception as e:  # noqa: BLE001
            real.append({"ticker": t, "status": f"error {type(e).__name__}"})
            continue
        if text is None:
            real.append({"ticker": t, "status": "invalid_after_repair"})
            continue

        class _R:
            cached = all(not a["parse_errors"] for a in rec["attempts"])
        r = _R()
        e1 = rw.check_rewrite(sk, text)
        if e1:
            n_v1_fail += 1
            real.append({"ticker": t, "status": "v1_fail", "cached": r.cached, "v1": "; ".join(e1[:5])})
            continue
        n_model += 1
        e2 = rw.check_bindings(sk, text)
        for e in e2:
            ref = e.split(": ", 1)[1].split(" ", 1)[0]
            seg = next((rw._clause_of(s, m.start(), m.end()) for s in text.values() if isinstance(s, str)
                        for m in rw.PH.finditer(s) if f"{m.group(1)}:{m.group(2)}" == ref), "")
            real.append({"ticker": t, "status": "v2_flag", "cached": r.cached, "error": e, "ref": ref,
                         "claim_text": sk["claims"].get(ref, {}).get("text"), "clause": seg})
        if not e2:
            real.append({"ticker": t, "status": "v2_pass", "cached": r.cached})
    rd = pd.DataFrame(real)
    rd.to_csv(a.out / "real_rewrites.csv", index=False, encoding="utf-8-sig")
    res["real"] = {"n_cases": len(cases), "n_model_pass_v1": n_model, "n_v1_fail": n_v1_fail,
                   "n_cases_v2_flagged": int(rd[rd["status"] == "v2_flag"]["ticker"].nunique()) if len(rd) else 0,
                   "n_v2_flags": int((rd["status"] == "v2_flag").sum()) if len(rd) else 0}
    (a.out / "x11_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
