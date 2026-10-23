"""quantmind.line_a.p4.x4v2 — X4 v2：语义蒸馏标签、MacBERT 多标签分类、盲标金标准 v2（T-049，规则先登记）.

- `sample_sets`：训练（2019～2023-06，约 3 万）、校准（2023 下半年，约 3 千）、金标准 v2（测试期：A 关键词分层 14×30，
  B 均匀随机 200），按模板去重，排除金标准 v1 用过的公告；
- `distill`：DeepSeek 按 14 类的语义定义给标题打多标签（每次 20 条，结构化输出，断点续跑）；
- `train_macbert` / `predict_macbert`：只输入标题的 14 类多标签分类器，阈值在校准集上按 F1 选；
- `score`：金标准 v2 上比较规则、DeepSeek 直接分类、MacBERT。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quantmind.line_a.p4.x4_dataset import TYPE_KEYWORDS, TYPES

SEED = 20261024
TYPE_KEYS = list(TYPES)
SEVERE = ("investigation", "penalty", "st_change", "regulatory_letter", "restructuring", "control_change")
COLS = ["announcementId", "ts_code", "date", "title", "keep", "split"]
LABEL_SYSTEM = ("你是 A 股上市公司公告的分类标注员。下面给出 14 个事件类别的定义和一批公告标题。"
                "按类别的定义（不是按标题里有没有某个词）判断每条标题属于哪些类别：可以属于多个，也可以都不属于（types 为空）。"
                "注意区分：计划与已实施、上市公司本身与子公司或股东、收到与回复、限制性股票回购注销与公司回购股份。"
                "只根据标题判断。类别：\n" + "\n".join(f"- {k}：{v[0]}" for k, v in TYPES.items()))


# 指南 v2（2026-10-23 补充登记）：定义 + 排除项，逐条来自 T-038 的复核原则（2026-10-22 写定，早于金标准 v2）
EXCLUSIONS = {
    "performance_forecast": "不含：业绩快报、业绩承诺实现情况、未来几年的经营预测、子公司的业绩、对监管函件的回复",
    "performance_express": "不含：业绩预告、月度产销快报或经营数据快报、子公司（另一上市公司）的快报",
    "repurchase": "不含：限制性股票回购注销、回购子公司股权、出售或处置已回购股份（含转给员工持股计划）",
    "holder_reduction": "不含：公司减持已回购的股份、减持可转债或其他债券",
    "holder_increase": "不含：增持可转债、认购定增免于要约的法律意见",
    "unlock": "不含：解除限售条件未成就、回购注销不符合解除限售条件的股票、回购股东名单",
    "dividend": "不含：因分红引起的转股价 / 发行价 / 回购价调整、转债暂停转股、利润分配权等股权交易",
    "st_change": "只算已被实施或已撤销及其后续进展；不含：可能被实施的风险提示、可能被终止上市的提示、"
                 "简称带 ST 但内容是其他事项的公告",
    "investigation": "只算公司、控股股东、实际控制人被立案；不含：高管个人被立案或留置、诉讼执行立案、"
                     "尽职调查承诺函、反倾销调查、事故调查",
    "penalty": "算公司、子公司、相关人员、控股股东、实控人收到行政处罚（含事先告知书）、市场禁入、纪律处分、公开谴责；"
               "不含：最近五年处罚情况说明、行政监管措施、刑事处罚",
    "regulatory_letter": "只算收到交易所或监管机构的问询函、关注函、监管函、警示函、责令改正等监管措施；"
                         "不含：对函件的回复或延期回复、发行审核问询函的回复、独立董事或审计委员会的督促函、"
                         "公司向股东发出的征询函",
    "litigation": "涉及诉讼或仲裁，主体不限",
    "restructuring": "不含：控股股东集团层面的重组、子公司之间的吸收合并、一般的对外投资或资产购买、重组完成后的业绩承诺核查",
    "control_change": "不含：控股股东名称或工商信息变更、表决权委托到期、权益变动触及 1%、未表明控制权变更的股份转让协议",
}
GUIDE_V2 = ("你是 A 股上市公司公告的分类标注员。下面给出 14 个事件类别的定义（含排除项）和一批公告标题。"
            "按定义判断每条标题属于哪些类别：可以属于多个，也可以都不属于（types 为空）。只根据标题判断。类别：\n"
            + "\n".join(f"- {k}：{v[0]}。{EXCLUSIONS[k]}" for k, v in TYPES.items()))
# 指南 v2.1（X4 v2.2，T-049，2026-10-24）：只补 restructuring 的范围。v2.1 难例重训时发现 DeepSeek 对"重组过程中的文件"
# 标得不一致（这类难例里只有约 30% 标成 restructuring），金标准是按"算"标的
EXCLUSIONS_V21 = EXCLUSIONS | {
    "restructuring": "算：重组过程中的公告和中介机构文件（预案、草案及修订、问询函及其回复、核查意见、法律意见书、"
                     "审核与注册进展、自查报告、风险提示、停复牌、终止）；"
                     "不含：控股股东集团层面的重组、子公司之间的吸收合并、一般的对外投资或资产购买、"
                     "重组完成后的业绩承诺核查和持续督导、重组发行股份的限售股上市流通（归 unlock）",
}
GUIDE_V21 = ("你是 A 股上市公司公告的分类标注员。下面给出 14 个事件类别的定义（含排除项）和一批公告标题。"
             "按定义判断每条标题属于哪些类别：可以属于多个，也可以都不属于（types 为空）。只根据标题判断。类别：\n"
             + "\n".join(f"- {k}：{v[0]}。{EXCLUSIONS_V21[k]}" for k, v in TYPES.items()))


class Item(BaseModel):
    id: int
    types: list[Literal[tuple(TYPE_KEYS)]] = Field(default_factory=list)  # type: ignore[valid-type]


class Batch(BaseModel):
    items: list[Item]


def _years(x4_dir: Path, years) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(Path(x4_dir) / f"x4_{y}.parquet", columns=COLS) for y in years], ignore_index=True)


def _strat(df: pd.DataFrame, n_per_type: int, n_uniform: int, rng: random.Random) -> pd.DataFrame:
    picked: dict[str, str] = {}
    for t, kw in TYPE_KEYWORDS.items():
        cand = df[df["title"].str.contains(kw, regex=True)]
        for aid in cand["announcementId"].sample(min(n_per_type, len(cand)), random_state=rng.randrange(2**31)):
            picked.setdefault(aid, f"kw_{t}")
    rest = df[~df["announcementId"].isin(picked)]
    for aid in rest["announcementId"].sample(min(n_uniform, len(rest)), random_state=rng.randrange(2**31)):
        picked.setdefault(aid, "uniform")
    out = df.set_index("announcementId").loc[list(picked)].reset_index()
    out["stratum"] = out["announcementId"].map(picked)
    return out


def sample_sets(x4_dir: Path, out_dir: Path, gold_v1: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    tr = _years(x4_dir, range(2019, 2024))
    tr = tr[tr["keep"]]
    train = _strat(tr[tr["split"] == "train"], 1000, 16000, rng)
    calib = _strat(tr[tr["split"] == "calib"], 100, 1600, rng)
    te = _years(x4_dir, range(2024, 2027))
    used = set(pd.read_csv(gold_v1, dtype=str)["announcementId"]) if Path(gold_v1).exists() else set()
    te = te[te["keep"] & (te["split"] == "test") & ~te["announcementId"].isin(used)]
    a = []
    for i, (t, kw) in enumerate(TYPE_KEYWORDS.items()):
        cand = te[te["title"].str.contains(kw, regex=True)]
        a.append(cand.sample(min(30, len(cand)), random_state=SEED + i).assign(gold_for=t, part="A"))
    a = pd.concat(a, ignore_index=True)
    b = te[~te["announcementId"].isin(set(a["announcementId"]))].sample(200, random_state=SEED + 99).assign(part="B")
    for name, d in (("train", train), ("calib", calib), ("gold_a", a), ("gold_b", b)):
        d.to_parquet(out_dir / f"{name}.parquet")
    return {"train": len(train), "calib": len(calib), "gold_a": len(a), "gold_b": len(b), "excluded_v1": len(used)}


def distill(llm, df: pd.DataFrame, out_jsonl: Path, *, batch: int = 20, log=print, system: str = LABEL_SYSTEM,
            workers: int = 1, llm_factory=None) -> pd.DataFrame:
    """每批 20 条标题；已写入 out_jsonl 的公告跳过（断点续跑）。返回 announcementId → types 列表。
    workers > 1 时并行，每个线程用 llm_factory() 建自己的 LLM 对象，写文件加锁。"""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from quantmind.line_a.harness.llm import BudgetExceededError, TLSInterceptionError
    from quantmind.line_a.harness.structured import chat_structured
    out_jsonl = Path(out_jsonl)
    done = {}
    if out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["announcementId"]] = r["types"]
    todo = df[~df["announcementId"].isin(done)]
    rows = list(todo[["announcementId", "title"]].itertuples(index=False))
    chunks = [rows[i:i + batch] for i in range(0, len(rows), batch)]
    lock, local = threading.Lock(), threading.local()
    stop = {"err": None}

    def one(k_chunk):
        k, chunk = k_chunk
        if stop["err"] is not None:
            return
        cl = llm if workers == 1 or llm_factory is None else getattr(local, "llm", None)
        if cl is None:
            cl = local.llm = llm_factory()
        user = "\n".join(f"{j}. {r.title}" for j, r in enumerate(chunk)) + \
            '\n输出 JSON：{"items": [{"id": 序号, "types": [类别英文键, ...]}, ...]}，每条标题都要有一项。'
        try:
            out, _ = chat_structured(cl, [{"role": "system", "content": system}, {"role": "user", "content": user}], Batch,
                                     thinking=None, max_tokens=3000, meta={"task": "x4v2_distill"})
        except (TLSInterceptionError, BudgetExceededError) as e:
            stop["err"] = e
            return
        except Exception as e:  # noqa: BLE001  单批失败（重试后仍断网等）：跳过，下一轮续标
            log(f"  batch {k} failed: {type(e).__name__}: {str(e)[:120]}")
            return
        got = {it["id"]: it["types"] for it in (out or {}).get("items", [])}
        with lock:
            for j, r in enumerate(chunk):
                if j in got:                                  # 缺的那条留到下次续跑时重标
                    f.write(json.dumps({"announcementId": r.announcementId, "types": got[j]}, ensure_ascii=False) + "\n")
                    done[r.announcementId] = got[j]
            f.flush()
            if k % 50 == 0:
                log(f"  distill batch {k}/{len(chunks)}")
    with out_jsonl.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, enumerate(chunks)))
    if stop["err"] is not None:
        raise stop["err"]
    return pd.Series(done, name="types")


def label_matrix(df: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    y = np.zeros((len(df), len(TYPE_KEYS)), dtype=np.float32)
    for i, aid in enumerate(df["announcementId"]):
        for t in labels.get(aid, []) or []:
            y[i, TYPE_KEYS.index(t)] = 1.0
    return y


MODEL = "hfl/chinese-macbert-base"
CFG = {"lr": 3e-5, "epochs": 3, "batch": 64, "warmup": 0.06, "max_len": 64, "seed": SEED}


def train_macbert(titles: list[str], y: np.ndarray, out_dir: Path, *, log=print) -> Path:
    import torch
    import transformers
    torch.manual_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(TYPE_KEYS), problem_type="multi_label_classification").cuda()
    steps = CFG["epochs"] * ((len(titles) + CFG["batch"] - 1) // CFG["batch"])
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
    sch = transformers.get_linear_schedule_with_warmup(opt, int(CFG["warmup"] * steps), steps)
    pos = y.mean(axis=0).clip(1e-3, None)
    pw = torch.tensor(np.clip((1 - pos) / pos, 1, 20), dtype=torch.float32).cuda()   # 稀有类别加权，上限 20
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    model.train()
    step = 0
    for ep in range(CFG["epochs"]):
        order = rng.permutation(len(titles))
        for i in range(0, len(order), CFG["batch"]):
            idx = order[i:i + CFG["batch"]]
            enc = tok([titles[k] for k in idx], truncation=True, max_length=CFG["max_len"], padding=True,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits
            loss = lossf(logits.float(), torch.tensor(y[idx]).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                log(f"  epoch {ep} step {step}/{steps} loss {loss.item():.4f}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "x4v2_config.json").write_text(json.dumps({"cfg": CFG, "types": TYPE_KEYS, "pos_weight": pw.tolist(),
                                                          "n_train": len(titles)}, ensure_ascii=False, indent=1))
    return out_dir


def predict_macbert(model_dir: Path, titles: list[str], batch: int = 256) -> np.ndarray:
    import torch
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(titles), batch):
            enc = tok(titles[i:i + batch], truncation=True, max_length=CFG["max_len"], padding=True,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(model(**enc).logits.float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TYPE_KEYS)))


def thresholds(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """每类在校准集上按 F1 选阈值；校准集没有正例的类别用 0.5。"""
    thr = np.full(p.shape[1], 0.5)
    for j in range(p.shape[1]):
        if y[:, j].sum() == 0:
            continue
        best = -1.0
        for t in np.linspace(0.05, 0.95, 37):
            pr = p[:, j] >= t
            tp = (pr & (y[:, j] == 1)).sum()
            f1 = 2 * tp / max(1, pr.sum() + y[:, j].sum())
            if f1 > best:
                best, thr[j] = f1, t
    return thr


def gold_pairs(gold_a: pd.DataFrame, gold_b: pd.DataFrame, labels_a: dict, labels_b: dict) -> pd.DataFrame:
    """把 A（单类判断）和 B（全类别）展开成 (announcementId, title, type, y, part) 的判断表。"""
    rows = [{"announcementId": r.announcementId, "title": r.title, "type": r.gold_for, "y": labels_a[r.announcementId],
             "part": "A"} for r in gold_a.itertuples() if r.announcementId in labels_a]
    for r in gold_b.itertuples():
        if r.announcementId not in labels_b:
            continue
        for t in TYPE_KEYS:
            rows.append({"announcementId": r.announcementId, "title": r.title, "type": t,
                         "y": t in labels_b[r.announcementId], "part": "B"})
    return pd.DataFrame(rows)


def _prf(y: pd.Series, p: pd.Series) -> dict:
    tp = int((y & p).sum())
    prec, rec = tp / max(1, int(p.sum())), tp / max(1, int(y.sum()))
    return {"n": int(len(y)), "n_pos": int(y.sum()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(1e-9, prec + rec)}


def score(pairs: pd.DataFrame, preds: dict[str, pd.Series]) -> dict:
    """preds：{系统名: 与 pairs 同索引的布尔预测}。"""
    out = {}
    y = pairs["y"].astype(bool)
    for name, p in preds.items():
        p = p.astype(bool)
        by = {t: _prf(y[pairs["type"] == t], p[pairs["type"] == t]) for t in TYPE_KEYS}
        sev = pairs["type"].isin(SEVERE)
        out[name] = {"all": _prf(y, p), "macro_f1": float(np.mean([v["f1"] for v in by.values() if v["n_pos"]])),
                     "severe": _prf(y[sev], p[sev]), "part_A": _prf(y[pairs["part"] == "A"], p[pairs["part"] == "A"]),
                     "part_B": _prf(y[pairs["part"] == "B"], p[pairs["part"] == "B"]), "by_type": by}
    return out
