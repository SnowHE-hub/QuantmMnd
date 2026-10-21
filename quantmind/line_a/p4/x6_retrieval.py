"""quantmind.line_a.p4.x6_retrieval — X6 检索与重排：年报内检索的开发集和对比（03_final_plan_v2 第 9.1、10.4 节 X6；P4）.

检索场景：角色在一份指定的年报里找证据（文档内检索），检索单位是 pdf_locate 切出的段落（与线上一致）。

开发集 v1（全部自动生成，可复现）：
- 文档：X13 的开发集 40 份年报作 X6 开发集，X13 的测试集 40 份作 X6 测试集；另外各加 10 份审计意见非标准的年报
  （保留意见、无法表示意见、带强调事项段；种子 20261021，排除已用过的股票），用来测"关键风险证据"；
- 事实查询：X13 的可用标准值（11 个财报字段），查询句式见 FACT_QUERIES（先定下，再看结果）；
  相关段落 = 去掉空白后包含该数值的段落，写法只认"千分位两位小数（元）"和"x.xx 亿元"两种；
  正文里一个都找不到的查询丢弃并计数；
- 风险查询：审计意见类型，标准答案来自 fina_audit 表；相关段落 = 含该意见类型说法的段落（规则见 OPINION_PATTERNS）。
指标：每类查询的 Recall@5（前 5 条里有任一相关段落）、MRR@10。
方法：bm25（线上基线）、dense:<模型>（句向量，余弦）、hybrid:<模型>（BM25 与向量的 RRF，k = 60）、
rerank:<模型>（BM25 前 50 条 + 交叉编码器重排）。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from quantmind.line_a.evidence.pdf_locate import parse_pdf
from quantmind.line_a.evidence.retrieval import BM25

SEED = 20261021
FACT_QUERIES = {
    "total_revenue": "{y}年营业总收入是多少",
    "revenue": "{y}年营业收入是多少",
    "operate_profit": "{y}年营业利润是多少",
    "n_income": "{y}年净利润是多少",
    "n_income_attr_p": "{y}年归属于上市公司股东的净利润",
    "total_assets": "{y}年末总资产是多少",
    "total_liab": "{y}年末负债合计",
    "total_hldr_eqy_exc_min_int": "{y}年末归属于上市公司股东的净资产",
    "total_cur_assets": "{y}年末流动资产合计",
    "total_cur_liab": "{y}年末流动负债合计",
    "n_cashflow_act": "{y}年经营活动产生的现金流量净额",
}
RISK_QUERY = "{y}年度财务报告的审计意见类型是什么"
OPINION_PATTERNS = {
    "标准无保留意见": r"标准的?无保留意见",
    "带强调事项段的无保留意见": r"强调事项",
    "保留意见": r"(?<!无)保留意见",
    "无法表示意见": r"无法表示意见",
    "否定意见": r"否定意见",
}
RRF_K = 60


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _forms(v: float) -> list[str]:
    a = abs(v)
    out = [f"{a:,.2f}"]
    if a >= 1e8:
        out.append(f"{a / 1e8:.2f}亿元")
    return out


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


# ---------------------------------------------------------------- 开发集
def risk_docs(lake: Path, n: int, seed: int, exclude: set) -> pd.DataFrame:
    """审计意见非标准的年报（报告年度 2019–2023）。"""
    from quantmind.line_a.p4.x13_pdf import annual_reports
    fa = pd.read_parquet(sorted((Path(lake) / "events").glob("fina_audit_2*.parquet"))[-1])
    fa = fa.assign(year=fa["end_date"].astype(str).str.replace("-", "").str[:4].astype(int),
                   mmdd=fa["end_date"].astype(str).str.replace("-", "").str[4:8])
    fa = fa[(fa["mmdd"] == "1231") & fa["year"].between(2019, 2023) & (fa["audit_result"] != "标准无保留意见")]
    ar = annual_reports(Path(lake) / "text/cninfo/_parts")
    m = ar.merge(fa[["ts_code", "year", "audit_result"]].drop_duplicates(["ts_code", "year"]), on=["ts_code", "year"])
    m = m[~m["ts_code"].isin(exclude)].drop_duplicates("ts_code")
    return m.sample(min(n, len(m)), random_state=seed).reset_index(drop=True)


def build_queries(docs: pd.DataFrame, gold: pd.DataFrame, paras: dict[str, list], audit: dict) -> pd.DataFrame:
    """docs：announcementId, ts_code, year；gold：X13 的 gold.csv；paras：{ann_id: [(para_id, text)]}；
    audit：{(ts_code, year): 审计意见类型}。"""
    rows, dropped = [], {"fact_no_match": 0, "risk_no_match": 0}
    g = gold[gold["value"].notna() & gold["excluded"].isna()] if len(gold) else gold
    for d in docs.itertuples(index=False):
        a = str(d.announcementId)
        ps = paras.get(a)
        if not ps:
            continue
        sq = [(pid, _squash(t)) for pid, t in ps]
        for r in g[g["ann_id"].astype(str) == a].itertuples(index=False):
            forms = _forms(float(r.value))
            rel = [pid for pid, t in sq if any(f in t for f in forms)]
            if not rel:
                dropped["fact_no_match"] += 1
                continue
            rows.append({"qid": _sha(f"{a}:{r.field}")[:12], "ann_id": a, "kind": "fact", "field": r.field,
                         "query": FACT_QUERIES[r.field].format(y=d.year), "relevant": rel})
        op = audit.get((d.ts_code, int(d.year)))
        if op in OPINION_PATTERNS:
            pat = re.compile(OPINION_PATTERNS[op])
            rel = [pid for pid, t in sq if pat.search(t)]
            if rel:
                rows.append({"qid": _sha(f"{a}:audit")[:12], "ann_id": a, "kind": "risk" if op != "标准无保留意见" else "audit_std",
                             "field": op, "query": RISK_QUERY.format(y=d.year), "relevant": rel})
            else:
                dropped["risk_no_match"] += 1
    q = pd.DataFrame(rows)
    q.attrs["dropped"] = dropped
    return q


def parse_docs(docs: pd.DataFrame, cache_dir: Path) -> dict[str, list]:
    out = {}
    for d in docs.itertuples(index=False):
        a = str(d.announcementId)
        p = next(Path(cache_dir).glob(f"*/{a}.pdf"), None)
        if p is None:
            continue
        doc = parse_pdf(p, a, tables=False)
        out[a] = [(f"{a}:{x.page}:{x.idx}", x.text) for x in doc.paras]
    return out


# ---------------------------------------------------------------- 检索方法
class Dense:
    def __init__(self, model: str, batch: int = 64):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device="cuda")
        self.m.max_seq_length = 512
        self.batch = batch

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.m.encode(texts, batch_size=self.batch, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False)


class Rerank:
    """交叉编码器重排：BM25 取前 RERANK_POOL 条，再按 (查询, 段落) 打分重排。"""
    def __init__(self, model: str, batch: int = 32):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(model, device="cuda", max_length=512)
        self.batch = batch

    def order(self, query: str, cands: list[tuple[str, str]]) -> list[str]:
        if not cands:
            return []
        s = self.m.predict([(query, t) for _, t in cands], batch_size=self.batch, show_progress_bar=False)
        return [cands[i][0] for i in np.argsort(-np.asarray(s))]


RERANK_POOL = 50


def _rank_bm25(ps: list, query: str, k: int) -> list[str]:
    return [h.para_id for h in BM25(ps).search(query, top_k=k)]


def _rrf(lists: list[list[str]], k: int) -> list[str]:
    sc: dict[str, float] = {}
    for lst in lists:
        for r, pid in enumerate(lst):
            sc[pid] = sc.get(pid, 0.0) + 1.0 / (RRF_K + r + 1)
    return [p for p, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))][:k]


def evaluate(queries: pd.DataFrame, paras: dict[str, list], method: str, *, dense: Dense | None = None,
             rerank: Rerank | None = None, k: int = 10) -> dict:
    """返回 {总体 / 每类查询: Recall@5, MRR@10, n}，以及每条查询的排名。"""
    per = []
    emb_cache: dict[str, np.ndarray] = {}
    for a, qs in queries.groupby("ann_id"):
        ps = paras[a]
        text = dict(ps)
        if method.startswith(("dense", "hybrid")):
            if a not in emb_cache:
                emb_cache = {a: dense.encode([t for _, t in ps])}
            qv = dense.encode(qs["query"].tolist())
            sims = qv @ emb_cache[a].T
        for j, q in enumerate(qs.itertuples(index=False)):
            if method == "bm25":
                ranked = _rank_bm25(ps, q.query, k)
            elif method.startswith("rerank"):
                pool = _rank_bm25(ps, q.query, RERANK_POOL)
                ranked = rerank.order(q.query, [(p, text[p]) for p in pool])[:k]
            else:
                order = np.argsort(-sims[j])[: max(k, 50)]
                dl = [ps[i][0] for i in order]
                ranked = dl[:k] if method.startswith("dense") else _rrf([_rank_bm25(ps, q.query, 50), dl[:50]], k)
            rel = set(q.relevant)
            rank = next((i + 1 for i, p in enumerate(ranked) if p in rel), None)
            per.append({"qid": q.qid, "kind": q.kind, "field": q.field, "rank": rank})
    df = pd.DataFrame(per)

    def agg(d):
        return {"n": int(len(d)), "recall@5": float((d["rank"].fillna(99) <= 5).mean()),
                "mrr@10": float(d["rank"].map(lambda r: 0.0 if pd.isna(r) else 1.0 / r).mean())}
    out = {"method": method, "all": agg(df), "by_kind": {kd: agg(g) for kd, g in df.groupby("kind")},
           "by_field": {f: agg(g) for f, g in df[df["kind"] == "fact"].groupby("field")}}
    out["per_query"] = df.to_dict("records")
    return out


def save_queries(q: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in q.to_dict("records"))
    path.write_text(text, encoding="utf-8")
    return _sha(text)
