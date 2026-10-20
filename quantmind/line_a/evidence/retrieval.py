"""quantmind.line_a.evidence.retrieval — 单只股票文档库内的 BM25 检索（03_final_plan_v2 第 5 节 F5(b)4 第一版）.

- 分词：中文按字二元组（不依赖分词词典，结果确定）；英文单词与数字整体保留（金额、比例、年份能精确命中）。
- 检索单位是段落（来自 pdf_locate 的定位结果），返回段落编号、页码和分数，原文由调用方按定位取回。
- 四种用途（查事实 / 查变化 / 查背景 / 查反证）共用这一个底层；"查反证"由调用方构造针对性的查询词。
向量检索和重排（BGE、Jina）是 P4 的候选增强，必须在这里的基线上实测之后才接入。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_CJK = re.compile(r"[一-鿿]+")
_WORD = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?%?")


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for m in _CJK.finditer(text):
        s = m.group()
        toks += [s] if len(s) == 1 else [s[i:i + 2] for i in range(len(s) - 1)]
    toks += [w.lower() for w in _WORD.findall(text)]
    return toks


@dataclass(frozen=True)
class Hit:
    para_id: str
    score: float


class BM25:
    def __init__(self, paras: list[tuple[str, str]], *, k1: float = 1.5, b: float = 0.75):
        """paras：[(段落编号, 文本)]。"""
        self.ids = [p for p, _ in paras]
        self.tf = [Counter(tokenize(t)) for _, t in paras]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        q = Counter(tokenize(query))
        out = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = c.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                out.append(Hit(self.ids[i], s))
        return sorted(out, key=lambda h: (-h.score, h.para_id))[:top_k]
