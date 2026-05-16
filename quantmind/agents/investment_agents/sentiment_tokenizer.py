"""稳定可导入的 jieba 分词器 — 供 sklearn TfidfVectorizer(analyzer=...) pickle 复用."""

from __future__ import annotations


def jieba_cut_analyzer(text: str) -> list[str]:
    import jieba

    return list(jieba.cut(str(text)))
