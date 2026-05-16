"""quantmind.kb.retriever — 混合检索器（ChromaDB + BM25 + BGE-Reranker）.

检索流程：
1. Dense 检索：ChromaDB 向量召回 top_k × 3 候选
2. Sparse 检索：BM25 召回 top_k × 3 候选
3. 候选合并：RRF（Reciprocal Rank Fusion）加权合并，去重
4. PIT 过滤：过滤 published_date > as_of 的文档
5. Reranker：BGE-Reranker-v2 精排，返回最终 top_k

Collection 命名：quantmind_kb_{collection_name}
"""

from __future__ import annotations

import threading
from datetime import date
from typing import Any

import numpy as np
from loguru import logger

from quantmind.kb.chunker import Chunk
from quantmind.kb.embedding_service import EmbeddingService, get_embedding_service

__all__ = ["HybridRetriever", "RetrievalResult"]

_DEFAULT_COLLECTION = "default"
_DEFAULT_CHROMA_DIR = ".cache/chromadb"
_RRF_K = 60  # RRF 常数


class RetrievalResult:
    """单条检索结果."""

    def __init__(
        self,
        text: str,
        metadata: dict[str, Any],
        score: float,
        retrieval_method: str = "hybrid",
    ) -> None:
        self.text = text
        self.metadata = metadata
        self.score = score
        self.retrieval_method = retrieval_method

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def ticker(self) -> str:
        return self.metadata.get("ticker", "")

    @property
    def published_date(self) -> date | None:
        d = self.metadata.get("published_date")
        if isinstance(d, str):
            try:
                from datetime import datetime
                return datetime.strptime(d, "%Y-%m-%d").date()
            except ValueError:
                return None
        if isinstance(d, date):
            return d
        return None

    def __repr__(self) -> str:
        return f"RetrievalResult(score={self.score:.3f}, source={self.source!r})"


class HybridRetriever:
    """混合检索器：Dense（ChromaDB）+ Sparse（BM25）+ Cross-Encoder Reranker.

    Args:
        collection_name: ChromaDB collection 名称（前缀 quantmind_kb_）
        chroma_dir:      ChromaDB 持久化目录
        embedding_svc:   EmbeddingService 实例，None = 使用全局单例
        use_reranker:    是否启用 BGE-Reranker-v2 精排，默认 True
        reranker_model:  Reranker 模型名称
        rrf_dense_weight: RRF 中 dense 检索的权重（sparse = 1-weight）
    """

    def __init__(
        self,
        collection_name: str = _DEFAULT_COLLECTION,
        chroma_dir: str = _DEFAULT_CHROMA_DIR,
        embedding_svc: EmbeddingService | None = None,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        rrf_dense_weight: float = 0.6,
    ) -> None:
        self.collection_name = f"quantmind_kb_{collection_name}"
        self.chroma_dir = chroma_dir
        self._embedding_svc = embedding_svc
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        self.rrf_dense_weight = rrf_dense_weight

        self._chroma_client = None
        self._chroma_collection = None
        self._bm25_index = None
        self._bm25_docs: list[dict] = []  # {text, metadata}
        self._reranker = None
        self._lock = threading.RLock()

    # ── 公开 API ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """向知识库添加文本块.

        Args:
            chunks: Chunk 列表

        Returns:
            实际添加数量
        """
        if not chunks:
            return 0

        coll = self._get_collection()
        emb_svc = self._get_embedding_svc()

        texts = [c.text for c in chunks]
        logger.info(f"[Retriever] embed_batch start: n_chunks={len(chunks)}")
        embeddings = emb_svc.embed_batch(texts)
        logger.info(f"[Retriever] embed_batch done: n_chunks={len(chunks)}")

        # 生成唯一 ID（基于内容哈希）
        import hashlib
        ids = [
            hashlib.md5(f"{c.source}_{c.chunk_idx}_{c.text[:50]}".encode()).hexdigest()
            for c in chunks
        ]

        # 序列化 metadata（ChromaDB 要求 value 为 str/int/float/bool）
        metadatas = [self._serialize_metadata(c.metadata) for c in chunks]

        logger.info(f"[Retriever] Chroma upsert start: n={len(chunks)}")
        try:
            coll.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"[Retriever] Chroma upsert done: n={len(chunks)}")
        except Exception as e:
            logger.error(f"[Retriever] ChromaDB upsert 失败: {e}")
            return 0

        # 更新 BM25 索引
        logger.info(f"[Retriever] BM25 rebuild start: n_new={len(chunks)}")
        self._rebuild_bm25(chunks)
        logger.info("[Retriever] BM25 rebuild done")

        logger.info(f"[Retriever] added {len(chunks)} chunks to '{self.collection_name}'")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        as_of: date | None = None,
        ticker_filter: str | None = None,
        source_type_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """混合检索.

        Args:
            query:               查询文本
            top_k:               最终返回数量
            as_of:               PIT 截止日期（过滤 published_date > as_of）
            ticker_filter:       只返回指定股票代码的文档
            source_type_filter:  只返回指定类型（news/report/filing）

        Returns:
            按相关性排序的 RetrievalResult 列表
        """
        candidate_k = top_k * 4  # 候选池大小
        if ticker_filter:
            candidate_k = max(candidate_k, top_k * 40, 200)
        dense_results = self._dense_retrieve(query, candidate_k)

        # 2. Sparse（BM25）检索
        sparse_results = self._sparse_retrieve(query, candidate_k)

        # 3. RRF 合并
        merged = self._rrf_merge(dense_results, sparse_results)

        # 4. PIT 过滤 + 字段过滤
        filtered = self._apply_filters(merged, as_of, ticker_filter, source_type_filter)

        # 取 top candidate_k 精排
        candidates = filtered[:candidate_k]

        # 5. Reranker 精排
        if self.use_reranker and candidates:
            candidates = self._rerank(query, candidates)

        return candidates[:top_k]

    def count(self) -> int:
        """返回集合中的文档数."""
        try:
            coll = self._get_collection()
            return coll.count()
        except Exception:
            return 0

    def fetch_by_metadata(
        self,
        where: dict,
        as_of: date | None = None,
        limit: int = 100,
    ) -> list[RetrievalResult]:
        """直接用 ChromaDB WHERE filter 拉取文档（不做向量检索）.

        适合按 ticker + doc_type 精确定位结构化快照文档。
        结果按 published_date 降序（最新在前）。

        Args:
            where:  ChromaDB where 字典，如 {"$and": [{"ticker": {"$eq": "600519.SH"}},
                    {"doc_type": {"$eq": "company_profile"}}]}
            as_of:  PIT 截止日期（过滤 published_date > as_of）
            limit:  最多返回文档数
        """
        try:
            coll = self._get_collection()
            results = coll.get(
                where=where,
                include=["documents", "metadatas"],
                limit=limit,
            )
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            items: list[RetrievalResult] = []
            for doc, meta in zip(docs, metas):
                r = RetrievalResult(text=doc, metadata=meta, score=1.0, retrieval_method="metadata_filter")
                if as_of is not None and r.published_date is not None:
                    if r.published_date > as_of:
                        continue
                items.append(r)
            # sort newest first
            items.sort(key=lambda x: (x.metadata.get("published_date") or ""), reverse=True)
            return items
        except Exception as e:
            logger.warning(f"[Retriever] fetch_by_metadata 失败: {e}")
            return []

    def clear(self) -> None:
        """清空集合."""
        try:
            client = self._get_chroma_client()
            client.delete_collection(self.collection_name)
            self._chroma_collection = None
            self._bm25_index = None
            self._bm25_docs = []
            logger.info(f"[Retriever] cleared collection '{self.collection_name}'")
        except Exception as e:
            logger.warning(f"[Retriever] clear 失败: {e}")

    # ── Dense 检索 ────────────────────────────────────────────────────────────

    def _dense_retrieve(self, query: str, k: int) -> list[dict]:
        """ChromaDB 向量检索."""
        try:
            coll = self._get_collection()
            if coll.count() == 0:
                return []

            emb_svc = self._get_embedding_svc()
            query_emb = emb_svc.embed(query)

            results = coll.query(
                query_embeddings=[query_emb],
                n_results=min(k, coll.count()),
                include=["documents", "metadatas", "distances"],
            )

            items = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, dists):
                # ChromaDB 返回 L2 距离，转为相似度分数
                score = 1.0 / (1.0 + dist)
                items.append({"text": doc, "metadata": meta, "score": score})

            return items
        except Exception as e:
            logger.warning(f"[Retriever] dense_retrieve 失败: {e}")
            return []

    # ── Sparse（BM25）检索 ────────────────────────────────────────────────────

    def _sparse_retrieve(self, query: str, k: int) -> list[dict]:
        """BM25 稀疏检索."""
        if not self._bm25_docs:
            return []
        try:
            from rank_bm25 import BM25Okapi
            if self._bm25_index is None:
                return []

            tokenized_query = self._tokenize(query)
            scores = self._bm25_index.get_scores(tokenized_query)

            top_indices = np.argsort(scores)[::-1][:k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append({
                        "text": self._bm25_docs[idx]["text"],
                        "metadata": self._bm25_docs[idx]["metadata"],
                        "score": float(scores[idx]),
                    })
            return results
        except Exception as e:
            logger.warning(f"[Retriever] sparse_retrieve 失败: {e}")
            return []

    def _rebuild_bm25(self, new_chunks: list[Chunk]) -> None:
        """重建 BM25 索引（追加新文档后重建）."""
        try:
            from rank_bm25 import BM25Okapi
            # 追加
            for c in new_chunks:
                self._bm25_docs.append({"text": c.text, "metadata": c.metadata})
            # 重建
            tokenized = [self._tokenize(d["text"]) for d in self._bm25_docs]
            self._bm25_index = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("[Retriever] rank-bm25 未安装，跳过 BM25 索引")
        except Exception as e:
            logger.warning(f"[Retriever] BM25 rebuild 失败: {e}")

    def _tokenize(self, text: str) -> list[str]:
        """中英文分词（jieba + 空格切分英文）."""
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            # 回退：空格+字符切分
            import re
            return re.findall(r"[\w一-鿿]+", text)

    # ── RRF 合并 ──────────────────────────────────────────────────────────────

    def _rrf_merge(
        self,
        dense: list[dict],
        sparse: list[dict],
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion 合并两路检索结果."""
        rrf_scores: dict[str, float] = {}
        text_to_item: dict[str, dict] = {}

        def _add(results: list[dict], weight: float) -> None:
            for rank, item in enumerate(results):
                key = item["text"][:100]  # 用前100字符做 key
                rrf_score = weight / (_RRF_K + rank + 1)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score
                if key not in text_to_item:
                    text_to_item[key] = item

        _add(dense, self.rrf_dense_weight)
        _add(sparse, 1.0 - self.rrf_dense_weight)

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for key, score in merged:
            item = text_to_item[key]
            results.append(RetrievalResult(
                text=item["text"],
                metadata=item.get("metadata", {}),
                score=score,
                retrieval_method="hybrid",
            ))
        return results

    # ── 过滤 ─────────────────────────────────────────────────────────────────

    def _apply_filters(
        self,
        results: list[RetrievalResult],
        as_of: date | None,
        ticker_filter: str | None,
        source_type_filter: str | None,
    ) -> list[RetrievalResult]:
        """应用 PIT 过滤和字段过滤."""
        filtered = []
        for r in results:
            # PIT 过滤
            if as_of is not None and r.published_date is not None:
                if r.published_date > as_of:
                    continue
            # ticker 过滤
            if ticker_filter:
                if (r.ticker or "") != ticker_filter:
                    continue
            # source_type 过滤
            if source_type_filter:
                st = r.metadata.get("source_type", "")
                if st and st != source_type_filter:
                    continue
            filtered.append(r)
        return filtered

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """BGE-Reranker-v2 精排."""
        if not candidates:
            return candidates

        try:
            reranker = self._get_reranker()
            if reranker is None:
                return candidates

            pairs = [[query, r.text] for r in candidates]
            scores = reranker.compute_score(pairs, normalize=True)

            if not isinstance(scores, (list, tuple)):
                scores = [scores]

            for r, s in zip(candidates, scores):
                r.score = float(s)
                r.retrieval_method = "reranked"

            return sorted(candidates, key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.warning(f"[Retriever] rerank 失败，使用原始排序: {e}")
            return candidates

    def _get_reranker(self):
        with self._lock:
            if self._reranker is None:
                try:
                    from FlagEmbedding import FlagReranker
                    logger.info(f"[Retriever] loading reranker {self.reranker_model_name}")
                    self._reranker = FlagReranker(
                        self.reranker_model_name,
                        use_fp16=True,
                    )
                    logger.info("[Retriever] reranker loaded OK")
                except ImportError:
                    logger.warning("[Retriever] FlagEmbedding 未安装，跳过 reranker")
                    self._reranker = None
                except Exception as e:
                    logger.warning(f"[Retriever] reranker 加载失败: {e}")
                    self._reranker = None
            return self._reranker

    # ── ChromaDB 初始化 ───────────────────────────────────────────────────────

    def _get_chroma_client(self):
        with self._lock:
            if self._chroma_client is None:
                try:
                    import chromadb
                    self._chroma_client = chromadb.PersistentClient(
                        path=self.chroma_dir
                    )
                    logger.debug(f"[Retriever] ChromaDB client @ {self.chroma_dir}")
                except Exception as e:
                    logger.error(f"[Retriever] ChromaDB init 失败: {e}")
                    raise
        return self._chroma_client

    def _get_collection(self):
        with self._lock:
            if self._chroma_collection is None:
                client = self._get_chroma_client()
                self._chroma_collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "l2"},
                )
                logger.debug(
                    f"[Retriever] collection '{self.collection_name}' "
                    f"count={self._chroma_collection.count()}"
                )
        return self._chroma_collection

    def _get_embedding_svc(self) -> EmbeddingService:
        if self._embedding_svc is None:
            self._embedding_svc = get_embedding_service()
        return self._embedding_svc

    # ── 工具 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _serialize_metadata(meta: dict) -> dict:
        """将 metadata value 序列化为 ChromaDB 支持的基础类型."""
        result = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                result[k] = v
            elif isinstance(v, date):
                result[k] = v.isoformat()
            elif v is None:
                result[k] = ""
            else:
                result[k] = str(v)
        return result
