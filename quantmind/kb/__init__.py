"""quantmind.kb — Phase 5 知识库与 RAG 模块.

组件：
- SmartChunker:     智能文本分块（500/100，保留 Markdown 表格）
- EmbeddingService: BGE-M3 嵌入服务（批量/缓存/CPU fallback）
- HybridRetriever:  混合检索（ChromaDB Dense + BM25 Sparse + BGE-Reranker）
- KBBuilder:        知识库构建器（新闻/报告/Snapshot 三路来源）
"""

from quantmind.kb.chunker import SmartChunker, Chunk
from quantmind.kb.embedding_service import EmbeddingService, get_embedding_service
from quantmind.kb.retriever import HybridRetriever, RetrievalResult
from quantmind.kb.builder import KBBuilder

__all__ = [
    "SmartChunker",
    "Chunk",
    "EmbeddingService",
    "get_embedding_service",
    "HybridRetriever",
    "RetrievalResult",
    "KBBuilder",
]
