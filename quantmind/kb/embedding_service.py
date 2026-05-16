"""quantmind.kb.embedding_service — BGE-M3 嵌入服务.

特性：
- 模型：BAAI/bge-m3（支持中英文，8192 token 上下文）
- 批量推理，batch_size=32
- MD5 本地缓存（避免重复计算）
- GPU 优先，CPU fallback
- 线程安全（lazy init）
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

__all__ = ["EmbeddingService", "get_embedding_service"]

_DEFAULT_MODEL = "BAAI/bge-m3"
_DEFAULT_CACHE_DIR = Path(".cache/embeddings")
_DEFAULT_BATCH_SIZE = 32

_GLOBAL_LOCK = threading.Lock()
_GLOBAL_SERVICE: EmbeddingService | None = None


class EmbeddingService:
    """BGE-M3 文本嵌入服务.

    Args:
        model_name:  HuggingFace 模型名，默认 BAAI/bge-m3
        cache_dir:   本地 embedding 缓存目录
        batch_size:  批处理大小，默认 32
        device:      "cuda" | "cpu" | None（自动检测）
        use_cache:   是否启用本地 MD5 缓存，默认 True
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        cache_dir: str | Path = _DEFAULT_CACHE_DIR,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: str | None = None,
        use_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.use_cache = use_cache
        self._device = device
        self._model = None  # lazy init
        self._lock = threading.Lock()

        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 公开 API ─────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """对单条文本生成 embedding.

        Args:
            text: 输入文本

        Returns:
            float 列表（维度取决于模型，bge-m3 为 1024）
        """
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding.

        先查本地缓存，未命中的再调用模型。

        Args:
            texts: 文本列表

        Returns:
            与输入等长的 embedding 列表
        """
        if not texts:
            return []

        logger.info(f"[EmbeddingService] embed_batch start: n={len(texts)}")
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # 查缓存
        if self.use_cache:
            for i, text in enumerate(texts):
                cached = self._load_cache(text)
                if cached is not None:
                    results[i] = cached
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts[:]

        # 批量推理未命中部分
        if uncached_texts:
            model = self._get_model()
            embeddings = self._run_model(model, uncached_texts)

            for local_i, (global_i, text) in enumerate(zip(uncached_indices, uncached_texts)):
                emb = embeddings[local_i]
                results[global_i] = emb
                if self.use_cache:
                    self._save_cache(text, emb)

        logger.info(
            f"[EmbeddingService] embed_batch done: n={len(texts)} "
            f"(uncached={len(uncached_texts)})"
        )
        return results  # type: ignore[return-value]

    @property
    def dim(self) -> int:
        """Embedding 维度（bge-m3 = 1024）."""
        return 1024

    # ── 模型懒初始化 ──────────────────────────────────────────────────────────

    def _get_model(self):
        with self._lock:
            if self._model is None:
                self._model = self._load_model()
        return self._model

    def _load_model(self):
        """加载 BGE-M3 模型."""
        try:
            from FlagEmbedding import BGEM3FlagModel
            device = self._resolve_device()
            logger.info(
                f"[EmbeddingService] embedding model load start: {self.model_name} (device={device})"
            )
            model = BGEM3FlagModel(
                self.model_name,
                use_fp16=(device != "cpu"),
                device=device,
            )
            logger.info(f"[EmbeddingService] embedding model load done: {self.model_name}")
            return model
        except ImportError:
            logger.error("[EmbeddingService] FlagEmbedding 未安装，pip install FlagEmbedding")
            raise
        except Exception as e:
            logger.error(f"[EmbeddingService] 模型加载失败：{e}")
            raise

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _run_model(self, model, texts: list[str]) -> list[list[float]]:
        """分批调用模型推理."""
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            try:
                output = model.encode(
                    batch,
                    batch_size=len(batch),
                    max_length=512,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                dense = output["dense_vecs"]
                # dense 可能是 np.ndarray，转为 list[list[float]]
                if hasattr(dense, "tolist"):
                    all_embeddings.extend(dense.tolist())
                else:
                    all_embeddings.extend([list(v) for v in dense])
            except Exception as e:
                logger.error(f"[EmbeddingService] batch embed 失败: {e}")
                # 回退：返回零向量
                all_embeddings.extend([[0.0] * self.dim] * len(batch))

        return all_embeddings

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _text_hash(self, text: str) -> str:
        return hashlib.md5((self.model_name + text).encode("utf-8")).hexdigest()

    def _cache_path(self, text: str) -> Path:
        h = self._text_hash(text)
        # 二级目录减少单目录文件数
        return self.cache_dir / h[:2] / f"{h}.json"

    def _load_cache(self, text: str) -> list[float] | None:
        path = self._cache_path(text)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_cache(self, text: str, embedding: list[float]) -> None:
        path = self._cache_path(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(embedding), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[EmbeddingService] 缓存写入失败: {e}")


# ── 全局单例 ────────────────────────────────────────────────────────────────


def get_embedding_service(
    model_name: str = _DEFAULT_MODEL,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> EmbeddingService:
    """获取全局 EmbeddingService 单例（线程安全懒初始化）."""
    global _GLOBAL_SERVICE
    with _GLOBAL_LOCK:
        if _GLOBAL_SERVICE is None:
            _GLOBAL_SERVICE = EmbeddingService(
                model_name=model_name,
                cache_dir=cache_dir,
                batch_size=batch_size,
                device=device,
            )
        return _GLOBAL_SERVICE
