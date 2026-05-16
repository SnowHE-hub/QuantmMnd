"""quantmind.kb.chunker — 智能文本分块器.

特性：
- RecursiveCharacterTextSplitter 500/100（中英文通用）
- Markdown 表格整体保留（不拆断）
- jieba 分词辅助识别中文句边界
- 每个 Chunk 携带 metadata（source, ticker, published_date, chunk_idx, total_chunks）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from loguru import logger

try:
    import jieba
    _JIEBA_OK = True
except ImportError:
    _JIEBA_OK = False
    logger.warning("[Chunker] jieba 未安装，回退到简单句子分割")

__all__ = ["SmartChunker", "Chunk"]

# 默认分块参数
_DEFAULT_CHUNK_SIZE = 500
_DEFAULT_CHUNK_OVERLAP = 100

# Markdown 表格行正则
_TABLE_ROW_RE = re.compile(r"^\|.+\|$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\|[-:| ]+\|$", re.MULTILINE)


@dataclass
class Chunk:
    """单个文本块."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_idx: int = 0
    total_chunks: int = 1

    # 便捷属性
    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def ticker(self) -> str:
        return self.metadata.get("ticker", "")

    @property
    def published_date(self) -> date | None:
        d = self.metadata.get("published_date")
        if isinstance(d, date):
            return d
        if isinstance(d, str):
            try:
                from datetime import datetime
                return datetime.strptime(d, "%Y-%m-%d").date()
            except ValueError:
                return None
        return None


class SmartChunker:
    """智能文本分块器.

    Args:
        chunk_size:    目标块大小（字符数），默认 500
        chunk_overlap: 相邻块重叠字符数，默认 100
        preserve_tables: 是否将 Markdown 表格整体保留，默认 True
    """

    def __init__(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        preserve_tables: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_tables = preserve_tables

    # ── 公开 API ─────────────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """将文本切分为 Chunk 列表.

        Args:
            text:     原始文本（支持 Markdown）
            metadata: 附加元数据（source/ticker/published_date 等）

        Returns:
            list[Chunk]
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}

        # 先抽出 Markdown 表格，用占位符替换
        segments, table_map = self._extract_tables(text) if self.preserve_tables else (text, {})

        # 递归分割
        raw_chunks = self._recursive_split(segments)

        # 还原表格占位符
        if table_map:
            raw_chunks = [self._restore_tables(c, table_map) for c in raw_chunks]

        # 过滤空块
        raw_chunks = [c for c in raw_chunks if c.strip()]

        # 包装为 Chunk 对象
        total = len(raw_chunks)
        chunks = []
        for idx, text_piece in enumerate(raw_chunks):
            chunks.append(Chunk(
                text=text_piece.strip(),
                metadata={**meta, "chunk_idx": idx, "total_chunks": total},
                chunk_idx=idx,
                total_chunks=total,
            ))

        logger.debug(f"[Chunker] '{meta.get('source', '?')}' → {total} chunks")
        return chunks

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[Chunk]:
        """批量分块文档列表.

        每个 document 需含 "text" 字段，其余字段作为 metadata。

        Args:
            documents: list of {"text": str, "source": str, ...}

        Returns:
            所有 Chunk 的列表
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            text = doc.get("text", "")
            meta = {k: v for k, v in doc.items() if k != "text"}
            all_chunks.extend(self.chunk_text(text, meta))
        return all_chunks

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _extract_tables(self, text: str) -> tuple[str, dict[str, str]]:
        """抽出 Markdown 表格并用占位符替换."""
        table_map: dict[str, str] = {}
        counter = [0]

        def _replace(m: re.Match) -> str:
            key = f"__TABLE_{counter[0]}__"
            table_map[key] = m.group(0)
            counter[0] += 1
            return f"\n{key}\n"

        # 匹配完整表格（至少含表头+分隔行+1行数据）
        table_pattern = re.compile(
            r"(?:^\|.+\|\n)+^\|[-:| ]+\|\n(?:^\|.+\|\n)*",
            re.MULTILINE,
        )
        result = table_pattern.sub(_replace, text)
        return result, table_map

    def _restore_tables(self, text: str, table_map: dict[str, str]) -> str:
        for key, table in table_map.items():
            text = text.replace(key, table)
        return text

    def _recursive_split(self, text: str) -> list[str]:
        """递归字符分割（中英文通用）."""
        # 分隔符优先级：双换行 > 单换行 > 句号/。 > 逗号/，> 空格 > 字符
        separators = ["\n\n", "\n", "。", ".", "！", "！", "；", ";", "，", ",", " ", ""]

        return self._split_with_separators(text, separators)

    def _split_with_separators(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else []

        if sep:
            parts = text.split(sep)
        else:
            # 按字符强制切分
            return self._hard_split(text)

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    # 当前块已满，先 flush
                    if len(current) > self.chunk_size and remaining_seps:
                        # 当前块超大，递归用更细分隔符
                        chunks.extend(self._split_with_separators(current, remaining_seps))
                    else:
                        chunks.append(current)
                    # 新块从 overlap 开始
                    overlap_text = self._get_overlap(current)
                    current = overlap_text + (sep if overlap_text else "") + part
                    if len(current) > self.chunk_size and remaining_seps:
                        chunks.extend(self._split_with_separators(current, remaining_seps))
                        current = ""
                else:
                    # part 本身就超大
                    if remaining_seps:
                        chunks.extend(self._split_with_separators(part, remaining_seps))
                    else:
                        chunks.extend(self._hard_split(part))
                    current = ""

        if current:
            if len(current) > self.chunk_size and remaining_seps:
                chunks.extend(self._split_with_separators(current, remaining_seps))
            else:
                chunks.append(current)

        return [c for c in chunks if c.strip()]

    def _hard_split(self, text: str) -> list[str]:
        """强制按字符数切分（最后手段）."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks

    def _get_overlap(self, text: str) -> str:
        """从文本末尾取 overlap 长度的文字."""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]
