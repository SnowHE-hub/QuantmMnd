"""quantmind.kb.builder — 知识库构建器.

数据来源（按 source_type）：
  - news:    AkShare 股票新闻（akshare.stock_news_em）
  - report:  研究报告摘要（目前无公开 API，从 data/reports/ 读取本地文件）
  - filing:  公司公告财务摘要（从 legacy Snapshot .pkl 提取）
  - snapshot: 结构化快照摘要（data/snapshots/<as_of>/ 下 parquet，见 build_from_snapshot_dir）

特性：
  - PIT metadata（published_date）
  - 增量构建（MD5 去重，不重复写入）
  - 支持本地目录批量导入
  - 支持 as_of 截止日期过滤
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime, timedelta
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from quantmind.kb.chunker import SmartChunker, Chunk
from quantmind.kb.retriever import HybridRetriever

__all__ = [
    "KBBuilder",
    "NEWS_TITLE_CANDIDATES",
    "NEWS_BODY_CANDIDATES",
    "NEWS_TIME_CANDIDATES",
    "NEWS_URL_CANDIDATES",
    "analyze_raw_news_records",
]

_DEFAULT_REPORTS_DIR = Path("data/reports")
_DEFAULT_CHROMA_DIR = ".cache/chromadb"

# ── News field candidates (AkShare / 东方财富列名可能变化) ───────────────

NEWS_TITLE_CANDIDATES: tuple[str, ...] = (
    "title",
    "标题",
    "news_title",
    "新闻标题",
)

NEWS_BODY_CANDIDATES: tuple[str, ...] = (
    "content",
    "内容",
    "news_content",
    "summary",
    "摘要",
    "新闻内容",
)

NEWS_TIME_CANDIDATES: tuple[str, ...] = (
    "publish_time",
    "发布时间",
    "time",
    "date",
    "日期",
)

NEWS_URL_CANDIDATES: tuple[str, ...] = (
    "url",
    "链接",
    "link",
    "新闻链接",
)

NEWS_SUMMARY_KEYS: frozenset[str] = frozenset({"summary", "摘要"})


def _nonempty_scalar(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, float) and math.isnan(val):
        return False
    if isinstance(val, (datetime, date)):
        return True
    if isinstance(val, bool):
        return True
    if isinstance(val, (int, float)):
        return True
    s = str(val).strip()
    return bool(s) and s.lower() != "nan"


def _union_record_keys(records: list[dict]) -> list[str]:
    keys: set[str] = set()
    for r in records:
        keys.update(str(k) for k in r.keys())
    return sorted(keys)


def _nonempty_count_for_key(records: list[dict], key: str) -> int:
    return sum(1 for r in records if _nonempty_scalar(r.get(key)))


def ordered_candidates_by_nonempty_rate(
    records: list[dict],
    candidates: tuple[str, ...],
) -> list[str]:
    present: list[tuple[int, int, str]] = []
    for i, name in enumerate(candidates):
        if not any(name in r for r in records):
            continue
        score = _nonempty_count_for_key(records, name)
        present.append((-score, i, name))
    present.sort()
    return [name for _, _, name in present]


def first_nonempty_for_row(row: dict, ordered_keys: list[str]) -> tuple[str | None, Any]:
    for k in ordered_keys:
        if k not in row:
            continue
        v = row.get(k)
        if _nonempty_scalar(v):
            return k, v
    return None, None


def coerce_news_date(val: Any) -> date | None:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if hasattr(val, "to_pydatetime"):
        try:
            return val.to_pydatetime().date()
        except Exception:
            pass
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    fmts = (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%d",
        "%Y/%m/%d",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_date_str(date_str: str) -> date | None:
    s = (date_str or "").strip()
    if not s:
        return None
    return coerce_news_date(s)


def _row_title_body_url_time(
    row: dict,
    title_order: list[str],
    body_order: list[str],
    url_order: list[str],
    time_order: list[str],
) -> tuple[str, str, str, date | None]:
    """Returns (title_s, body_s, url_s, pub_date)."""
    _, tv = first_nonempty_for_row(row, title_order)
    _, bv = first_nonempty_for_row(row, body_order)
    _, uv = first_nonempty_for_row(row, url_order)
    _, time_v = first_nonempty_for_row(row, time_order)
    title_s = str(tv).strip() if _nonempty_scalar(tv) else ""
    body_s = str(bv).strip() if _nonempty_scalar(bv) else ""
    url_s = str(uv).strip() if _nonempty_scalar(uv) else ""
    pub_date = coerce_news_date(time_v) if time_v is not None else None
    return title_s, body_s, url_s, pub_date


def _assemble_news_text(title_s: str, body_s: str) -> str:
    parts: list[str] = []
    if title_s:
        parts.append(title_s)
    if body_s:
        parts.append(body_s)
    return "\n\n".join(parts)


def analyze_raw_news_records(
    news_list: list[dict],
    ticker: str,
    cutoff: date,
    as_of: date,
    chunker: SmartChunker,
    seen_hashes: set[str] | None,
    enable_dedup: bool,
) -> dict[str, Any]:
    """Fetch-free analytics: builds docs / chunks / counts without embedding or Chroma.

    seen_hashes: when enable_dedup and set, updates like KBBuilder._dedup_documents.
    """
    stats: dict[str, Any] = {
        "ticker": ticker,
        "raw_docs": len(news_list),
        "raw_field_names": _union_record_keys(news_list),
    }

    title_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_TITLE_CANDIDATES)
    body_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_BODY_CANDIDATES)
    url_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_URL_CANDIDATES)
    time_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_TIME_CANDIDATES)

    stats["resolved_title_fields"] = title_order
    stats["resolved_body_fields"] = body_order
    stats["resolved_url_fields"] = url_order
    stats["resolved_time_fields"] = time_order

    def _col_counts(keys: tuple[str, ...]) -> dict[str, int]:
        return {k: _nonempty_count_for_key(news_list, k) for k in keys if any(k in r for r in news_list)}

    stats["title_like_nonempty"] = _col_counts(NEWS_TITLE_CANDIDATES)
    stats["content_like_nonempty"] = {k: v for k, v in _col_counts(NEWS_BODY_CANDIDATES).items() if k not in NEWS_SUMMARY_KEYS}
    stats["summary_like_nonempty"] = {k: v for k, v in _col_counts(NEWS_BODY_CANDIDATES).items() if k in NEWS_SUMMARY_KEYS}
    stats["url_like_nonempty"] = _col_counts(NEWS_URL_CANDIDATES)
    stats["publish_time_like_nonempty"] = _col_counts(NEWS_TIME_CANDIDATES)

    date_parse_ok = 0
    drop_before_cutoff = 0
    drop_after_asof = 0
    docs_pre_filter: list[dict] = []

    for item in news_list:
        t_s, b_s, u_s, pub_date = _row_title_body_url_time(
            item, title_order, body_order, url_order, time_order,
        )
        _, time_v = first_nonempty_for_row(item, time_order)
        if time_v is not None and coerce_news_date(time_v) is not None:
            date_parse_ok += 1

        if pub_date is not None and pub_date < cutoff:
            drop_before_cutoff += 1
            continue
        if pub_date is not None and pub_date > as_of:
            drop_after_asof += 1
            continue

        text = _assemble_news_text(t_s, b_s)
        src = u_s if u_s else f"news_{ticker}"
        docs_pre_filter.append({
            "text": text.strip(),
            "source": src,
            "source_type": "news",
            "ticker": ticker,
            "published_date": pub_date.isoformat() if pub_date else "",
            "title": t_s,
        })

    stats["date_parse_success_rows"] = date_parse_ok
    stats["drop_before_cutoff"] = drop_before_cutoff
    stats["drop_after_asof"] = drop_after_asof
    stats["docs_before_date_filter"] = len(news_list)
    stats["docs_after_date_filter"] = len(docs_pre_filter)
    stats["empty_text_docs"] = sum(1 for d in docs_pre_filter if not (d.get("text") or "").strip())
    stats["docs_before_dedup"] = len(docs_pre_filter)

    if not enable_dedup:
        deduped = list(docs_pre_filter)
    else:
        target: set[str] = seen_hashes if seen_hashes is not None else set()
        deduped = []
        for doc in docs_pre_filter:
            h = hashlib.md5(doc.get("text", "").encode("utf-8")).hexdigest()
            if h in target:
                continue
            target.add(h)
            deduped.append(doc)

    stats["docs_after_dedup"] = len(deduped)
    stats["dedup_dropped"] = (len(docs_pre_filter) - len(deduped)) if enable_dedup else 0

    chunks = chunker.chunk_documents(deduped)
    stats["len_chunks"] = len(chunks)
    return stats


class KBBuilder:
    """知识库构建器.

    Args:
        collection_name: ChromaDB 集合名称（会被加 quantmind_kb_ 前缀）
        chroma_dir:      ChromaDB 持久化目录
        chunker:         SmartChunker 实例（None = 默认参数）
        dedup:           是否启用 MD5 去重，默认 True
    """

    def __init__(
        self,
        collection_name: str = "default",
        chroma_dir: str = _DEFAULT_CHROMA_DIR,
        chunker: SmartChunker | None = None,
        dedup: bool = True,
    ) -> None:
        self.collection_name = collection_name
        self.chroma_dir = chroma_dir
        self.chunker = chunker or SmartChunker()
        self.dedup = dedup
        self._retriever: HybridRetriever | None = None
        self._seen_hashes: set[str] = set()

    # ── 公开 API ─────────────────────────────────────────────────────────────

    def build_from_news(
        self,
        tickers: list[str],
        as_of: date | None = None,
        days_back: int = 90,
        *,
        raise_on_error: bool = False,
    ) -> int:
        """从 AkShare 抓取股票新闻并写入知识库.

        Args:
            tickers:   股票代码列表（如 ["600519", "000858"]）
            as_of:     截止日期（PIT）
            days_back: 往前抓取天数
            raise_on_error: True 时将单只抓取/写入异常抛出给调用方（用于批量脚本重试）。

        Returns:
            写入的 Chunk 总数
        """
        as_of = as_of or date.today()
        cutoff = as_of - timedelta(days=days_back)
        total = 0

        for ticker in tickers:
            try:
                news_list = self._fetch_news(ticker, as_of)
                raw_n = len(news_list)
                raw_fields = _union_record_keys(news_list)
                logger.info(
                    f"[KBBuilder] news {ticker}: provider rows={raw_n}, "
                    f"field_names={raw_fields}"
                )

                title_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_TITLE_CANDIDATES)
                body_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_BODY_CANDIDATES)
                url_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_URL_CANDIDATES)
                time_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_TIME_CANDIDATES)

                def _col_counts(keys: tuple[str, ...]) -> dict[str, int]:
                    return {k: _nonempty_count_for_key(news_list, k) for k in keys if any(k in r for r in news_list)}

                title_counts = _col_counts(NEWS_TITLE_CANDIDATES)
                body_all = _col_counts(NEWS_BODY_CANDIDATES)
                content_counts = {k: v for k, v in body_all.items() if k not in NEWS_SUMMARY_KEYS}
                summary_counts = {k: v for k, v in body_all.items() if k in NEWS_SUMMARY_KEYS}

                logger.info(
                    f"[KBBuilder] news {ticker}: resolved_fields "
                    f"title_order={title_order}, body_order={body_order}, "
                    f"time_order={time_order}, url_order={url_order}"
                )
                logger.info(
                    f"[KBBuilder] news {ticker}: nonempty title-like={title_counts}, "
                    f"content-like={content_counts}, summary-like={summary_counts}, "
                    f"url-like={_col_counts(NEWS_URL_CANDIDATES)}, "
                    f"time-like={_col_counts(NEWS_TIME_CANDIDATES)}"
                )

                date_parse_ok = 0
                drop_before_cutoff = 0
                drop_after_asof = 0
                docs: list[dict] = []

                for item in news_list:
                    t_s, b_s, u_s, pub_date = _row_title_body_url_time(
                        item, title_order, body_order, url_order, time_order,
                    )
                    _, time_v = first_nonempty_for_row(item, time_order)
                    if time_v is not None and coerce_news_date(time_v) is not None:
                        date_parse_ok += 1

                    if pub_date is not None and pub_date < cutoff:
                        drop_before_cutoff += 1
                        continue
                    if pub_date is not None and pub_date > as_of:
                        drop_after_asof += 1
                        continue

                    text = _assemble_news_text(t_s, b_s)
                    src = u_s if u_s else f"news_{ticker}"
                    docs.append({
                        "text": text.strip(),
                        "source": src,
                        "source_type": "news",
                        "ticker": ticker,
                        "published_date": pub_date.isoformat() if pub_date else "",
                        "title": t_s,
                    })

                n_before_dedup = len(docs)
                empty_text_docs = sum(1 for d in docs if not (d.get("text") or "").strip())

                logger.info(
                    f"[KBBuilder] news {ticker}: date_parse_ok={date_parse_ok}, "
                    f"drop_before_cutoff={drop_before_cutoff}, drop_after_asof={drop_after_asof}, "
                    f"docs_after_date_filter={n_before_dedup} "
                    f"(before_date_filter_rows={raw_n})"
                )
                logger.info(
                    f"[KBBuilder] news {ticker}: empty_text_docs={empty_text_docs}, "
                    f"docs_before_dedup={n_before_dedup}"
                )

                deduped_docs = self._dedup_documents(docs)
                n_after_dedup = len(deduped_docs)
                logger.info(
                    f"[KBBuilder] news {ticker}: docs_after_dedup={n_after_dedup}, "
                    f"dedup_dropped={n_before_dedup - n_after_dedup}"
                )

                chunks = self.chunker.chunk_documents(deduped_docs)
                n_chunks = len(chunks)
                logger.info(f"[KBBuilder] news {ticker}: len(chunks)={n_chunks}")

                n_written = self._get_retriever().add_chunks(chunks)
                total += n_written
                logger.info(
                    f"[KBBuilder] news {ticker}: add_chunks_returned={n_written} "
                    f"(raw_rows={raw_n}, docs_after_date_filter={n_before_dedup})"
                )
                if n_chunks > 0 and n_written == 0:
                    logger.warning(
                        f"[KBBuilder] news {ticker}: len(chunks)={n_chunks} 但 add_chunks 返回 0 —— "
                        "可能为 embedding 失败或 Chroma upsert 失败，请查看 [EmbeddingService] / [Retriever] 日志"
                    )
            except Exception as e:
                logger.warning(f"[KBBuilder] 新闻抓取失败 {ticker}: {e}")
                if raise_on_error:
                    raise

        return total

    def build_from_reports_dir(
        self,
        reports_dir: str | Path | None = None,
        as_of: date | None = None,
        extensions: tuple[str, ...] = (".txt", ".md", ".pdf"),
    ) -> int:
        """从本地目录批量导入研究报告.

        Args:
            reports_dir: 报告目录（默认 data/reports/）
            as_of:       截止日期（PIT）
            extensions:  支持的文件扩展名

        Returns:
            写入的 Chunk 总数
        """
        reports_dir = Path(reports_dir or _DEFAULT_REPORTS_DIR)
        as_of = as_of or date.today()
        total = 0

        if not reports_dir.exists():
            logger.info(f"[KBBuilder] reports_dir 不存在，跳过: {reports_dir}")
            return 0

        for fpath in reports_dir.rglob("*"):
            if fpath.suffix.lower() not in extensions:
                continue
            try:
                text = self._read_file(fpath)
                if not text:
                    continue

                # 从文件名推断 ticker 和日期
                ticker, pub_date = self._infer_from_filename(fpath.name)

                # PIT 过滤
                if pub_date and pub_date > as_of:
                    continue

                docs = [{
                    "text": text,
                    "source": str(fpath),
                    "source_type": "report",
                    "ticker": ticker or "",
                    "published_date": pub_date.isoformat() if pub_date else "",
                    "title": fpath.stem,
                }]

                chunks = self._dedup_and_chunk(docs)
                n = self._get_retriever().add_chunks(chunks)
                total += n
                logger.debug(f"[KBBuilder] report {fpath.name} → {n} chunks")
            except Exception as e:
                logger.warning(f"[KBBuilder] 文件导入失败 {fpath}: {e}")

        logger.info(f"[KBBuilder] reports_dir: {total} chunks total")
        return total

    def build_from_snapshot(
        self,
        snapshot_path: str | Path,
        tickers: list[str] | None = None,
        as_of: date | None = None,
    ) -> int:
        """从 Snapshot 提取财务摘要写入知识库.

        Args:
            snapshot_path: Snapshot HDF5/pickle 路径
            tickers:       指定股票列表（None = 全部）
            as_of:         截止日期（PIT）

        Returns:
            写入的 Chunk 总数
        """
        as_of = as_of or date.today()
        total = 0

        try:
            import pickle
            with open(snapshot_path, "rb") as f:
                snapshot = pickle.load(f)

            if tickers is None:
                tickers = list(getattr(snapshot, "tickers", {}).keys())

            for ticker in tickers:
                text = self._snapshot_to_text(snapshot, ticker, as_of)
                if not text:
                    continue

                docs = [{
                    "text": text,
                    "source": f"snapshot:{ticker}",
                    "source_type": "filing",
                    "ticker": ticker,
                    "published_date": as_of.isoformat(),
                    "title": f"{ticker} 财务摘要 {as_of}",
                }]

                chunks = self._dedup_and_chunk(docs)
                n = self._get_retriever().add_chunks(chunks)
                total += n
        except FileNotFoundError:
            logger.warning(f"[KBBuilder] Snapshot 文件不存在: {snapshot_path}")
        except Exception as e:
            logger.warning(f"[KBBuilder] Snapshot 导入失败: {e}")

        return total

    def build_from_snapshot_dir(
        self,
        snapshot_dir: str | Path,
        tickers: list[str] | None = None,
        as_of: date | str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """从 ``data/snapshots/<as_of>/`` 下的 parquet 生成短摘要文档并写入 KB.

        不整表 dump；缺失的 parquet 在 strict=False 下告警并跳过（见 snapshot_parquet 模块）。
        dry_run 时不触碰 EmbeddingService / Chroma / BM25。
        """
        from quantmind.kb.snapshot_parquet import generate_snapshot_documents

        snapshot_dir = Path(snapshot_dir)
        docs, stats = generate_snapshot_documents(snapshot_dir, tickers, as_of=as_of)
        doc_type_counts = Counter(str(d.get("doc_type") or "") for d in docs)
        stats["doc_type_counts"] = {k: v for k, v in doc_type_counts.items() if k}
        if not docs:
            logger.info(
                f"[KBBuilder] build_from_snapshot_dir: no documents from {snapshot_dir} "
                f"(check parquet files / tickers)"
            )

        chunks = self._dedup_and_chunk(docs)
        out: dict[str, Any] = {
            **stats,
            "n_chunks": len(chunks),
            "chunks_written": 0,
            "dry_run": dry_run,
        }

        if dry_run:
            return out

        n_written = 0
        if chunks:
            n_written = self._get_retriever().add_chunks(chunks)

        out["chunks_written"] = n_written
        if not dry_run:
            try:
                out["final_collection_count"] = self.count()
            except Exception:  # noqa: BLE001
                out["final_collection_count"] = None
        logger.info(
            f"[KBBuilder] snapshot_dir {snapshot_dir}: n_docs={stats['n_docs']} "
            f"n_chunks={len(chunks)} add_chunks_returned={n_written}"
        )
        return out

    def build_from_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> int:
        """直接从 document list 构建（通用接口）.

        每个 document 须含 "text" 字段，其余字段作为 metadata。

        Args:
            documents: list[{"text": str, "source": str, ...}]

        Returns:
            写入 Chunk 总数
        """
        chunks = self._dedup_and_chunk(documents)
        n = self._get_retriever().add_chunks(chunks)
        logger.info(f"[KBBuilder] build_from_documents: {len(documents)} docs → {n} chunks")
        return n

    def count(self) -> int:
        """返回知识库中的文档总数."""
        return self._get_retriever().count()

    def get_retriever(self) -> HybridRetriever:
        """获取内部 Retriever 实例（用于检索）."""
        return self._get_retriever()

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                collection_name=self.collection_name,
                chroma_dir=self.chroma_dir,
                use_reranker=False,  # Builder 写入时不需要 reranker
            )
        return self._retriever

    def _dedup_documents(self, documents: list[dict]) -> list[dict]:
        if not self.dedup:
            return list(documents)
        deduped: list[dict] = []
        for doc in documents:
            h = hashlib.md5(doc.get("text", "").encode("utf-8")).hexdigest()
            if h in self._seen_hashes:
                continue
            self._seen_hashes.add(h)
            deduped.append(doc)
        return deduped

    def _dedup_and_chunk(self, documents: list[dict]) -> list[Chunk]:
        """去重后分块."""
        deduped = self._dedup_documents(documents)
        return self.chunker.chunk_documents(deduped)

    def _fetch_news(self, ticker: str, as_of: date, timeout: int = 30) -> list[dict]:
        """通过 AkShare 抓取新闻（PIT 安全）.

        Args:
            ticker:  股票代码
            as_of:   截止日期
            timeout: 网络请求超时秒数（默认 30s），防止 AkShare 无限挂起
        """
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"AkShare 请求超时（>{timeout}s）")

        try:
            import akshare as ak
            code = ticker.split(".")[0]

            # 设置超时（仅 Unix 可用）
            old_handler = None
            try:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)
            except (AttributeError, OSError):
                pass  # Windows 无 SIGALRM，跳过

            try:
                df = ak.stock_news_em(symbol=code)
            finally:
                try:
                    signal.alarm(0)  # 取消超时
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)
                except (AttributeError, OSError):
                    pass

            if df is None or df.empty:
                return []
            return df.to_dict("records")
        except ImportError:
            logger.warning("[KBBuilder] akshare 未安装，跳过新闻抓取")
            return []
        except TimeoutError as e:
            logger.warning(f"[KBBuilder] {e}，跳过 {ticker} 新闻")
            return []
        except Exception as e:
            logger.warning(f"[KBBuilder] akshare news 失败 ({ticker}): {e}")
            return []

    def _read_file(self, path: Path) -> str:
        """读取文件文本内容（支持 .txt/.md，.pdf 用 pdfplumber）."""
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            try:
                return path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""
        elif suffix == ".pdf":
            try:
                import pdfplumber
                pages = []
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            pages.append(t)
                return "\n\n".join(pages)
            except ImportError:
                logger.warning("[KBBuilder] pdfplumber 未安装，跳过 PDF")
                return ""
            except Exception as e:
                logger.warning(f"[KBBuilder] PDF 读取失败 {path}: {e}")
                return ""
        return ""

    def _infer_from_filename(self, filename: str) -> tuple[str | None, date | None]:
        """从文件名推断 ticker 和发布日期.

        期望格式：{ticker}_{YYYYMMDD}_*.txt 或 {ticker}_{YYYY-MM-DD}_*.md
        """
        import re
        ticker_match = re.search(r"(\d{6}(?:\.[A-Z]+)?)", filename)
        ticker = ticker_match.group(1) if ticker_match else None

        date_match = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2})", filename)
        pub_date = None
        if date_match:
            raw = date_match.group(1).replace("_", "-").replace("", "-")
            try:
                pub_date = datetime.strptime(raw, "%Y-%m-%d").date()
            except ValueError:
                try:
                    pub_date = datetime.strptime(raw, "%Y%m%d").date()
                except ValueError:
                    pass

        return ticker, pub_date

    def _snapshot_to_text(self, snapshot: Any, ticker: str, as_of: date) -> str:
        """从 Snapshot 对象提取财务文本摘要."""
        try:
            # 尝试访问 snapshot 的财务数据
            lines = [f"股票代码：{ticker}", f"数据截止：{as_of}"]

            if hasattr(snapshot, "financials"):
                fin = snapshot.financials.get(ticker, {})
                if fin:
                    lines.append("=== 财务数据 ===")
                    for k, v in list(fin.items())[:20]:  # 最多20个字段
                        lines.append(f"{k}: {v}")

            if hasattr(snapshot, "basics"):
                basics = snapshot.basics.get(ticker, {})
                if basics:
                    lines.append("=== 公司基础信息 ===")
                    for k, v in list(basics.items())[:10]:
                        lines.append(f"{k}: {v}")

            return "\n".join(lines) if len(lines) > 2 else ""
        except Exception:
            return ""

    @staticmethod
    def _parse_date(date_str: str) -> date | None:
        return _parse_date_str(date_str)
