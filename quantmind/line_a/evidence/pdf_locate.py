"""quantmind.line_a.evidence.pdf_locate — 公告 PDF 的文本层解析与段落、表格定位（03_final_plan_v2 第 5 节 F5(b)1）.

- 只读文本层（PyMuPDF）。没有文本层的页面标为 needs_ocr，留给版面 / OCR 补救（PaddleOCR-VL 或 MinerU，按需、只对这些页）。
- 每个段落记录：文档编号、页码、段落序号、页内字符区间、文本、文本 sha256；
  每个表格记录：页码、表格序号、行列数和单元格文本。
- 文档层记录原文件 sha256、页数、解析器名称与版本。引用证据时用（文档编号, 页码, 段落序号）加文本 sha256，
  校验器可以据此逐字核对引文是否真的出现在原文里。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

PARSER = "pymupdf-text-v1"


@dataclass
class Para:
    doc_id: str
    page: int
    idx: int
    start: int
    end: int
    text: str
    sha256: str


@dataclass
class Table:
    doc_id: str
    page: int
    idx: int
    n_rows: int
    n_cols: int
    cells: list[list[str]]


@dataclass
class ParsedDoc:
    doc_id: str
    file_sha256: str
    n_pages: int
    parser: str
    paras: list[Para] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    needs_ocr: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _sha(s: str | bytes) -> str:
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


def _split_paras(page_text: str) -> list[tuple[int, int, str]]:
    """按空行或"句末标点 + 换行"切段；同一段内的换行（PDF 折行）去掉。返回（起, 止, 文本）。"""
    out, start = [], 0
    for m in re.finditer(r"\n\s*\n|(?<=[。；！？:：])\n", page_text):
        seg = page_text[start:m.start()]
        if seg.strip():
            out.append((start, m.start(), re.sub(r"\s*\n\s*", "", seg).strip()))
        start = m.end()
    seg = page_text[start:]
    if seg.strip():
        out.append((start, len(page_text), re.sub(r"\s*\n\s*", "", seg).strip()))
    return out


def parse_pdf(path: Path, doc_id: str, *, min_chars_per_page: int = 20, tables: bool = True) -> ParsedDoc:
    import fitz
    data = Path(path).read_bytes()
    doc = fitz.open(stream=data, filetype="pdf")
    pd_ = ParsedDoc(doc_id, _sha(data), doc.page_count, f"{PARSER}/fitz-{fitz.VersionBind}")
    for pno in range(doc.page_count):
        page = doc[pno]
        txt = page.get_text("text")
        if len(txt.strip()) < min_chars_per_page:
            pd_.needs_ocr.append(pno + 1)
            continue
        for i, (s, e, t) in enumerate(_split_paras(txt)):
            pd_.paras.append(Para(doc_id, pno + 1, i, s, e, t, _sha(t)))
        if tables:
            try:
                for j, tb in enumerate(page.find_tables().tables):
                    cells = [[("" if c is None else str(c)) for c in row] for row in tb.extract()]
                    pd_.tables.append(Table(doc_id, pno + 1, j, len(cells), max((len(r) for r in cells), default=0),
                                            cells))
            except Exception:  # noqa: BLE001  表格检测失败不影响正文
                pass
    return pd_


def verify_quote(doc: ParsedDoc, page: int, idx: int, quote: str) -> bool:
    """校验器用：引文（去掉空白后）是否出现在指定段落里。"""
    q = re.sub(r"\s+", "", quote)
    for p in doc.paras:
        if p.page == page and p.idx == idx:
            return q in re.sub(r"\s+", "", p.text)
    return False
