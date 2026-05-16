#!/usr/bin/env python3
"""将研报元数据 / 摘要摄入 Chroma KB（reports，source_type=report）。

分层执行顺序：
1. Tushare ``pro.report_rc``（积分接口，默认由 --source 控制）
2. AkShare ``stock_research_report_em``
3. 巨潮年报 PDF 摘要（仅 ``--source cninfo``）

环境变量：
- ``TUSHARE_TOKEN`` — Tushare；不得打印。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from loguru import logger

from quantmind.kb.builder import KBBuilder
from quantmind.kb.chunker import SmartChunker


def _code_from_ts(ts: str) -> str:
    ts = ts.strip().upper()
    return ts.split(".")[0] if "." in ts else ts


def _parse_as_of(d: date | str | None):
    from datetime import date as date_cls

    if d is None:
        return date_cls.today()
    if isinstance(d, date_cls):
        return d
    return datetime.strptime(str(d).strip()[:10], "%Y-%m-%d").date()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ticker-file",
        type=Path,
        default=None,
        help="Ticker 列表：parquet(ts_code)/txt 一行一只；省略则用 data/snapshots/<as-of>/universe.parquet",
    )
    p.add_argument("--as-of", default=None, help="PIT 截止 YYYY-MM-DD（默认今日）")
    p.add_argument(
        "--source",
        choices=("all", "tushare", "akshare", "cninfo"),
        default="all",
        help="数据源：all=tushare+akshare（不含 cninfo）；cninfo=仅年报 PDF",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research_reports"),
        help="写入摄入摘要日志 / manifest",
    )
    p.add_argument("--top-n", type=int, default=5, help="每只票各层最多保留条数")
    p.add_argument("--sleep", type=float, default=0.0, help="股票间隔秒（降频）")
    p.add_argument("--collection-name", default="default")
    p.add_argument("--chroma-dir", default=".cache/chromadb")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--chunk-overlap", type=int, default=100)
    return p.parse_args()


def load_tickers(ticker_file: Path | None, as_of_date) -> list[str]:
    if ticker_file is not None:
        if not ticker_file.is_file():
            raise SystemExit(f"--ticker-file 不存在: {ticker_file}")
        if ticker_file.suffix.lower() == ".txt":
            lines = ticker_file.read_text(encoding="utf-8").splitlines()
            return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        df = pd.read_parquet(ticker_file)
        col = "ts_code" if "ts_code" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()

    uf = _ROOT / "data" / "snapshots" / as_of_date.isoformat() / "universe.parquet"
    if not uf.is_file():
        raise SystemExit(
            f"默认 universe 不存在: {uf}，请传入 --ticker-file 或先下载该日快照"
        )
    df = pd.read_parquet(uf)
    col = "ts_code" if "ts_code" in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def _normalize_pub_date(row_val: Any, as_of_cutoff) -> str:
    """返回 YYYY-MM-DD 或空字符串."""
    if row_val is None or (isinstance(row_val, float) and pd.isna(row_val)):
        return ""
    if isinstance(row_val, (int, float)) and not pd.isna(row_val):
        row_val = str(int(row_val))
    s = str(row_val).strip().replace("-", "")
    if s.isdigit() and len(s) == 8:
        try:
            dt = datetime.strptime(s, "%Y%m%d").date()
            if dt > as_of_cutoff:
                return ""
            return dt.isoformat()
        except ValueError:
            return ""
    try:
        d = pd.to_datetime(row_val, errors="coerce")
        if pd.isna(d):
            return ""
        dt = pd.Timestamp(d).date()
        if dt > as_of_cutoff:
            return ""
        return dt.isoformat()
    except Exception:
        return ""


def layer_tushare_report_rc(ts_code: str, as_of_cutoff, top_n: int) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    token = os.environ.get("TUSHARE_TOKEN") or ""
    if not token:
        logger.warning("[ingest_rc] TUSHARE_TOKEN 未设置，跳过 Tushare 层")
        return docs

    try:
        import tushare as ts
    except ImportError:
        logger.warning("[ingest_rc] tushare 未安装，跳过 Tushare 层")
        return docs

    pro = ts.pro_api(token)
    end_s = as_of_cutoff.strftime("%Y%m%d")
    start_s = (as_of_cutoff - timedelta(days=730)).strftime("%Y%m%d")
    try:
        df = pro.report_rc(ts_code=ts_code, start_date=start_s, end_date=end_s)
    except Exception as e:
        logger.warning(f"[ingest_rc] Tushare report_rc {ts_code} 失败: {e}")
        return docs

    if df is None or df.empty:
        return docs

    sort_col = "report_date" if "report_date" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False).head(top_n)

    for _, row in df.iterrows():
        rdate = ""
        if "report_date" in row.index:
            rdate_raw = row.get("report_date")
            rdate = _normalize_pub_date(rdate_raw, as_of_cutoff)
        organ = row.get("organ", "") if "organ" in row.index else ""
        author = row.get("author", "") if "author" in row.index else ""
        rec = row.get("rec_val", "") if "rec_val" in row.index else ""
        tp = row.get("target_price", "")
        cp = row.get("close_price", "")
        text = (
            f"{ts_code} {rdate or row.get('report_date', '')} {organ} 给出 {rec} 评级，"
            f"目标价 {tp}，当时收盘价 {cp}。分析师：{author}。"
        ).strip()
        src = (
            f"tushare:report_rc:{ts_code}:"
            f"{row.get('report_date', '')}:{organ}:{author}"
        )
        docs.append({
            "text": text,
            "source": src[:500],
            "source_type": "report",
            "doc_type": "analyst_recommendation",
            "ticker": ts_code,
            "published_date": rdate,
            "title": f"Tushare 分析师评级 {organ} {rec}".strip(),
        })
    return docs


def layer_akshare_reports(ts_code: str, as_of_cutoff, top_n: int) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    try:
        import akshare as ak
    except ImportError:
        logger.warning("[ingest_rc] akshare 未安装，跳过研报层 AkShare")
        return docs

    sym = _code_from_ts(ts_code)
    try:
        df = ak.stock_research_report_em(symbol=sym)
    except Exception as e:
        logger.warning(f"[ingest_rc] AkShare stock_research_report_em {sym} 失败: {e}")
        return docs

    if df is None or df.empty:
        return docs

    date_col = "报告日期" if "报告日期" in df.columns else None
    if date_col:
        df = df.sort_values(date_col, ascending=False)
    df = df.head(top_n)

    for _, row in df.iterrows():
        r_raw = row.get("报告日期", "") if "报告日期" in row.index else ""
        pname = row.get("机构名称", "")
        title = row.get("报告标题", "")
        rating = row.get("评级", "")
        analyst = row.get("分析师", "")
        pub = _normalize_pub_date(r_raw, as_of_cutoff)
        if not pub:
            pub = ""
        text = (
            f"{pname}于{r_raw}发布研究报告《{title}》，评级为{rating}，分析师：{analyst}。"
        )
        docs.append({
            "text": text.strip(),
            "source": f"akshare:stock_research_report_em:{sym}:{title}:{r_raw}"[:480],
            "source_type": "report",
            "doc_type": "research_report_meta",
            "ticker": ts_code,
            "published_date": pub or "",
            "title": str(title or "")[:200],
        })
    return docs


def _http_get_pdf_text(url: str, max_chars: int = 3000) -> str:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/120 Safari/537.36")
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
    except Exception as e:
        logger.warning(f"[ingest_rc] PDF 下载失败 {url}: {e}")
        return ""

    txt = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:8]:
                t = page.extract_text() or ""
                txt += "\n" + t
                if len(txt) >= max_chars:
                    break
    except ImportError:
        logger.warning("[ingest_rc] pdfplumber 未安装，跳过 PDF 解析")
        return ""
    except Exception as e:
        logger.warning(f"[ingest_rc] PDF 解析失败: {e}")
        return ""

    txt = txt.strip().replace("\r", "")
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "\n...[truncated]"
    return txt


def _guess_url_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cn = str(c)
        low = cn.lower()
        if "链接" in cn or "url" in low or "pdf" in low or "地址" in cn:
            return c
    return None


def layer_cninfo_latest_annual(ts_code: str, as_of_cutoff, top_n: int) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    sym = _code_from_ts(ts_code)
    df: pd.DataFrame | None = None
    try:
        import akshare as ak
        fn = getattr(ak, "stock_notice_report", None)
        if callable(fn):
            df = fn(symbol=sym, keyword="年度报告")  # type: ignore[operator]
    except TypeError:
        try:
            import akshare as ak

            df = ak.stock_notice_report(symbol=sym, keyword="年度报告")  # type: ignore[misc]
        except Exception:
            df = None
    except Exception as e:
        logger.warning(f"[ingest_rc] stock_notice_report {sym} 不可用: {e}")
        df = None

    if df is None or df.empty:
        logger.warning(f"[ingest_rc] cninfo/AkShare 未返回公告表 {sym}，跳过年报 PDF")
        return docs

    url_col = _guess_url_column(df)
    if not url_col:
        logger.warning(f"[ingest_rc] 无法在公告表中识别 URL 列，列名={list(df.columns)}")
        return docs

    sort_col = None
    for candidate in ("公告日期", "公告时间", "发布时间"):
        if candidate in df.columns:
            sort_col = candidate
            break
    sub = df.copy()
    if sort_col:
        sub["_ts"] = pd.to_datetime(sub[sort_col], errors="coerce")
        sub = sub.dropna(subset=["_ts"])
        sub["_d"] = sub["_ts"].dt.date
        sub = sub[sub["_d"] <= as_of_cutoff]
        sub = sub.sort_values("_ts", ascending=False)
    sub = sub.head(top_n)

    for _, row in sub.iterrows():
        url = str(row.get(url_col, "") or "").strip()
        if not url.startswith("http"):
            continue
        ndate = ""
        if sort_col and sort_col in row.index:
            ndate = _normalize_pub_date(row.get(sort_col), as_of_cutoff)
        excerpt = _http_get_pdf_text(url)
        if not excerpt:
            continue
        text = f"{ts_code} 年度报告节选（PDF，公告日 {ndate}）：\n{excerpt}"
        docs.append({
            "text": text,
            "source": url[:480],
            "source_type": "report",
            "doc_type": "annual_report",
            "ticker": ts_code,
            "published_date": ndate or "",
            "title": f"{ts_code} 年报摘要 PDF",
        })
    return docs


def main() -> None:
    args = parse_args()
    from datetime import date as date_cls

    as_cut = _parse_as_of(args.as_of) if args.as_of else date_cls.today()
    tickers = load_tickers(args.ticker_file, as_cut)

    chunker = SmartChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    builder = KBBuilder(
        collection_name=args.collection_name,
        chroma_dir=args.chroma_dir,
        chunker=chunker,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / f"manifest_{as_cut.isoformat()}.json"

    want_tush = args.source in ("all", "tushare")
    want_ak = args.source in ("all", "akshare")
    want_cn = args.source == "cninfo"

    stats: dict[str, Any] = {
        "as_of": as_cut.isoformat(),
        "n_tickers": len(tickers),
        "chunks_written": 0,
        "per_ticker": {},
    }

    total_chunks = 0
    kb0 = builder.count()

    for i, tk in enumerate(tickers, start=1):
        if args.sleep > 0 and i > 1:
            time.sleep(float(args.sleep))
        per: dict[str, int] = {"tushare": 0, "akshare": 0, "cninfo": 0}
        all_docs: list[dict[str, Any]] = []
        try:
            if want_tush:
                d = layer_tushare_report_rc(tk, as_cut, args.top_n)
                per["tushare"] = len(d)
                all_docs.extend(d)
            if want_ak:
                d = layer_akshare_reports(tk, as_cut, args.top_n)
                per["akshare"] = len(d)
                all_docs.extend(d)
            if want_cn:
                d = layer_cninfo_latest_annual(tk, as_cut, args.top_n)
                per["cninfo"] = len(d)
                all_docs.extend(d)
            if all_docs:
                nw = builder.build_from_documents(all_docs)
                total_chunks += int(nw)
        except Exception as e:
            logger.error(f"[ingest_rc] {tk} EXCEPTION: {e}")
        stats["per_ticker"][tk] = per

    stats["chunks_written"] = total_chunks
    stats["collection_count_after"] = builder.count()
    stats["kb_started_at"] = kb0
    manifest_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        f"[ingest_rc] 完成 manifest={manifest_path.resolve()} chunks≈{total_chunks} KB={stats['collection_count_after']}"
    )


if __name__ == "__main__":
    main()
