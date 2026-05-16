#!/usr/bin/env python3
"""KB 检索 + DashScope 情绪打分 → sentiment_scores_{as_of}.parquet."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.agents.llm_client import LLMClient  # noqa: E402
from quantmind.kb.retriever import HybridRetriever  # noqa: E402


def _parse_score(text: str) -> float:
    if not text:
        return float("nan")
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", "."))
    if not m:
        return float("nan")
    try:
        v = float(m.group(0))
        return max(-1.0, min(1.0, v))
    except ValueError:
        return float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD PIT 截止")
    p.add_argument("--provider", default="dashscope")
    p.add_argument("--model", default="qwen-plus")
    p.add_argument("--universe-parquet", type=Path, default=None)
    p.add_argument("--kb-collection", default="default")
    p.add_argument("--top-news", type=int, default=5)
    p.add_argument("--max-chars", type=int, default=3500)
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--timeout-skip", type=float, default=10.0, help="仅文档用途；LLMClient 内部超时随 SDK")
    args = p.parse_args()

    del args.timeout_skip

    as_of = date.fromisoformat(str(args.as_of)[:10])
    univ_path = args.universe_parquet or (_ROOT / f"data/snapshots/{as_of.isoformat()}/universe.parquet")
    if not univ_path.is_file():
        logger.error("universe parquet missing: {}", univ_path)
        return 1
    uni = pd.read_parquet(univ_path)
    col = "ticker" if "ticker" in uni.columns else "ts_code"
    tickers = uni[col].astype(str).tolist()
    if args.max_tickers:
        tickers = tickers[: int(args.max_tickers)]

    use_kb = args.provider.lower() != "none"
    retriever = HybridRetriever(collection_name=args.kb_collection, use_reranker=False) if use_kb else None
    llm = LLMClient(provider=args.provider, model=args.model)

    kb_ok = False
    if use_kb and retriever is not None:
        try:
            kb_ok = retriever.count() > 0
        except Exception:
            kb_ok = False
        if not kb_ok:
            logger.warning("KB empty or unreadable — sentiment_score will be NaN (no retrieval).")

    rows: list[dict[str, object]] = []
    for i, tk in enumerate(tickers):
        news_parts: list[str] = []
        try:
            if kb_ok and retriever is not None:
                hits = retriever.retrieve(
                    f"{tk} 公司近期公告或新闻报道",
                    top_k=int(args.top_news),
                    as_of=as_of,
                    ticker_filter=tk,
                )
                news_parts = [h.text for h in hits if h.text][: int(args.top_news)]
        except Exception as e:  # noqa: BLE001
            logger.warning("{} retrieval failed: {}", tk, str(e)[:120])
        news_text = "\n".join(news_parts)[: int(args.max_chars)]
        score = float("nan")
        if news_parts:
            prompt_user = (
                f"以下是{tk}近期新闻与检索摘要，请从 -1（极度悲观）到 +1（极度乐观）打分，"
                f"只输出一个浮点数，不要其它文字。\n\n{news_text}"
            )
            resp = llm.chat(system="你是严谨的卖方研究员。", user=prompt_user, temperature=0.2)
            if resp is not None and resp.content:
                score = _parse_score(resp.content)
        rows.append({
            "ticker": tk,
            "sentiment_score": score,
            "news_count": len(news_parts),
        })
        if (i + 1) % 20 == 0:
            logger.info("sentiment progress {}/{}", i + 1, len(tickers))

    out_dir = _ROOT / "data/features"
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / f"sentiment_scores_{as_of.isoformat()}.parquet"
    pd.DataFrame(rows).to_parquet(outp, index=False)
    logger.info("wrote {} rows → {}", len(rows), outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
