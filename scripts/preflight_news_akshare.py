"""AkShare 新闻预检：真实拉取行数 + 按 KBBuilder 规则模拟 PIT 过滤后的条数（不写 KB）."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.kb.builder import (  # noqa: E402
    NEWS_BODY_CANDIDATES,
    NEWS_TIME_CANDIDATES,
    NEWS_TITLE_CANDIDATES,
    NEWS_URL_CANDIDATES,
    KBBuilder,
    _row_title_body_url_time,
    ordered_candidates_by_nonempty_rate,
)
from quantmind.kb.chunker import SmartChunker  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tickers",
        required=True,
        help="逗号分隔，如 000858.SZ,600036.SH",
    )
    p.add_argument("--as-of", default="2024-12-31")
    p.add_argument("--days-back", type=int, default=90)
    p.add_argument("--sleep", type=float, default=2.0, help="每只股票之间的间隔（秒）")
    p.add_argument("--collection", default="default")
    p.add_argument("--chroma-dir", default=".cache/chromadb")
    return p.parse_args()


def _simulate_kept(news_list: list[dict], ticker: str, cutoff: date, as_of: date) -> dict:
    title_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_TITLE_CANDIDATES)
    body_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_BODY_CANDIDATES)
    url_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_URL_CANDIDATES)
    time_order = ordered_candidates_by_nonempty_rate(news_list, NEWS_TIME_CANDIDATES)
    drop_before = drop_after = kept = 0
    no_date = 0
    for item in news_list:
        _, _, _, pub = _row_title_body_url_time(
            item, title_order, body_order, url_order, time_order,
        )
        if pub is None:
            no_date += 1
        if pub is not None and pub < cutoff:
            drop_before += 1
            continue
        if pub is not None and pub > as_of:
            drop_after += 1
            continue
        kept += 1
    return {
        "raw": len(news_list),
        "kept_after_pit": kept,
        "drop_before_cutoff": drop_before,
        "drop_after_asof": drop_after,
        "rows_without_parseable_date": no_date,
    }


def main() -> None:
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    as_of = date.fromisoformat(args.as_of)
    cutoff = as_of - timedelta(days=args.days_back)

    chunker = SmartChunker()
    builder = KBBuilder(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        chunker=chunker,
    )

    print(f"as_of={as_of}  cutoff={cutoff}  days_back={args.days_back}")
    print(f"tickers={len(tickers)}")
    print("-" * 72)
    for i, tk in enumerate(tickers):
        if i and args.sleep > 0:
            time.sleep(args.sleep)
        code = tk.split(".")[0]
        try:
            news_list = builder._fetch_news(tk, as_of)
        except Exception as e:
            print(f"{tk} (sym={code})  FETCH_ERROR: {e}")
            continue
        stats = _simulate_kept(news_list, tk, cutoff, as_of)
        print(
            f"{tk}  raw_rows={stats['raw']}  "
            f"kept_after_pit={stats['kept_after_pit']}  "
            f"drop_before={stats['drop_before_cutoff']}  "
            f"drop_after_asof={stats['drop_after_asof']}  "
            f"no_parse_date={stats['rows_without_parseable_date']}",
        )


if __name__ == "__main__":
    main()
