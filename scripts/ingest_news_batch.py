"""逐只抓取 AkShare 新闻并写入 KB（股票间 sleep，失败可记录后重试）。

说明：AkShare EM 为实时资讯；默认 as_of=今天，`build_from_news` 不致将新闻全部丢弃。

用法示例：
  # 初次全量 pending
  PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \\
    python -u scripts/ingest_news_batch.py \\
      --ticker-file reports/news_fullbatch/pending_tickers.txt \\
      --sleep 2 --failed-output reports/news_fullbatch/failed_tickers.txt

  # 仅用失败列表重跑
  PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \\
    python -u scripts/ingest_news_batch.py \\
      --retry-failed \\
      --failed-output reports/news_fullbatch/failed_tickers.txt \\
      --sleep 2

Ctrl+C 后可用 ``--retry-failed``（或手动 ``--ticker-file failed_tickers.txt``）续跑。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.kb.builder import KBBuilder  # noqa: E402
from quantmind.kb.chunker import SmartChunker  # noqa: E402


def parse_tickers_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tickers", help="逗号分隔 ticker（与 ticker-file/retry-failed 三选一）")
    src.add_argument("--ticker-file", type=Path, help="每行一只 ticker")

    src.add_argument(
        "--retry-failed",
        action="store_true",
        help=f"等价于读取 --failed-output 中的 ticker 列表",
    )

    p.add_argument(
        "--failed-output",
        type=Path,
        default=Path("reports/news_fullbatch/failed_tickers.txt"),
        help="本批失败 ticker 写入此文件（重试时用 --retry-failed 读取同一路径）",
    )
    p.add_argument("--batch-size", type=int, default=50, help="每 ingest 几只打印一批摘要（不改变顺序）")

    p.add_argument(
        "--as-of",
        default="",
        help="PIT 截止日 YYYY-MM-DD；空=今天（实时新闻 recommended）",
    )
    p.add_argument("--days-back", type=int, default=90)
    p.add_argument("--sleep", type=float, default=2.0)
    p.add_argument("--collection", default="default")
    p.add_argument("--chroma-dir", default=".cache/chromadb")
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--chunk-overlap", type=int, default=100)
    return p.parse_args()


def _collect_tickers(args: argparse.Namespace) -> list[str]:
    if args.retry_failed:
        if not args.failed_output.is_file():
            raise SystemExit(f"--retry-failed 需要已有的失败列表: {args.failed_output}")
        return parse_tickers_file(args.failed_output)
    assert args.tickers or args.ticker_file
    if args.ticker_file is not None:
        if not args.ticker_file.is_file():
            raise SystemExit(f"ticker-file 不存在: {args.ticker_file}")
        return parse_tickers_file(args.ticker_file)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    return tickers


def main() -> None:
    args = parse_args()
    tickers = _collect_tickers(args)
    total_n = len(tickers)
    if total_n == 0:
        print("[ingest_news_batch] ticker 列表为空，退出", flush=True)
        return

    as_of = (
        datetime.strptime(args.as_of.strip(), "%Y-%m-%d").date()
        if args.as_of.strip()
        else date.today()
    )

    chunker = SmartChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    builder = KBBuilder(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        chunker=chunker,
    )
    kb0 = builder.count()

    failed_path = args.failed_output
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text("", encoding="utf-8")

    print(
        f"[ingest_news_batch] as_of={as_of} days_back={args.days_back} "
        f"tickers={total_n} sleep={args.sleep}s batch_hint={args.batch_size} KB_start={kb0}",
        flush=True,
    )

    batch_chunks = 0
    failures: list[str] = []

    for i, tk in enumerate(tickers, start=1):
        if i > 1 and args.sleep > 0:
            time.sleep(float(args.sleep))
        chunks = 0
        try:
            chunks = builder.build_from_news([tk], as_of=as_of, days_back=args.days_back, raise_on_error=True)
        except Exception as e:
            print(f"[{i}/{total_n}] {tk}  EXCEPTION  err={e!s}", flush=True)
            failures.append(tk)
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            with failed_path.open("a", encoding="utf-8") as fh:
                fh.write(tk + "\n")
                fh.flush()

        kb_total = builder.count()
        print(f"[{i}/{total_n}] {tk}  写入={chunks} chunks  KB总数={kb_total}", flush=True)
        batch_chunks += chunks

        if i % args.batch_size == 0 or i == total_n:
            print(
                f"[ingest_news_batch] -- 进度小结 已处理{i}/{total_n} 本摘要窗chunks合计={batch_chunks}",
                flush=True,
            )

    kb1 = builder.count()
    fail_n = len(failures)
    print(
        f"[ingest_news_batch] DONE  KB {kb0}->{kb1}  "
        f"本次失败={fail_n} 见 {failed_path.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
