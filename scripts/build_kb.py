"""scripts/build_kb.py — 知识库构建 CLI.

用法：
    # 干跑（查看将会处理哪些数据，不写入 ChromaDB）
    python scripts/build_kb.py --tickers 600519.SH --doc-types news --start 2024-06-01 --end 2024-06-30 --dry-run

    # 实际构建（新闻，指定日期区间）
    python scripts/build_kb.py --tickers 600519.SH,300750.SZ --doc-types news --start 2024-01-01 --end 2024-12-31

    # 从新闻构建（指定股票 + 截止日期 + 回溯天数，兼容旧用法）
    python scripts/build_kb.py --source news --tickers 600519,000858 --as-of 2024-06-30

    # 从本地报告目录构建
    python scripts/build_kb.py --source reports --reports-dir data/reports --as-of 2024-06-30

    # 从 Snapshot 构建财务摘要
    python scripts/build_kb.py --source snapshot --snapshot-path data/snapshots/snapshot.pkl

    # 全量构建（三路合并）
    python scripts/build_kb.py --source all --tickers 600519,000858 --as-of 2024-06-30

    # 查询知识库状态
    python scripts/build_kb.py --status
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# 确保项目根目录在 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantMind Phase 5 — 知识库构建工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── 数据来源（新旧两种写法均支持）─────────────────────────────────────
    parser.add_argument(
        "--source",
        choices=["news", "reports", "snapshot", "all"],
        default=None,
        help="数据来源（news/reports/snapshot/all）；与 --doc-types 二选一",
    )
    parser.add_argument(
        "--doc-types",
        default=None,
        help="文档类型，逗号分隔（news,reports,snapshot）；与 --source 效果相同",
    )
    # ── 股票 ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tickers",
        default=None,
        help="股票代码列表，逗号分隔（如 600519.SH,300750.SZ）",
    )
    # ── 日期（新旧两种写法）────────────────────────────────────────────────
    parser.add_argument(
        "--as-of",
        default=None,
        help="数据截止日期 YYYY-MM-DD（默认今天）；与 --end 等价",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="数据开始日期 YYYY-MM-DD（与 --end 一起指定日期区间）",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="数据结束日期 YYYY-MM-DD（等同于 --as-of）",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="新闻往前抓取天数（默认 90；--start 优先级更高）",
    )
    # ── 路径 ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--reports-dir",
        default="data/reports",
        help="本地报告目录（默认 data/reports）",
    )
    parser.add_argument(
        "--snapshot-path",
        default=None,
        help="Snapshot pickle 路径（legacy .pkl 财务摘要）",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Parquet snapshot 目录，如 data/snapshots/2024-12-31",
    )
    parser.add_argument(
        "--snapshot-as-of",
        default=None,
        help="Snapshot PIT 日期 YYYY-MM-DD（默认与 --as-of 或目录名一致）",
    )
    parser.add_argument(
        "--snapshot-tickers",
        default=None,
        help="仅摄入的 ticker 列表（逗号分隔）；默认复用 --tickers",
    )
    # ── ChromaDB ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--collection",
        default="default",
        help="ChromaDB 集合名称（默认 default）",
    )
    parser.add_argument(
        "--chroma-dir",
        default=".cache/chromadb",
        help="ChromaDB 持久化目录（默认 .cache/chromadb）",
    )
    # ── 分块 ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="分块大小（默认 500）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="分块重叠（默认 100）",
    )
    # ── 操作模式 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑：列出待处理任务但不实际写入 ChromaDB",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="查询知识库当前状态后退出",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="清空知识库后重新构建",
    )
    return parser.parse_args()


def print_status(collection: str, chroma_dir: str) -> None:
    """打印知识库状态."""
    from quantmind.kb.retriever import HybridRetriever
    r = HybridRetriever(collection_name=collection, chroma_dir=chroma_dir)
    count = r.count()
    print(f"\n{'='*50}")
    print(f"QuantMind 知识库状态")
    print(f"{'='*50}")
    print(f"Collection : quantmind_kb_{collection}")
    print(f"ChromaDB   : {chroma_dir}")
    print(f"文档总数   : {count}")
    print(f"{'='*50}\n")


def _resolve_snapshot_parquet_dir(args: argparse.Namespace) -> Path | None:
    """Parquet 快照目录：优先 --snapshot-dir；若 --snapshot-path 指向已存在目录则同等对待."""
    if args.snapshot_dir:
        p = Path(args.snapshot_dir)
        return p if p.is_dir() else None
    if args.snapshot_path:
        p = Path(args.snapshot_path)
        if p.is_dir():
            return p
    return None


def _resolve_source(args: argparse.Namespace) -> str:
    """统一 --source 与 --doc-types 的解析结果."""
    if args.doc_types:
        # 规范化：news,reports,snapshot → 多选；单个也支持
        types = {t.strip().lower() for t in args.doc_types.split(",")}
        if types == {"news", "reports", "snapshot"} or "all" in types:
            return "all"
        if len(types) == 1:
            return types.pop()
        # 多个但非全集：返回逗号合并（main() 里会逐一处理）
        return ",".join(sorted(types))
    return args.source or "all"


def _dry_run_report(
    source: str,
    tickers: list[str],
    start_date: date,
    end_date: date,
    args: argparse.Namespace,
) -> None:
    """打印 dry-run 任务预览，不写入 KB."""
    days = (end_date - start_date).days + 1
    sources_list = [s for s in ("news", "reports", "snapshot") if s in source or source == "all"]

    print(f"\n{'='*60}")
    print("QuantMind 知识库构建 — 干跑预览（DRY RUN）")
    print(f"{'='*60}")
    print(f"日期区间   : {start_date}  →  {end_date}  ({days} 天)")
    print(f"目标股票   : {', '.join(tickers) if tickers else '（未指定）'}")
    print(f"数据来源   : {', '.join(sources_list)}")
    print(f"Collection : quantmind_kb_{args.collection}")
    print(f"ChromaDB   : {args.chroma_dir}")
    print(f"分块大小   : {args.chunk_size}  重叠: {args.chunk_overlap}")
    print()

    if "news" in sources_list:
        est_news = len(tickers) * days // 7 * 3  # 粗估：每周3篇/只
        print(f"  📰 新闻   : {len(tickers)} 只 × ~{days//7*3} 篇 ≈ {est_news} 篇（预估 chunks ≈ {est_news * 2}）")
        for t in tickers:
            print(f"             {t}: news, {start_date} ~ {end_date}")

    if "reports" in sources_list:
        reports_dir = Path(args.reports_dir)
        if reports_dir.exists():
            files = list(reports_dir.glob("*.html")) + list(reports_dir.glob("*.pdf")) + list(reports_dir.glob("*.txt"))
            print(f"  📄 报告   : {reports_dir} → {len(files)} 个文件（实际 chunks 依文件大小而定）")
        else:
            print(f"  📄 报告   : {reports_dir} 目录不存在，将跳过")

    if "snapshot" in sources_list:
        if args.snapshot_dir:
            print(f"  💾 Snapshot (parquet): {args.snapshot_dir}")
            st = [t.strip() for t in (args.snapshot_tickers or "").split(",") if t.strip()] or tickers
            print(f"             tickers: {st or '（未指定，可能仅有指数环境文档）'}")
        else:
            snap_path = args.snapshot_path or "data/snapshots/*.pkl（自动发现）"
            print(f"  💾 Snapshot (legacy .pkl): {snap_path}")
            if tickers:
                print(f"             将提取 {len(tickers)} 只股票的财务摘要")

    print()
    print("  ✅  干跑完成，无数据被写入 ChromaDB。")
    print("  ℹ️   去掉 --dry-run 参数即可开始实际构建。")
    print(f"{'='*60}\n")


def _snapshot_dry_run_real_parse(
    args: argparse.Namespace,
    snap_root: Path,
    as_of: date,
    tickers_snap: list[str] | None,
) -> None:
    """干跑：解析 parquet 并统计文档数（不写 Chroma / 不加载嵌入）。"""
    from quantmind.kb.builder import KBBuilder
    from quantmind.kb.chunker import SmartChunker

    snap_as_of: date = as_of
    if args.snapshot_as_of:
        snap_as_of = datetime.strptime(args.snapshot_as_of, "%Y-%m-%d").date()

    chunker = SmartChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    builder = KBBuilder(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        chunker=chunker,
    )
    res = builder.build_from_snapshot_dir(
        snap_root,
        tickers=tickers_snap if tickers_snap else None,
        as_of=snap_as_of,
        dry_run=True,
    )
    n_docs = int(res.get("n_docs", 0))
    n_chunks = int(res.get("n_chunks", 0))
    missing = res.get("parquet_missing") or []
    loaded = res.get("parquet_loaded") or []

    dtc = res.get("doc_type_counts") or {}

    print(f"\n{'='*60}")
    print("Snapshot parquet — 干跑解析（DRY RUN，未写入 ChromaDB）")
    print(f"{'='*60}")
    print(f"目录       : {snap_root.resolve()}")
    print(f"as_of      : {res.get('as_of', snap_as_of)}")
    print(f"n_docs     : {n_docs}")
    print(f"n_chunks   : {n_chunks}（分块后）")
    print(f"tickers    : {res.get('n_tickers_ingested', 0)}")
    print(f"parquet 已加载: {len(loaded)} 个")
    if missing:
        print(f"parquet 缺失（已跳过）: {', '.join(missing)}")
    if dtc:
        print("doc_type 文档条数:")
        for k in sorted(dtc.keys()):
            print(f"  {k}: {dtc[k]}")
    print(f"{'='*60}\n")


def main() -> None:
    args = parse_args()

    # ── 日期解析 ─────────────────────────────────────────────────────────
    # 优先级: --end > --as-of > today
    as_of: date
    if args.end:
        as_of = datetime.strptime(args.end, "%Y-%m-%d").date()
    elif args.as_of:
        as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    else:
        as_of = date.today()

    # 开始日期: --start > as_of - days_back
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        days_back = (as_of - start_date).days
    else:
        days_back = args.days_back
        start_date = as_of - timedelta(days=days_back)

    # ── 查询状态 ─────────────────────────────────────────────────────────
    if args.status:
        print_status(args.collection, args.chroma_dir)
        return

    # ── 解析 ticker / source ─────────────────────────────────────────────
    tickers: list[str] = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    source = _resolve_source(args)

    # ── 干跑 ─────────────────────────────────────────────────────────────
    if args.dry_run:
        sources_list = [s for s in ("news", "reports", "snapshot") if s in source or source == "all"]
        snap_root = _resolve_snapshot_parquet_dir(args)
        st_from_arg: list[str] = []
        if args.snapshot_tickers:
            st_from_arg = [t.strip() for t in args.snapshot_tickers.split(",") if t.strip()]
        tickers_snap = st_from_arg if st_from_arg else list(tickers)
        did_snap_parse = False
        if "snapshot" in sources_list and snap_root is not None:
            _snapshot_dry_run_real_parse(args, snap_root, as_of, tickers_snap or None)
            did_snap_parse = True
        elif "snapshot" in sources_list:
            logger.warning(
                "[BuildKB] dry-run snapshot: 未找到 parquet 目录（请使用 --snapshot-dir "
                "或让 --snapshot-path 指向快照日期目录）"
            )
        if not (did_snap_parse and sources_list == ["snapshot"]):
            _dry_run_report(source, tickers, start_date, as_of, args)
        return

    # ── 初始化 Builder ───────────────────────────────────────────────────
    from quantmind.kb.builder import KBBuilder
    from quantmind.kb.chunker import SmartChunker

    chunker = SmartChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    builder = KBBuilder(
        collection_name=args.collection,
        chroma_dir=args.chroma_dir,
        chunker=chunker,
    )

    # 清空
    if args.clear:
        logger.info("[BuildKB] 清空知识库...")
        builder.get_retriever().clear()

    total = 0
    t_run0 = time.perf_counter()

    # ── 新闻来源 ─────────────────────────────────────────────────────────
    if "news" in source or source == "all":
        if not tickers:
            logger.warning("[BuildKB] --source news/all 需要指定 --tickers")
        else:
            logger.info("[BuildKB] source=news start")
            t0 = time.perf_counter()
            logger.info(
                f"[BuildKB] 抓取新闻: tickers={tickers}, "
                f"as_of={as_of}, days_back={days_back}"
            )
            n = builder.build_from_news(
                tickers=tickers,
                as_of=as_of,
                days_back=days_back,
            )
            logger.info(f"[BuildKB] 新闻 → {n} chunks")
            total += n
            logger.info(
                f"[BuildKB] source=news end elapsed_s={time.perf_counter() - t0:.2f}"
            )

    # ── 报告目录来源 ─────────────────────────────────────────────────────
    if "reports" in source or source == "all":
        logger.info("[BuildKB] source=reports start")
        t0 = time.perf_counter()
        logger.info(f"[BuildKB] 导入报告目录: {args.reports_dir}")
        n = builder.build_from_reports_dir(
            reports_dir=args.reports_dir,
            as_of=as_of,
        )
        logger.info(f"[BuildKB] 研究报告 → {n} chunks")
        total += n
        logger.info(
            f"[BuildKB] source=reports end elapsed_s={time.perf_counter() - t0:.2f}"
        )

    # ── Snapshot 来源（parquet 目录优先，其次 legacy .pkl）──────────────────
    if "snapshot" in source or source == "all":
        logger.info("[BuildKB] source=snapshot start")
        t0 = time.perf_counter()
        n_snap = 0
        used_parquet = False

        st_from_arg: list[str] = []
        if args.snapshot_tickers:
            st_from_arg = [t.strip() for t in args.snapshot_tickers.split(",") if t.strip()]
        st_eff = st_from_arg if st_from_arg else list(tickers)

        snap_root = _resolve_snapshot_parquet_dir(args)
        if snap_root is not None:
            if snap_root.is_dir():
                snap_as_of: date = as_of
                if args.snapshot_as_of:
                    snap_as_of = datetime.strptime(args.snapshot_as_of, "%Y-%m-%d").date()
                tickers_snap = st_eff if st_eff else None
                logger.info(
                    f"[BuildKB] 从 parquet snapshot 目录摄入: {snap_root} "
                    f"as_of={snap_as_of} tickers={tickers_snap or 'full_universe'}"
                )
                res = builder.build_from_snapshot_dir(
                    snap_root,
                    tickers=tickers_snap,
                    as_of=snap_as_of,
                    dry_run=False,
                )
                n_snap = int(res.get("chunks_written", 0))
                used_parquet = True
                logger.info(
                    f"[BuildKB] Snapshot parquet: docs={res.get('n_docs', 0)} "
                    f"chunks_written={n_snap}"
                )
            else:
                logger.warning(
                    f"[BuildKB] snapshot: parquet 目录无效: {snap_root}，跳过 parquet 摄入"
                )
        else:
            logger.info(
                "[BuildKB] snapshot: 未提供 --snapshot-dir（parquet），将尝试 legacy data/snapshots/*.pkl"
            )

        if not used_parquet:
            snap_path = args.snapshot_path
            if not snap_path:
                default_snaps = list(Path("data/snapshots").glob("*.pkl"))
                if default_snaps:
                    snap_path = str(default_snaps[0])
                    logger.info(f"[BuildKB] 自动发现 legacy Snapshot: {snap_path}")
                else:
                    logger.info(
                        "[BuildKB] snapshot skipped: no valid --snapshot-dir, "
                        "no --snapshot-path, no data/snapshots/*.pkl found"
                    )

            if snap_path:
                logger.info(f"[BuildKB] 从 legacy pickle Snapshot 提取: {snap_path}")
                n_pkl = builder.build_from_snapshot(
                    snapshot_path=snap_path,
                    tickers=tickers or None,
                    as_of=as_of,
                )
                n_snap += n_pkl
                logger.info(f"[BuildKB] legacy Snapshot .pkl → {n_pkl} chunks")

        total += n_snap
        logger.info(f"[BuildKB] Snapshot 合计 → {n_snap} chunks")
        logger.info(
            f"[BuildKB] source=snapshot end elapsed_s={time.perf_counter() - t0:.2f}"
        )

    run_elapsed = time.perf_counter() - t_run0
    logger.info(f"[BuildKB] run total elapsed_s={run_elapsed:.2f}")

    # ── 汇报 ─────────────────────────────────────────────────────────────
    final_count = builder.count()
    print(f"\n{'='*50}")
    print(f"QuantMind 知识库构建完成")
    print(f"{'='*50}")
    print(f"本次新增    : {total} chunks")
    print(f"知识库总量  : {final_count} chunks")
    print(f"Collection  : quantmind_kb_{args.collection}")
    print(f"ChromaDB    : {args.chroma_dir}")
    print(f"日期区间    : {start_date} → {as_of}")
    print(f"总耗时(s)  : {run_elapsed:.2f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
