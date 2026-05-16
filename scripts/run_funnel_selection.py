"""scripts/run_funnel_selection.py — 全市场漏斗选股 CLI.

用法：
    python scripts/run_funnel_selection.py \\
        --date 2024-12-31 \\
        --top-n 15 \\
        --universe csi300 \\
        --output data/recommendations/2024-12-31/funnel_candidates.json

    # 跳过基本面层（数据不足时）
    python scripts/run_funnel_selection.py \\
        --date 2024-12-31 \\
        --universe full_a \\
        --skip-layers 4 \\
        --top-n 15

    # 快速模式：只用CSI300价格，跳过LLM
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \\
    python scripts/run_funnel_selection.py \\
        --date 2024-12-31 \\
        --universe csi300 \\
        --provider none
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="全市场漏斗选股系统")
    p.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="快照日期 YYYY-MM-DD（默认今天）",
    )
    p.add_argument(
        "--universe",
        default="csi300",
        choices=["csi300", "csi1000", "full_a", "custom"],
        help="股票池（默认 csi300）；custom 时需 --custom-universe",
    )
    p.add_argument("--top-n", type=int, default=15, help="最终输出数量")
    p.add_argument("--lgbm-top", type=int, default=50, help="LGBM保留数量")
    p.add_argument(
        "--skip-layers",
        type=int,
        nargs="*",
        default=[],
        help="跳过的层编号，例如：--skip-layers 4",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSON 路径（默认 data/recommendations/{date}/funnel_candidates.json）",
    )
    p.add_argument("--provider", default="none", help="LLM provider（none/dashscope/openai）")
    p.add_argument("--model", default="qwen-plus", help="LLM model")
    p.add_argument(
        "--price-panel",
        type=Path,
        default=None,
        help="日线价格面板路径（parquet）",
    )
    p.add_argument(
        "--fundamentals",
        type=Path,
        default=None,
        help="基本面数据路径（parquet）",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=_ROOT / "models/lgbm_v1_final.pkl",
        help="LightGBM 模型路径",
    )
    p.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="测试模式：Layer1 入口随机抽样至多 N 只（避免首次全市场耗时过长）",
    )
    p.add_argument(
        "--custom-universe",
        type=Path,
        default=None,
        help="universe=custom 时：每行一个 ts_code（如 600519.SH）或 csv 首列",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="追加写入 Loguru 日志文件路径",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.universe == "custom":
        if args.custom_universe is None or not args.custom_universe.is_file():
            print(
                "错误：universe=custom 时必须提供有效的 --custom-universe 文件路径",
                file=sys.stderr,
            )
            sys.exit(2)

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logger.add(args.log, encoding="utf-8", level="DEBUG")

    logger.info(f"[FunnelCLI] ===== 漏斗选股系统 =====")
    logger.info(f"[FunnelCLI] 日期: {args.date} | Universe: {args.universe} | TopN: {args.top_n}")
    if args.skip_layers:
        logger.info(f"[FunnelCLI] 跳过层: {args.skip_layers}")

    t0 = time.monotonic()

    from quantmind.data.shared_cache import SharedDataCache
    from quantmind.selection.funnel_selector import FunnelResult, FunnelSelector
    from quantmind.selection.lazy_data_engine import LazyDataEngine

    shared_cache = SharedDataCache.get_instance()
    data_engine = LazyDataEngine(
        args.date,
        shared_cache=shared_cache,
        universe=args.universe,
        custom_universe_file=args.custom_universe if args.universe == "custom" else None,
    )

    selector = FunnelSelector(
        as_of=args.date,
        data_engine=data_engine,
        lgbm_model_path=args.model_path,
        provider=args.provider,
        model_name=args.model,
        universe=args.universe,
        # 兼容旧参数（传给内部 price panel 路径等）
        price_panel_path=args.price_panel,
        fundamentals_path=args.fundamentals,
        custom_universe_file=args.custom_universe if args.universe == "custom" else None,
    )

    result = selector.run(
        skip_layers=args.skip_layers,
        top_n=args.top_n,
        lgbm_top=args.lgbm_top,
        max_tickers=args.max_tickers,
    )

    if isinstance(result, FunnelResult):
        candidates = result.candidates
        stats = result  # to_output_json 接受 FunnelResult
    else:
        candidates, stats = result

    # 生成输出
    output_data = selector.to_output_json(candidates, stats)

    # 确定输出路径
    if args.output is None:
        out_dir = _ROOT / "data" / "recommendations" / args.date
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "funnel_candidates.json"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_path = args.output

    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.monotonic() - t0

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"漏斗选股完成 — {args.date}")
    print(f"{'='*60}")
    if isinstance(result, FunnelResult):
        for ls in result.layer_stats:
            layer_names = {1:"基础质量",2:"流动性",3:"趋势",4:"基本面",5:"LGBM",6:"LLM精排"}
            flag = " (跳过)" if ls.skipped else ""
            print(f"  Layer{ls.layer} {layer_names.get(ls.layer,'')}: {ls.n_in} → {ls.n_out}{flag}")
        skipped = [ls.layer for ls in result.layer_stats if ls.skipped]
        if skipped:
            print(f"  跳过层:         {skipped}")
        cr = result.cache_stats.get("hit_rate", 0)
        print(f"  缓存命中率:     {cr:.1%}")
    else:
        s = stats
        print(f"  Layer1 基础质量: {s.layer1_in} → {s.layer1_out}")
        print(f"  Layer6 LLM精排:  {s.layer6_in} → {s.layer6_out}")
    print(f"  最终候选:       {len(candidates)}只")
    print(f"  耗时:           {elapsed:.1f}s")
    print(f"  输出:           {output_path}")
    print(f"{'='*60}")

    if candidates is not None and len(candidates) > 0:
        print("\n最终候选股票：")
        for _, row in candidates.iterrows():
            score = f"lgbm={row.get('lgbm_score', 'N/A'):.3f}" if pd.notna(row.get('lgbm_score')) else ""
            print(f"  {row['ticker']:15s} {score}")

    logger.info(f"[FunnelCLI] 结果写入 → {output_path}")


if __name__ == "__main__":
    main()
