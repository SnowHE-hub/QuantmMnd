"""scripts/run_investment_pipeline.py — 投资分析完整 Pipeline.

流程：
  Step 1: 读取候选股票（来自 daily_update.py 输出）
  Step 2: 对每只股票运行 6 个 Agent + StrategyAgent
  Step 3: 对 BUY 信号做历史回测验证
  Step 4: 生成最终 Markdown 报告

用法：
    python scripts/run_investment_pipeline.py \\
        --date 2024-12-31 \\
        --provider dashscope \\
        --model qwen-plus \\
        --candidates-json data/recommendations/2024-12-31/top10.json \\
        --output-dir reports/investment_pipeline/ \\
        --top-n 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.agents.investment_agents import (
    MomentumAgent,
    QualityAgent,
    RiskAgent,
    SentimentAgent,
    StrategyAgent,
    ValuationAgent,
)
from quantmind.agents.investment_agents.strategy_agent import InvestmentStrategy
from scripts.validate_strategies import (
    StrategyValidationResult,
    batch_validate,
)

# 价格面板优先级：alpha 长表 > 旧 daily_prices > CSI300 宽表
_PRICE_FILE_LONG = _ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
_PRICE_FILE_LONG_FALLBACK = _ROOT / "data" / "raw" / "daily_prices_panel.parquet"
_PRICE_FILE_WIDE = _ROOT / "data" / "prices" / "csi300_daily_adj_close.parquet"

# 因子面板优先级：alpha v3 > 旧 test/val 拼接
_PANEL_FILES = [
    _ROOT / "data/panel/alpha_panel_v3.parquet",
    _ROOT / "data/panel/test.parquet",
    _ROOT / "data/panel/val.parquet",
]


def _load_price_df() -> pd.DataFrame:
    """加载复权收盘价宽表（index=trade_date，columns=tickers）.

    优先从 alpha_prices_panel（长表）pivot；退回 CSI300 宽表。
    """
    import pyarrow.parquet as _pq

    for longp in [_PRICE_FILE_LONG, _PRICE_FILE_LONG_FALLBACK]:
        if not longp.is_file():
            continue
        try:
            schema_names = _pq.ParquetFile(str(longp)).schema_arrow.names
            price_col = "adj_close" if "adj_close" in schema_names else "close"
            df = pd.read_parquet(longp, columns=["trade_date", "ts_code", price_col])
            wide = df.pivot_table(index="trade_date", columns="ts_code", values=price_col, aggfunc="last")
            wide.index = pd.to_datetime(wide.index)
            wide.columns.name = None
            logger.info(f"[pipeline] 价格宽表 from {longp.name}: shape={wide.shape}")
            return wide
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[pipeline] 加载长表价格失败 {longp}: {e}")
    # 退回 CSI300 宽表
    if _PRICE_FILE_WIDE.is_file():
        df = pd.read_parquet(_PRICE_FILE_WIDE)
        logger.warning("[pipeline] 回退至 CSI300 宽表价格（覆盖范围有限）")
        return df
    logger.error("[pipeline] 未找到任何价格文件，返回空 DataFrame")
    return pd.DataFrame()


# ── Step 1: 候选股票 ──────────────────────────────────────────────────────────

def load_candidates(json_path: Path) -> list[dict[str, Any]]:
    """从 top10.json 读取候选股票列表."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        candidates = data.get("top10", data.get("candidates", []))
    else:
        candidates = data
    return candidates


def retrieve_context(ticker: str, as_of: str, chroma_dir: str = ".cache/chromadb") -> dict:
    """从 KB 检索股票上下文，失败则返回空 dict."""
    try:
        from quantmind.kb.retriever import HybridRetriever

        retriever = HybridRetriever(
            collection_name="default",
            chroma_dir=chroma_dir,
            use_reranker=False,
        )
        if retriever.count() == 0:
            logger.warning(f"[pipeline] ChromaDB 为空，{ticker} 使用空 context")
            return _empty_context(ticker, as_of)

        # 避免重复导入 retrieve_stock_context 的函数
        from scripts.retrieve_stock_context import (
            SNAPSHOT_DOC_TYPES,
            retrieve_market_context,
            retrieve_news,
            retrieve_reports,
            retrieve_snapshot_by_doc_type,
        )
        from datetime import date as _date

        as_of_date = _date.fromisoformat(as_of) if as_of else None
        news_items = retrieve_news(retriever, ticker, as_of_date, 5)
        report_items = retrieve_reports(retriever, ticker, as_of_date, 5)
        snapshot_results: dict[str, list] = {}
        for dt in SNAPSHOT_DOC_TYPES:
            snapshot_results[dt] = retrieve_snapshot_by_doc_type(
                retriever, ticker, as_of_date, dt, 3
            )
        market_items = retrieve_market_context(retriever, as_of_date)

        ctx: dict[str, Any] = {
            "ticker": ticker,
            "as_of": as_of,
            "news_context": news_items,
            "news_count": len(news_items),
            "report_context": report_items,
            "report_count": len(report_items),
            "market_context": market_items,
        }
        for dt in SNAPSHOT_DOC_TYPES:
            ctx[f"snapshot_{dt}"] = snapshot_results[dt]

        return ctx

    except Exception as e:
        logger.warning(f"[pipeline] retrieve_context({ticker}) 失败: {e}，使用空 context")
        return _empty_context(ticker, as_of)


def _empty_context(ticker: str, as_of: str) -> dict:
    """无法检索时的空 context 结构."""
    return {
        "ticker": ticker,
        "as_of": as_of,
        "news_context": [],
        "news_count": 0,
        "report_context": [],
        "report_count": 0,
        "market_context": [],
        "snapshot_company_profile": [],
        "snapshot_latest_market_metrics": [],
        "snapshot_financial_indicator_summary": [],
        "snapshot_northbound_summary": [],
        "snapshot_margin_summary": [],
    }


# ── Step 2: 运行 6 个 Agent ───────────────────────────────────────────────────

def run_six_agents(
    ticker: str,
    as_of: str,
    context: dict,
    provider: str = "none",
    model: str = "qwen-plus",
    agent_version_overrides: dict[str, str] | None = None,
) -> list:
    """运行 5 个分析 Agent，返回 AgentSignal 列表.

    agent_version_overrides: 覆盖各 Agent 的模型版本，例如
        {"MomentumAgent": "lgbm_v2", "SentimentAgent": "tfidf_v2"}
    """
    overrides = agent_version_overrides or {}
    agents = [
        ValuationAgent(ticker, as_of, context,
                       model_version=overrides.get("ValuationAgent", "active")),
        MomentumAgent(ticker, as_of, context,
                      model_version=overrides.get("MomentumAgent", "active")),
        QualityAgent(ticker, as_of, context,
                     model_version=overrides.get("QualityAgent", "active")),
        SentimentAgent(ticker, as_of, context, provider=provider, model=model,
                       model_version=overrides.get("SentimentAgent", "active")),
        RiskAgent(ticker, as_of, context,
                  model_version=overrides.get("RiskAgent", "active")),
    ]
    signals = []
    for agent in agents:
        try:
            sig = agent.analyze()
            signals.append(sig)
            logger.info(
                f"  [{sig.agent_name}] signal={sig.signal:+.2f} conf={sig.confidence:.1%} | {sig.summary[:40]}"
            )
        except Exception as e:
            logger.error(f"  [{type(agent).__name__}] 失败: {e}")
            # 降级：插入零信号
            from quantmind.agents.investment_agents.base_agent import AgentSignal
            signals.append(AgentSignal(
                agent_name=type(agent).__name__,
                ticker=ticker,
                signal=0.0,
                confidence=0.0,
                summary=f"分析失败: {str(e)[:30]}",
                evidence={"error": str(e)},
                warnings=[f"Agent执行失败"],
            ))
    return signals


def save_strategy(strategy: InvestmentStrategy, output_dir: Path) -> None:
    """实时保存单只股票策略（防止中途失败丢失数据）."""
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy_dict = _strategy_to_dict(strategy)
    out_path = output_dir / f"{strategy.ticker}_strategy.json"
    out_path.write_text(
        json.dumps(strategy_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _strategy_to_dict(s: InvestmentStrategy) -> dict:
    """将 InvestmentStrategy 序列化为 JSON 可序列化 dict."""
    return {
        "ticker": s.ticker,
        "as_of": s.as_of,
        "rating": s.rating,
        "composite_signal": s.composite_signal,
        "confidence": s.confidence,
        "entry_price_range": list(s.entry_price_range),
        "target_price_1m": s.target_price_1m,
        "target_price_3m": s.target_price_3m,
        "stop_loss_price": s.stop_loss_price,
        "position_size": s.position_size,
        "holding_horizon": s.holding_horizon,
        "investment_thesis": s.investment_thesis,
        "key_risks": s.key_risks,
        "key_catalysts": s.key_catalysts,
        "agent_signals": s.agent_signals,
        "llm_used": s.llm_used,
    }


# ── Step 4: 报告生成 ──────────────────────────────────────────────────────────

def generate_final_report(
    acceptable: list[StrategyValidationResult],
    watchlist: list[StrategyValidationResult],
    avoid: list[StrategyValidationResult],
    all_strategies: list[InvestmentStrategy],
    output_dir: Path,
    date_str: str,
) -> Path:
    """生成最终 Markdown 投资推荐报告."""
    strategy_map = {s.ticker: s for s in all_strategies}
    lines = []

    lines.append(f"# QuantMind 投资推荐报告 — {date_str}\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    lines.append(f"> 模型：LightGBM + 6 Agent 投资分析系统\n\n")

    # ── ✅ 可接受的投资机会 ─────────────────────────────────────────────────
    lines.append(f"## ✅ 可接受的投资机会（{len(acceptable)}只）\n")
    if acceptable:
        lines.append("| 股票 | 评级 | 综合信号 | 历史胜率 | 期望月收益 | 建议仓位 | 止损价 | 1月目标价 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in acceptable:
            s = strategy_map.get(r.ticker)
            stop = f"¥{s.stop_loss_price:.2f}" if s else "-"
            target = f"¥{s.target_price_1m:.2f}" if s else "-"
            signal_str = f"{s.composite_signal:+.2f}" if s else "-"
            lines.append(
                f"| {r.ticker} | {r.rating} | {signal_str} "
                f"| {r.win_rate:.0%} | {r.avg_return:+.1%} "
                f"| {r.position_size} | {stop} | {target} |"
            )
        lines.append("")

        # 详细分析
        for r in acceptable:
            s = strategy_map.get(r.ticker)
            if not s:
                continue
            lines.append(f"### 详细分析：{r.ticker}\n")
            lines.append(f"**综合信号**：{s.composite_signal:+.2f}（{s.rating}）\n")
            lines.append("| Agent | 信号 | 置信度 | 摘要 |")
            lines.append("|---|---|---|---|")
            for name, info in s.agent_signals.items():
                lines.append(
                    f"| {name} | {info['signal']:+.2f} | {info['confidence']:.0%} | {info['summary'][:40]} |"
                )
            lines.append("")

            if s.investment_thesis:
                lines.append(f"**投资逻辑**：{s.investment_thesis}\n")

            lines.append(
                f"**历史验证**（回溯{r.n_signals}次信号）：\n"
                f"- 信号次数：{r.n_signals}次，胜率：{r.win_rate:.0%}，"
                f"平均月收益：{r.avg_return:+.1%}，最大单次亏损：{r.max_loss:.1%}\n"
            )
            lines.append("**建议操作**：")
            lines.append(f"- 入场价格区间：¥{s.entry_price_range[0]:.2f} ~ ¥{s.entry_price_range[1]:.2f}")
            lines.append(f"- 1个月目标价：¥{s.target_price_1m:.2f}")
            lines.append(f"- 止损位：¥{s.stop_loss_price:.2f}")
            lines.append(f"- 建议仓位：{s.position_size}")
            if s.key_risks:
                lines.append(f"- 主要风险：{'；'.join(s.key_risks[:2])}")
            lines.append("")

    else:
        lines.append("*本次无满足条件的投资机会*\n")

    # ── 👀 观察名单 ─────────────────────────────────────────────────────────
    lines.append(f"## 👀 观察名单（{len(watchlist)}只）\n")
    if watchlist:
        lines.append("| 股票 | 评级 | 综合信号 | 历史胜率 | 期望月收益 | 建议 |")
        lines.append("|---|---|---|---|---|---|")
        for r in watchlist:
            s = strategy_map.get(r.ticker)
            signal_str = f"{s.composite_signal:+.2f}" if s else "-"
            lines.append(
                f"| {r.ticker} | {r.rating} | {signal_str} "
                f"| {r.win_rate:.0%} | {r.avg_return:+.1%} | {r.action} |"
            )
        lines.append("")
    else:
        lines.append("*无观察名单股票*\n")

    # ── ❌ 暂时回避 ─────────────────────────────────────────────────────────
    lines.append(f"## ❌ 暂时回避（{len(avoid)}只）\n")
    if avoid:
        lines.append("| 股票 | 评级 | 综合信号 | 原因 |")
        lines.append("|---|---|---|---|")
        for r in avoid:
            s = strategy_map.get(r.ticker)
            signal_str = f"{s.composite_signal:+.2f}" if s else "-"
            reason = r.final_recommendation[:60] + "..." if len(r.final_recommendation) > 60 else r.final_recommendation
            lines.append(f"| {r.ticker} | {r.rating} | {signal_str} | {reason} |")
        lines.append("")
    else:
        lines.append("*无回避名单股票*\n")

    # ── 📊 评估摘要 ─────────────────────────────────────────────────────────
    total = len(acceptable) + len(watchlist) + len(avoid)
    n_accept = len(acceptable)
    pct = f"{n_accept / total * 100:.0f}%" if total > 0 else "0%"
    lines.append("## 📊 本次评估摘要\n")
    lines.append(f"- 候选总数：{total}只")
    lines.append(f"- 可接受：{n_accept}只（{pct}）")
    lines.append(f"- 观察名单：{len(watchlist)}只")
    lines.append(f"- 回避：{len(avoid)}只")
    lines.append(f"- 数据截止：{date_str}")
    lines.append(f"- 分析框架：ValuationAgent + MomentumAgent + QualityAgent + SentimentAgent + RiskAgent + StrategyAgent")
    lines.append("")
    lines.append("> **免责声明**：本报告由量化模型自动生成，仅供参考，不构成投资建议。")
    lines.append("> 投资有风险，决策需谨慎。过往表现不代表未来收益。")

    md_text = "\n".join(lines)
    report_path = output_dir / "final_recommendations.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md_text, encoding="utf-8")
    return report_path


# ── 主 Pipeline ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantMind 投资分析完整 Pipeline")
    p.add_argument("--date", required=True, help="分析日期 YYYY-MM-DD")
    p.add_argument("--provider", default="none", help="LLM provider (none/dashscope/openai)")
    p.add_argument("--model", default="qwen-plus")
    p.add_argument(
        "--candidates-json",
        type=Path,
        help="候选股票 JSON 路径（top10.json）",
    )
    p.add_argument(
        "--tickers-from-file",
        type=Path,
        default=None,
        help="与 candidates-json 相同，显式指定漏斗输出 JSON（含 candidates 列表）",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "reports/investment_pipeline",
    )
    p.add_argument("--top-n", type=int, default=10, help="处理前 N 只候选股票")
    p.add_argument("--chroma-dir", default=".cache/chromadb")
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="跳过历史回测验证（快速模式）",
    )
    # 新增参数
    p.add_argument(
        "--universe",
        default="alpha",
        choices=["full_a", "csi300", "csi1000", "alpha"],
        help="股票池（默认 alpha，即 1374 只 alpha 宇宙）",
    )
    p.add_argument(
        "--skip-funnel-layers",
        type=int,
        nargs="*",
        default=[],
        help="漏斗选股跳过的层编号（仅 universe≠csi300 时有效）",
    )
    p.add_argument(
        "--agent-model-versions",
        default=None,
        help='各 Agent 使用的模型版本，逗号分隔，例如："MomentumAgent=lgbm_v2,SentimentAgent=tfidf_v2"',
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用磁盘缓存，强制重新拉取所有数据",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    date_str = args.date
    output_dir = args.output_dir / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[pipeline] ===== QuantMind 投资分析 Pipeline =====")
    logger.info(f"[pipeline] 日期: {date_str} | Provider: {args.provider} | TopN: {args.top_n}")

    # ── 初始化共享缓存和懒加载引擎 ───────────────────────────────────────────
    from quantmind.data.shared_cache import SharedDataCache
    from quantmind.selection.lazy_data_engine import LazyDataEngine

    if args.no_cache:
        SharedDataCache.reset_instance()

    shared_cache = SharedDataCache.get_instance()
    data_engine = LazyDataEngine(date_str, shared_cache=shared_cache, universe=args.universe)
    logger.info(f"[pipeline] 共享缓存已初始化（no_cache={args.no_cache}）")

    # ── 解析 Agent 模型版本覆盖 ───────────────────────────────────────────────
    agent_version_overrides: dict[str, str] = {}
    if args.agent_model_versions:
        for part in args.agent_model_versions.split(","):
            part = part.strip()
            if "=" in part:
                agent, ver = part.split("=", 1)
                agent_version_overrides[agent.strip()] = ver.strip()
    if agent_version_overrides:
        logger.info(f"[pipeline] Agent 版本覆盖: {agent_version_overrides}")

    # ── Step 0: 漏斗选股（若 universe != csi300）──────────────────────────────
    funnel_tickers: list[str] | None = None
    # alpha 宇宙：top10.json 已由 daily_update 生成，跳过漏斗选股；其他大池（full_a/csi1000）才需要漏斗
    _skip_funnel = args.universe in ("csi300", "alpha")
    if not _skip_funnel:
        logger.info(f"[pipeline] 使用漏斗选股 universe={args.universe}")
        from quantmind.selection.funnel_selector import FunnelSelector
        selector = FunnelSelector(
            as_of=date_str,
            data_engine=data_engine,
            universe=args.universe,
            provider=args.provider,
            model_name=args.model,
        )
        funnel_result = selector.run(
            skip_layers=args.skip_funnel_layers,
            top_n=args.top_n * 2,
        )
        funnel_tickers = funnel_result.candidates["ticker"].tolist()
        # 保存漏斗结果
        funnel_out = output_dir / "funnel_candidates.json"
        funnel_out.write_text(
            json.dumps(funnel_result.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[pipeline] 漏斗选股完成 → {funnel_out}")

    # ── Step 1: 读取候选股票 ──────────────────────────────────────────────────
    candidates_json = args.tickers_from_file or args.candidates_json
    if candidates_json is None:
        candidates_json = _ROOT / f"data/recommendations/{date_str}/top10.json"

    if not candidates_json.exists():
        logger.error(f"[pipeline] 候选文件不存在: {candidates_json}")
        logger.info("[pipeline] 请先运行 daily_update.py 生成候选股票")
        sys.exit(1)

    candidates = load_candidates(candidates_json)
    tickers = [c["ticker"] for c in candidates[: args.top_n]]
    # 若漏斗选股有结果，合并（漏斗结果优先）
    if funnel_tickers:
        tickers = funnel_tickers[: args.top_n]
    logger.info(f"[pipeline] 候选股票: {tickers}")

    # 读取价格面板（alpha 长表 → pivot 宽表）
    price_df = _load_price_df()

    # 读取因子面板（alpha_panel_v3 优先）
    panel_df = None
    panel_parts = []
    for pf in _PANEL_FILES:
        if pf.exists():
            panel_parts.append(pd.read_parquet(pf))
            logger.info(f"[pipeline] 加载因子面板: {pf.name} → shape={panel_parts[-1].shape}")
            break  # 只取优先级最高的一个（alpha_panel_v3 已包含全部数据）
    if panel_parts:
        panel_df = panel_parts[0]
        logger.info(f"[pipeline] 因子面板: {panel_df.shape}")

    # ── Step 2: 对每只股票运行 6 个 Agent ─────────────────────────────────────
    all_strategies: list[InvestmentStrategy] = []
    ticker_dir = output_dir / "strategies"

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"\n[pipeline] [{i}/{len(tickers)}] 分析 {ticker} ...")
        t0 = time.monotonic()

        # 优先从 LazyDataEngine 获取 KB 上下文（复用已缓存数据）
        kb_contexts = data_engine.get_kb_context([ticker])
        context = kb_contexts.get(ticker) or retrieve_context(ticker, date_str, chroma_dir=args.chroma_dir)

        # 运行 5 个基础 Agent（支持 Agent 版本覆盖）
        signals = run_six_agents(
            ticker, date_str, context, args.provider, args.model,
            agent_version_overrides=agent_version_overrides,
        )

        # 运行 StrategyAgent
        strategy_agent = StrategyAgent(
            ticker=ticker,
            as_of=date_str,
            context=context,
            agent_signals=signals,
            provider=args.provider,
            model=args.model,
        )
        strategy = strategy_agent.analyze_with_llm()
        all_strategies.append(strategy)

        # 实时保存（防止中途失败）
        save_strategy(strategy, ticker_dir)

        elapsed = time.monotonic() - t0
        logger.info(
            f"[pipeline] {ticker} 完成: 评级={strategy.rating} 信号={strategy.composite_signal:+.2f} "
            f"耗时={elapsed:.1f}s"
        )

    # 保存汇总策略 JSON
    strategies_data = [_strategy_to_dict(s) for s in all_strategies]
    strategies_path = output_dir / "strategies.json"
    strategies_path.write_text(
        json.dumps(strategies_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"[pipeline] 策略汇总 → {strategies_path}")

    # ── Step 3: 历史回测验证 ──────────────────────────────────────────────────
    if args.skip_validation:
        results = [
            StrategyValidationResult(
                ticker=s.ticker,
                rating=s.rating,
                validation_status="WATCHLIST",
                n_signals=0,
                win_rate=0.0,
                avg_return=0.0,
                max_loss=0.0,
                hit_stop=0.0,
                expected_value=0.0,
                final_recommendation="跳过历史验证",
                action="等待机会",
                position_size=s.position_size,
            )
            for s in all_strategies
        ]
    else:
        logger.info(f"\n[pipeline] === Step 3: 历史回测验证 ===")
        results = batch_validate(strategies_data, price_df, panel_df)

        # 保存验证结果
        from dataclasses import asdict
        validations_path = output_dir / "validations.json"
        validations_path.write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[pipeline] 验证结果 → {validations_path}")

    # ── Step 4: 生成最终报告 ──────────────────────────────────────────────────
    logger.info(f"\n[pipeline] === Step 4: 生成最终报告 ===")
    acceptable = [r for r in results if r.validation_status == "ACCEPTABLE"]
    watchlist = [r for r in results if r.validation_status == "WATCHLIST"]
    avoid = [r for r in results if r.validation_status == "AVOID"]

    # 限制可接受数量 ≤ 8
    if len(acceptable) > 8:
        logger.warning(f"[pipeline] 可接受股票{len(acceptable)}只 > 8，截取前8只")
        overflow = acceptable[8:]
        acceptable = acceptable[:8]
        watchlist = overflow + watchlist

    report_path = generate_final_report(
        acceptable, watchlist, avoid, all_strategies, output_dir, date_str
    )
    logger.info(f"[pipeline] 最终报告 → {report_path}")

    # 打印摘要
    cache_info = shared_cache.stats()
    print(f"\n{'='*60}")
    print(f"QuantMind 投资分析 Pipeline 完成 — {date_str}")
    print(f"{'='*60}")
    print(f"  ✅ 可接受：{len(acceptable)}只")
    print(f"  👀 观察：{len(watchlist)}只")
    print(f"  ❌ 回避：{len(avoid)}只")
    print(f"  📄 报告：{report_path}")
    print(f"  缓存命中率: {cache_info['hit_rate']:.1%} (hits={cache_info['hits']}, misses={cache_info['misses']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
