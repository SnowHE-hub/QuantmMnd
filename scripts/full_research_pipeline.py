"""scripts/full_research_pipeline.py — 一键研究 + 回测流水线.

功能
====
PIT 快照加载 → LightGBM 评分 → 并行 Agent 研究 → LLM 重排 →
回测（默认过去 2 年，月度调仓，等权组合）→ DSR 统计检验 →
输出整合 HTML 报告 ``reports/research_{tickers}_{date}.html``

用法
====
::

    # 使用默认 CSI300 宇宙，过去 2 年回测
    python scripts/full_research_pipeline.py

    # 指定标的 + 日期
    python scripts/full_research_pipeline.py --tickers 600519.SH,300750.SZ --as-of 2024-06-28

    # 只做研究，不运行回测
    python scripts/full_research_pipeline.py --tickers 600519.SH --no-backtest

    # 不运行 Agent（仅 LGBM + LLM 重排）
    python scripts/full_research_pipeline.py --no-agent --max-tickers 10

    # 完整模式（含回测、DSR、Agent）
    python scripts/full_research_pipeline.py \\
        --tickers 600519.SH,300750.SZ,601318.SH \\
        --as-of 2024-06-28 \\
        --backtest-start 2022-06-28 \\
        --backtest-end 2024-06-28 \\
        --provider ollama --model qwen2.5:7b \\
        --max-workers 3

CLI 参数
========
--tickers          逗号分隔股票代码；省略时从 LGBM 粗排结果取 Top-N
--as-of            截面日期 YYYY-MM-DD（默认：最近交易日）
--universe         股票池（csi300 / csi500）
--max-tickers      Agent 研究最多处理多少只股票（默认 5）
--backtest-start   回测开始日期（默认 as-of 前 2 年）
--backtest-end     回测结束日期（默认 as-of）
--rebalance-freq   调仓频率 M/Q（默认 M）
--no-agent         跳过 Agent 研究步骤
--no-backtest      跳过回测步骤
--no-llm           跳过 LLM 重排（仅 LGBM 排名）
--lgbm-top         LGBM 粗排保留只数（默认 50）
--llm-top          LLM 重排输出只数（默认 10）
--max-workers      并行 Agent 线程数（默认 3）
--provider         LLM 提供商 ollama/deepseek/openai（默认 ollama）
--model            LLM 模型名（默认 qwen2.5:7b）
--output-dir       报告输出目录（默认 reports）
--force            强制重建因子（即使缓存存在）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantMind 一键研究 + 回测流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", default=None,
        help="逗号分隔股票代码；省略时从 LGBM Top-N 自动选择",
    )
    parser.add_argument("--as-of", default=None, help="截面日期 YYYY-MM-DD")
    parser.add_argument("--universe", default="csi300", help="股票池（默认 csi300）")
    parser.add_argument(
        "--max-tickers", type=int, default=5,
        help="Agent 研究最多处理只数（默认 5）",
    )
    parser.add_argument(
        "--backtest-start", default=None, help="回测开始日期（默认 as-of 前 2 年）"
    )
    parser.add_argument(
        "--backtest-end", default=None, help="回测结束日期（默认 as-of）"
    )
    parser.add_argument(
        "--rebalance-freq", default="M", choices=["M", "Q"],
        help="调仓频率（默认 M=月度）",
    )
    parser.add_argument("--no-agent", action="store_true", help="跳过 Agent 研究")
    parser.add_argument("--no-backtest", action="store_true", help="跳过回测")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 重排")
    parser.add_argument(
        "--lgbm-top", type=int, default=50, help="LGBM 粗排保留只数（默认 50）"
    )
    parser.add_argument(
        "--llm-top", type=int, default=10, help="LLM 重排输出只数（默认 10）"
    )
    parser.add_argument(
        "--max-workers", type=int, default=3, help="并行 Agent 线程数（默认 3）"
    )
    parser.add_argument("--provider", default="ollama", help="LLM 提供商")
    parser.add_argument("--model", default="qwen2.5:7b", help="LLM 模型名")
    parser.add_argument("--output-dir", default="reports", help="报告输出目录")
    parser.add_argument("--force", action="store_true", help="强制重建因子缓存")
    parser.add_argument(
        "--n-sim-dsr", type=int, default=100,
        help="DSR 模拟策略数（默认 100）",
    )
    return parser.parse_args()


# ============================================================================
# 步骤函数
# ============================================================================


def step_resolve_date(args: argparse.Namespace) -> date:
    """解析目标截面日期（最近交易日）."""
    if args.as_of:
        return datetime.strptime(args.as_of, "%Y-%m-%d").date()

    today = date.today()
    try:
        from quantmind.data.calendar import list_sse_trade_dates
        dates = list_sse_trade_dates(
            start=str(today - timedelta(days=30)),
            end=str(today),
        )
        if dates:
            return datetime.strptime(str(dates[-1]), "%Y%m%d").date()
    except Exception:
        pass

    # 回退：找最近工作日
    d = today
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def step_build_features(as_of: date, args: argparse.Namespace) -> tuple[Path, Any]:
    """加载或构建因子截面；返回 (parquet_path, feature_df)."""
    from quantmind.features.pipeline import FeaturePipeline

    settings_data_dir = None
    try:
        from quantmind.core.config import get_settings
        settings_data_dir = Path(get_settings().data.dir)
    except Exception:
        settings_data_dir = _ROOT / "data"

    feat_path = settings_data_dir / "features" / f"{args.universe}_{as_of.isoformat()}.parquet"

    if feat_path.exists() and not args.force:
        logger.info(f"[FRP] 因子文件已存在，直接加载: {feat_path}")
        import pandas as pd
        return feat_path, pd.read_parquet(feat_path)

    logger.info(f"[FRP] 构建因子截面 as_of={as_of}")
    pipe = FeaturePipeline()
    feat_df = pipe.run_single(as_of, universe=args.universe)
    saved_path = pipe.save(feat_df, as_of, universe=args.universe)
    return saved_path, feat_df


def step_lgbm_rank(
    feat_df: Any, as_of: date, args: argparse.Namespace
) -> list[dict]:
    """LightGBM 粗排；返回 Top-N 候选列表（含因子暴露）."""
    import numpy as np

    model_path = _ROOT / "models" / "lgbm_ranker.pkl"
    if not model_path.exists():
        logger.warning(f"[FRP] LGBM 模型文件不存在: {model_path}，退化为等权候选")
        tickers = feat_df.index.tolist()
        return [
            {"lgbm_rank": i + 1, "ticker": t, "lgbm_score": 0.0,
             "key_factors": {}, "shap_values": {}}
            for i, t in enumerate(tickers[: args.lgbm_top])
        ]

    try:
        from quantmind.models.lgbm_ranker import LGBMRankerModel
        model = LGBMRankerModel.load(model_path)
        feat_names = model._feature_names
        import pandas as pd
        X = feat_df.reindex(columns=feat_names).fillna(0.0)
        scores = model.predict(X)

        top_n = min(args.lgbm_top, len(scores))
        idx = np.argsort(scores)[::-1][:top_n]
        tickers = feat_df.index.tolist()

        # SHAP（取 Top-10）
        shap_map: dict[str, dict] = {}
        try:
            top10_idx = idx[:10]
            top10_tickers = [tickers[i] for i in top10_idx]
            X_top = X.iloc[top10_idx]
            shap_vals = model.explain(X_top, tickers=top10_tickers)
            shap_map = shap_vals if isinstance(shap_vals, dict) else {}
        except Exception:
            pass

        candidates = []
        for rank, i in enumerate(idx, start=1):
            t = tickers[i]
            row = feat_df.iloc[i]
            candidates.append({
                "lgbm_rank": rank,
                "ticker": t,
                "lgbm_score": float(scores[i]),
                "key_factors": {
                    k: float(row[k])
                    for k in [
                        "pb", "pe_ttm", "roe_ttm", "momentum_6m",
                        "volatility_3m", "accruals",
                    ]
                    if k in row.index and not np.isnan(row[k])
                },
                "shap_values": shap_map.get(t, {}),
            })
        logger.info(f"[FRP] LGBM 粗排完成，Top-{top_n} 候选")
        return candidates

    except Exception as exc:
        logger.warning(f"[FRP] LGBM 粗排失败（{exc}），退化为等权候选")
        tickers_all = feat_df.index.tolist()
        return [
            {"lgbm_rank": i + 1, "ticker": t, "lgbm_score": 0.0,
             "key_factors": {}, "shap_values": {}}
            for i, t in enumerate(tickers_all[: args.lgbm_top])
        ]


def step_llm_rerank(
    candidates: list[dict],
    feat_df: Any,
    as_of: date,
    args: argparse.Namespace,
) -> list[dict]:
    """LLM 精排候选列表；--no-llm 时直接截取 Top-N."""
    if args.no_llm:
        logger.info("[FRP] --no-llm，使用 LGBM 排名前 N")
        result = []
        for item in candidates[: args.llm_top]:
            result.append({**item, "llm_rank": item["lgbm_rank"],
                           "reason": "（LGBM 排名）", "is_fallback": True})
        return result

    try:
        import numpy as np
        from quantmind.models.llm_reranker import LLMListwiseReranker, RerankCandidate

        reranker = LLMListwiseReranker(provider=args.provider, model=args.model)

        rc_list: list[RerankCandidate] = []
        for item in candidates:
            t = item["ticker"]
            row = feat_df.loc[t] if t in feat_df.index else None
            def _f(k):  # noqa: E731
                if row is None or k not in row.index:
                    return float("nan")
                v = row[k]
                return float(v) if not np.isnan(v) else float("nan")

            rc_list.append(
                RerankCandidate(
                    ticker=t,
                    lgbm_score=item["lgbm_score"],
                    lgbm_rank=item["lgbm_rank"],
                    pe_ttm=_f("pe_ttm"),
                    pb=_f("pb"),
                    roe_ttm=_f("roe_ttm"),
                    accruals=_f("accruals"),
                    distance_to_52w_high=_f("distance_to_52w_high"),
                    momentum_6m=_f("momentum_6m"),
                    volatility_3m=_f("volatility_3m"),
                    shap_values=item.get("shap_values", {}),
                )
            )

        rerank_results = reranker.rerank(
            rc_list,
            top_n=args.llm_top,
            as_of_date=as_of.isoformat(),
        )

        out = []
        cand_map = {c["ticker"]: c for c in candidates}
        for rr in rerank_results:
            base = cand_map.get(rr.ticker, {})
            out.append({
                **base,
                "llm_rank": rr.rank,
                "reason": rr.reason,
                "portfolio_thesis": getattr(rr, "portfolio_thesis", ""),
                "risk_warnings": getattr(rr, "risk_warnings", []),
                "is_fallback": rr.is_fallback,
            })
        logger.info(f"[FRP] LLM 重排完成，最终 Top-{len(out)}")
        return out

    except Exception as exc:
        logger.warning(f"[FRP] LLM 重排失败（{exc}），退化为 LGBM 排名前 N")
        return [
            {**item, "llm_rank": item["lgbm_rank"],
             "reason": f"LLM rerank failed: {exc}", "is_fallback": True}
            for item in candidates[: args.llm_top]
        ]


def _run_single_agent(
    ticker: str,
    as_of: date,
    args: argparse.Namespace,
) -> tuple[str, Any, str | None]:
    """单股 Agent 研究（在线程池中执行）."""
    try:
        from quantmind.agents.orchestrator import run_research
        state = run_research(
            ticker=ticker,
            as_of=as_of,
            provider=args.provider,
            model=args.model,
            max_iterations=2,  # 全管线中缩短迭代以节省时间
        )
        return ticker, state, None
    except Exception as exc:
        return ticker, None, str(exc)


def step_agent_research(
    tickers: list[str],
    as_of: date,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """并行 Agent 研究；返回 {ticker: AgentState | None}."""
    if args.no_agent:
        logger.info("[FRP] --no-agent，跳过 Agent 研究")
        return {}

    research_tickers = tickers[: args.max_tickers]
    logger.info(
        f"[FRP] 启动并行 Agent 研究，"
        f"tickers={research_tickers}, max_workers={args.max_workers}"
    )

    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_run_single_agent, t, as_of, args): t
            for t in research_tickers
        }
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                t, state, err = fut.result()
                if err:
                    logger.warning(f"[FRP] Agent 研究失败 {t}: {err}")
                    results[t] = None
                else:
                    logger.info(f"[FRP] Agent 研究完成: {t}")
                    results[t] = state
            except Exception as exc:
                logger.warning(f"[FRP] Agent 线程异常 {ticker}: {exc}")
                results[ticker] = None

    return results


def step_backtest(
    as_of: date,
    feat_path: Path,
    args: argparse.Namespace,
) -> tuple[Any, dict]:
    """运行等权组合回测（默认 2 年，月度调仓）; 返回 (BacktestResult, stat_dict)."""
    if args.no_backtest:
        logger.info("[FRP] --no-backtest，跳过回测")
        return None, {}

    import numpy as np

    bt_end = (
        datetime.strptime(args.backtest_end, "%Y-%m-%d").date()
        if args.backtest_end
        else as_of
    )
    bt_start = (
        datetime.strptime(args.backtest_start, "%Y-%m-%d").date()
        if args.backtest_start
        else date(bt_end.year - 2, bt_end.month, bt_end.day)
    )

    logger.info(f"[FRP] 回测区间 {bt_start}~{bt_end}，调仓={args.rebalance_freq}")

    # 尝试加载价格数据
    prices_df = None
    try:
        from quantmind.core.config import get_settings
        data_dir = Path(get_settings().data.dir)
        # 尝试从 features parquet 的 sibling 目录找 prices panel
        prices_path = data_dir / "prices" / "panel.parquet"
        if prices_path.exists():
            import pandas as pd
            prices_df = pd.read_parquet(prices_path)
            logger.info(f"[FRP] 加载价格面板: {prices_path}")
    except Exception:
        pass

    if prices_df is None:
        logger.warning("[FRP] 未找到价格面板，使用 demo 模拟数据")
        prices_df = _make_demo_prices(bt_start, bt_end)

    try:
        from quantmind.backtest import (
            BacktestConfig,
            BacktestEngine,
            EqualWeightStrategy,
            bootstrap_ci,
            deflated_sharpe_ratio,
            probabilistic_sharpe_ratio,
        )

        config = BacktestConfig()
        strategy = EqualWeightStrategy(rebalance_freq=args.rebalance_freq, config=config)

        # 提取 universe（从 feat_path）
        universe: list[str]
        try:
            import pandas as pd
            feat_df = pd.read_parquet(feat_path)
            universe = feat_df.index.tolist()
        except Exception:
            universe = list(
                prices_df.index.get_level_values("ticker").unique()
                if hasattr(prices_df.index, "names")
                else prices_df.columns
            )[:300]

        engine = BacktestEngine(config=config, prices_df=prices_df)
        result = engine.run(strategy, bt_start, bt_end, universe)
        logger.info(f"[FRP] 回测完成，Sharpe={result.metrics.get('sharpe_ratio', float('nan')):.3f}")

        # 统计显著性检验
        stat_results: dict = {}
        if hasattr(result, "returns") and result.returns is not None and not result.returns.empty:
            ret = result.returns.dropna()

            stat_results["PSR"] = probabilistic_sharpe_ratio(ret)

            candidate_sr = float(result.metrics.get("sharpe_ratio", 0))
            rng = np.random.default_rng(42)
            sim_sharpes = []
            for _ in range(args.n_sim_dsr):
                sim_ret = rng.normal(0.0003, 0.015, len(ret))
                sr = float(
                    np.mean(sim_ret) / (np.std(sim_ret) + 1e-10) * np.sqrt(252)
                )
                sim_sharpes.append(sr)
            sim_sharpes.append(candidate_sr)
            stat_results["DSR"] = deflated_sharpe_ratio(
                sim_sharpes, candidate_sr, n_obs=len(ret)
            )

            def _sharpe_fn(r: np.ndarray) -> float:  # noqa: E731
                rf = 0.025 / 252
                excess = r - rf
                return float(np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(252))

            stat_results["Bootstrap Sharpe CI"] = bootstrap_ci(
                _sharpe_fn, ret.values, n_bootstrap=500
            )

        return result, stat_results

    except Exception as exc:
        logger.error(f"[FRP] 回测失败: {exc}\n{traceback.format_exc()}")
        return None, {}


def _make_demo_prices(start: date, end: date) -> "pd.DataFrame":
    """生成用于 demo 的模拟价格面板."""
    import numpy as np
    import pandas as pd

    tickers = [
        "600519.SH", "000858.SZ", "600036.SH", "000333.SZ",
        "601318.SH", "600276.SH", "000001.SZ", "600000.SH",
        "002415.SZ", "300750.SZ",
    ]
    dates = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(42)
    records = []
    for t in tickers:
        p = rng.uniform(10.0, 200.0)
        for dt in dates:
            p = max(p * (1 + rng.normal(5e-4, 0.02)), 0.1)
            records.append({
                "date": dt,
                "ticker": t,
                "open": p * (1 + rng.normal(0, 0.005)),
                "high": p * (1 + abs(rng.normal(0, 0.01))),
                "low": p * (1 - abs(rng.normal(0, 0.01))),
                "close": p,
                "pre_close": p * (1 - rng.normal(5e-4, 0.02)),
                "volume": int(rng.integers(100_000, 10_000_000)),
            })
    df = pd.DataFrame(records).set_index(["date", "ticker"])
    return df


# ============================================================================
# HTML 报告生成
# ============================================================================

_CSS = """
body{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f6fa;margin:0;padding:24px}
h1,h2,h3{color:#1a1a2e}
.card{background:#fff;border-radius:10px;padding:20px;margin:16px 0;box-shadow:0 2px 10px rgba(0,0,0,.08)}
.metric{display:inline-block;padding:14px 20px;border-radius:8px;margin:6px;text-align:center;min-width:120px;color:#fff}
table{width:100%;border-collapse:collapse}
th{background:#1a1a2e;color:#fff;padding:9px;text-align:left;font-weight:500}
td{padding:8px;border-bottom:1px solid #eee}
tr:hover td{background:#f8f9ff}
pre{background:#f8f9fa;padding:12px;border-radius:6px;overflow-x:auto;font-size:.85em}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.8em;font-weight:bold}
.nav-section{border-left:4px solid #6c5ce7;padding-left:12px;margin:8px 0}
details summary{cursor:pointer;font-weight:600;color:#1a1a2e;padding:8px 0}
"""


def _fmt_pct(v) -> str:
    if v is None:
        return "N/A"
    try:
        f = float(v)
        return f"{f:.1%}" if not (f != f) else "N/A"
    except Exception:
        return "N/A"


def _fmt_n(v, dec: int = 3) -> str:
    if v is None:
        return "N/A"
    try:
        f = float(v)
        return f"{f:.{dec}f}" if not (f != f) else "N/A"
    except Exception:
        return "N/A"


def _rating_color(rating: str) -> str:
    return {
        "strong_buy": "#00b894", "buy": "#00cec9",
        "hold": "#fdcb6e", "sell": "#e17055", "strong_sell": "#d63031",
    }.get(rating.lower(), "#636e72")


def _build_section_overview(top_list: list[dict], feat_df: Any) -> str:
    """Section 1: 市场概览 — LGBM Top 推荐 + 因子暴露."""
    import numpy as np

    rows = ""
    for item in top_list:
        t = item["ticker"]
        kf = item.get("key_factors", {})
        llm_r = item.get("llm_rank", item.get("lgbm_rank", "—"))
        fb = "⚠️" if item.get("is_fallback") else ""
        rows += f"""
<tr>
  <td><b>{llm_r}</b></td>
  <td><b>{t}</b></td>
  <td>{_fmt_n(item.get('lgbm_score'), 4)}</td>
  <td>{_fmt_n(kf.get('pe_ttm'))}</td>
  <td>{_fmt_n(kf.get('pb'))}</td>
  <td>{_fmt_pct(kf.get('roe_ttm'))}</td>
  <td>{_fmt_pct(kf.get('momentum_6m'))}</td>
  <td>{_fmt_pct(kf.get('volatility_3m'))}</td>
  <td>{item.get('reason', '')[:60]}{fb}</td>
</tr>"""

    # SHAP 摘要（Top-5）
    shap_html = ""
    for item in top_list[:5]:
        shap = item.get("shap_values") or {}
        if shap:
            sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            shap_rows = "".join(
                f"<tr><td>{k}</td><td style='color:{'#e17055' if v<0 else '#00b894'}'>{v:+.3f}</td></tr>"
                for k, v in sorted_shap
            )
            shap_html += f"""
<details style="margin:8px 0">
  <summary>{item['ticker']} SHAP Top-5</summary>
  <table style="max-width:400px"><tr><th>因子</th><th>SHAP值</th></tr>{shap_rows}</table>
</details>"""

    return f"""
<div class="card">
  <h2>📊 Section 1 · 市场概览与推荐排行</h2>
  <table>
    <tr>
      <th>排名</th><th>代码</th><th>LGBM分</th>
      <th>PE(TTM)</th><th>PB</th><th>ROE</th>
      <th>动量6M</th><th>波动3M</th><th>推荐理由</th>
    </tr>
    {rows}
  </table>
  {f'<h3 style="margin-top:20px">SHAP 因子贡献</h3>{shap_html}' if shap_html else ''}
</div>"""


def _build_section_research(agent_results: dict, top_list: list[dict]) -> str:
    """Section 2: 个股研究报告（Agent 输出）."""
    if not agent_results:
        return """<div class="card"><h2>🔬 Section 2 · 个股研究</h2><p style="color:#636e72">未运行 Agent 研究（--no-agent）</p></div>"""

    cards = ""
    for ticker, state in agent_results.items():
        if state is None:
            cards += f'<details><summary>{ticker} ❌ Agent 研究失败</summary><p>运行时出错，请检查日志。</p></details>'
            continue

        report = getattr(state, "report", None)
        feedback = getattr(state, "critic_feedback", None)
        fa = getattr(state, "fundamentals", {}).get(ticker)
        ta = getattr(state, "technicals", {}).get(ticker)

        rating_str = report.rating.value if report else "hold"
        r_color = _rating_color(rating_str)
        q_score = feedback.overall_quality_score if feedback else 0
        q_color = "#00b894" if q_score >= 7 else ("#fdcb6e" if q_score >= 5 else "#d63031")
        passed = feedback.passed if feedback else False

        summary_html = ""
        if report and report.executive_summary:
            summary_html = "<ul>" + "".join(f"<li>{s}</li>" for s in report.executive_summary[:3]) + "</ul>"

        risks_html = ""
        if report and report.risk_warnings:
            risks_html = "<ul>" + "".join(f"<li>⚠️ {r}</li>" for r in report.risk_warnings[:3]) + "</ul>"

        fa_html = ""
        if fa:
            fa_html = f"""
<div class="nav-section">
  <b>盈利分析</b><p>{getattr(fa,'profitability_analysis','N/A')}</p>
  <b>估值分析</b><p>{getattr(fa,'valuation_analysis','N/A')}</p>
</div>"""

        ta_html = ""
        if ta:
            sig = getattr(ta, "signal", "neutral")
            sig_color = "#00b894" if sig == "bullish" else ("#d63031" if sig == "bearish" else "#636e72")
            ta_html = f"""
<div class="nav-section">
  <b>信号：</b><span style="color:{sig_color};font-weight:bold">{sig.upper()}</span>
  <p>{getattr(ta,'summary','N/A')}</p>
</div>"""

        cards += f"""
<details>
  <summary>
    {ticker}
    <span class="badge" style="background:{r_color};color:#fff;margin-left:8px">{rating_str.upper()}</span>
    <span class="badge" style="background:{q_color};color:#fff;margin-left:4px">质量 {q_score:.1f}</span>
    {'<span class="badge" style="background:#00b894;color:#fff;margin-left:4px">✅ Critic通过</span>' if passed else ''}
  </summary>
  <div style="padding:16px">
    {summary_html}
    {fa_html}
    {ta_html}
    {f'<div class="nav-section"><b>风险提示</b>{risks_html}</div>' if risks_html else ''}
    <p style="font-size:.85em;color:#b2bec3">迭代次数: {getattr(state,'iteration_count','-')}</p>
  </div>
</details>"""

    return f"""<div class="card"><h2>🔬 Section 2 · 个股深度研究</h2>{cards}</div>"""


def _build_section_backtest(result: Any, stat_results: dict) -> str:
    """Section 3: 回测结果（净值曲线 + 性能指标）."""
    if result is None:
        return """<div class="card"><h2>📈 Section 3 · 回测结果</h2><p style="color:#636e72">未运行回测（--no-backtest）</p></div>"""

    import json as _json

    m = getattr(result, "metrics", {}) or {}
    nav = getattr(result, "nav_series", None)

    nav_json = "[]"
    if nav is not None and not nav.empty:
        base = float(nav.iloc[0]) if len(nav) > 0 else 1.0
        nav_json = _json.dumps([
            {"x": str(idx.date() if hasattr(idx, "date") else idx),
             "y": round(float(v) / base, 6)}
            for idx, v in nav.items()
        ])

    # 统计检验行
    stat_rows = ""
    for test_name, tr in stat_results.items():
        if not tr:
            continue
        sig = "✅ 显著" if tr.get("is_significant") else "❌ 不显著"
        pv = tr.get("p_value", float("nan"))
        interp = tr.get("interpretation", "")[:100]
        stat_rows += f"<tr><td>{test_name}</td><td>{_fmt_n(pv)}</td><td>{sig}</td><td style='font-size:.85em;color:#636e72'>{interp}</td></tr>"

    return f"""
<div class="card">
  <h2>📈 Section 3 · 回测结果</h2>
  <p style="color:#636e72">{getattr(result,'strategy_name','—')} | {getattr(result,'start_date','?')} ~ {getattr(result,'end_date','?')}</p>

  <div>
    <span class="metric" style="background:#00b894">
      <div style="font-size:.75em">总收益</div>
      <div style="font-size:1.4em;font-weight:bold">{_fmt_pct(m.get('total_return'))}</div>
    </span>
    <span class="metric" style="background:#6c5ce7">
      <div style="font-size:.75em">Sharpe</div>
      <div style="font-size:1.4em;font-weight:bold">{_fmt_n(m.get('sharpe_ratio'))}</div>
    </span>
    <span class="metric" style="background:#e17055">
      <div style="font-size:.75em">最大回撤</div>
      <div style="font-size:1.4em;font-weight:bold">{_fmt_pct(m.get('max_drawdown'))}</div>
    </span>
    <span class="metric" style="background:#0984e3">
      <div style="font-size:.75em">年化收益</div>
      <div style="font-size:1.4em;font-weight:bold">{_fmt_pct(m.get('cagr'))}</div>
    </span>
    <span class="metric" style="background:#fd79a8">
      <div style="font-size:.75em">Sortino</div>
      <div style="font-size:1.4em;font-weight:bold">{_fmt_n(m.get('sortino_ratio'))}</div>
    </span>
  </div>

  <canvas id="navChart" height="80" style="margin-top:16px"></canvas>

  <h3 style="margin-top:20px">性能指标明细</h3>
  <table>
    <tr><th>类别</th><th>指标</th><th>值</th></tr>
    <tr><td>收益</td><td>总收益率</td><td>{_fmt_pct(m.get('total_return'))}</td></tr>
    <tr><td>收益</td><td>CAGR</td><td>{_fmt_pct(m.get('cagr'))}</td></tr>
    <tr><td>风险</td><td>年化波动率</td><td>{_fmt_pct(m.get('volatility'))}</td></tr>
    <tr><td>风险</td><td>最大回撤</td><td>{_fmt_pct(m.get('max_drawdown'))}</td></tr>
    <tr><td>风险</td><td>95% VaR（日）</td><td>{_fmt_pct(m.get('VaR_95'))}</td></tr>
    <tr><td>风险调整</td><td>Sharpe Ratio</td><td>{_fmt_n(m.get('sharpe_ratio'))}</td></tr>
    <tr><td>风险调整</td><td>Sortino Ratio</td><td>{_fmt_n(m.get('sortino_ratio'))}</td></tr>
    <tr><td>风险调整</td><td>Calmar Ratio</td><td>{_fmt_n(m.get('calmar_ratio'))}</td></tr>
    <tr><td>归因</td><td>Beta</td><td>{_fmt_n(m.get('beta'))}</td></tr>
    <tr><td>归因</td><td>Jensen Alpha</td><td>{_fmt_n(m.get('Jensen_alpha'))}</td></tr>
  </table>

  {f'<h3 style="margin-top:20px">统计显著性检验</h3><table><tr><th>检验</th><th>p-value</th><th>结论</th><th>解读</th></tr>{stat_rows}</table>' if stat_rows else ''}
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
(function(){{
  const navData = {nav_json};
  if (!navData.length) return;
  const ctx = document.getElementById('navChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: navData.map(d => d.x),
      datasets: [{{
        label: '策略净值（归一化）',
        data: navData.map(d => d.y),
        borderColor: '#6c5ce7',
        fill: false, pointRadius: 0, borderWidth: 1.8
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{legend: {{position: 'top'}}}},
      scales: {{x: {{ticks: {{maxTicksLimit: 10}}}}}}
    }}
  }});
}})();
</script>"""


def _build_section_risk(result: Any, stat_results: dict, top_list: list[dict]) -> str:
    """Section 4: 风险分析（回撤分析 + 集中度 + DSR 结论）."""
    import json as _json

    if result is None:
        return """<div class="card"><h2>🛡️ Section 4 · 风险分析</h2><p style="color:#636e72">需要回测数据（--no-backtest 时不可用）</p></div>"""

    m = getattr(result, "metrics", {}) or {}
    nav = getattr(result, "nav_series", None)

    # 回撤时序
    dd_json = "[]"
    if nav is not None and not nav.empty:
        nav_norm = nav / nav.cummax()
        dd_data = (nav_norm - 1.0)
        dd_json = _json.dumps([
            {"x": str(idx.date() if hasattr(idx, "date") else idx), "y": round(float(v), 6)}
            for idx, v in dd_data.items()
        ])

    # DSR 结论
    dsr = stat_results.get("DSR", {})
    psr = stat_results.get("PSR", {})
    ci = stat_results.get("Bootstrap Sharpe CI", {})

    dsr_html = ""
    if dsr:
        sig = "✅ 显著（alpha 可能为真实）" if dsr.get("is_significant") else "❌ 不显著（可能源于数据挖掘偏差）"
        dsr_html = f"""
<div class="nav-section" style="border-color:#e17055">
  <b>DSR（去通货膨胀 Sharpe 比率）</b>
  <p>结论：{sig}</p>
  <p>p-value={_fmt_n(dsr.get('p_value'))} | 观测 Sharpe={_fmt_n(dsr.get('observed_sr'))} | DSR阈值={_fmt_n(dsr.get('dsr_threshold'))}</p>
  <p style="font-size:.85em;color:#636e72">{dsr.get('interpretation','')}</p>
</div>"""

    psr_html = ""
    if psr:
        psr_html = f"""
<div class="nav-section" style="border-color:#0984e3">
  <b>PSR（概率 Sharpe 比率）</b>
  <p>PSR={_fmt_n(psr.get('psr_value'))} | Benchmark Sharpe={_fmt_n(psr.get('benchmark_sr'))}</p>
  <p style="font-size:.85em;color:#636e72">{psr.get('interpretation','')}</p>
</div>"""

    ci_html = ""
    if ci:
        lo, hi = ci.get("ci_lower"), ci.get("ci_upper")
        ci_html = f"""
<div class="nav-section" style="border-color:#00b894">
  <b>Bootstrap Sharpe 95% CI</b>
  <p>[{_fmt_n(lo)}, {_fmt_n(hi)}]</p>
</div>"""

    # 集中度风险（Top-N 中相同行业）
    sectors: dict[str, list] = {}
    for item in top_list:
        # ticker 前缀推断行业（简化）
        prefix = item["ticker"][:3]
        sectors.setdefault(prefix[:1], []).append(item["ticker"])

    conc_html = ""
    for sector, ts in sectors.items():
        if len(ts) >= 2:
            conc_html += f"<li>前缀 <b>{sector}...</b>：{', '.join(ts)}</li>"

    return f"""
<div class="card">
  <h2>🛡️ Section 4 · 风险分析</h2>

  <h3>回撤走势</h3>
  <canvas id="ddChart" height="60"></canvas>

  <h3 style="margin-top:24px">统计健壮性检验</h3>
  {dsr_html}
  {psr_html}
  {ci_html}

  <h3 style="margin-top:20px">关键风险指标</h3>
  <table style="max-width:500px">
    <tr><th>指标</th><th>值</th></tr>
    <tr><td>最大回撤</td><td>{_fmt_pct(m.get('max_drawdown'))}</td></tr>
    <tr><td>95% VaR（日）</td><td>{_fmt_pct(m.get('VaR_95'))}</td></tr>
    <tr><td>Calmar Ratio</td><td>{_fmt_n(m.get('calmar_ratio'))}</td></tr>
    <tr><td>Beta（vs CSI300）</td><td>{_fmt_n(m.get('beta'))}</td></tr>
  </table>

  {f'<h3 style="margin-top:20px">集中度警告</h3><ul>{conc_html}</ul>' if conc_html else ''}
</div>

<script>
(function(){{
  const ddData = {dd_json};
  if (!ddData.length) return;
  const ctx2 = document.getElementById('ddChart').getContext('2d');
  new Chart(ctx2, {{
    type: 'line',
    data: {{
      labels: ddData.map(d => d.x),
      datasets: [{{
        label: '回撤',
        data: ddData.map(d => d.y * 100),
        borderColor: '#e17055',
        backgroundColor: 'rgba(225,112,85,0.1)',
        fill: true, pointRadius: 0, borderWidth: 1.5
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{legend: {{position: 'top'}}}},
      scales: {{
        x: {{ticks: {{maxTicksLimit: 10}}}},
        y: {{ticks: {{callback: v => v + '%'}}}}
      }}
    }}
  }});
}})();
</script>"""


def generate_html_report(
    as_of: date,
    top_list: list[dict],
    feat_df: Any,
    agent_results: dict,
    backtest_result: Any,
    stat_results: dict,
    runtime_seconds: float,
    args: argparse.Namespace,
) -> str:
    """整合 4 个 Section 生成完整 HTML 报告."""
    tickers_str = ", ".join(item["ticker"] for item in top_list[:5])
    if len(top_list) > 5:
        tickers_str += f" +{len(top_list)-5}"

    s1 = _build_section_overview(top_list, feat_df)
    s2 = _build_section_research(agent_results, top_list)
    s3 = _build_section_backtest(backtest_result, stat_results)
    s4 = _build_section_risk(backtest_result, stat_results, top_list)

    toc = """
<nav style="background:#1a1a2e;color:#fff;border-radius:8px;padding:14px 20px;margin-bottom:20px;display:flex;gap:20px;flex-wrap:wrap">
  <a href="#s1" style="color:#a29bfe;text-decoration:none">📊 市场概览</a>
  <a href="#s2" style="color:#74b9ff;text-decoration:none">🔬 个股研究</a>
  <a href="#s3" style="color:#55efc4;text-decoration:none">📈 回测结果</a>
  <a href="#s4" style="color:#fd79a8;text-decoration:none">🛡️ 风险分析</a>
</nav>"""

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantMind Research Pipeline — {as_of}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>🧠 QuantMind 一键研究报告</h1>
<p style="color:#636e72">
  截面日期：<b>{as_of}</b> |
  股票池：<b>{args.universe}</b> |
  推荐：<b>{tickers_str}</b> |
  耗时：<b>{runtime_seconds:.1f}s</b> |
  生成：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</p>

{toc}

<div id="s1">{s1}</div>
<div id="s2">{s2}</div>
<div id="s3">{s3}</div>
<div id="s4">{s4}</div>

<footer style="text-align:center;padding:24px;color:#b2bec3;font-size:.85em;margin-top:32px">
  QuantMind Full Research Pipeline | LightGBM + Multi-Agent + Backtest | PIT-Strict
</footer>
</body>
</html>"""


# ============================================================================
# 主流程
# ============================================================================


def main() -> None:
    args = parse_args()
    t_start = time.time()

    logger.info("=" * 64)
    logger.info("QuantMind Full Research Pipeline")
    logger.info("=" * 64)

    # ── 1. 解析日期 ─────────────────────────────────────────────
    as_of = step_resolve_date(args)
    logger.info(f"[FRP] 截面日期: {as_of}")

    # ── 2. 构建因子 ─────────────────────────────────────────────
    feat_path, feat_df = step_build_features(as_of, args)
    logger.info(f"[FRP] 因子截面: shape={feat_df.shape}")

    # ── 3. LGBM 粗排 ────────────────────────────────────────────
    lgbm_candidates = step_lgbm_rank(feat_df, as_of, args)

    # ── 4. 确定目标股票 ─────────────────────────────────────────
    if args.tickers:
        focus_tickers = [t.strip() for t in args.tickers.split(",")]
        # 在候选列表中置顶用户指定标的
        focus_set = set(focus_tickers)
        ordered = [c for c in lgbm_candidates if c["ticker"] in focus_set]
        rest = [c for c in lgbm_candidates if c["ticker"] not in focus_set]
        lgbm_candidates = ordered + rest
    else:
        focus_tickers = [c["ticker"] for c in lgbm_candidates[: args.max_tickers]]

    # ── 5. LLM 重排 ─────────────────────────────────────────────
    top_list = step_llm_rerank(lgbm_candidates, feat_df, as_of, args)

    # ── 6. Agent 研究 ────────────────────────────────────────────
    agent_tickers = [item["ticker"] for item in top_list[: args.max_tickers]]
    agent_results = step_agent_research(agent_tickers, as_of, args)

    # ── 7. 回测 ─────────────────────────────────────────────────
    backtest_result, stat_results = step_backtest(as_of, feat_path, args)

    # ── 8. 生成报告 ──────────────────────────────────────────────
    runtime = time.time() - t_start
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ticker_tag = "_".join(
        item["ticker"].replace(".", "") for item in top_list[:3]
    )
    report_name = f"research_{ticker_tag}_{as_of.isoformat()}.html"
    report_path = out_dir / report_name

    html = generate_html_report(
        as_of=as_of,
        top_list=top_list,
        feat_df=feat_df,
        agent_results=agent_results,
        backtest_result=backtest_result,
        stat_results=stat_results,
        runtime_seconds=runtime,
        args=args,
    )
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"[FRP] HTML 报告 → {report_path}")

    # ── JSON 摘要 ────────────────────────────────────────────────
    json_path = out_dir / f"research_{ticker_tag}_{as_of.isoformat()}.json"
    summary = {
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now().isoformat(),
        "universe": args.universe,
        "runtime_seconds": round(runtime, 1),
        "top_list": [
            {k: v for k, v in item.items() if k != "shap_values"}
            for item in top_list
        ],
        "backtest_metrics": (
            getattr(backtest_result, "metrics", {}) if backtest_result else {}
        ),
        "stat_tests": {
            name: {k: (float(v) if isinstance(v, float) else v)
                   for k, v in tr.items() if not callable(v)}
            for name, tr in stat_results.items()
        },
        "agent_tickers_done": [t for t, s in agent_results.items() if s is not None],
        "agent_tickers_failed": [t for t, s in agent_results.items() if s is None],
        "steps": {
            "features_built": True,
            "lgbm_ranked": len(lgbm_candidates) > 0,
            "llm_reranked": not args.no_llm,
            "agent_research": not args.no_agent,
            "backtest_run": backtest_result is not None,
            "stat_tests_run": len(stat_results) > 0,
        },
    }
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ── 控制台摘要 ───────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("QuantMind Full Research Pipeline — 完成")
    print("=" * 64)
    print(f"截面日期  : {as_of}")
    print(f"推荐股票  :")
    for item in top_list[:10]:
        fb = " ⚠️LLM降级" if item.get("is_fallback") else ""
        print(f"  #{item.get('llm_rank','-'):>2}  {item['ticker']}  LGBM={item['lgbm_score']:.4f}  {item.get('reason','')[:50]}{fb}")
    if backtest_result:
        m = getattr(backtest_result, "metrics", {}) or {}
        print(f"\n回测结果  :")
        print(f"  总收益={_fmt_pct(m.get('total_return'))} | CAGR={_fmt_pct(m.get('cagr'))} | "
              f"Sharpe={_fmt_n(m.get('sharpe_ratio'))} | MaxDD={_fmt_pct(m.get('max_drawdown'))}")
    if stat_results:
        print(f"\n统计检验  :")
        for k, v in stat_results.items():
            if v:
                sig = "显著" if v.get("is_significant") else "不显著"
                print(f"  {k}: {sig} (p={_fmt_n(v.get('p_value'))})")
    print(f"\n总耗时    : {runtime:.1f}s")
    print(f"HTML 报告 : {report_path}")
    print(f"JSON 摘要 : {json_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
