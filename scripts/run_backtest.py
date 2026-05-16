"""scripts/run_backtest.py — 回测集成 CLI.

用法：
    python scripts/run_backtest.py --strategy equal_weight --start 2022-01-01 --end 2024-06-30
    python scripts/run_backtest.py --strategy lgbm --start 2022-01-01 --end 2024-06-30 --universe csi300
    python scripts/run_backtest.py --strategy agent --start 2023-01-01 --end 2024-06-30 --llm-model qwen2.5:7b
    python scripts/run_backtest.py --strategy walk_forward --start 2020-01-01 --end 2024-06-30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantMind Phase 6 — 回测引擎 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strategy", default="equal_weight",
                        choices=["equal_weight", "lgbm", "agent", "walk_forward"],
                        help="策略类型")
    parser.add_argument("--start", default="2022-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2024-06-30", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--universe", default="csi300",
                        help="股票池（csi300/csi500 或逗号分隔的股票代码）")
    parser.add_argument("--initial-capital", type=float, default=10_000_000,
                        help="初始资金（默认 1000万）")
    parser.add_argument("--execution-mode", default="open_price",
                        choices=["open_price", "close_price", "vwap"],
                        help="撮合模式（默认 open_price）")
    parser.add_argument("--rebalance-freq", default="Q",
                        choices=["M", "Q"],
                        help="调仓频率（M=月, Q=季）")
    parser.add_argument("--features-path", default=None,
                        help="因子特征文件路径（parquet）")
    parser.add_argument("--prices-path", default=None,
                        help="价格数据文件路径（parquet），MultiIndex date/ticker")
    parser.add_argument("--lgbm-model-path", default=None,
                        help="LGBM 模型路径")
    parser.add_argument("--llm-provider", default="ollama",
                        help="LLM 提供商（ollama/deepseek）")
    parser.add_argument("--llm-model", default="qwen2.5:7b",
                        help="LLM 模型名")
    parser.add_argument("--top-n", type=int, default=10,
                        help="LGBM：做多标的数量（等权）；其它策略可扩展")
    parser.add_argument("--rank-pool", type=int, default=50,
                        help="LGBM：截面打分后截取前 N 再取长多头")
    parser.add_argument("--panel-dir", type=Path, default=_ROOT / "data/panel")
    parser.add_argument("--prices-dir", type=Path, default=_ROOT / "data/prices")
    parser.add_argument("--output-dir", default="reports",
                        help="HTML 报告输出目录")
    parser.add_argument("--run-stat-tests", action="store_true",
                        help="运行统计显著性检验")
    parser.add_argument("--n-sim-strategies", type=int, default=10,
                        help="DSR 检验的模拟策略数量（默认 10）")
    parser.add_argument(
        "--max-industry-stocks",
        type=int,
        default=None,
        help="LGBM：单一申万行业最多持仓数量（默认不限）",
    )
    parser.add_argument(
        "--reversal-filter-pct",
        type=float,
        default=None,
        help="LGBM：剔除最近5个交易日涨幅大于该值的候选（如 0.10）；默认关闭",
    )
    parser.add_argument(
        "--weighting",
        default="equal",
        choices=["equal", "inverse_vol"],
        help="LGBM：持仓权重（等权或逆波动率）",
    )
    parser.add_argument(
        "--adj-close-path",
        default=None,
        help="LGBM：复权收盘价宽表（用于反转过滤）；默认 data/prices/csi300_daily_adj_close.parquet",
    )
    return parser.parse_args()


# ── 策略工厂 ──────────────────────────────────────────────────────────────────

def load_ohlcv_for_engine(path: Path) -> pd.DataFrame:
    """Load Phase B OHLCV parquet → MultiIndex(date, ticker) for BacktestEngine."""
    df = pd.read_parquet(path).reset_index()
    rename_map = {"trade_date": "date", "ts_code": "ticker"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "volume" not in df.columns and "vol" in df.columns:
        df["volume"] = pd.to_numeric(df["vol"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def load_benchmark_from_index_parquet(path: Path, code: str = "000300.SH") -> pd.Series:
    df = pd.read_parquet(path).reset_index()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    sub = df.loc[df["ts_code"] == code].set_index("trade_date")["close"].sort_index()
    return sub


def load_csi300_universe_tickers() -> list[str]:
    univ = _ROOT / "data/snapshots/2024-12-31/universe.parquet"
    if not univ.is_file():
        return []
    u = pd.read_parquet(univ)
    col = "ticker" if "ticker" in u.columns else "ts_code"
    return u[col].astype(str).tolist()


def load_prices_features(args) -> tuple:
    """加载价格和特征数据."""
    prices_df = None
    features_df = None

    if args.prices_path and Path(args.prices_path).exists():
        logger.info(f"[RunBacktest] 加载价格数据: {args.prices_path}")
        prices_df = pd.read_parquet(args.prices_path)
    else:
        logger.warning("[RunBacktest] 未找到价格数据，使用模拟数据（demo 模式）")
        prices_df = _make_demo_prices(args)

    if args.features_path and Path(args.features_path).exists():
        logger.info(f"[RunBacktest] 加载特征数据: {args.features_path}")
        features_df = pd.read_parquet(args.features_path)

    return prices_df, features_df


def _make_demo_prices(args) -> "pd.DataFrame":
    """生成 demo 用的模拟价格数据."""
    import numpy as np
    import pandas as pd

    tickers = ["600519.SH", "000858.SZ", "600036.SH", "000333.SZ",
               "601318.SH", "600276.SH", "000001.SZ", "600000.SH",
               "002415.SZ", "300750.SZ"]

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    dates = pd.date_range(start, end, freq="B")  # 工作日

    rng = np.random.default_rng(42)
    records = []
    for ticker in tickers:
        price = rng.uniform(10, 200)
        for dt in dates:
            price *= (1 + rng.normal(0.0005, 0.02))
            records.append({
                "date": dt,
                "ticker": ticker,
                "open": price * (1 + rng.normal(0, 0.005)),
                "high": price * (1 + abs(rng.normal(0, 0.01))),
                "low": price * (1 - abs(rng.normal(0, 0.01))),
                "close": price,
                "volume": rng.integers(100000, 10000000),
                "pre_close": price * (1 - rng.normal(0.0005, 0.02)),
            })

    df = pd.DataFrame(records)
    df = df.set_index(["date", "ticker"])
    return df


# ── HTML 报告生成 ─────────────────────────────────────────────────────────────

def build_html_report(result, stat_results: dict, strategy: str, as_of: date) -> str:
    """生成 HTML 回测报告."""
    m = result.metrics if hasattr(result, "metrics") else {}
    nav = result.nav_series if hasattr(result, "nav_series") else None
    fills = result.fills if hasattr(result, "fills") else []

    # NAV 图表数据
    nav_data = "[]"
    bench_data = "[]"
    if nav is not None and not nav.empty:
        nav_data = json.dumps([
            {"x": str(idx.date() if hasattr(idx, "date") else idx), "y": round(v, 4)}
            for idx, v in nav.items()
        ])
    if hasattr(result, "benchmark_series") and not result.benchmark_series.empty:
        bench_normalized = result.benchmark_series / result.benchmark_series.iloc[0]
        nav_normalized = nav / nav.iloc[0] if nav is not None and not nav.empty else None
        bench_data = json.dumps([
            {"x": str(idx.date() if hasattr(idx, "date") else idx), "y": round(v, 4)}
            for idx, v in bench_normalized.items()
        ])

    def _fmt(v, pct=False, decimals=3):
        if v is None or (isinstance(v, float) and (v != v)):  # NaN
            return "N/A"
        if pct:
            return f"{v:.1%}"
        return f"{v:.{decimals}f}"

    # 统计检验结果表格
    stat_rows = ""
    for test_name, test_result in stat_results.items():
        sig = "✅显著" if test_result.get("is_significant") else "❌不显著"
        pval = test_result.get("p_value", float("nan"))
        interp = test_result.get("interpretation", "")
        stat_rows += f"""
<tr>
  <td>{test_name}</td>
  <td>{_fmt(pval)}</td>
  <td>{sig}</td>
  <td style="font-size:0.85em;color:#636e72">{interp[:80]}</td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantMind 回测报告 — {strategy}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body{{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f6fa;margin:0;padding:20px}}
  h1,h2{{color:#1a1a2e}} .card{{background:#fff;border-radius:8px;padding:20px;margin:16px 0;box-shadow:0 2px 8px rgba(0,0,0,0.08)}}
  .metric{{display:inline-block;padding:14px 20px;border-radius:8px;margin:6px;text-align:center;min-width:120px;color:#fff}}
  table{{width:100%;border-collapse:collapse}} th{{background:#1a1a2e;color:#fff;padding:8px}} td{{padding:7px;border-bottom:1px solid #eee}}
</style>
</head>
<body>
<h1>📈 QuantMind 回测报告</h1>
<p style="color:#636e72">策略：{strategy} | 区间：{getattr(result,'start_date','?')}~{getattr(result,'end_date','?')} | 生成：{as_of}</p>

<div>
  <span class="metric" style="background:#00b894">
    <div style="font-size:0.75em">总收益</div>
    <div style="font-size:1.4em;font-weight:bold">{_fmt(m.get('total_return',0), pct=True)}</div>
  </span>
  <span class="metric" style="background:#6c5ce7">
    <div style="font-size:0.75em">Sharpe</div>
    <div style="font-size:1.4em;font-weight:bold">{_fmt(m.get('sharpe_ratio',0))}</div>
  </span>
  <span class="metric" style="background:#e17055">
    <div style="font-size:0.75em">最大回撤</div>
    <div style="font-size:1.4em;font-weight:bold">{_fmt(m.get('max_drawdown',0), pct=True)}</div>
  </span>
  <span class="metric" style="background:#0984e3">
    <div style="font-size:0.75em">年化收益</div>
    <div style="font-size:1.4em;font-weight:bold">{_fmt(m.get('cagr',0), pct=True)}</div>
  </span>
</div>

<div class="card">
  <h2>净值曲线</h2>
  <canvas id="navChart" height="80"></canvas>
</div>

<div class="card">
  <h2>性能指标</h2>
  <table>
    <tr><th>指标类</th><th>指标</th><th>值</th><th>说明</th></tr>
    <tr><td>收益</td><td>总收益率</td><td>{_fmt(m.get('total_return',0),pct=True)}</td><td>累计持有期收益</td></tr>
    <tr><td>收益</td><td>CAGR</td><td>{_fmt(m.get('cagr',0),pct=True)}</td><td>复合年化增长率</td></tr>
    <tr><td>风险</td><td>年化波动率</td><td>{_fmt(m.get('volatility',0),pct=True)}</td><td>日收益率标准差×√252</td></tr>
    <tr><td>风险</td><td>最大回撤</td><td>{_fmt(m.get('max_drawdown',0),pct=True)}</td><td>峰谷最大跌幅</td></tr>
    <tr><td>风险</td><td>95% VaR（日）</td><td>{_fmt(m.get('VaR_95',0),pct=True)}</td><td>95%置信度1日损失上限</td></tr>
    <tr><td>风险调整</td><td>Sharpe Ratio</td><td>{_fmt(m.get('sharpe_ratio',0))}</td><td>(E[r]-rf)/σ×√252，rf=2.5%</td></tr>
    <tr><td>风险调整</td><td>Sortino Ratio</td><td>{_fmt(m.get('sortino_ratio',0))}</td><td>只考虑下行风险的 Sharpe</td></tr>
    <tr><td>风险调整</td><td>Calmar Ratio</td><td>{_fmt(m.get('calmar_ratio',0))}</td><td>年化收益/最大回撤</td></tr>
    <tr><td>归因</td><td>Beta</td><td>{_fmt(m.get('beta',float('nan')))}</td><td>相对沪深300的系统风险</td></tr>
    <tr><td>归因</td><td>Jensen Alpha</td><td>{_fmt(m.get('Jensen_alpha',float('nan')))}</td><td>扣除市场风险后的纯 alpha</td></tr>
  </table>
</div>

{'<div class="card"><h2>统计显著性检验</h2><table><tr><th>检验方法</th><th>p-value</th><th>结论</th><th>解读</th></tr>' + stat_rows + '</table></div>' if stat_rows else ''}

<div class="card">
  <h2>综合评价</h2>
  <p>{m.get('interpretation', 'N/A')}</p>
</div>

<script>
const navData = {nav_data};
const benchData = {bench_data};
if (navData.length > 0) {{
  const ctx = document.getElementById('navChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: navData.map(d => d.x),
      datasets: [
        {{label:'策略净值', data: navData.map((d,i) => navData[0].y > 0 ? d.y/navData[0].y : 1),
          borderColor:'#6c5ce7', fill:false, pointRadius:0, borderWidth:1.5}},
        {{label:'基准', data: benchData.map(d => d.y),
          borderColor:'#b2bec3', fill:false, pointRadius:0, borderWidth:1, borderDash:[4,4]}}
      ]
    }},
    options: {{responsive:true, scales:{{x:{{ticks:{{maxTicksLimit:8}}}}}}, plugins:{{legend:{{position:'top'}}}}}}
  }});
}}
</script>

<footer style="text-align:center;padding:20px;color:#b2bec3;font-size:0.85em">
  QuantMind Phase 6 | Backtest Engine | T+1 / 涨跌停 / PIT Strict
</footer>
</body>
</html>"""
    return html


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    import numpy as np
    import pandas as pd
    args = parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    logger.info(f"[RunBacktest] strategy={args.strategy} | {start}~{end}")

    if args.strategy == "lgbm":
        prices_df, features_df = None, None
    else:
        prices_df, features_df = load_prices_features(args)

    # 解析 universe
    universe: list[str]
    if args.universe.startswith("csi"):
        if args.strategy == "lgbm":
            universe = load_csi300_universe_tickers() or []
        elif prices_df is not None and isinstance(prices_df.index, pd.MultiIndex):
            universe = list(prices_df.index.get_level_values("ticker").unique())[:300]
        elif prices_df is not None:
            universe = list(prices_df.columns)[:300]
        else:
            universe = []
    else:
        universe = [t.strip() for t in args.universe.split(",")]

    from quantmind.backtest import (
        BacktestEngine, BacktestConfig, EqualWeightStrategy,
        PerformanceMetrics,
        deflated_sharpe_ratio, probabilistic_sharpe_ratio,
        bootstrap_ci, ic_ttest_newey_west,
    )

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        execution_mode=args.execution_mode,
    )

    result = None
    stat_results: dict = {}

    if args.strategy == "equal_weight":
        strategy = EqualWeightStrategy(
            rebalance_freq=args.rebalance_freq,
            config=config,
        )
        engine = BacktestEngine(config=config, prices_df=prices_df)
        result = engine.run(strategy, start, end, universe)

    elif args.strategy == "agent":
        from quantmind.backtest import AgentBacktester
        backtester = AgentBacktester(
            features_df=features_df,
            prices_df=prices_df,
            lgbm_model_path=args.lgbm_model_path,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
        )
        agent_result = backtester.run(
            start_date=start,
            end_date=end,
            universe=universe,
            rebalance_freq=args.rebalance_freq,
        )
        print("\n" + agent_result.summary_table().to_string(index=False))
        # 用 agent 结果构造兼容的 result
        from quantmind.backtest.engine import BacktestResult
        dummy = BacktestResult(
            strategy_name="Agent",
            start_date=start,
            end_date=end,
            universe_size=len(universe),
            config=config,
            returns=agent_result.agent_returns,
            metrics=agent_result.metrics.get("agent", {}),
        )
        result = dummy

    elif args.strategy == "walk_forward":
        from quantmind.backtest import WalkForwardValidator, WalkForwardConfig
        wf_config = WalkForwardConfig()
        validator = WalkForwardValidator(config=wf_config, prices_df=prices_df)
        wf_result = validator.validate(
            strategy_class=EqualWeightStrategy,
            prices_df=prices_df,
        )
        print(f"\nWalk-Forward 结果: {json.dumps(wf_result.summary, indent=2, default=str)}")
        # 构造兼容 result
        from quantmind.backtest.engine import BacktestResult
        result = BacktestResult(
            strategy_name="WalkForward",
            start_date=start,
            end_date=end,
            universe_size=len(universe),
            config=config,
            metrics=wf_result.summary,
        )

    elif args.strategy == "lgbm":
        from quantmind.backtest.lgbm_strategy import LGBMStrategy

        price_path = Path(args.prices_path) if args.prices_path else args.prices_dir / "csi300_daily_ohlcv.parquet"
        bench_path = args.prices_dir / "index_daily.parquet"

        try:
            prices_df = load_ohlcv_for_engine(price_path)
        except FileNotFoundError:
            logger.warning("[RunBacktest] missing {}; using demo prices", price_path)
            prices_df = _make_demo_prices(args)

        benchmark_series = None
        if bench_path.is_file():
            try:
                benchmark_series = load_benchmark_from_index_parquet(bench_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("[RunBacktest] benchmark load failed: {}", str(e)[:120])

        universe_lgb = load_csi300_universe_tickers()
        avail = set(prices_df.index.get_level_values("ticker"))
        if universe_lgb:
            universe_lgb = [t for t in universe_lgb if t in avail]
        else:
            universe_lgb = sorted(avail)[:300]

        model_path = args.lgbm_model_path or str(_ROOT / "models/lgbm_v1_final.pkl")
        reb = "quarterly" if args.rebalance_freq == "Q" else "monthly"
        adj_default = _ROOT / "data/prices/csi300_daily_adj_close.parquet"
        adj_path = Path(args.adj_close_path) if args.adj_close_path else adj_default
        adj_existing = adj_path if adj_path.is_file() else None

        strategy = LGBMStrategy(
            model_path=model_path,
            panel_dir=args.panel_dir,
            top_n=args.rank_pool,
            long_n=args.top_n,
            rebalance=reb,
            price_path=price_path,
            adj_close_path=adj_existing,
            max_industry_stocks=args.max_industry_stocks,
            reversal_pct_cutoff=args.reversal_filter_pct,
            weighting=args.weighting,
            config=config,
        )
        engine = BacktestEngine(
            config=config,
            prices_df=prices_df,
            benchmark_df=benchmark_series,
        )
        result = engine.run(strategy, start, end, universe_lgb)

    else:
        logger.error("[RunBacktest] unknown strategy: {}", args.strategy)

    if result is None:
        logger.error("[RunBacktest] 回测结果为空")
        return

    # ── 统计显著性检验 ────────────────────────────────────────────────────────
    if args.run_stat_tests and hasattr(result, "returns") and not result.returns.empty:
        logger.info("[RunBacktest] 运行统计显著性检验...")
        ret = result.returns.dropna()

        # PSR
        stat_results["PSR"] = probabilistic_sharpe_ratio(ret)

        # DSR（模拟 N 个随机策略）
        sim_sharpes = []
        rng = np.random.default_rng(42)
        for _ in range(args.n_sim_strategies):
            sim_ret = rng.normal(0.0003, 0.015, len(ret))
            sim_sr = float(np.mean(sim_ret) / (np.std(sim_ret) + 1e-10) * np.sqrt(252))
            sim_sharpes.append(sim_sr)
        candidate_sr = result.metrics.get("sharpe_ratio", 0)
        sim_sharpes.append(candidate_sr)
        stat_results["DSR"] = deflated_sharpe_ratio(sim_sharpes, candidate_sr, n_obs=len(ret))

        # Bootstrap CI for Sharpe
        def _sharpe(r):
            rf = 0.025 / 252
            excess = r - rf
            return np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(252)

        stat_results["Bootstrap Sharpe CI"] = bootstrap_ci(
            _sharpe, ret.values, n_bootstrap=500
        )

    # ── 打印摘要 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"回测完成：{result.strategy_name}")
    print(f"{'='*60}")
    print(PerformanceMetrics().generate_report(result))

    # ── 生成 HTML 报告 ────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"backtest_{args.strategy}_{end}.html"
    output_path = output_dir / filename

    html = build_html_report(result, stat_results, args.strategy, end)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"[RunBacktest] HTML 报告 → {output_path}")
    print(f"\n报告已保存 → {output_path}")

    if args.strategy == "lgbm":
        rdir = _ROOT / "reports" / "backtest"
        rdir.mkdir(parents=True, exist_ok=True)
        alt_html = rdir / "lgbm_engine_backtest_report.html"
        alt_html.write_text(html, encoding="utf-8")
        m = getattr(result, "metrics", {}) or {}
        tr = m.get("turnover_rate")
        export = {
            "total_return": m.get("total_return"),
            "cagr": m.get("cagr"),
            "sharpe": m.get("sharpe_ratio"),
            "max_drawdown": m.get("max_drawdown"),
            "turnover": tr,
            "turnover_annual_pct": float(tr) * 100.0 if tr is not None and np.isfinite(tr) else None,
            "volatility": m.get("volatility"),
            "sortino_ratio": m.get("sortino_ratio"),
        }
        (rdir / "lgbm_engine_metrics.json").write_text(
            json.dumps(export, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("[RunBacktest] LGBM 报告 → {} | {}", alt_html, rdir / "lgbm_engine_metrics.json")


if __name__ == "__main__":
    main()
