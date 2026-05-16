"""QuantMind 全系统 End-to-End 演示。

三个系统串联，输出完整投资报告：

  System 1 — 因子选股系统
    Regime 检测 → LGBM LambdaRanker（Top-50）→ HRP 仓位优化 → Top-N 候选

  System 2 — 投资分析系统
    6-Agent 深度分析（估值 / 动量 / 风险 / 质量 / 情绪 / 策略）

  System 3 — 回测验证系统
    历史胜率 / 期望收益 / 最大损失 / ACCEPTABLE/WATCHLIST/AVOID 评级

  输出: reports/demo/<date>/final_investment_report.html

用法:
  python scripts/run_full_system_demo.py --date 2025-12-31 --top 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default="2025-12-31",
                   help="演示日期 YYYY-MM-DD（需有对应快照）")
    p.add_argument("--top", type=int, default=15,
                   help="最终推荐股票数（默认 15）")
    p.add_argument("--lgbm-model", type=Path,
                   default=Path("models/lgbm_v3_top18.pkl"))
    p.add_argument("--position-sizing", choices=["equal", "hrp", "kelly", "blend"],
                   default="hrp")
    p.add_argument("--kelly-fraction", type=float, default=0.5)
    p.add_argument("--agent-provider", default="none",
                   help="LLM provider（none = 仅 ML+Rules，dashscope/ollama = 启用 LLM）")
    p.add_argument("--agent-model", default="qwen-plus")
    p.add_argument("--universe", default="alpha")
    p.add_argument("--out", type=Path, default=None,
                   help="输出目录（默认 reports/demo/<date>/）")
    p.add_argument("--force-features", action="store_true",
                   help="强制重建因子（忽略缓存）")
    return p.parse_args(argv)


# ─────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────

class _Timer:
    def __init__(self, label: str):
        self.label = label
        self._t0 = time.monotonic()

    def done(self, note: str = "") -> float:
        elapsed = time.monotonic() - self._t0
        tag = f" ({note})" if note else ""
        print(f"    ✓ {self.label} — {elapsed:.1f}s{tag}")
        return elapsed


def _print_stage(n: int, title: str):
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  System {n}: {title}")
    print(bar)


# ─────────────────────────────────────────────
# System 1: 因子选股
# ─────────────────────────────────────────────

def system1_factor_ranking(
    as_of: date,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    """
    1a. 检测市场 Regime
    1b. 构建因子截面（FeaturePipeline）
    1c. LGBM 打分 → Top-50
    1d. HRP/Kelly 仓位优化 → Top-N 候选
    """
    _print_stage(1, "因子选股系统")

    # ── 1a Regime 检测 ──
    t = _Timer("Regime 检测")
    from scripts.daily_update import _detect_market_regime, _resolve_regime_model
    regime_name, regime_hint = _detect_market_regime()
    model_path = _resolve_regime_model(args, regime_hint)
    t.done(f"Regime={regime_name} → {model_path.name}")
    regime_info = {
        "name": regime_name,
        "hint": regime_hint,
        "model": model_path.name,
    }

    # ── 1b 因子构建 ──
    t = _Timer("因子截面构建（FeaturePipeline）")
    feat_file = ROOT / "data" / "features" / f"{args.universe}_{as_of.isoformat()}.parquet"
    if feat_file.exists() and not args.force_features:
        feat_df = pd.read_parquet(feat_file)
        t.done(f"读取缓存 shape={feat_df.shape}")
    else:
        from quantmind.features import FeaturePipeline
        pipe = FeaturePipeline()
        feat_df = pipe.run_single(as_of, universe=args.universe)
        pipe.save(feat_df, as_of, universe=args.universe)
        t.done(f"新建 shape={feat_df.shape}")

    # ── 1c LGBM 打分 ──
    t = _Timer("LGBM 粗排")
    from quantmind.models.factor_model import FactorModel
    from quantmind.utils.score_order import order_preserving_pct_rank

    model = FactorModel.load(model_path)
    feat_names = getattr(model, "_feature_names", None) or model.feature_names
    missing = [c for c in feat_names if c not in feat_df.columns]
    if missing:
        print(f"  ⚠ 缺失特征 {missing[:3]}，补零")
        for c in missing:
            feat_df[c] = 0.0
    X = feat_df[list(feat_names)].to_numpy(dtype=np.float32, copy=True)
    raw_scores = model.predict(X)
    pct_scores = order_preserving_pct_rank(raw_scores)
    score_series = pd.Series(pct_scores, index=feat_df.index, name="score")
    top50 = score_series.nlargest(50)
    t.done(f"全截面 {len(feat_df)} 只 → Top-50")

    # ── 1d HRP/Kelly 仓位优化 ──
    t = _Timer(f"仓位优化（{args.position_sizing}）")
    top_tickers = list(top50.index.astype(str))
    weights = _compute_weights_for_demo(
        top_tickers, as_of, args.position_sizing, args.kelly_fraction
    )

    # 取 HRP 权重最大的 Top-N 作为最终候选
    w_series = pd.Series(weights).sort_values(ascending=False)
    final_tickers = list(w_series.head(args.top).index)
    final_weights = {t: weights[t] for t in final_tickers}
    # 重新归一化
    total_w = sum(final_weights.values())
    final_weights = {t: v / total_w for t, v in final_weights.items()}
    t.done(f"Top-50 → Top-{len(final_tickers)} 最终候选")

    # ── 组装因子得分表 ──
    factor_table = []
    for ticker in final_tickers:
        row = {"ticker": ticker, "lgbm_score": round(float(score_series.get(ticker, 0)), 4)}
        for col in ["log_market_cap", "pe_ttm", "momentum_3m", "volatility_3m",
                    "roe_ttm", "turnover_3m_avg", "margin_buy_intensity", "earnings_yield"]:
            row[col] = round(float(feat_df[col].get(ticker, float("nan"))), 4) if col in feat_df.columns else None
        row["weight"] = round(final_weights.get(ticker, 0), 5)
        factor_table.append(row)

    result = {
        "as_of": str(as_of),
        "regime": regime_info,
        "top50_count": len(top50),
        "final_tickers": final_tickers,
        "final_weights": final_weights,
        "factor_table": factor_table,
    }
    (out_dir / "system1_factor_ranking.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  → 最终候选：{final_tickers[:5]}…")
    return result


def _compute_weights_for_demo(tickers, as_of, method, kelly_fraction):
    from quantmind.portfolio.position_sizing import hrp_weights, kelly_weights, blend_weights
    import pyarrow.parquet as pq

    if method == "equal":
        return {t: 1.0 / len(tickers) for t in tickers}

    price_path = ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
    try:
        schema_names = pq.ParquetFile(str(price_path)).schema_arrow.names
        price_col = "adj_close" if "adj_close" in schema_names else "close"
        prices = pd.read_parquet(price_path, columns=["ts_code", "trade_date", price_col])
        prices = prices[prices["ts_code"].isin(tickers)]
        wide = prices.pivot_table(index="trade_date", columns="ts_code",
                                  values=price_col, aggfunc="last")
        wide.index = pd.to_datetime(wide.index)
        cutoff = pd.Timestamp(as_of)
        hist = wide[wide.index < cutoff].tail(252)
        rets = hist.pct_change().dropna(how="all")

        if method in ("hrp", "blend"):
            w_hrp = hrp_weights(rets, min_periods=63)
        if method in ("kelly", "blend"):
            mu = rets.mean() * 252
            cov = rets.cov() * 252
            w_kelly = kelly_weights(mu, cov.values, fraction=kelly_fraction)
            w_kelly.index = mu.index

        if method == "hrp":
            w = w_hrp
        elif method == "kelly":
            w = w_kelly
        else:
            w = blend_weights(w_hrp, w_kelly, alpha=0.5)

        missing = [t for t in tickers if t not in w.index]
        for mt in missing:
            w[mt] = 1.0 / len(tickers)
        w = w.reindex(tickers).fillna(0)
        w /= w.sum()
        return w.to_dict()
    except Exception as e:
        print(f"  ⚠ 仓位优化失败（{e}），等权回退")
        return {t: 1.0 / len(tickers) for t in tickers}


# ─────────────────────────────────────────────
# System 2: 投资分析
# ─────────────────────────────────────────────

def system2_investment_analysis(
    as_of: date,
    tickers: list[str],
    weights: dict[str, float],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    """6-Agent 深度投资分析（逐股运行）。"""
    _print_stage(2, "投资分析系统（6-Agent）")

    from scripts.run_investment_pipeline import _load_price_df, run_six_agents
    from quantmind.agents.investment_agents.strategy_agent import StrategyAgent

    price_df = _load_price_df()

    strategies: list[dict] = []
    agent_scores_table: list[dict] = []

    for i, ticker in enumerate(tickers):
        t = _Timer(f"[{i+1}/{len(tickers)}] {ticker}")
        try:
            context = {
                "price_df": price_df,
                "position_weight": weights.get(ticker, 0.0),
            }
            agent_signals = run_six_agents(
                ticker=ticker,
                as_of=str(as_of),
                context=context,
                provider=args.agent_provider,
                model=args.agent_model,
            )

            strategy_agent = StrategyAgent(
                ticker=ticker,
                as_of=str(as_of),
                context=context,
                agent_signals=agent_signals,
                provider=args.agent_provider,
                model=args.agent_model,
            )
            strategy = strategy_agent.analyze_with_llm()

            strat_dict = strategy.model_dump() if hasattr(strategy, "model_dump") else vars(strategy)
            strat_dict["ticker"] = ticker
            strat_dict["as_of"] = str(as_of)
            strat_dict["weight"] = weights.get(ticker, 0.0)
            strategies.append(strat_dict)

            scores_row = {"ticker": ticker, "weight": round(weights.get(ticker, 0), 5)}
            for sig in agent_signals:
                agent_name = type(sig).__name__.replace("Signal", "").lower()
                scores_row[f"{agent_name}_score"] = round(float(getattr(sig, "score", 0) or 0), 3)
            scores_row["composite"] = round(
                np.mean([float(getattr(s, "score", 0) or 0) for s in agent_signals]), 3
            )
            scores_row["rating"] = (
                strat_dict.get("rating")
                or strat_dict.get("comprehensive_rating")
                or "—"
            )
            agent_scores_table.append(scores_row)
            elapsed = t.done(f"综合评级={scores_row.get('rating','—')}")
        except Exception as e:
            print(f"  ⚠ {ticker} 分析失败：{e}")
            strategies.append({"ticker": ticker, "error": str(e)})
            agent_scores_table.append({"ticker": ticker, "error": str(e)})

    result = {
        "as_of": str(as_of),
        "strategies": strategies,
        "agent_scores_table": agent_scores_table,
    }

    strat_out = out_dir / "system2_strategies.json"
    strat_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  → {len(strategies)} 只股票分析完成，保存至 {strat_out.name}")
    return result


# ─────────────────────────────────────────────
# System 3: 回测验证
# ─────────────────────────────────────────────

def system3_backtest_validation(
    as_of: date,
    tickers: list[str],
    strategies: list[dict],
    out_dir: Path,
) -> dict:
    """历史回测验证：胜率、期望收益、最大损失、评级。"""
    _print_stage(3, "回测验证系统")

    from scripts.validate_strategies import batch_validate, _load_wide_price_df

    t = _Timer("加载价格面板")
    try:
        price_df = _load_wide_price_df()
    except Exception:
        from scripts.run_investment_pipeline import _load_price_df
        price_df = _load_price_df()
    t.done(f"shape={price_df.shape}")

    # 加载因子面板（取最近一期）
    t = _Timer("加载因子面板（作为回测 features）")
    try:
        panel = pd.read_parquet(ROOT / "data" / "panel" / "alpha_panel_v3.parquet")
    except Exception:
        panel = None
    t.done()

    t = _Timer("批量历史回测")
    try:
        validation_results = batch_validate(
            strategies=strategies,
            price_df=price_df,
            panel_df=panel,
            only_buy_signals=False,
        )
        t.done(f"{len(validation_results)} 条回测结果")
    except Exception as e:
        print(f"  ⚠ batch_validate 失败（{e}），逐股简单回测")
        validation_results = _simple_backtest(tickers, as_of, price_df)

    # 汇总
    validation_table = []
    for r in validation_results:
        if hasattr(r, "__dict__"):
            row = {k: v for k, v in r.__dict__.items() if not k.startswith("_")}
        elif isinstance(r, dict):
            row = r
        else:
            row = {"raw": str(r)}
        validation_table.append(row)

    result = {"as_of": str(as_of), "validations": validation_table}
    (out_dir / "system3_validations.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  → 回测完成，保存至 system3_validations.json")
    return result


def _simple_backtest(tickers: list[str], as_of: date, price_df: pd.DataFrame) -> list[dict]:
    """当 batch_validate 不可用时的简化回测（1年历史）。"""
    results = []
    cutoff = pd.Timestamp(as_of)
    start = cutoff - pd.Timedelta(days=365)
    for ticker in tickers:
        if ticker not in price_df.columns:
            results.append({"ticker": ticker, "validation_status": "NO_DATA"})
            continue
        prices = price_df.loc[(price_df.index >= start) & (price_df.index <= cutoff), ticker].dropna()
        if len(prices) < 20:
            results.append({"ticker": ticker, "validation_status": "INSUFFICIENT"})
            continue
        monthly = prices.resample("ME").last().pct_change().dropna()
        win_rate = float((monthly > 0).mean())
        exp_ret  = float(monthly.mean())
        max_loss = float(monthly.min())
        ann_ret  = float((1 + exp_ret) ** 12 - 1)
        if win_rate >= 0.55 and ann_ret > 0.10:
            status = "ACCEPTABLE"
        elif win_rate >= 0.45 and ann_ret > 0:
            status = "WATCHLIST"
        else:
            status = "AVOID"
        results.append({
            "ticker": ticker,
            "win_rate": round(win_rate, 3),
            "expected_monthly_return": round(exp_ret, 4),
            "max_monthly_loss": round(max_loss, 4),
            "ann_return_1y": round(ann_ret, 4),
            "validation_status": status,
        })
    return results


# ─────────────────────────────────────────────
# HTML 报告生成
# ─────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>QuantMind 全系统投资报告 {as_of}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:'Microsoft YaHei',sans-serif; background:#0d1117; color:#c9d1d9; padding:20px; }}
h1   {{ font-size:22px; color:#58a6ff; margin:16px 0 6px; }}
h2   {{ font-size:16px; color:#3fb950; margin:20px 0 8px; border-left:3px solid #3fb950; padding-left:10px; }}
h3   {{ font-size:14px; color:#8b949e; margin:14px 0 6px; }}
.header {{ display:flex; align-items:center; gap:20px; background:#161b22; padding:16px 20px; border-radius:10px; margin-bottom:20px; }}
.header h1 {{ margin:0; }}
.badge {{ padding:4px 12px; border-radius:20px; font-size:12px; font-weight:bold; }}
.badge-bull  {{ background:#0d2626; color:#3fb950; border:1px solid #3fb950; }}
.badge-bear  {{ background:#2d1b1b; color:#f85149; border:1px solid #f85149; }}
.badge-norm  {{ background:#1c2128; color:#d29922; border:1px solid #d29922; }}
.regime-meta {{ font-size:12px; color:#8b949e; }}
.funnel-row  {{ display:flex; align-items:center; gap:8px; font-size:13px; color:#8b949e; margin:8px 0; }}
.funnel-n    {{ font-size:20px; font-weight:bold; color:#58a6ff; min-width:60px; }}
.funnel-arr  {{ color:#444; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; margin:10px 0; }}
th    {{ background:#161b22; padding:8px 10px; text-align:left; color:#8b949e; font-weight:normal; border-bottom:1px solid #30363d; }}
td    {{ padding:7px 10px; border-bottom:1px solid #21262d; }}
tr:hover td {{ background:#161b22; }}
.tag-acceptable {{ color:#3fb950; font-weight:bold; }}
.tag-watchlist  {{ color:#d29922; }}
.tag-avoid      {{ color:#f85149; }}
.tag-buy        {{ color:#3fb950; font-weight:bold; }}
.tag-hold       {{ color:#d29922; }}
.tag-sell       {{ color:#f85149; }}
.score-bar      {{ display:inline-block; height:8px; border-radius:4px; vertical-align:middle; margin-left:6px; }}
.grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }}
.card {{ background:#161b22; border-radius:8px; padding:14px; }}
.card-title {{ font-size:12px; color:#8b949e; margin-bottom:8px; }}
.card-val   {{ font-size:22px; font-weight:bold; color:#58a6ff; }}
.radar-wrap {{ display:flex; flex-wrap:wrap; gap:12px; }}
.radar-item {{ background:#161b22; border-radius:8px; padding:10px; }}
.radar-item h4 {{ font-size:13px; margin-bottom:4px; color:#c9d1d9; }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>⚡ QuantMind 全系统投资报告</h1>
    <div class="regime-meta">演示日期：{as_of} · 宇宙：Alpha 1374 · 模型：{model_name}</div>
  </div>
  <div class="badge {regime_badge_cls}">{regime_label}</div>
  <div class="regime-meta">子模型: {sub_model}</div>
</div>

<!-- 系统1 -->
<h2>System 1 — 因子选股系统</h2>
<div class="funnel-row">
  <span class="funnel-n">1374</span><span>Alpha 宇宙</span>
  <span class="funnel-arr">→</span>
  <span class="funnel-n">50</span><span>LGBM Top-50</span>
  <span class="funnel-arr">→</span>
  <span class="funnel-n" style="color:#3fb950">{final_n}</span><span>HRP 权重筛选后最终候选</span>
</div>

<table>
<thead><tr>
  <th>#</th><th>股票</th><th>LGBM分</th>
  <th>PE</th><th>ROE</th><th>3M动量</th><th>波动率3M</th>
  <th>北向强度</th><th>融资强度</th>
  <th>HRP权重</th>
</tr></thead>
<tbody>{factor_rows}</tbody>
</table>

<!-- 系统2 -->
<h2>System 2 — 投资分析系统（6-Agent）</h2>
<h3>Agent 信号矩阵</h3>
<table>
<thead><tr>
  <th>股票</th><th>估值Agent</th><th>动量Agent</th>
  <th>风险Agent</th><th>质量Agent</th><th>情绪Agent</th>
  <th>综合评分</th><th>投资评级</th><th>建议仓位</th>
</tr></thead>
<tbody>{agent_rows}</tbody>
</table>

<h3>Agent 雷达图</h3>
<div class="radar-wrap" id="radar-container">{radar_placeholders}</div>
<script>{radar_scripts}</script>

<!-- 系统3 -->
<h2>System 3 — 回测验证系统</h2>
<table>
<thead><tr>
  <th>股票</th><th>1年胜率</th><th>月均收益</th><th>年化收益</th>
  <th>最大单月损失</th><th>验证结论</th>
</tr></thead>
<tbody>{backtest_rows}</tbody>
</table>

<!-- 最终推荐 -->
<h2>📋 最终投资建议</h2>
<div class="grid-3" style="margin-bottom:20px">
{summary_cards}
</div>

<table>
<thead><tr>
  <th>#</th><th>股票</th><th>仓位权重</th><th>投资评级</th>
  <th>验证结论</th><th>综合评分</th><th>1年胜率</th><th>年化收益</th>
</tr></thead>
<tbody>{final_rows}</tbody>
</table>

<p style="margin-top:20px; font-size:11px; color:#555;">
  生成时间：{gen_time} | QuantMind v1.0 | 仅供研究参考，不构成投资建议
</p>
</body>
</html>
"""


def _score_bar(score: float, positive_color="#3fb950", negative_color="#f85149") -> str:
    """生成小的评分条 HTML。"""
    s = max(-1.0, min(1.0, float(score) if score == score else 0))
    pct = int(abs(s) * 60)
    color = positive_color if s >= 0 else negative_color
    return (f'<span style="color:{color}">{s:+.2f}</span>'
            f'<span class="score-bar" style="width:{pct}px;background:{color};opacity:0.5"></span>')


def _rating_tag(rating: str) -> str:
    r = str(rating).upper()
    cls = "tag-buy" if "BUY" in r else "tag-sell" if "SELL" in r else "tag-hold"
    return f'<span class="{cls}">{rating}</span>'


def _validation_tag(status: str) -> str:
    s = str(status).upper()
    cls = "tag-acceptable" if "ACCEPTABLE" in s else "tag-avoid" if "AVOID" in s else "tag-watchlist"
    label = {"ACCEPTABLE": "✅ 通过", "WATCHLIST": "⚠ 观察", "AVOID": "❌ 回避"}.get(s, status)
    return f'<span class="{cls}">{label}</span>'


def build_html_report(
    s1: dict, s2: dict, s3: dict, args: argparse.Namespace
) -> str:
    from datetime import datetime

    as_of = s1["as_of"]
    regime = s1["regime"]
    regime_badge = {"bull_low_vol": "badge-bull", "bear_crisis": "badge-bear"}.get(
        regime["name"], "badge-norm"
    )
    regime_label_map = {
        "bull_low_vol": "🐂 牛市低波",
        "normal": "〰 震荡正常",
        "bear_crisis": "🐻 熊市危机",
        "unknown": "❓ 未知",
    }

    # ── System 1 factor rows ──
    factor_rows = []
    for i, row in enumerate(s1["factor_table"], 1):
        t = row["ticker"]
        def _f(v): return f"{v:.3f}" if v is not None and v == v else "—"
        factor_rows.append(
            f"<tr><td>{i}</td><td><b>{t}</b></td>"
            f"<td>{_score_bar(row.get('lgbm_score',0))}</td>"
            f"<td>{_f(row.get('pe_ttm'))}</td>"
            f"<td>{_f(row.get('roe_ttm'))}</td>"
            f"<td>{_f(row.get('momentum_3m'))}</td>"
            f"<td>{_f(row.get('volatility_3m'))}</td>"
            f"<td>{_f(row.get('north_bound_net_inflow_30d'))}</td>"
            f"<td>{_f(row.get('margin_buy_intensity'))}</td>"
            f"<td><b>{row['weight']*100:.1f}%</b></td>"
            f"</tr>"
        )

    # ── System 2 agent rows & radars ──
    agent_rows = []
    radar_placeholders = []
    radar_scripts_parts = []
    agent_score_lookup: dict[str, dict] = {
        row["ticker"]: row for row in s2.get("agent_scores_table", [])
    }
    strat_lookup: dict[str, dict] = {
        s.get("ticker", ""): s for s in s2.get("strategies", [])
    }

    for i, row in enumerate(s2.get("agent_scores_table", [])):
        t = row.get("ticker", "")
        composite = row.get("composite", 0)
        rating = row.get("rating") or "—"
        strat = strat_lookup.get(t, {})
        weight = row.get("weight", 0)
        agent_rows.append(
            f"<tr>"
            f"<td><b>{t}</b></td>"
            f"<td>{_score_bar(row.get('valuation_score',0))}</td>"
            f"<td>{_score_bar(row.get('momentum_score',0))}</td>"
            f"<td>{_score_bar(-row.get('risk_score',0))}</td>"  # risk: lower=better
            f"<td>{_score_bar(row.get('quality_score',0))}</td>"
            f"<td>{_score_bar(row.get('sentiment_score',0))}</td>"
            f"<td>{_score_bar(composite)}</td>"
            f"<td>{_rating_tag(rating)}</td>"
            f"<td>{weight*100:.1f}%</td>"
            f"</tr>"
        )

        # Radar chart
        radar_id = f"radar_{i}"
        radar_placeholders.append(
            f'<div class="radar-item"><h4>{t}</h4>'
            f'<div id="{radar_id}" style="width:200px;height:180px"></div></div>'
        )
        val_scores = [
            round(row.get("valuation_score", 0) or 0, 3),
            round(row.get("momentum_score", 0) or 0, 3),
            round(-(row.get("risk_score", 0) or 0), 3),
            round(row.get("quality_score", 0) or 0, 3),
            round(row.get("sentiment_score", 0) or 0, 3),
        ]
        # 映射到 0-100 分
        mapped = [int((v + 1) * 50) for v in val_scores]
        radar_scripts_parts.append(f"""
(function(){{
  var c = echarts.init(document.getElementById('{radar_id}'));
  c.setOption({{
    backgroundColor:'#161b22',
    radar:{{ indicator:[
      {{name:'估值',max:100}},{{name:'动量',max:100}},
      {{name:'风控',max:100}},{{name:'质量',max:100}},{{name:'情绪',max:100}}
    ], radius:65, splitLine:{{lineStyle:{{color:'#30363d'}}}},
    name:{{textStyle:{{color:'#8b949e',fontSize:10}}}}}},
    series:[{{type:'radar',data:[{{value:{json.dumps(mapped)},
      areaStyle:{{color:'rgba(63,185,80,0.2)'}},
      lineStyle:{{color:'#3fb950'}},
      itemStyle:{{color:'#3fb950'}}}}]}}]
  }});
}})();""")

    # ── System 3 backtest rows ──
    backtest_rows = []
    validation_lookup: dict[str, dict] = {}
    for v in s3.get("validations", []):
        ticker = v.get("ticker", "")
        validation_lookup[ticker] = v
        wr = v.get("win_rate", None)
        mr = v.get("expected_monthly_return", None)
        ar = v.get("ann_return_1y", None)
        ml = v.get("max_monthly_loss", None)
        status = v.get("validation_status", "—")
        def _p(v2): return f"{v2*100:.1f}%" if v2 is not None and v2 == v2 else "—"
        backtest_rows.append(
            f"<tr><td><b>{ticker}</b></td>"
            f"<td>{_p(wr)}</td><td>{_p(mr)}</td><td>{_p(ar)}</td>"
            f"<td style='color:#f85149'>{_p(ml)}</td>"
            f"<td>{_validation_tag(status)}</td></tr>"
        )

    # ── Summary cards ──
    acceptable = sum(1 for v in s3.get("validations", [])
                     if "ACCEPTABLE" in str(v.get("validation_status", "")).upper())
    watchlist  = sum(1 for v in s3.get("validations", [])
                     if "WATCHLIST"  in str(v.get("validation_status", "")).upper())
    avoid      = sum(1 for v in s3.get("validations", [])
                     if "AVOID"      in str(v.get("validation_status", "")).upper())
    avg_wr = np.nanmean([v.get("win_rate") for v in s3.get("validations", [])
                         if v.get("win_rate") is not None])
    buy_count = sum(1 for r in s2.get("agent_scores_table", [])
                    if "BUY" in str(r.get("rating", "")).upper())

    summary_cards = "".join([
        f'<div class="card"><div class="card-title">✅ 回测通过</div>'
        f'<div class="card-val" style="color:#3fb950">{acceptable}</div>'
        f'<div class="card-title" style="margin-top:4px">只股票</div></div>',
        f'<div class="card"><div class="card-title">Agent 评级 BUY</div>'
        f'<div class="card-val" style="color:#58a6ff">{buy_count}</div>'
        f'<div class="card-title" style="margin-top:4px">只股票</div></div>',
        f'<div class="card"><div class="card-title">平均1年胜率</div>'
        f'<div class="card-val">{avg_wr*100:.1f}%</div>'
        f'<div class="card-title" style="margin-top:4px">候选组合</div></div>',
    ])

    # ── Final merged rows ──
    final_rows = []
    # 综合排序：回测通过 > 评级 BUY > composite score
    all_tickers = s1["final_tickers"]
    def _final_sort_key(ticker):
        v = validation_lookup.get(ticker, {})
        a = agent_score_lookup.get(ticker, {})
        vs = 0 if "ACCEPTABLE" in str(v.get("validation_status","")).upper() else (
             1 if "WATCHLIST"  in str(v.get("validation_status","")).upper() else 2)
        rs = 0 if "BUY" in str(a.get("rating","")).upper() else 1
        cs = -(a.get("composite", 0) or 0)
        return (vs, rs, cs)

    ranked_tickers = sorted(all_tickers, key=_final_sort_key)
    for rank, ticker in enumerate(ranked_tickers, 1):
        v  = validation_lookup.get(ticker, {})
        a  = agent_score_lookup.get(ticker, {})
        w  = s1["final_weights"].get(ticker, 0)
        wr = v.get("win_rate")
        ar = v.get("ann_return_1y")
        def _p(v2): return f"{v2*100:.1f}%" if v2 is not None and v2 == v2 else "—"
        final_rows.append(
            f"<tr><td>{rank}</td><td><b>{ticker}</b></td>"
            f"<td><b>{w*100:.1f}%</b></td>"
            f"<td>{_rating_tag(a.get('rating','—'))}</td>"
            f"<td>{_validation_tag(v.get('validation_status','—'))}</td>"
            f"<td>{_score_bar(a.get('composite',0))}</td>"
            f"<td>{_p(wr)}</td><td>{_p(ar)}</td>"
            f"</tr>"
        )

    return _HTML.format(
        as_of=as_of,
        model_name=regime["model"],
        regime_badge_cls=regime_badge,
        regime_label=regime_label_map.get(regime["name"], regime["name"]),
        sub_model=regime["model"],
        final_n=len(s1["final_tickers"]),
        factor_rows="\n".join(factor_rows),
        agent_rows="\n".join(agent_rows),
        radar_placeholders="\n".join(radar_placeholders),
        radar_scripts="".join(radar_scripts_parts),
        backtest_rows="\n".join(backtest_rows),
        summary_cards=summary_cards,
        final_rows="\n".join(final_rows),
        gen_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    as_of = date.fromisoformat(args.date)

    out_dir = args.out if args.out else ROOT / "reports" / "demo" / str(as_of)
    if args.out and not args.out.is_absolute():
        out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    lgbm_path = args.lgbm_model if args.lgbm_model.is_absolute() else ROOT / args.lgbm_model
    args.lgbm_model = lgbm_path  # absolute

    t_total = time.monotonic()
    print(f"\n{'='*60}")
    print(f"  QuantMind 全系统演示  |  日期: {as_of}  |  Top-{args.top}")
    print(f"{'='*60}")

    # ── 三系统串行执行 ──
    s1 = system1_factor_ranking(as_of, args, out_dir)
    s2 = system2_investment_analysis(as_of, s1["final_tickers"], s1["final_weights"], args, out_dir)
    s3 = system3_backtest_validation(as_of, s1["final_tickers"], s2["strategies"], out_dir)

    # ── 生成最终 HTML 报告 ──
    print(f"\n{'═'*60}")
    print("  生成最终投资报告...")
    t = _Timer("HTML 报告")
    html = build_html_report(s1, s2, s3, args)
    report_path = out_dir / "final_investment_report.html"
    report_path.write_text(html, encoding="utf-8")
    t.done(str(report_path))

    total = time.monotonic() - t_total
    print(f"\n{'='*60}")
    print(f"  ✅ 全系统演示完成！  总耗时: {total:.1f}s")
    print(f"  📄 报告: {report_path}")
    print(f"{'='*60}\n")

    # 打印最终推荐摘要
    print("最终推荐（按综合评分排序）：")
    agent_lookup = {r["ticker"]: r for r in s2.get("agent_scores_table", [])}
    val_lookup   = {v.get("ticker",""):v for v in s3.get("validations", [])}
    for i, ticker in enumerate(s1["final_tickers"][:args.top], 1):
        a = agent_lookup.get(ticker, {})
        v = val_lookup.get(ticker, {})
        w = s1["final_weights"].get(ticker, 0)
        print(f"  {i:2d}. {ticker}  权重={w*100:.1f}%  "
              f"评级={a.get('rating','—'):6s}  "
              f"回测={v.get('validation_status','—'):10s}  "
              f"胜率={v.get('win_rate',float('nan'))*100 if v.get('win_rate') else 0:.0f}%")


if __name__ == "__main__":
    main()
