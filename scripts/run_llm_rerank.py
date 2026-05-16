#!/usr/bin/env python3
"""Phase 4.1 — LLM Listwise Reranker 演示.

流程：
  1. 加载最新截面面板数据
  2. 加载 models/lgbm_ranker.pkl，获取 direction（auto_flip 信息）
  3. 用 LGBM 对该截面打分，取 direction 修正后的 Top-N
  4. 计算 SHAP 因子贡献（LightGBM built-in pred_contrib）
  5. 准备 RerankCandidate（附加因子值 + SHAP）
  6. 调用 LLMListwiseReranker.rerank() → Top-K
  7. 输出 reports/llm_rerank_demo.html（含排名对比、SHAP、组合分析）

用法::

    python scripts/run_llm_rerank.py
    python scripts/run_llm_rerank.py --provider deepseek --model deepseek-chat
    python scripts/run_llm_rerank.py --as-of 2024-03-31
    python scripts/run_llm_rerank.py --lgbm-top 30 --llm-top 15
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"

DEFAULT_PANEL  = FEATURES_DIR / "csi300_2019Q1_2024Q2.parquet"
DEFAULT_MODEL  = MODELS_DIR / "lgbm_ranker.pkl"
DEFAULT_META   = MODELS_DIR / "lgbm_ranker.meta.json"
DEFAULT_REPORT = REPORTS_DIR / "llm_rerank_demo.html"

LLM_FACTOR_COLS = [
    "pe_ttm", "pb", "roe_ttm", "accruals",
    "distance_to_52w_high", "momentum_6m", "volatility_3m",
]


# ============================================================================
# HTML 报告
# ============================================================================

_CSS = """
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
    margin: 0; padding: 24px; background: #f8f9fa; color: #333;
  }
  h1 { color: #1a252f; border-bottom: 4px solid #8e44ad; padding-bottom: 12px; margin-top: 0; }
  h2 { color: #1a252f; margin-top: 28px; }
  .stats-row { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0; }
  .stat-card {
    background: white; border-radius: 10px; padding: 18px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); min-width: 140px; text-align: center;
  }
  .stat-val { font-size: 2em; font-weight: 700; color: #8e44ad; line-height: 1.1; }
  .stat-val.green  { color: #27ae60; }
  .stat-val.blue   { color: #2980b9; }
  .stat-val.orange { color: #e67e22; }
  .stat-val.red    { color: #e74c3c; }
  .stat-lbl { font-size: 0.82em; color: #777; margin-top: 4px; }
  .panel-info {
    background: white; border-radius: 10px; padding: 16px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px;
    border-left: 5px solid #8e44ad;
  }
  .panel-info p { margin: 6px 0; font-size: 0.9em; color: #555; }
  .thesis-box {
    background: #f0f7ff; border-radius: 10px; padding: 18px 22px;
    border-left: 5px solid #2980b9; margin-bottom: 20px;
  }
  .thesis-box h3 { margin-top: 0; color: #2980b9; }
  .risk-box {
    background: #fff8f0; border-radius: 10px; padding: 18px 22px;
    border-left: 5px solid #e67e22; margin-bottom: 20px;
  }
  .risk-box h3 { margin-top: 0; color: #e67e22; }
  .risk-box li { margin: 6px 0; }
  table {
    width: 100%; border-collapse: collapse;
    background: white; border-radius: 10px; overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }
  thead tr { background: #1a252f; color: white; }
  th { padding: 12px 10px; text-align: center; font-size: 12px; font-weight: 600; }
  td { padding: 9px 10px; text-align: center; font-size: 12px; border-bottom: 1px solid #f0f0f0; }
  tr:hover td { background: #fafafa; }
  .rank-up   { color: #27ae60; font-weight: 700; }
  .rank-down { color: #e74c3c; font-weight: 700; }
  .rank-same { color: #888; }
  .fallback-badge {
    display: inline-block; padding: 2px 8px; background: #e67e22;
    color: white; border-radius: 10px; font-size: 11px;
  }
  .reason-cell { text-align: left; max-width: 220px; font-size: 11px; color: #555; }
  .shap-cell   { text-align: left; max-width: 180px; font-size: 11px; color: #2980b9; }
  footer { margin-top: 32px; color: #999; font-size: 0.8em; text-align: center; }
</style>
"""


def _rank_arrow(lgbm_rank: int, llm_rank: int) -> str:
    diff = lgbm_rank - llm_rank
    if diff > 3:
        return f'<span class="rank-up">↑{diff}</span>'
    elif diff < -3:
        return f'<span class="rank-down">↓{abs(diff)}</span>'
    else:
        return f'<span class="rank-same">→{diff:+d}</span>'


def _fmt_factor(val: float, pct: bool = False, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<span style="color:#ccc">—</span>'
    if pct:
        return f"{val * 100:.{decimals}f}%"
    return f"{val:.{decimals}f}"


def _shap_html(shap_vals: dict[str, float]) -> str:
    if not shap_vals:
        return '<span style="color:#ccc">—</span>'
    top2 = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
    parts = []
    for k, v in top2:
        color = "#27ae60" if v > 0 else "#e74c3c"
        parts.append(f'<span style="color:{color}">{k}({v:+.3f})</span>')
    return ", ".join(parts)


def build_html_report(
    results,
    candidates,
    as_of_date: str,
    provider: str,
    model: str,
    lgbm_top: int,
    meta: dict,
    elapsed_s: float,
    shap_computed: bool,
) -> str:
    is_any_fallback = any(r.is_fallback for r in results)
    cand_map = {c.ticker: c for c in candidates}

    # 提取组合分析（从第一个有效结果取）
    portfolio_thesis = ""
    risk_warnings: list[str] = []
    for r in results:
        if r.portfolio_thesis:
            portfolio_thesis = r.portfolio_thesis
            risk_warnings = r.risk_warnings
            break

    parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="UTF-8">',
        "<title>QuantMind — LLM Rerank Demo</title>",
        _CSS,
        "</head><body>",
        "<h1>🤖 QuantMind — LLM Listwise Reranker 演示 (Phase 4.1)</h1>",
    ]

    # 信息面板
    direction = meta.get("direction", 1)
    regime_note = meta.get("regime_note", "")
    parts.append('<div class="panel-info">')
    parts.append(f"<p><b>截止日期：</b>{as_of_date} &nbsp;|&nbsp; "
                 f"<b>LGBM 粗排：</b>Top-{lgbm_top} &nbsp;|&nbsp; <b>LLM 精排：</b>Top-{len(results)}</p>")
    parts.append(f"<p><b>Provider：</b>{provider} &nbsp;|&nbsp; <b>Model：</b>{model} &nbsp;|&nbsp; "
                 f"<b>推理耗时：</b>{elapsed_s:.1f}s</p>")
    parts.append(f"<p><b>LGBM direction：</b>{direction:+d} &nbsp;|&nbsp; "
                 f"<b>有效 IC_IR：</b>{meta.get('effective_ic_ir', 0):+.4f} &nbsp;|&nbsp; "
                 f"<b>SHAP：</b>{'✅ 已计算' if shap_computed else '—'}</p>")
    if regime_note:
        parts.append(f'<p style="color:#e67e22"><b>⚠ Regime：</b>{regime_note[:120]}…</p>')
    if is_any_fallback:
        parts.append('<p><span class="fallback-badge">⚠ LLM JSON 解析失败，已降级为 LGBM 顺序</span></p>')
    parts.append("</div>")

    # 组合投资逻辑
    if portfolio_thesis:
        parts.append('<div class="thesis-box">')
        parts.append("<h3>📊 组合投资逻辑（LLM 分析）</h3>")
        parts.append(f"<p>{portfolio_thesis}</p>")
        parts.append("</div>")

    # 风险提示
    if risk_warnings:
        parts.append('<div class="risk-box">')
        parts.append("<h3>⚠ 主要风险</h3><ul>")
        for rw in risk_warnings:
            parts.append(f"<li>{rw}</li>")
        parts.append("</ul></div>")

    # 统计卡片
    parts.append('<div class="stats-row">')
    def card(val, label, cls=""):
        return (f'<div class="stat-card"><div class="stat-val {cls}">{val}</div>'
                f'<div class="stat-lbl">{label}</div></div>')

    rank_changes = [c.lgbm_rank - r.rank
                    for r in results if r.ticker in cand_map
                    for c in [cand_map[r.ticker]]]
    avg_change = float(np.mean(rank_changes)) if rank_changes else 0
    big_moves = sum(1 for d in rank_changes if abs(d) > 5)

    parts += [
        card(f"{len(results)}", "LLM 精排数量", "green"),
        card(f"{lgbm_top}", "LGBM 粗排数量", "blue"),
        card(f"{avg_change:+.1f}", "平均排名变化", "orange" if avg_change > 0 else ""),
        card(f"{big_moves}", "大幅移位(>5位)", "orange"),
        card(f"{elapsed_s:.1f}s", "LLM 推理耗时", "blue"),
        card("✓" if not is_any_fallback else "⚠",
             "LLM解析状态", "green" if not is_any_fallback else "red"),
    ]
    parts.append("</div>")

    # 排名对比表
    shap_header = "<th>SHAP主导因子</th>" if shap_computed else ""
    parts.append(f"""
    <table>
      <thead><tr>
        <th>LLM</th><th>代码</th><th>LGBM</th><th>变化</th>
        <th>PE</th><th>PB</th><th>ROE%</th><th>Accruals</th>
        <th>52W高%</th><th>动量6M%</th><th>波动3M%</th>
        {shap_header}<th>LLM 理由</th>
      </tr></thead><tbody>
    """)

    for r in results:
        c = cand_map.get(r.ticker)
        reason_html = (f'<span class="reason-cell">{r.reason}</span>'
                       if r.reason else '<span style="color:#ccc">—</span>')
        shap_td = (f'<td class="shap-cell">{_shap_html(c.shap_values if c else {})}</td>'
                   if shap_computed else "")
        fallback_tag = ' <span class="fallback-badge">fb</span>' if r.is_fallback else ""

        parts.append(f"""
        <tr>
          <td><b>{r.rank}</b>{fallback_tag}</td>
          <td><b>{r.ticker}</b></td>
          <td>{r.lgbm_rank}</td>
          <td>{_rank_arrow(r.lgbm_rank, r.rank)}</td>
          <td>{_fmt_factor(c.pe_ttm if c else float("nan"))}</td>
          <td>{_fmt_factor(c.pb if c else float("nan"))}</td>
          <td>{_fmt_factor(c.roe_ttm if c else float("nan"), pct=True)}</td>
          <td>{_fmt_factor(c.accruals if c else float("nan"), decimals=3)}</td>
          <td>{_fmt_factor(c.distance_to_52w_high if c else float("nan"), pct=True)}</td>
          <td>{_fmt_factor(c.momentum_6m if c else float("nan"), pct=True)}</td>
          <td>{_fmt_factor(c.volatility_3m if c else float("nan"), pct=True)}</td>
          {shap_td}
          <td>{reason_html}</td>
        </tr>""")

    parts.append("</tbody></table>")
    parts.append(f"""
    <footer>QuantMind Phase 4.1 — LLM Listwise Reranker |
      scripts/run_llm_rerank.py | {as_of_date}</footer>
    </body></html>""")
    return "\n".join(parts)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.1 — LLM Listwise Reranker 演示")
    p.add_argument("--panel", default=None)
    p.add_argument("--model-path", default=None, dest="model_path")
    p.add_argument("--as-of", default=None, dest="as_of")
    p.add_argument("--provider", default="ollama")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--lgbm-top", type=int, default=50, dest="lgbm_top")
    p.add_argument("--llm-top", type=int, default=30, dest="llm_top")
    p.add_argument("--no-shap", action="store_true", dest="no_shap",
                   help="跳过 SHAP 计算（加速调试）")
    p.add_argument("--report-out", default=None, dest="report_out")
    return p.parse_args()


def main() -> None:
    import time
    from quantmind.models.lgbm_ranker import LGBMRankerModel
    from quantmind.models.llm_reranker import LLMListwiseReranker, RerankCandidate

    args = parse_args()

    print(f"\n{'='*65}")
    print("QuantMind — Phase 4.1  LLM Listwise Reranker 演示")
    print(f"{'='*65}")

    panel_path = Path(args.panel) if args.panel else DEFAULT_PANEL
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")
    model_path = Path(args.model_path) if args.model_path else DEFAULT_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_factor_model.py first.")
    report_out = Path(args.report_out) if args.report_out else DEFAULT_REPORT
    report_out = report_out if report_out.is_absolute() else PROJECT_ROOT / report_out

    print(f"面板    : {panel_path}")
    print(f"模型    : {model_path}")
    print(f"Provider: {args.provider}  Model: {args.model}")

    meta: dict = {}
    if DEFAULT_META.exists():
        with open(DEFAULT_META, encoding="utf-8") as f:
            meta = json.load(f)

    # [1] 加载面板
    print("\n[1/6] 加载面板 …")
    panel = pd.read_parquet(panel_path)
    dates = sorted(panel.index.get_level_values("as_of").unique())
    as_of = pd.Timestamp(args.as_of) if args.as_of else dates[-1]
    as_of_str = str(as_of)[:10]
    xs = panel.xs(as_of, level="as_of")
    print(f"  截面日期: {as_of_str}  股票数: {len(xs)}")

    # [2] 加载模型
    print("\n[2/6] 加载 LGBM 模型 …")
    lgbm_model = LGBMRankerModel.load(model_path)
    feature_cols = meta.get("feature_cols") or lgbm_model._feature_names or []
    feature_cols = [f for f in feature_cols if f in xs.columns]
    direction = getattr(lgbm_model, "direction", meta.get("direction", 1))
    print(f"  特征: {len(feature_cols)} 个  direction={direction:+d}")

    # [3] LGBM 推断
    print(f"\n[3/6] LGBM 推断，取 Top-{args.lgbm_top} …")
    X_all = xs[feature_cols].fillna(0.0).values.astype("float32")
    scores_all = lgbm_model.predict(X_all)
    rank_idx = np.argsort(scores_all)[::-1][:args.lgbm_top]
    top_tickers = [str(t) for t in xs.index[rank_idx].tolist()]
    top_scores  = scores_all[rank_idx]
    X_top = X_all[rank_idx]
    print(f"  分数范围: [{top_scores.min():.4f}, {top_scores.max():.4f}]")

    # [4] SHAP 计算
    shap_computed = False
    shap_dict: dict[str, dict[str, float]] = {}
    if not args.no_shap:
        print("\n[4/6] 计算 SHAP 因子贡献 …")
        try:
            shap_dict = lgbm_model.explain(X_top, tickers=top_tickers)
            shap_computed = True
            print(f"  ✅ SHAP 完成，覆盖 {len(shap_dict)} 只股票")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠  SHAP 计算失败（{e}），跳过")
    else:
        print("\n[4/6] 跳过 SHAP（--no-shap）")

    # [5] 构造候选
    print(f"\n[5/6] 准备 LLM 候选（附加因子数据）…")

    def safe_get(row: pd.Series, col: str) -> float:
        v = row.get(col, float("nan")) if hasattr(row, "get") else float("nan")
        try:
            fv = float(v)
            return fv if not np.isnan(fv) else float("nan")
        except (TypeError, ValueError):
            return float("nan")

    candidates: list[RerankCandidate] = []
    for lgbm_rank, (ticker, score) in enumerate(zip(top_tickers, top_scores), start=1):
        row = xs.loc[ticker] if ticker in xs.index else pd.Series(dtype=float)
        candidates.append(RerankCandidate(
            ticker=ticker,
            lgbm_score=float(score),
            lgbm_rank=lgbm_rank,
            pe_ttm=safe_get(row, "pe_ttm"),
            pb=safe_get(row, "pb"),
            roe_ttm=safe_get(row, "roe_ttm"),
            accruals=safe_get(row, "accruals"),
            distance_to_52w_high=safe_get(row, "distance_to_52w_high"),
            momentum_6m=safe_get(row, "momentum_6m"),
            volatility_3m=safe_get(row, "volatility_3m"),
            shap_values=shap_dict.get(ticker, {}),
        ))
    print(f"  候选: {len(candidates)} 只  SHAP 已填充: {sum(1 for c in candidates if c.shap_values)}")

    # [6] LLM 重排
    print(f"\n[6/6] LLM 重排（{args.provider}/{args.model}，目标 Top-{args.llm_top}）…")
    reranker = LLMListwiseReranker(
        provider=args.provider, model=args.model, batch_size=args.lgbm_top
    )
    t0 = time.monotonic()
    results = reranker.rerank(candidates, top_n=args.llm_top, as_of_date=as_of_str)
    elapsed = time.monotonic() - t0

    is_fallback = any(r.is_fallback for r in results)
    print(f"  完成：{len(results)} 只  耗时 {elapsed:.1f}s  "
          f"解析: {'✅ 成功' if not is_fallback else '⚠ 降级'}")

    # 第一个结果的组合分析
    thesis = next((r.portfolio_thesis for r in results if r.portfolio_thesis), "")
    risks  = next((r.risk_warnings for r in results if r.risk_warnings), [])
    if thesis:
        print(f"\n  组合逻辑: {thesis[:80]}…")
    if risks:
        print(f"  风险提示: {', '.join(risks[:2])}…")

    print(f"\n  Top-10 预览（LLM排名 → LGBM排名）：")
    for r in results[:10]:
        diff = r.lgbm_rank - r.rank
        arrow = f"↑{diff}" if diff > 0 else (f"↓{abs(diff)}" if diff < 0 else "→0")
        print(f"    LLM#{r.rank:2d} LGBM#{r.lgbm_rank:2d} {arrow:4s}  {r.ticker}"
              f"  {(r.reason or '')[:28]}")

    # 构建报告
    report_out.parent.mkdir(parents=True, exist_ok=True)
    html = build_html_report(
        results=results, candidates=candidates,
        as_of_date=as_of_str, provider=args.provider, model=args.model,
        lgbm_top=args.lgbm_top, meta=meta, elapsed_s=elapsed,
        shap_computed=shap_computed,
    )
    report_out.write_text(html, encoding="utf-8")

    print(f"\n{'='*65}")
    print(f"🎉  Phase 4.1 完成")
    print(f"{'='*65}")
    print(f"  报告: {report_out}")
    print(f"  LGBM Top-{args.lgbm_top} → LLM Top-{len(results)}  "
          f"SHAP: {'✅' if shap_computed else '—'}  "
          f"组合分析: {'✅' if thesis else '—'}")


if __name__ == "__main__":
    main()
