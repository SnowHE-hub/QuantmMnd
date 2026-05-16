"""scripts/run_agent_research.py — Agent 研究流程 CLI 入口.

用法：
    python scripts/run_agent_research.py --ticker 600519.SH --as-of 2024-06-30
    python scripts/run_agent_research.py --ticker 000858.SZ --provider ollama --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# 确保项目根目录在 sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantMind Agent 研究流程 — 自动生成投资研究报告"
    )
    parser.add_argument("--ticker", required=True, help="股票代码（如 600519.SH）")
    parser.add_argument(
        "--as-of", default=None,
        help="数据截止日期 YYYY-MM-DD（默认今天）"
    )
    parser.add_argument(
        "--query", default=None,
        help="自定义查询文本（默认自动生成）"
    )
    parser.add_argument(
        "--provider", default="ollama",
        help="LLM 提供商（ollama / deepseek / openai）"
    )
    parser.add_argument(
        "--model", default="qwen2.5:7b",
        help="模型名称（默认 qwen2.5:7b）"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="最大 Critic 迭代次数（默认 3）"
    )
    parser.add_argument(
        "--output-dir", default="reports",
        help="HTML 报告输出目录"
    )
    return parser.parse_args()


def build_html_report(state, ticker: str, as_of: date) -> str:
    """把 AgentState 渲染为 HTML 报告."""
    report = state.report
    fa = state.fundamentals.get(ticker)
    ta = state.technicals.get(ticker)
    sa = state.sentiments.get(ticker)
    feedback = state.critic_feedback

    def _fmt_json(d) -> str:
        try:
            return json.dumps(d, ensure_ascii=False, indent=2)
        except Exception:
            return str(d)

    def _section(title: str, content: str, color: str = "#1a1a2e") -> str:
        return f"""
<div style="margin:20px 0;padding:16px;border-radius:8px;background:#fff;box-shadow:0 2px 8px rgba(0,0,0,0.1)">
  <h2 style="color:{color};margin:0 0 12px 0;font-size:1.2em">{title}</h2>
  <div style="line-height:1.6;color:#333">{content}</div>
</div>"""

    # 评级颜色
    rating_str = report.rating.value if report else "hold"
    rating_colors = {
        "strong_buy": "#00b894", "buy": "#00cec9",
        "hold": "#fdcb6e", "sell": "#e17055", "strong_sell": "#d63031"
    }
    rating_color = rating_colors.get(rating_str, "#636e72")

    # 质量分数
    quality = feedback.overall_quality_score if feedback else 0
    quality_color = "#00b894" if quality >= 7 else ("#fdcb6e" if quality >= 5 else "#d63031")

    # Issue 表格
    issue_rows = ""
    if feedback and feedback.issues:
        for issue in feedback.issues:
            sev_color = {"critical": "#d63031", "major": "#e17055", "minor": "#fdcb6e"}.get(
                issue.severity.value, "#636e72"
            )
            issue_rows += f"""<tr>
<td style="color:{sev_color};font-weight:bold">{issue.severity.value}</td>
<td>{issue.type.value}</td>
<td>{issue.description}</td>
<td>{issue.fix_action.instruction if issue.fix_action else ''}</td>
</tr>"""

    # 执行历史
    history_rows = ""
    for h in state.history:
        status_icon = "✅" if h.success else "❌"
        history_rows += f"<tr><td>{status_icon}</td><td>{h.agent.value}</td><td>{h.started_at.strftime('%H:%M:%S')}</td><td>{h.input_summary}</td></tr>"

    # 财务比率
    ratios_html = ""
    if fa and fa.ratios:
        for k, v in fa.ratios.items():
            ratios_html += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantMind — {ticker} 投资研究报告</title>
<style>
  body{{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f6fa;margin:0;padding:20px}}
  h1{{color:#1a1a2e}} table{{width:100%;border-collapse:collapse}}
  th{{background:#1a1a2e;color:#fff;padding:8px;text-align:left}}
  td{{padding:7px;border-bottom:1px solid #eee}}
  .stat-card{{display:inline-block;padding:12px 24px;border-radius:8px;margin:6px;text-align:center;min-width:120px}}
  pre{{background:#f8f9fa;padding:12px;border-radius:6px;overflow-x:auto;font-size:0.85em}}
</style>
</head>
<body>
<h1>📊 {ticker} 投资研究报告</h1>
<p style="color:#636e72">生成时间：{date.today()}  截止日期：{as_of}  迭代次数：{state.iteration_count}</p>

<div style="margin:16px 0">
  <span class="stat-card" style="background:{rating_color};color:#fff">
    <div style="font-size:0.8em">综合评级</div>
    <div style="font-size:1.4em;font-weight:bold">{rating_str.upper()}</div>
  </span>
  <span class="stat-card" style="background:{quality_color};color:#fff">
    <div style="font-size:0.8em">质量分数</div>
    <div style="font-size:1.4em;font-weight:bold">{quality:.1f}/10</div>
  </span>
  <span class="stat-card" style="background:#6c5ce7;color:#fff">
    <div style="font-size:0.8em">Critic</div>
    <div style="font-size:1.4em;font-weight:bold">{'✅通过' if (feedback and feedback.passed) else '⚠️待改进'}</div>
  </span>
  {'<span class="stat-card" style="background:#00b894;color:#fff"><div style="font-size:0.8em">置信度</div><div style="font-size:1.4em;font-weight:bold">' + str(round(report.confidence,2)) + '</div></span>' if report else ''}
</div>

{_section("执行摘要", '<ul>' + ''.join(f'<li>{s}</li>' for s in (report.executive_summary if report else [])) + '</ul>')}

{_section("基本面分析", f'''
<h3>盈利质量</h3><p>{fa.profitability_analysis if fa else "N/A"}</p>
<h3>偿债能力</h3><p>{fa.solvency_analysis if fa else "N/A"}</p>
<h3>成长分析</h3><p>{fa.growth_analysis if fa else "N/A"}</p>
<h3>估值分析</h3><p>{fa.valuation_analysis if fa else "N/A"}</p>
<h3>财务比率</h3><table><tr><th>指标</th><th>值</th></tr>{ratios_html}</table>
<h3>风险点</h3><ul>{''.join(f'<li>{r}</li>' for r in (fa.key_risks if fa else []))}</ul>
''') if fa else _section("基本面分析", "数据不可用")}

{_section("技术分析", f'''
<p><b>信号：</b><span style="color:{{"bullish":"#00b894","bearish":"#d63031"}.get(ta.signal,"#636e72") if ta else "#636e72"}">{ta.signal.upper() if ta else "N/A"}</span></p>
<p>{ta.summary if ta else "N/A"}</p>
<h3>技术指标</h3><table><tr><th>指标</th><th>值</th></tr>
{''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in (ta.indicators.items() if ta else []))}
</table>
''') if ta else _section("技术分析", "数据不可用")}

{_section("情绪分析", f'''
<p><b>情绪分数：</b>{sa.aggregated_sentiment:.2f if sa else "N/A"}</p>
<p>{sa.summary if sa else "N/A"}</p>
<h3>催化剂</h3><ul>{''.join(f'<li>{c}</li>' for c in (sa.catalysts if sa else []))}</ul>
<h3>风险</h3><ul>{''.join(f'<li>{r}</li>' for r in (sa.risks if sa else []))}</ul>
''') if sa else _section("情绪分析", "数据不可用")}

{f'''<div style="margin:20px 0;padding:16px;border-radius:8px;background:#fff;box-shadow:0 2px 8px rgba(0,0,0,0.1)">
<h2 style="color:#d63031;margin:0 0 12px 0">⚠️ Critic 审查结果</h2>
<p>通过：{"✅" if feedback.passed else "❌"} | 质量分：{feedback.overall_quality_score:.1f} | {feedback.approval_message}</p>
<table><tr><th>严重度</th><th>类型</th><th>描述</th><th>修复建议</th></tr>{issue_rows}</table>
</div>''' if feedback else ''}

{_section("风险提示", '<ul>' + ''.join(f'<li>{r}</li>' for r in (report.risk_warnings if report else [])) + '</ul>', "#d63031")}

{_section("执行历史", f'<table><tr><th>状态</th><th>Agent</th><th>时间</th><th>摘要</th></tr>{history_rows}</table>')}

<footer style="text-align:center;padding:20px;color:#b2bec3;font-size:0.85em">
  QuantMind Phase 4 | Multi-Agent Research System | Powered by LangGraph + Ollama
</footer>
</body>
</html>"""
    return html


def main() -> None:
    args = parse_args()

    # 解析日期
    as_of: date
    if args.as_of:
        from datetime import datetime
        as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    else:
        as_of = date.today()

    logger.info(f"Starting research: ticker={args.ticker}, as_of={as_of}, provider={args.provider}/{args.model}")

    # 运行研究流程
    from quantmind.agents.orchestrator import run_research
    state = run_research(
        ticker=args.ticker,
        as_of=as_of,
        query_text=args.query,
        provider=args.provider,
        model=args.model,
        max_iterations=args.max_iterations,
    )

    # 生成 HTML 报告
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"agent_research_{args.ticker.replace('.', '_')}_{as_of}.html"
    output_path = output_dir / filename

    html = build_html_report(state, args.ticker, as_of)
    output_path.write_text(html, encoding="utf-8")

    # 控制台输出
    print(f"\n{'='*60}")
    print(f"QuantMind Agent Research — {args.ticker}")
    print(f"{'='*60}")
    if state.report:
        r = state.report
        print(f"评级：{r.rating.value.upper()}")
        print(f"置信度：{r.confidence:.2f}")
        print(f"迭代次数：{state.iteration_count}")
        if r.executive_summary:
            print("\n执行摘要：")
            for s in r.executive_summary:
                print(f"  • {s}")
        if r.risk_warnings:
            print("\n风险提示：")
            for w in r.risk_warnings[:3]:
                print(f"  ⚠️  {w}")
    if state.critic_feedback:
        fb = state.critic_feedback
        print(f"\nCritic: {'✅ 通过' if fb.passed else '❌ 未通过'} (质量分 {fb.overall_quality_score:.1f}/10)")
    print(f"\n报告已保存 → {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
