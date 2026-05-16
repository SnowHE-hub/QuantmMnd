"""quantmind.agents.report_agent — ReportAgent.

整合所有 Agent 输出，生成9章节结构化投资报告，并附4张 Plotly 图表。
同时输出 .md 和 .html 两个版本。
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

from quantmind.agents.base import BaseAgent
from quantmind.core.state import (
    AgentState,
    AgentType,
    InvestmentReport,
    Recommendation,
)

__all__ = ["ReportAgent"]

_SYSTEM_PROMPT = """你是一位顶级投资银行的首席研究员。
请把以下结构化分析数据整合为一份专业、完整、可直接呈递客户的投资研究报告。

报告必须包含9个章节，每章节输出在 JSON 的对应字段中：
1. executive_summary: 执行摘要（List[str]，3-5个要点，每条≤50字）
2. company_overview: 公司概况（含主营业务、行业地位、核心竞争力）
3. financial_analysis: 财务分析（含关键比率解读，引用具体数字）
4. valuation_analysis: 估值分析（DCF结果解读 + 可比估值 + 综合判断）
5. technical_quant: 技术面与量化信号（综合技术指标和LGBM因子信号）
6. sentiment_catalysts: 情绪面与催化剂（近期重要事件和情绪分析）
7. risk_assessment: 风险评估（3-5个具体风险点，含发生概率和影响程度）
8. investment_recommendation: 投资建议（评级/目标价区间/建议持有期/仓位建议）
9. appendix: 附录（数据来源、分析方法说明）

评级规则：
- strong_buy: 基本面优秀+技术突破+情绪正面
- buy: 基本面良好，无明显负面因素
- hold: 基本面一般或已充分定价
- sell: 基本面恶化或估值过高
- strong_sell: 基本面恶化+技术破位+情绪悲观

输出严格 JSON：
{
  "rating": "strong_buy|buy|hold|sell|strong_sell",
  "target_price_low": null或数字,
  "target_price_high": null或数字,
  "confidence": 0.0-1.0,
  "horizon_days": 60,
  "executive_summary": ["要点1", "要点2", "要点3"],
  "company_overview": "公司概况文本",
  "financial_analysis": "财务分析文本",
  "valuation_analysis": "估值分析文本",
  "technical_quant": "技术量化文本",
  "sentiment_catalysts": "情绪催化剂文本",
  "risk_assessment": "风险评估文本",
  "investment_recommendation": "投资建议文本",
  "appendix": "附录文本"
}"""


class ReportAgent(BaseAgent):
    agent_type = AgentType.REPORT
    description = "报告生成 Agent：整合所有分析，生成9章节专业报告 + 4图表"
    system_prompt = _SYSTEM_PROMPT

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        output_dir: str = "reports",
        **kwargs,
    ) -> None:
        super().__init__(provider=provider, model=model, **kwargs)
        self.output_dir = Path(output_dir)

    def format_input(self, state: AgentState) -> str:
        ticker = state.query.tickers[0] if state.query.tickers else "unknown"
        as_of = state.query.as_of or date.today()

        fa = state.fundamentals.get(ticker)
        ta = state.technicals.get(ticker)
        sa = state.sentiments.get(ticker)

        parts = [
            f"股票代码：{ticker}",
            f"数据截止：{as_of}",
            f"分析期限：{state.query.horizon_days}天",
            "",
        ]

        # 基本面摘要
        if fa:
            parts.append("=== 基本面数据 ===")
            parts.append(f"财务比率：{json.dumps(fa.ratios, ensure_ascii=False)}")
            parts.append(f"DCF公允价值：{fa.dcf_value}")
            parts.append(f"盈利分析：{fa.profitability_analysis}")
            parts.append(f"估值分析：{fa.valuation_analysis}")
            parts.append(f"成长分析：{fa.growth_analysis}")
            parts.append(f"风险点：{fa.key_risks}")
            parts.append(f"综合评语：{fa.summary}")
            parts.append(f"置信度：{fa.confidence}")
        else:
            parts.append("=== 基本面数据：不可用 ===")

        # 技术面摘要
        if ta:
            parts.append("\n=== 技术面数据 ===")
            parts.append(f"信号：{ta.signal}")
            parts.append(f"技术指标：{json.dumps(ta.indicators, ensure_ascii=False)}")
            parts.append(f"LGBM量化分数：{ta.quant_score}")
            parts.append(f"技术观点：{ta.summary}")
        else:
            parts.append("\n=== 技术面数据：不可用 ===")

        # 情绪面摘要
        if sa:
            parts.append("\n=== 情绪面数据 ===")
            parts.append(f"情绪分数：{sa.aggregated_sentiment:.2f}")
            parts.append(f"催化剂：{sa.catalysts}")
            parts.append(f"风险：{sa.risks}")
            parts.append(f"情绪摘要：{sa.summary}")
        else:
            parts.append("\n=== 情绪面数据：不可用 ===")

        # Critic 审查结果
        if state.critic_feedback:
            fb = state.critic_feedback
            parts.append(f"\n=== Critic 审查 ===")
            parts.append(f"通过：{fb.passed}，质量分：{fb.overall_quality_score:.1f}")
            if fb.issues:
                parts.append(f"遗留问题：{[i.description for i in fb.issues[:3]]}")

        parts.append("\n请生成完整的9章节投资报告，输出 JSON。")
        return "\n".join(parts)

    def parse_output(self, llm_response: str) -> dict[str, Any]:
        text = llm_response.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return {}

    def _execute(self, state: AgentState) -> AgentState:
        ticker = state.query.tickers[0] if state.query.tickers else "unknown"
        as_of = state.query.as_of or date.today()

        # LLM 生成报告正文
        prompt = self.format_input(state)
        resp = self.llm_chat(prompt, max_tokens=3000)
        parsed = self.parse_output(resp.content)

        fa = state.fundamentals.get(ticker)
        ta = state.technicals.get(ticker)
        sa = state.sentiments.get(ticker)

        # 解析评级
        rating_str = parsed.get("rating", "hold")
        try:
            rating = Recommendation(rating_str)
        except ValueError:
            rating = Recommendation.HOLD

        # 目标价
        tp_low = parsed.get("target_price_low")
        tp_high = parsed.get("target_price_high")
        target_price = None
        if tp_low and tp_high:
            target_price = (float(tp_low) + float(tp_high)) / 2

        # 风险汇总
        risk_warnings: list[str] = []
        if fa:
            risk_warnings.extend(fa.key_risks[:3])
        if sa:
            risk_warnings.extend(sa.risks[:2])

        exec_summary = parsed.get("executive_summary", [])
        if not exec_summary:
            exec_summary = []
            if fa and fa.summary:
                exec_summary.append(f"基本面：{fa.summary}")
            if ta and ta.summary:
                exec_summary.append(f"技术面：{ta.summary}")
            if sa and sa.summary:
                exec_summary.append(f"情绪面：{sa.summary}")

        # 9章节 Markdown
        full_md = self._build_markdown(ticker, as_of, parsed, fa, ta, sa, rating, tp_low, tp_high)

        state.report = InvestmentReport(
            query=state.query,
            as_of=as_of,
            rating=rating,
            target_price=target_price,
            confidence=float(parsed.get("confidence", 0.5)),
            horizon_days=int(parsed.get("horizon_days", state.query.horizon_days)),
            thesis=parsed.get("investment_recommendation", ""),
            fundamental=fa,
            technical=ta,
            sentiment=sa,
            risk_warnings=risk_warnings,
            executive_summary=exec_summary,
            full_markdown=full_md,
            data_sources=["tushare_snapshot", "ollama_llm"],
            iteration_count=state.iteration_count,
        )

        # 保存文件
        self._save_report(state.report, ticker, as_of, state, parsed)

        logger.info(f"[ReportAgent] report saved: {ticker}, rating={rating.value}")
        return state

    # ── Markdown 生成 ─────────────────────────────────────────────────────────

    def _build_markdown(
        self,
        ticker: str,
        as_of: date,
        parsed: dict,
        fa, ta, sa,
        rating: Recommendation,
        tp_low, tp_high,
    ) -> str:
        today = date.today()
        tp_str = f"{tp_low}~{tp_high}" if tp_low and tp_high else "未设定"

        sections = [
            f"# {ticker} 投资研究报告",
            f"> **截止日期：{as_of}** | 生成日期：{today} | 评级：**{rating.value.upper()}** | 目标价：{tp_str}",
            "",
            "---",
            "",
            "## 一、执行摘要",
            "\n".join(f"- {s}" for s in parsed.get("executive_summary", [])),
            "",
            "## 二、公司概况",
            parsed.get("company_overview", "_数据不可用_"),
            "",
            "## 三、财务分析",
            parsed.get("financial_analysis", "_数据不可用_"),
        ]

        # 财务比率表格
        if fa and fa.ratios:
            sections.append("\n| 指标 | 值 |")
            sections.append("|------|------|")
            for k, v in fa.ratios.items():
                sections.append(f"| {k} | {v:.4f} |")

        sections += [
            "",
            "## 四、估值分析",
            parsed.get("valuation_analysis", "_数据不可用_"),
            "",
            "## 五、技术面与量化信号",
            parsed.get("technical_quant", "_数据不可用_"),
            "",
            "## 六、情绪面与催化剂",
            parsed.get("sentiment_catalysts", "_数据不可用_"),
            "",
            "## 七、风险评估",
            parsed.get("risk_assessment", "_数据不可用_"),
            "",
            "## 八、投资建议",
            f"**综合评级：{rating.value.upper()}**\n\n",
            parsed.get("investment_recommendation", "_数据不可用_"),
            "",
            "## 九、附录",
            parsed.get("appendix", "数据来源：Tushare 行情数据库 | 分析模型：QuantMind LLM Multi-Agent"),
        ]

        return "\n".join(sections)

    # ── HTML + Plotly 图表生成 ────────────────────────────────────────────────

    def _save_report(
        self,
        report: InvestmentReport,
        ticker: str,
        as_of: date,
        state: AgentState,
        parsed: dict,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        safe_ticker = ticker.replace(".", "_")

        # 保存 Markdown
        md_path = self.output_dir / f"agent_research_{safe_ticker}_{as_of}.md"
        md_path.write_text(report.full_markdown, encoding="utf-8")

        # 生成 HTML（含 Plotly 图表）
        html_path = self.output_dir / f"agent_research_{safe_ticker}_{as_of}.html"
        html = self._build_html(report, ticker, as_of, state, parsed)
        html_path.write_text(html, encoding="utf-8")

        logger.info(f"[ReportAgent] saved → {md_path.name} + {html_path.name}")

    def _build_html(
        self,
        report: InvestmentReport,
        ticker: str,
        as_of: date,
        state: AgentState,
        parsed: dict,
    ) -> str:
        """构建含 Plotly 图表的完整 HTML 报告."""
        fa = report.fundamental
        ta = report.technical
        sa = report.sentiment

        charts_js = self._build_charts_js(fa, ta, state)

        rating_colors = {
            "strong_buy": "#00b894", "buy": "#55efc4",
            "hold": "#fdcb6e", "sell": "#e17055", "strong_sell": "#d63031",
        }
        rc = rating_colors.get(report.rating.value, "#636e72")

        md_content = report.full_markdown.replace("\n", "<br>").replace("**", "").replace("# ", "<h3>").replace("## ", "<h4>")

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantMind — {ticker} 投资研究报告</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  *{{box-sizing:border-box}} body{{font-family:'PingFang SC','Microsoft YaHei',sans-serif;background:#f0f2f5;margin:0;padding:0}}
  .container{{max-width:1200px;margin:0 auto;padding:20px}}
  .header{{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:32px;border-radius:12px;margin-bottom:20px}}
  .header h1{{margin:0 0 8px 0;font-size:1.8em}} .header p{{margin:0;opacity:.8}}
  .badge{{display:inline-block;padding:6px 16px;border-radius:20px;font-weight:bold;font-size:1.1em;margin:8px 4px}}
  .card{{background:#fff;border-radius:10px;padding:20px;margin:12px 0;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
  .card h2{{color:#1a1a2e;margin:0 0 12px 0;font-size:1.1em;border-bottom:2px solid #eee;padding-bottom:8px}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
  .stat{{text-align:center;padding:12px;background:#f8f9fa;border-radius:8px}}
  .stat-val{{font-size:1.5em;font-weight:bold}} .stat-label{{font-size:.8em;color:#636e72}}
  table{{width:100%;border-collapse:collapse}} th{{background:#1a1a2e;color:#fff;padding:8px;font-size:.9em}}
  td{{padding:7px 8px;border-bottom:1px solid #eee;font-size:.9em}}
  .chart-container{{height:350px}}
  .section-badge{{display:inline-block;background:#1a1a2e;color:#fff;padding:3px 10px;border-radius:4px;font-size:.8em;margin-bottom:8px}}
  ul{{padding-left:20px;line-height:1.8}} li{{margin:4px 0}}
  @media(max-width:768px){{.grid-2,.grid-4{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="container">
  <!-- 标题栏 -->
  <div class="header">
    <h1>📊 {ticker} 投资研究报告</h1>
    <p>截止日期：{as_of} | 生成时间：{date.today()} | 迭代次数：{state.iteration_count}</p>
    <div>
      <span class="badge" style="background:{rc}">{report.rating.value.upper()}</span>
      {'<span class="badge" style="background:#6c5ce7">目标价 ' + str(parsed.get("target_price_low","?")) + '~' + str(parsed.get("target_price_high","?")) + '</span>' if parsed.get("target_price_low") else ""}
      <span class="badge" style="background:#0984e3">置信度 {report.confidence:.0%}</span>
      <span class="badge" style="background:#00cec9">持有期 {report.horizon_days}天</span>
    </div>
  </div>

  <!-- 核心指标 -->
  <div class="grid-4">
    <div class="stat"><div class="stat-val" style="color:{rc}">{report.rating.value.upper()}</div><div class="stat-label">综合评级</div></div>
    <div class="stat"><div class="stat-val">{report.confidence:.0%}</div><div class="stat-label">置信度</div></div>
    <div class="stat"><div class="stat-val">{fa.ratios.get("roe","N/A") if fa and fa.ratios else "N/A"}</div><div class="stat-label">ROE</div></div>
    <div class="stat"><div class="stat-val">{"%.2f" % sa.aggregated_sentiment if sa else "N/A"}</div><div class="stat-label">情绪分数</div></div>
  </div>

  <!-- 执行摘要 -->
  <div class="card">
    <h2>一、执行摘要</h2>
    <ul>{"".join(f"<li>{s}</li>" for s in report.executive_summary)}</ul>
  </div>

  <!-- 图表区 -->
  <div class="grid-2">
    <div class="card"><h2>股价走势</h2><div class="chart-container" id="chart-price"></div></div>
    <div class="card"><h2>财务比率 vs 同业</h2><div class="chart-container" id="chart-radar"></div></div>
    <div class="card"><h2>DCF 敏感性分析</h2><div class="chart-container" id="chart-dcf"></div></div>
    <div class="card"><h2>因子贡献（SHAP）</h2><div class="chart-container" id="chart-shap"></div></div>
  </div>

  <!-- 分析章节 -->
  <div class="grid-2">
    <div class="card">
      <h2>三、财务分析</h2>
      <p>{parsed.get("financial_analysis","数据不可用")}</p>
      {self._ratios_table(fa)}
    </div>
    <div class="card">
      <h2>四、估值分析</h2>
      <p>{parsed.get("valuation_analysis","数据不可用")}</p>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>五、技术面与量化信号</h2>
      <p>{parsed.get("technical_quant","数据不可用")}</p>
      {self._tech_table(ta)}
    </div>
    <div class="card">
      <h2>六、情绪面与催化剂</h2>
      <p>{parsed.get("sentiment_catalysts","数据不可用")}</p>
      {'<h4>催化剂</h4><ul>' + "".join(f"<li>{c}</li>" for c in sa.catalysts) + '</ul>' if sa and sa.catalysts else ""}
    </div>
  </div>

  <div class="card">
    <h2>七、风险评估</h2>
    <p>{parsed.get("risk_assessment","数据不可用")}</p>
    <ul>{"".join(f"<li>⚠️ {r}</li>" for r in report.risk_warnings)}</ul>
  </div>

  <div class="card" style="border-left:4px solid {rc}">
    <h2>八、投资建议</h2>
    <p><strong>评级：<span style="color:{rc}">{report.rating.value.upper()}</span></strong></p>
    <p>{parsed.get("investment_recommendation","数据不可用")}</p>
  </div>

  <div class="card" style="background:#f8f9fa">
    <h2>九、附录</h2>
    <p style="color:#636e72;font-size:.9em">{parsed.get("appendix","数据来源：Tushare | 模型：QuantMind Multi-Agent")}</p>
  </div>

  <!-- Critic 审查 -->
  {self._critic_section(state)}

  <footer style="text-align:center;padding:20px;color:#b2bec3;font-size:.8em">
    QuantMind Phase 4 | Multi-Agent Research System | LangGraph + Ollama
  </footer>
</div>

<script>
{charts_js}
</script>
</body>
</html>"""

    def _ratios_table(self, fa) -> str:
        if not fa or not fa.ratios:
            return ""
        rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in fa.ratios.items())
        return f"<table><tr><th>指标</th><th>值</th></tr>{rows}</table>"

    def _tech_table(self, ta) -> str:
        if not ta or not ta.indicators:
            return ""
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in list(ta.indicators.items())[:8])
        return f"<table><tr><th>指标</th><th>值</th></tr>{rows}</table>"

    def _critic_section(self, state: AgentState) -> str:
        fb = state.critic_feedback
        if not fb:
            return ""
        color = "#00b894" if fb.passed else "#d63031"
        issue_rows = "".join(
            f"<tr><td style='color:{'#d63031' if i.severity.value=='critical' else '#e17055'}'>{i.severity.value}</td>"
            f"<td>{i.type.value}</td><td>{i.description}</td></tr>"
            for i in fb.issues
        )
        return f"""<div class="card">
  <h2 style="color:{color}">Critic 审查结果</h2>
  <p>{'✅ 通过' if fb.passed else '❌ 未通过'} | 质量分：{fb.overall_quality_score:.1f}/10 | {fb.approval_message}</p>
  {f'<table><tr><th>严重度</th><th>类型</th><th>描述</th></tr>{issue_rows}</table>' if fb.issues else ''}
</div>"""

    def _build_charts_js(self, fa, ta, state: AgentState) -> str:
        """生成4张 Plotly 图表的 JS 代码."""

        # 图1：股价走势（从 state snapshot 读价格数据）
        price_js = self._price_chart_js(state)

        # 图2：财务比率雷达图
        radar_js = self._radar_chart_js(fa)

        # 图3：DCF 敏感性热力图
        dcf_js = self._dcf_sensitivity_js(fa)

        # 图4：SHAP 因子贡献条形图
        shap_js = self._shap_bar_js(ta)

        return "\n".join([price_js, radar_js, dcf_js, shap_js])

    def _price_chart_js(self, state: AgentState) -> str:
        """股价走势 + 60日均线."""
        try:
            if state.snapshot and state.query.tickers:
                from quantmind.agents.tools.data_tools import fetch_price_history
                from datetime import timedelta
                as_of = state.query.as_of or date.today()
                start = as_of - timedelta(days=120)
                prices = fetch_price_history(state.query.tickers[0], start, as_of, as_of)
                if not prices.empty and "close" in prices.columns and "trade_date" in prices.columns:
                    dates = prices["trade_date"].astype(str).tolist()
                    closes = prices["close"].round(2).tolist()
                    ma60 = prices["close"].rolling(60, min_periods=1).mean().round(2).tolist()
                    return f"""
Plotly.newPlot('chart-price', [
  {{x:{json.dumps(dates)}, y:{json.dumps(closes)}, type:'scatter', name:'收盘价', line:{{color:'#0984e3'}}}},
  {{x:{json.dumps(dates)}, y:{json.dumps(ma60)}, type:'scatter', name:'MA60', line:{{color:'#e17055',dash:'dash'}}}}
], {{margin:{{t:20,b:40}}, legend:{{orientation:'h',y:-0.2}}, plot_bgcolor:'#fafafa'}});"""
        except Exception:
            pass
        return "document.getElementById('chart-price').innerHTML='<p style=\"text-align:center;padding:80px;color:#b2bec3\">价格数据不可用</p>';"

    def _radar_chart_js(self, fa) -> str:
        """财务比率雷达图（vs 行业中位数基准）."""
        if not fa or not fa.ratios:
            return "document.getElementById('chart-radar').innerHTML='<p style=\"text-align:center;padding:80px;color:#b2bec3\">财务数据不可用</p>';"

        metrics = ["roe", "roa", "gross_margin", "net_margin", "current_ratio"]
        labels = ["ROE", "ROA", "毛利率", "净利率", "流动比率"]
        # 行业中位数基准（示范值）
        industry_bench = [0.12, 0.06, 0.35, 0.10, 1.5]

        values = [fa.ratios.get(m, 0) for m in metrics]
        # 归一化（相对于基准）
        norm_vals = [min(v / max(b, 0.001), 2.0) for v, b in zip(values, industry_bench)]
        norm_bench = [1.0] * len(metrics)

        return f"""
Plotly.newPlot('chart-radar', [
  {{type:'scatterpolar', r:{json.dumps(norm_vals + [norm_vals[0]])}, theta:{json.dumps(labels + [labels[0]])},
   fill:'toself', name:'本公司', line:{{color:'#6c5ce7'}}}},
  {{type:'scatterpolar', r:{json.dumps(norm_bench + [norm_bench[0]])}, theta:{json.dumps(labels + [labels[0]])},
   fill:'toself', name:'行业基准', line:{{color:'#b2bec3',dash:'dash'}}}}
], {{polar:{{radialaxis:{{visible:true,range:[0,2]}}}}, margin:{{t:20}}, legend:{{orientation:'h',y:-0.2}}}});"""

    def _dcf_sensitivity_js(self, fa) -> str:
        """DCF 敏感性分析热力图（WACC × 增长率）."""
        try:
            from quantmind.agents.tools.analysis_tools import compute_dcf_valuation
            if fa:
                wacc_range = [0.08, 0.09, 0.10, 0.11, 0.12]
                growth_range = [0.03, 0.04, 0.05, 0.06, 0.07]
                # 使用 fa.dcf_assumptions 或 fa.ratios 来 proxy financials
                financials_proxy = {f"financial_indicators__roe_ttm": fa.ratios.get("roe", 0.15)}
                z = []
                for g in growth_range:
                    row = []
                    for w in wacc_range:
                        dcf = compute_dcf_valuation(financials_proxy, wacc=w, growth=g)
                        v = dcf.get("intrinsic_value") or 0
                        row.append(round(float(v) / 1e8, 1) if v else 0)  # 亿元
                    z.append(row)
                wacc_labels = [f"{w:.0%}" for w in wacc_range]
                growth_labels = [f"{g:.0%}" for g in growth_range]
                return f"""
Plotly.newPlot('chart-dcf', [{{
  type:'heatmap', z:{json.dumps(z)},
  x:{json.dumps(wacc_labels)}, y:{json.dumps(growth_labels)},
  colorscale:'RdYlGn', text:{json.dumps(z)}, texttemplate:'%{{text}}亿',
  colorbar:{{title:'公允价值(亿)'}}
}}], {{margin:{{t:20,b:60}}, xaxis:{{title:'WACC'}}, yaxis:{{title:'增长率'}}}});"""
        except Exception:
            pass
        return "document.getElementById('chart-dcf').innerHTML='<p style=\"text-align:center;padding:80px;color:#b2bec3\">DCF数据不可用</p>';"

    def _shap_bar_js(self, ta) -> str:
        """因子贡献条形图（SHAP 值）."""
        if not ta or not ta.quant_explanation:
            return "document.getElementById('chart-shap').innerHTML='<p style=\"text-align:center;padding:80px;color:#b2bec3\">SHAP数据不可用</p>';"

        shap = ta.quant_explanation
        sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        factors = [k for k, _ in sorted_shap]
        values = [v for _, v in sorted_shap]
        colors = ["#00b894" if v >= 0 else "#d63031" for v in values]

        return f"""
Plotly.newPlot('chart-shap', [{{
  type:'bar', x:{json.dumps(values)}, y:{json.dumps(factors)}, orientation:'h',
  marker:{{color:{json.dumps(colors)}}},
  text:{json.dumps([f"{v:.4f}" for v in values])}, textposition:'outside'
}}], {{margin:{{t:20,l:120,r:60}}, xaxis:{{title:'SHAP值（贡献度）'}}, plot_bgcolor:'#fafafa'}});"""
