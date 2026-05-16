"""quantmind.agents.rag_report_agent — 基于 KB 检索结果的 RAG 投资报告生成器.

不依赖 BaseAgent / AgentState / LangGraph。
输入：retrieve_stock_context JSON（dict）
输出：结构化报告 dict（含 report_markdown）

设计原则：
- 严格只使用 context JSON 中的信息，不编造数据
- news_context 标注为"当前舆情"，不当作历史事实
- report_context 为空时明确写"未检索到本地研究报告"
- snapshot 数据优先使用 as_of 最接近目标日期的条目
- 不给确定性买卖建议，使用研究性措辞
- provider=none 时使用模板生成，保证离线可运行
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from loguru import logger

from quantmind.agents.llm_client import LLMClient, LLMClientResponse, build_client

__all__ = ["RAGReportAgent"]

# ── 投资建议等级 ──────────────────────────────────────────────────────────────
RATING_LABELS = ("观察", "谨慎关注", "积极关注", "不建议")

# ── LLM system prompt ────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """你是一位专业的股票研究员，负责根据结构化数据生成投资分析报告。

【输出格式要求】
- 直接输出 Markdown 文本，不要用 ```markdown 或 ``` 代码块包裹整个报告。
- 报告必须以 # 一级标题开头（如：# 600519.SH 投资分析报告）。
- 如果有多期趋势数据，在对应章节中分析变化方向（改善/恶化/稳定），并用表格展示。

【内容要求】
1. 严格只使用给定的证据摘要中的信息，不要凭空编造数据。
2. news 信息是"当前舆情"，不是历史事实，必须明确标注为"当前市场舆情"。
3. 如果某类数据为空，要明确写出"数据缺失"或"未检索到"，不要跳过或编造。
4. 投资建议使用研究性措辞：观察 / 谨慎关注 / 积极关注 / 不建议。
5. 不给出确定性收益承诺，不写"必须买入/卖出"。
6. 报告结尾必须包含免责声明：本报告不构成投资建议。

请按如下12个章节组织 Markdown 报告，每章节用 ## 标题开始：
1. 执行摘要
2. 公司概况
3. 行情与估值
4. 财务分析
5. 北向资金
6. 融资融券
7. 市场环境
8. 新闻与事件（当前舆情）
9. 本地研究报告
10. 风险因素
11. 投资建议
12. 数据来源与限制"""


class RAGReportAgent:
    """单股票 RAG 投资分析报告生成器.

    Args:
        llm_client: LLMClient 实例。None = 自动构建 provider=none
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client

    # ── 公开 API ─────────────────────────────────────────────────────────────

    def generate_report(
        self,
        context: dict[str, Any],
        language: str = "zh",
        use_llm: bool = True,
        llm_provider: str | None = None,
        llm_model: str | None = None,
    ) -> dict[str, Any]:
        """生成投资分析报告.

        Args:
            context:      retrieve_stock_context JSON dict
            language:     报告语言（当前仅支持 zh）
            use_llm:      是否尝试调用 LLM
            llm_provider: 覆盖 client 的 provider
            llm_model:    覆盖 client 的 model

        Returns:
            {
              "ticker", "as_of", "report_markdown",
              "data_coverage", "warnings", "used_sources",
              "llm_used", "provider", "model"
            }
        """
        ticker = context.get("ticker", "")
        as_of = context.get("as_of", "")
        warnings: list[str] = []
        used_sources: list[str] = []

        # ── 1. 构建证据摘要 ────────────────────────────────────────────────
        evidence = self._build_evidence_summary(context, warnings, used_sources)

        # ── 2. 数据覆盖度 ──────────────────────────────────────────────────
        coverage = self._build_coverage(context)

        # ── 3. 生成报告 ────────────────────────────────────────────────────
        client = self._get_client(llm_provider, llm_model)
        llm_used = False
        provider_used = client.provider
        model_used = client.model

        report_md: str = ""
        if use_llm and not client.is_none_provider:
            prompt = self._build_llm_prompt(ticker, as_of, evidence, context)
            try:
                resp: LLMClientResponse | None = client.chat(
                    system=_SYSTEM_PROMPT,
                    user=prompt,
                    max_tokens=4096,
                    temperature=0.3,
                )
                if resp and resp.content:
                    report_md = self._postprocess_llm_output(resp.content, ticker, as_of, coverage, warnings)
                    llm_used = True
                    logger.info(f"[RAGReportAgent] LLM report generated: {ticker}")
                else:
                    warnings.append("LLM 返回为空，回退模板生成")
            except Exception as e:
                warnings.append(f"LLM 调用异常，回退模板生成: {e}")
                logger.warning(f"[RAGReportAgent] LLM error, fallback template: {e}")

        if not report_md:
            report_md = self._generate_template_report(ticker, as_of, evidence, context, coverage, warnings)
            provider_used = "none"
            model_used = "template"

        return {
            "ticker": ticker,
            "as_of": as_of,
            "report_markdown": report_md,
            "data_coverage": coverage,
            "warnings": warnings,
            "used_sources": used_sources,
            "llm_used": llm_used,
            "provider": provider_used,
            "model": model_used,
        }

    # ── 证据摘要构建 ─────────────────────────────────────────────────────────

    def _build_evidence_summary(
        self,
        context: dict[str, Any],
        warnings: list[str],
        used_sources: list[str],
    ) -> dict[str, Any]:
        """从 context 中提取关键证据，不引入外部信息."""
        ev: dict[str, Any] = {}

        # company_profile：取 as_of 最近的条目
        cp_all = context.get("snapshot_company_profile", [])
        cp = self._pick_latest(cp_all)
        if cp:
            ev["company_profile"] = cp
            used_sources.append(f"snapshot/company_profile as_of={cp.get('as_of','')}")
        else:
            warnings.append("snapshot_company_profile 无数据")

        # latest_market_metrics — keep latest + all for trend
        mm_all = context.get("snapshot_latest_market_metrics", [])
        mm = self._pick_latest(mm_all)
        if mm:
            ev["market_metrics"] = mm
            ev["market_metrics_all"] = mm_all
            used_sources.append(f"snapshot/latest_market_metrics as_of={mm.get('as_of','')} (共 {len(mm_all)} 期)")
        else:
            warnings.append("snapshot_latest_market_metrics 无数据")

        # financial_indicator_summary — keep latest + all for trend
        fi_all = context.get("snapshot_financial_indicator_summary", [])
        fi = self._pick_latest(fi_all)
        if fi:
            ev["financial_summary"] = fi
            ev["financial_summary_all"] = fi_all
            used_sources.append(f"snapshot/financial_indicator_summary as_of={fi.get('as_of','')} (共 {len(fi_all)} 期)")
        else:
            warnings.append("snapshot_financial_indicator_summary 无数据")

        # northbound_summary — keep latest + all for trend
        nb_all = context.get("snapshot_northbound_summary", [])
        nb = self._pick_latest(nb_all)
        if nb:
            ev["northbound"] = nb
            ev["northbound_all"] = nb_all
            used_sources.append(f"snapshot/northbound_summary as_of={nb.get('as_of','')} (共 {len(nb_all)} 期)")
        else:
            warnings.append("snapshot_northbound_summary 无数据")

        # margin_summary — keep latest + all for trend
        mg_all = context.get("snapshot_margin_summary", [])
        mg = self._pick_latest(mg_all)
        if mg:
            ev["margin"] = mg
            ev["margin_all"] = mg_all
            used_sources.append(f"snapshot/margin_summary as_of={mg.get('as_of','')} (共 {len(mg_all)} 期)")
        else:
            warnings.append("snapshot_margin_summary 无数据")

        # market_context
        mkt = self._pick_latest(context.get("market_context", []))
        if mkt:
            ev["market_context"] = mkt
            used_sources.append(f"market_index_context as_of={mkt.get('as_of','')}")
        else:
            warnings.append("market_index_context 无数据")

        # news（最多 5 条）
        news_list = context.get("news_context", [])
        if news_list:
            ev["news"] = news_list[:5]
            used_sources.append(f"news x{len(news_list[:5])} (当前舆情)")
        else:
            warnings.append("news_context 无数据（未检索到新闻）")

        # reports（全部）
        report_list = context.get("report_context", [])
        if report_list:
            ev["reports"] = report_list
            used_sources.append(f"report x{len(report_list)}")
        else:
            warnings.append("report_context 无数据（未检索到本地研究报告）")

        return ev

    def _pick_latest(self, items: list[dict]) -> dict | None:
        """从多个快照条目中选取 as_of 最新的一条."""
        if not items:
            return None
        def _sort_key(x: dict) -> str:
            return x.get("as_of") or x.get("published_date") or ""
        return sorted(items, key=_sort_key, reverse=True)[0]

    def _sort_by_date(self, items: list[dict]) -> list[dict]:
        """按 as_of / published_date 升序排列（旧→新），方便趋势对比."""
        return sorted(
            items,
            key=lambda x: x.get("as_of") or x.get("published_date") or "",
        )

    def _build_trend_table(
        self,
        items: list[dict],
        fields: list[tuple[str, str]],
        caption: str = "",
    ) -> str:
        """Build a multi-period trend Markdown table from snapshot items.

        Args:
            items:   list of snapshot dicts (each has "as_of" and "text")
            fields:  list of (field_key, column_label) pairs
            caption: optional table caption line

        Returns:
            Markdown table string, or empty string if no data.
        """
        rows = self._sort_by_date(items)
        if not rows:
            return ""
        field_keys = [k for k, _ in fields]
        field_labels = [lbl for _, lbl in fields]
        header = "| 数据期 | " + " | ".join(field_labels) + " |"
        sep    = "|--------|" + "|".join(["-------"] * len(fields)) + "|"
        lines = [caption, header, sep] if caption else [header, sep]
        for item in rows:
            as_of = item.get("as_of", "")
            text = item.get("text", "")
            vals = [self._extract_field(text, k) or "—" for k in field_keys]
            lines.append("| " + as_of + " | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _build_coverage(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "collection_count": context.get("collection_count", 0),
            "news_count": context.get("news_count", 0),
            "report_count": context.get("report_count", 0),
            "snapshot_company_profile_count": context.get("snapshot_company_profile_count", 0),
            "snapshot_latest_market_metrics_count": context.get("snapshot_latest_market_metrics_count", 0),
            "snapshot_financial_indicator_summary_count": context.get("snapshot_financial_indicator_summary_count", 0),
            "snapshot_northbound_summary_count": context.get("snapshot_northbound_summary_count", 0),
            "snapshot_margin_summary_count": context.get("snapshot_margin_summary_count", 0),
            "market_context_count": context.get("market_count", 0),
        }

    # ── LLM prompt 构建 ──────────────────────────────────────────────────────

    def _build_llm_prompt(
        self,
        ticker: str,
        as_of: str,
        evidence: dict[str, Any],
        context: dict[str, Any],
    ) -> str:
        parts = [
            f"股票代码：{ticker}",
            f"数据截止日期（as_of）：{as_of}",
            f"KB 总文档数：{context.get('collection_count', 0)}",
            "",
            "=== 证据摘要（仅使用以下信息生成报告）===",
        ]

        cp = evidence.get("company_profile")
        if cp:
            parts += ["", "【公司概况】", cp.get("text", "")]

        mm = evidence.get("market_metrics")
        if mm:
            parts += ["", "【行情与估值（最新期）】", mm.get("text", "")]
        mm_all = evidence.get("market_metrics_all", [])
        if len(mm_all) > 1:
            trend = self._build_trend_table(
                mm_all,
                [("pe", "PE"), ("pb", "PB"), ("total_mv", "总市值(万)"), ("turnover_rate", "换手率")],
            )
            if trend:
                parts += ["", "【估值多期趋势（旧→新）】", trend]

        fi = evidence.get("financial_summary")
        if fi:
            parts += ["", "【财务指标（最新期）】", fi.get("text", "")]
        fi_all = evidence.get("financial_summary_all", [])
        if len(fi_all) > 1:
            trend = self._build_trend_table(
                fi_all,
                [
                    ("roe", "ROE(%)"),
                    ("grossprofit_margin", "毛利率(%)"),
                    ("netprofit_margin", "净利率(%)"),
                    ("debt_to_assets", "资负率(%)"),
                    ("or_yoy", "营收同比(%)"),
                ],
            )
            if trend:
                parts += ["", "【财务指标多期趋势（旧→新）】", trend]

        nb = evidence.get("northbound")
        if nb:
            parts += ["", "【北向资金（最新期）】", nb.get("text", "")]
        nb_all = evidence.get("northbound_all", [])
        if len(nb_all) > 1:
            trend = self._build_trend_table(
                nb_all,
                [("hold_ratio", "持股比(%)"), ("hold_vol", "持股量(股)")],
            )
            if trend:
                parts += ["", "【北向资金多期趋势（旧→新）】", trend]

        mg = evidence.get("margin")
        if mg:
            parts += ["", "【融资融券（最新期）】", mg.get("text", "")]
        mg_all = evidence.get("margin_all", [])
        if len(mg_all) > 1:
            trend = self._build_trend_table(
                mg_all,
                [("rzye", "融资余额(元)"), ("rzmre", "融资买入(元)")],
            )
            if trend:
                parts += ["", "【融资融券多期趋势（旧→新）】", trend]

        mkt = evidence.get("market_context")
        if mkt:
            parts += ["", "【市场环境】", mkt.get("text", "")]

        news = evidence.get("news", [])
        if news:
            parts += ["", "【当前市场舆情（published_date 为实时抓取时间，非历史事实）】"]
            for n in news:
                parts.append(f"- {n.get('text_preview','')[:150]}  [来源: {n.get('source','')}]")

        reports = evidence.get("reports", [])
        if reports:
            parts += ["", "【本地研究报告】"]
            for rp in reports:
                parts.append(f"- 标题：{rp.get('title','')}  ticker：{rp.get('ticker','')}")
                parts.append(f"  内容摘要：{rp.get('text_preview','')[:200]}")
        else:
            parts += ["", "【本地研究报告】", "（无）未检索到该股票对应的本地研究报告。"]

        parts += [
            "",
            "=== 输出要求 ===",
            f"1. 生成 {ticker} 的完整12章节投资分析报告（纯 Markdown 格式）。",
            "2. 直接输出 Markdown 文本，不要用 ```markdown 或 ``` 代码块包裹整个报告。",
            "3. 报告必须以 # 开头（一级标题）。",
            "4. 只能使用以上证据摘要中的信息，不要引入外部知识或凭空编造数据。",
            "5. news 必须明确标注为当前市场舆情，不得当作历史事实。",
            "6. 如有多期趋势数据，在对应章节中分析趋势变化方向（改善/恶化/稳定）。",
            "7. 不要给确定性买卖建议，使用研究性措辞：观察/谨慎关注/积极关注/不建议。",
            "8. 结尾必须加入免责声明。",
        ]
        return "\n".join(parts)

    # ── LLM 输出后处理 ────────────────────────────────────────────────────────

    def _postprocess_llm_output(
        self,
        content: str,
        ticker: str,
        as_of: str,
        coverage: dict[str, Any],
        warnings: list[str],
    ) -> str:
        """确保 LLM 输出包含必要的免责声明和数据覆盖度节."""
        md = content.strip()
        # Strip ```markdown ... ``` wrapping that some LLMs add
        if md.startswith("```markdown"):
            md = md[len("```markdown"):].strip()
        elif md.startswith("```"):
            md = md[3:].strip()
        if md.endswith("```"):
            md = md[:-3].strip()
        if not md.startswith("#"):
            today = date.today().isoformat()
            md = f"# {ticker} 投资分析报告\n> 数据截止：**{as_of}** | 生成日期：{today}\n\n" + md
        if "不构成投资建议" not in md:
            md += "\n\n> **免责声明**：本报告由 QuantMind RAG 系统自动生成，仅供研究参考，不构成投资建议。"
        md += "\n\n" + self._build_coverage_section(coverage, warnings)
        return md

    # ── 模板报告生成 ─────────────────────────────────────────────────────────

    def _generate_template_report(
        self,
        ticker: str,
        as_of: str,
        evidence: dict[str, Any],
        context: dict[str, Any],
        coverage: dict[str, Any],
        warnings: list[str],
    ) -> str:
        """在不调用 LLM 的情况下，从模板生成结构化报告."""
        today = date.today().isoformat()
        sections: list[str] = [
            f"# {ticker} 投资分析报告",
            f"> 数据截止：**{as_of}** | 生成日期：{today} | 生成方式：模板（provider=none）",
            "",
        ]

        # 1. 执行摘要
        sections.append("## 1. 执行摘要")
        exec_points = self._build_exec_summary(ticker, as_of, evidence, context)
        for pt in exec_points:
            sections.append(f"- {pt}")
        sections.append("")

        # 2. 公司概况
        sections.append("## 2. 公司概况")
        cp = evidence.get("company_profile")
        if cp:
            sections.append(self._clean_text(cp.get("text", "")))
        else:
            sections.append("_公司概况数据暂缺（snapshot_company_profile 无数据）。_")
        sections.append("")

        # 3. 行情与估值
        sections.append("## 3. 行情与估值")
        mm = evidence.get("market_metrics")
        mm_all = evidence.get("market_metrics_all", [])
        if mm:
            sections.append(self._clean_text(mm.get("text", "")))
            sections.append("")
            sections.append(self._build_market_table(mm.get("text", "")))
            if len(mm_all) > 1:
                trend = self._build_trend_table(
                    mm_all,
                    [("pe", "PE"), ("pb", "PB"), ("total_mv", "总市值(万)"), ("turnover_rate", "换手率")],
                    "**估值多期趋势**",
                )
                if trend:
                    sections.append("")
                    sections.append(trend)
        else:
            sections.append("_行情与估值数据暂缺。_")
        sections.append("")

        # 4. 财务分析
        sections.append("## 4. 财务分析")
        fi = evidence.get("financial_summary")
        fi_all = evidence.get("financial_summary_all", [])
        if fi:
            sections.append(self._clean_text(fi.get("text", "")))
            sections.append("")
            sections.append(self._build_financial_table(fi.get("text", "")))
            sections.append("")
            sections.append(self._build_financial_commentary(fi.get("text", "")))
            if len(fi_all) > 1:
                trend = self._build_trend_table(
                    fi_all,
                    [
                        ("roe", "ROE(%)"),
                        ("grossprofit_margin", "毛利率(%)"),
                        ("netprofit_margin", "净利率(%)"),
                        ("debt_to_assets", "资负率(%)"),
                        ("or_yoy", "营收同比(%)"),
                    ],
                    "**财务指标多期趋势**",
                )
                if trend:
                    sections.append("")
                    sections.append(trend)
        else:
            sections.append("_财务指标数据暂缺（snapshot_financial_indicator_summary 无数据）。_")
        sections.append("")

        # 5. 北向资金
        sections.append("## 5. 北向资金")
        nb = evidence.get("northbound")
        nb_all = evidence.get("northbound_all", [])
        if nb:
            sections.append(self._clean_text(nb.get("text", "")))
            if len(nb_all) > 1:
                trend = self._build_trend_table(
                    nb_all,
                    [("hold_ratio", "持股比(%)"), ("hold_vol", "持股量(股)"), ("hold_amount", "持股市值")],
                    "**北向资金多期变化**",
                )
                if trend:
                    sections.append("")
                    sections.append(trend)
        else:
            sections.append("_北向持股数据暂缺（可能该股不在陆股通标的范围，或数据未摄入）。_")
        sections.append("")

        # 6. 融资融券
        sections.append("## 6. 融资融券")
        mg = evidence.get("margin")
        mg_all = evidence.get("margin_all", [])
        if mg:
            sections.append(self._clean_text(mg.get("text", "")))
            if len(mg_all) > 1:
                trend = self._build_trend_table(
                    mg_all,
                    [("rzye", "融资余额(元)"), ("rzmre", "融资买入(元)"), ("rqye", "融券余额(元)")],
                    "**融资融券多期变化**",
                )
                if trend:
                    sections.append("")
                    sections.append(trend)
        else:
            sections.append("_融资融券数据暂缺。_")
        sections.append("")

        # 7. 市场环境
        sections.append("## 7. 市场环境")
        mkt = evidence.get("market_context")
        if mkt:
            sections.append(self._clean_text(mkt.get("text", "")))
        else:
            sections.append("_市场指数数据暂缺。_")
        sections.append("")

        # 8. 新闻与事件（当前舆情）
        sections.append("## 8. 新闻与事件")
        sections.append(
            "> ⚠️ **时态说明**：以下新闻为**当前市场舆情**，"
            "其 `published_date` 为实时抓取时间（非 as_of 历史时点已知信息），"
            "不代表截止 " + as_of + " 时已知事件。"
        )
        sections.append("")
        news_list = evidence.get("news", [])
        if news_list:
            for i, n in enumerate(news_list, 1):
                pub = n.get("published_date", "")
                src = n.get("source", "")
                preview = n.get("text_preview", "")[:200]
                sections.append(f"**{i}.** {preview}")
                sections.append(f"   _来源：{src}  发布时间：{pub}_")
                sections.append("")
        else:
            sections.append("_未检索到相关新闻。_")
        sections.append("")

        # 9. 本地研究报告
        sections.append("## 9. 本地研究报告")
        report_list = evidence.get("reports", [])
        if report_list:
            for rp in report_list:
                title = rp.get("title", "")
                rp_ticker = rp.get("ticker", "")
                pub = rp.get("published_date", "")
                sections.append(f"**报告标题**：{title}")
                sections.append(f"**股票代码**：{rp_ticker}  **发布日期**：{pub}")
                sections.append("")
                text = rp.get("text_preview", "")
                if text:
                    sections.append(text[:400])
                sections.append("")
        else:
            sections.append("_未检索到该股票对应的本地研究报告。如需报告分析，请向 `data/reports/` 目录添加相关文件并重新摄入 KB。_")
        sections.append("")

        # 10. 风险因素
        sections.append("## 10. 风险因素")
        risks = self._build_risks(evidence, context)
        for r in risks:
            sections.append(f"- {r}")
        sections.append("")

        # 11. 投资建议
        sections.append("## 11. 投资建议")
        rating, rationale, key_metrics, uncertainties = self._build_recommendation(evidence, ticker, as_of)
        sections.append(f"**研究性评级**：{rating}")
        sections.append("")
        sections.append(f"**主要依据**：{rationale}")
        sections.append("")
        sections.append("**关键跟踪指标**：")
        for m in key_metrics:
            sections.append(f"- {m}")
        sections.append("")
        sections.append("**不确定性**：")
        for u in uncertainties:
            sections.append(f"- {u}")
        sections.append("")
        sections.append(
            "> **免责声明**：本报告由 QuantMind RAG 系统自动生成，"
            "基于有限数据集，仅供研究参考，**不构成任何投资建议**。"
            "投资有风险，入市需谨慎。"
        )
        sections.append("")

        # 12. 数据来源与限制
        sections.append("## 12. 数据来源与限制")
        sections.append(self._build_coverage_section(coverage, warnings))

        return "\n".join(sections)

    # ── 各章节构建辅助 ────────────────────────────────────────────────────────

    def _build_exec_summary(
        self,
        ticker: str,
        as_of: str,
        evidence: dict[str, Any],
        context: dict[str, Any],
    ) -> list[str]:
        points: list[str] = [
            f"本报告基于 QuantMind KB（总文档 {context.get('collection_count', 0)} 条）对 **{ticker}** 进行结构化 RAG 分析，数据截止 **{as_of}**。",
        ]
        cp = evidence.get("company_profile")
        if cp:
            text = cp.get("text", "")
            name = self._extract_field(text, "名称")
            industry = self._extract_field(text, "行业")
            pts = []
            if name:
                pts.append(f"公司名称：{name}")
            if industry:
                pts.append(f"所属行业：{industry}")
            if pts:
                points.append("  ".join(pts))

        fi = evidence.get("financial_summary")
        if fi:
            text = fi.get("text", "")
            roe = self._extract_field(text, "roe")
            if roe:
                points.append(f"最新 ROE：{roe}（截至报告期）。")

        mm = evidence.get("market_metrics")
        if mm:
            text = mm.get("text", "")
            pe = self._extract_field(text, "pe")
            pb = self._extract_field(text, "pb")
            vals = []
            if pe:
                vals.append(f"PE={pe}")
            if pb:
                vals.append(f"PB={pb}")
            if vals:
                points.append(f"当前估值指标：{', '.join(vals)}（截至 {mm.get('as_of','')}）。")

        if context.get("news_count", 0) > 0:
            points.append(
                f"当前舆情：检索到 {context['news_count']} 条实时新闻（published_date 为今日，非 as_of 历史信息）。"
            )
        if not context.get("report_count", 0):
            points.append(f"本地研究报告：**未检索到** {ticker} 对应的研究报告。")

        return points

    def _build_financial_table(self, text: str) -> str:
        """从财务指标文本中提取 KV，生成 Markdown 表格."""
        rows = []
        metric_names = {
            "roe": "ROE（净资产收益率）",
            "roa": "ROA（总资产收益率）",
            "grossprofit_margin": "毛利率",
            "netprofit_margin": "净利率",
            "debt_to_assets": "资产负债率",
            "eps": "EPS（每股收益）",
            "bps": "BPS（每股净资产）",
            "q_profit_yoy": "归母净利润同比",
            "q_gr_yoy": "营收同比",
            "roe_yoy": "ROE 同比变化",
            "profit_to_gr": "净利润/营收",
            "tr_yoy": "总营收同比",
        }
        for key, label in metric_names.items():
            val = self._extract_field(text, key)
            if val:
                rows.append(f"| {label} | {val} |")

        if not rows:
            return ""
        header = "| 财务指标 | 值 |\n|---------|-------|"
        return header + "\n" + "\n".join(rows)

    def _build_market_table(self, text: str) -> str:
        """从行情文本中提取 KV，生成 Markdown 表格."""
        rows = []
        fields = {
            "pe": "市盈率 (PE)",
            "pb": "市净率 (PB)",
            "total_mv": "总市值（万元）",
            "circ_mv": "流通市值（万元）",
            "turnover_rate": "换手率",
            "volume_ratio": "量比",
            "close": "最新收盘价",
            "pct_chg": "涨跌幅（%）",
        }
        for key, label in fields.items():
            val = self._extract_field(text, key)
            if val:
                rows.append(f"| {label} | {val} |")

        if not rows:
            return ""
        header = "| 市场指标 | 值 |\n|---------|-------|"
        return header + "\n" + "\n".join(rows)

    def _build_financial_commentary(self, text: str) -> str:
        """根据财务指标文本生成简单分析文字."""
        parts = []
        roe = self._extract_field(text, "roe")
        gm = self._extract_field(text, "grossprofit_margin")
        da = self._extract_field(text, "debt_to_assets")
        profit_yoy = self._extract_field(text, "q_profit_yoy")
        or_yoy = self._extract_field(text, "or_yoy")

        if roe:
            roe_f = self._pct_to_float(roe)
            if roe_f is not None:
                if roe_f >= 0.15:
                    parts.append(f"盈利能力：ROE={roe}%，处于较高水平（≥15%），说明公司资本运用效率较强。")
                elif roe_f >= 0.08:
                    parts.append(f"盈利能力：ROE={roe}%，处于中等水平（8%~15%），需结合行业对比。")
                else:
                    parts.append(f"盈利能力：ROE={roe}%，低于 8%，需关注盈利能力改善情况。")
            else:
                parts.append(f"盈利能力：ROE={roe}。")

        if gm:
            gm_f = self._pct_to_float(gm)
            if gm_f is not None and gm_f >= 0.5:
                parts.append(f"毛利率：{gm}%（超过 50%，产品定价能力强）。")
            elif gm_f is not None and gm_f >= 0.25:
                parts.append(f"毛利率：{gm}%（处于中高水平）。")
            else:
                parts.append(f"毛利率：{gm}%。")

        if da:
            da_f = self._pct_to_float(da)
            if da_f is not None:
                if da_f >= 0.70:
                    parts.append(f"偿债风险：资产负债率={da}%，较高（≥70%），利率上行时风险加大，需关注债务结构。")
                elif da_f >= 0.50:
                    parts.append(f"资产结构：资产负债率={da}%，中等水平（50%~70%），符合行业常规。")
                else:
                    parts.append(f"资产结构：资产负债率={da}%，偏低，财务杠杆保守，偿债压力较小。")
            else:
                parts.append(f"资产负债率：{da}。")

        if profit_yoy:
            parts.append(f"成长性：归母净利润同比={profit_yoy}%，反映利润增长趋势。")
        elif or_yoy:
            parts.append(f"成长性：营收同比={or_yoy}%，反映收入增长趋势。")

        return "\n\n".join(parts) if parts else "_财务指标解读：字段值不足，无法生成详细解读。_"

    def _build_risks(self, evidence: dict[str, Any], context: dict[str, Any]) -> list[str]:
        risks = []

        # 估值风险
        mm = evidence.get("market_metrics")
        if mm:
            pe = self._extract_field(mm.get("text", ""), "pe")
            if pe:
                try:
                    pe_f = float(pe)
                    if pe_f > 50:
                        risks.append(f"估值风险：当前 PE={pe}，处于高位，若业绩不及预期存在估值回调风险。")
                    elif pe_f > 30:
                        risks.append(f"估值风险：PE={pe}，处于中高水平，需关注业绩兑现情况。")
                except ValueError:
                    pass

        # 财务风险
        fi = evidence.get("financial_summary")
        if fi:
            da = self._extract_field(fi.get("text", ""), "debt_to_assets")
            if da:
                da_f = self._pct_to_float(da)
                if da_f is not None and da_f >= 0.7:
                    risks.append(f"财务风险：资产负债率={da}%，较高（≥70%），利率上行或融资收紧时风险加大。")
                elif da_f is not None and da_f >= 0.5:
                    risks.append(f"财务风险：资产负债率={da}%，中等偏高，需持续关注债务结构。")

        # 资金面风险（北向）
        nb = evidence.get("northbound")
        if nb:
            text = nb.get("text", "")
            ratio = self._extract_field(text, "hold_ratio")
            if ratio:
                risks.append(f"外资风险：北向持股比例={ratio}，外资集中持有时波动敏感性较高。")
        else:
            risks.append("外资数据缺失：北向持股数据不可用，无法评估外资流入流出风险。")

        # 融资融券风险
        mg = evidence.get("margin")
        if mg:
            rzye = self._extract_field(mg.get("text", ""), "rzye")
            if rzye:
                risks.append(f"资金面风险：融资余额={rzye}，融资盘集中时下跌容易引发连锁抛售。")

        # 新闻事件风险
        news = evidence.get("news", [])
        if news:
            risks.append("舆情风险：当前存在实时新闻信息，需关注可能影响股价的重大事件（具体见第8节）。")
        else:
            risks.append("新闻数据缺失：未检索到近期新闻，舆情风险难以评估。")

        # 数据缺失风险
        if not context.get("report_count", 0):
            risks.append("信息风险：本地研究报告缺失，分析依赖结构化快照数据，深度不足。")

        # 通用风险
        risks.append("市场系统性风险：报告期内宏观经济政策变化、行业政策调整等系统性因素均可能影响股价。")

        return risks

    def _build_recommendation(
        self,
        evidence: dict[str, Any],
        ticker: str,
        as_of: str,
    ) -> tuple[str, str, list[str], list[str]]:
        """生成研究性评级（不给确定性买卖建议）."""
        score = 0
        rationale_parts = []

        fi = evidence.get("financial_summary")
        if fi:
            roe = self._extract_field(fi.get("text", ""), "roe")
            if roe:
                roe_f = self._pct_to_float(roe)
                if roe_f is not None:
                    if roe_f >= 0.15:
                        score += 1
                        rationale_parts.append(f"ROE={roe}% 处于高水平")
                    elif roe_f < 0.05:
                        score -= 1
                        rationale_parts.append(f"ROE={roe}% 偏低")

        mm = evidence.get("market_metrics")
        if mm:
            pe = self._extract_field(mm.get("text", ""), "pe")
            if pe:
                try:
                    pe_f = float(pe)
                    if pe_f > 60:
                        score -= 1
                        rationale_parts.append(f"PE={pe} 估值偏高")
                    elif pe_f < 20:
                        score += 1
                        rationale_parts.append(f"PE={pe} 估值偏低")
                except ValueError:
                    pass

        if not fi and not mm:
            rationale_parts.append("核心数据不足，无法进行量化评估")

        if score >= 1:
            rating = "积极关注"
        elif score <= -1:
            rating = "谨慎关注"
        else:
            rating = "观察"

        if not rationale_parts:
            rationale_parts = ["数据不足，结论不确定，仅为模板占位"]

        rationale = "；".join(rationale_parts) + f"（基于 as_of={as_of} 数据，仅供参考）。"

        key_metrics = []
        if fi:
            key_metrics.append("ROE / 净利润同比（下一报告期）")
        if mm:
            key_metrics.append("PE / PB 估值区间")
        key_metrics.append("北向资金持仓变化")
        key_metrics.append("融资余额趋势")
        key_metrics.append("行业政策与宏观环境")

        uncertainties = [
            "本报告无 LLM 深度分析（provider=none），结论基于有限指标机械评分",
            "新闻为实时舆情，非 as_of 历史信息，存在信息时态偏差",
            "研究报告缺失（或有限），深度分析不足" if not evidence.get("reports") else "研究报告数量有限",
            "量化评分模型简单，不能替代专业分析师判断",
        ]

        return rating, rationale, key_metrics, uncertainties

    def _build_coverage_section(
        self, coverage: dict[str, Any], warnings: list[str]
    ) -> str:
        lines = [
            "### 数据覆盖度",
            "",
            f"- KB 总文档数：{coverage.get('collection_count', 0)}",
            f"- 新闻条数：{coverage.get('news_count', 0)}（当前舆情，非 as_of 历史数据）",
            f"- 研究报告条数：{coverage.get('report_count', 0)}",
            f"- snapshot/company_profile：{coverage.get('snapshot_company_profile_count', 0)}",
            f"- snapshot/latest_market_metrics：{coverage.get('snapshot_latest_market_metrics_count', 0)}",
            f"- snapshot/financial_indicator_summary：{coverage.get('snapshot_financial_indicator_summary_count', 0)}",
            f"- snapshot/northbound_summary：{coverage.get('snapshot_northbound_summary_count', 0)}",
            f"- snapshot/margin_summary：{coverage.get('snapshot_margin_summary_count', 0)}",
            f"- market_index_context：{coverage.get('market_context_count', 0)}",
            "",
        ]
        if warnings:
            lines.append("### 数据警告")
            lines.append("")
            for w in warnings:
                lines.append(f"- ⚠️ {w}")
            lines.append("")
        lines += [
            "---",
            "",
            "> **免责声明**：本报告由 QuantMind RAG 系统自动生成，基于有限数据集，"
            "仅供研究参考，**不构成任何投资建议**。投资有风险，入市需谨慎。",
        ]
        return "\n".join(lines)

    # ── 工具方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pct_to_float(val_str: str) -> float | None:
        """Parse a percentage value that may be stored as 13.63 (%) or 0.1363 (decimal).
        Returns value in [0, 1] range. Returns None on error."""
        try:
            v = float(val_str)
            # If |value| > 1, assume it is already expressed as percentage points
            return v / 100 if abs(v) > 1 else v
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_field(text: str, key: str) -> str | None:
        """从结构化文本中提取 key: value 格式的值."""
        if not text:
            return None
        patterns = [
            rf"{re.escape(key)}[：:=]\s*([^\n,，]+)",
            rf"{re.escape(key)}=([^\s,，\n]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理文本，保留换行，去除多余空行."""
        if not text:
            return ""
        lines = [ln.rstrip() for ln in text.splitlines()]
        cleaned: list[str] = []
        prev_empty = False
        for ln in lines:
            if not ln.strip():
                if not prev_empty:
                    cleaned.append("")
                prev_empty = True
            else:
                cleaned.append(ln)
                prev_empty = False
        return "\n".join(cleaned).strip()

    # ── LLM client 获取 ───────────────────────────────────────────────────────

    def _get_client(
        self, provider: str | None, model: str | None
    ) -> LLMClient:
        if provider is not None:
            return build_client(provider=provider, model=model)
        if self._llm_client is not None:
            return self._llm_client
        return build_client(provider="none")
