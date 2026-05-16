"""智能问答引擎：解析意图 → 读取本地流水线结果 → 生成图表与 LLM 摘要。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.utils import charts
from app.utils.data_loader import (
    list_top10_dates,
    load_final_recommendations_md,
    load_funnel_candidates,
    load_price_history_with_vol,
    load_strategies_json,
    load_strategy_detail,
    load_top10,
    load_validations_json,
)
from app.utils.llm_commentator import LLMCommentator, cached_llm_comment, stock_report_cached


@dataclass
class QAResult:
    text_response: str
    charts: list = field(default_factory=list)
    full_report: str = ""


_NAME_TO_TICKER = {
    "茅台": "600519.SH",
    "贵州茅台": "600519.SH",
    "格力": "000651.SZ",
    "格力电器": "000651.SZ",
    "宁德时代": "300750.SZ",
    "比亚迪": "002594.SZ",
    "招商银行": "600036.SH",
    "宁德": "300750.SZ",
}


def _resolve_pipeline_date(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    ds = list_top10_dates()
    return ds[-1] if ds else None


def _validation_for_ticker(date_str: str, ticker: str) -> dict[str, Any]:
    for row in load_validations_json(date_str):
        if str(row.get("ticker")) == ticker:
            return row
    return {
        "win_rate": 0.0,
        "avg_return": 0.0,
        "n_signals": 0,
        "max_loss": 0.0,
        "final_recommendation": "（未找到 validations.json 条目）",
    }


def _strategy_row(date_str: str, ticker: str) -> dict[str, Any]:
    detail = load_strategy_detail(date_str, ticker)
    if detail:
        return detail
    for row in load_strategies_json(date_str):
        if str(row.get("ticker")) == ticker:
            return row
    return {}


class QAEngine:
    def __init__(self, pipeline_date: str | None = None) -> None:
        self.commentator = LLMCommentator()
        self.pipeline_date = pipeline_date

    def answer(self, question: str) -> QAResult:
        date_str = _resolve_pipeline_date(self.pipeline_date)
        intent = self._parse_intent(question)
        if not date_str:
            return QAResult(
                text_response="当前仓库缺少 `data/recommendations/*/top10.json`，无法对齐流水线日期。请先运行推荐流水线。",
                charts=[],
                full_report="",
            )

        if intent["type"] == "compare":
            return self._compare_stocks(intent["tickers"], question, date_str)
        if intent["type"] == "find_stocks":
            return self._find_stocks(intent, question, date_str)
        if intent["type"] == "single_stock" and intent["tickers"]:
            return self._analyze_single_stock(intent["tickers"][0], question, date_str)
        return self._general_market_answer(question, date_str)

    def _parse_intent(self, question: str) -> dict[str, Any]:
        q = question.strip()
        tickers = re.findall(r"\b([036]\d{5})\.(SH|SZ)\b", q.upper())
        tickers = [f"{a}.{b}" for a, b in tickers]
        for name, tk in _NAME_TO_TICKER.items():
            if name in q and tk not in tickers:
                tickers.append(tk)

        if len(tickers) >= 2 and any(w in q for w in ("对比", "比较", "vs", "VS")):
            qtype = "compare"
        elif len(tickers) >= 1:
            qtype = "single_stock"
        elif any(w in q for w in ("找", "推荐", "筛选", "哪些")):
            qtype = "find_stocks"
        else:
            qtype = "market_overview"

        return {"type": qtype, "tickers": tickers[:6]}

    def _analyze_single_stock(self, ticker: str, question: str, date_str: str) -> QAResult:
        row = _strategy_row(date_str, ticker)
        if not row or not row.get("ticker"):
            px = load_price_history_with_vol(ticker, days=66)
            fig_p = charts.price_volume_ma(px, _asof_timestamp(date_str))
            hint = (
                f"未在 `reports/investment_pipeline/{date_str}/` 找到 **{ticker}** 的策略 JSON。\n\n"
                "可先运行投资流水线生成 strategies，或更换「流水线日期」。下方仅展示本地价格走势。"
            )
            return QAResult(text_response=hint, charts=[fig_p], full_report="")

        val = _validation_for_ticker(date_str, ticker)
        sigs = row.get("agent_signals") or {}
        full_report = stock_report_cached(ticker, sigs, row, val)

        px = load_price_history_with_vol(ticker, days=66)
        fig_px = charts.price_volume_ma(px, _asof_timestamp(date_str))
        fig_rd = charts.plot_agent_radar_themed(sigs, ticker, float(row.get("composite_signal") or 0))

        q_ans = cached_llm_comment(
            (
                f'用户问题："{question}"\n'
                f"标的 {ticker} 评级 {row.get('rating')} ，综合信号 {float(row.get('composite_signal') or 0):+.2f}。\n"
                f"历史胜率 {float(val.get('win_rate') or 0):.0%}，期望收益约 {float(val.get('avg_return') or 0):.2%}。\n"
                "请用 2~3 段中文直接回答用户问题，给出明确但不过度夸大的建议，并提示风险。"
            ),
            cache_key=f"qa_focus_{ticker}_{hash(question) % 10**8}",
            max_tokens=280,
        )

        return QAResult(text_response=q_ans, charts=[fig_px, fig_rd], full_report=full_report)

    def _compare_stocks(self, tickers: list[str], question: str, date_str: str) -> QAResult:
        take = tickers[:5]
        by_tk: dict[str, dict] = {}
        lines = []
        for tk in take:
            row = _strategy_row(date_str, tk)
            if row.get("agent_signals"):
                by_tk[tk] = row["agent_signals"]
                lines.append(f"- {tk}: 评级 {row.get('rating')} ，信号 {float(row.get('composite_signal') or 0):+.2f}")
            else:
                lines.append(f"- {tk}: 无本地策略文件")
        if by_tk:
            fig = charts.radar_compare(by_tk)
            chart_list = [fig]
        else:
            chart_list = []
        blob = "\n".join(lines)
        txt = cached_llm_comment(
            f'用户问题："{question}"\n已知摘要：\n{blob}\n请对比各标的的优劣与风险，2~4 句中文。',
            cache_key=f"qa_cmp_{'_'.join(take)}",
            max_tokens=320,
        )
        return QAResult(text_response=txt, charts=chart_list, full_report=blob)

    def _find_stocks(self, intent: dict[str, Any], question: str, date_str: str) -> QAResult:
        raw_top = load_top10(date_str)
        rows = raw_top.get("top10") if isinstance(raw_top, dict) else None
        tickers = [str(x.get("ticker")) for x in rows if isinstance(x, dict)][:8] if isinstance(rows, list) else []
        funnel = load_funnel_candidates(date_str)
        fs = funnel.get("funnel_stats") or {}
        layer6 = fs.get("layer6_out") or fs.get("layer6_in")
        summary = json.dumps(
            {"question": question, "top10_head": tickers, "funnel_final": layer6},
            ensure_ascii=False,
        )
        txt = cached_llm_comment(
            "基于本地漏斗/Top10 候选（未实时筛选因子），用简短中文回答用户的选股诉求，并列出可关注的 ticker 列表：\n"
            + summary,
            cache_key=f"qa_find_{date_str}",
            max_tokens=300,
        )
        return QAResult(text_response=txt, charts=[], full_report=json.dumps(tickers, ensure_ascii=False))

    def _general_market_answer(self, question: str, date_str: str) -> QAResult:
        raw = load_top10(date_str)
        md = load_final_recommendations_md(date_str)
        snippet = json.dumps(raw, ensure_ascii=False)[:1800]
        prompt = (
            f'用户问题："{question}"\n'
            f"流水线日期 {date_str}。\n"
            f"top10.json 摘要（截断）：{snippet}\n"
            "若有 final_recommendations.md，请结合市场情绪做概述。\n"
            f"markdown 报告摘录（截断）：{md[:1200]}"
        )
        txt = cached_llm_comment(
            prompt, cache_key=f"qa_mkt_{date_str}_{hash(question) % 10**8}", max_tokens=320
        )
        return QAResult(text_response=txt, charts=[], full_report="")


def _asof_timestamp(date_str: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(date_str)
    except Exception:
        return None

