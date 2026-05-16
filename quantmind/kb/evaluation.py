"""quantmind.kb.evaluation — RAG 质量评估框架.

指标：
- hit_rate_at_k:      命中率（golden answer 出现在 top-k 结果中的比例）
- mrr:                Mean Reciprocal Rank（平均倒数排名）
- pit_correctness:    PIT 正确性测试（截止日期后文档不应出现）
- faithfulness_score: 忠实度（检索结果是否包含 golden answer 的关键词）

使用方式::

    from quantmind.kb.evaluation import KBEvaluator, GoldenCase

    evaluator = KBEvaluator(retriever)
    results = evaluator.run(golden_cases)
    evaluator.print_report(results)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from loguru import logger

__all__ = ["GoldenCase", "EvalResult", "KBEvaluator"]


@dataclass
class GoldenCase:
    """单条黄金测试用例.

    Attributes:
        query:           查询文本
        expected_tickers: 期望检索到的股票代码列表
        expected_keywords: 期望出现在检索结果中的关键词列表
        as_of:           PIT 截止日期
        source_type:     期望的文档类型（news/report/filing/None=任意）
        description:     用例描述（用于报告）
        pit_forbidden_date: 此日期后的文档不应出现（测试 PIT 过滤）
    """

    query: str
    expected_tickers: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    as_of: date | None = None
    source_type: str | None = None
    description: str = ""
    pit_forbidden_date: date | None = None  # 若设置，验证无此日期后的结果


@dataclass
class EvalResult:
    """单个测试用例的评估结果."""

    case: GoldenCase
    retrieved_texts: list[str]
    retrieved_tickers: list[str]
    retrieved_dates: list[date | None]

    # 指标
    hit: bool = False               # top-k 中是否命中任一 keyword
    reciprocal_rank: float = 0.0    # 1/首个命中位置
    faithfulness: float = 0.0       # keyword 覆盖率
    pit_violation: bool = False     # 是否出现 PIT 违规
    error: str = ""                 # 异常信息


class KBEvaluator:
    """知识库 RAG 质量评估器.

    Args:
        retriever:      HybridRetriever 实例
        top_k:          检索数量，默认 5
    """

    def __init__(self, retriever: Any, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k

    # ── 主评估入口 ────────────────────────────────────────────────────────────

    def run(self, cases: list[GoldenCase]) -> list[EvalResult]:
        """运行所有测试用例.

        Args:
            cases: GoldenCase 列表

        Returns:
            EvalResult 列表
        """
        results = []
        for i, case in enumerate(cases):
            logger.debug(f"[KBEval] case {i+1}/{len(cases)}: {case.query[:40]}")
            result = self._evaluate_case(case)
            results.append(result)
        return results

    def _evaluate_case(self, case: GoldenCase) -> EvalResult:
        """评估单个用例."""
        try:
            retrieved = self.retriever.retrieve(
                query=case.query,
                top_k=self.top_k,
                as_of=case.as_of,
                ticker_filter=case.expected_tickers[0] if len(case.expected_tickers) == 1 else None,
                source_type_filter=case.source_type,
            )
        except Exception as e:
            return EvalResult(
                case=case,
                retrieved_texts=[],
                retrieved_tickers=[],
                retrieved_dates=[],
                error=str(e),
            )

        texts = [r.text for r in retrieved]
        tickers = [r.ticker for r in retrieved]
        dates = [r.published_date for r in retrieved]

        combined_text = " ".join(texts).lower()

        # hit_rate：任一关键词出现在 top-k 文本中
        hit = False
        if case.expected_keywords:
            hit = any(kw.lower() in combined_text for kw in case.expected_keywords)
        elif case.expected_tickers:
            hit = any(t in tickers for t in case.expected_tickers)

        # MRR：找到首个命中位置
        rr = 0.0
        for rank, (text, ticker) in enumerate(zip(texts, tickers), start=1):
            kw_match = case.expected_keywords and any(
                kw.lower() in text.lower() for kw in case.expected_keywords
            )
            ticker_match = case.expected_tickers and ticker in case.expected_tickers
            if kw_match or ticker_match:
                rr = 1.0 / rank
                break

        # faithfulness：关键词覆盖率
        faith = 0.0
        if case.expected_keywords:
            found = sum(1 for kw in case.expected_keywords if kw.lower() in combined_text)
            faith = found / len(case.expected_keywords)

        # PIT 违规检测
        pit_violation = False
        if case.pit_forbidden_date:
            for d in dates:
                if d and d > case.pit_forbidden_date:
                    pit_violation = True
                    break

        return EvalResult(
            case=case,
            retrieved_texts=texts,
            retrieved_tickers=tickers,
            retrieved_dates=dates,
            hit=hit,
            reciprocal_rank=rr,
            faithfulness=faith,
            pit_violation=pit_violation,
        )

    # ── 聚合指标 ──────────────────────────────────────────────────────────────

    @staticmethod
    def hit_rate_at_k(results: list[EvalResult]) -> float:
        """命中率（top-k 中至少有1个相关文档的比例）."""
        if not results:
            return 0.0
        return sum(r.hit for r in results) / len(results)

    @staticmethod
    def mrr(results: list[EvalResult]) -> float:
        """Mean Reciprocal Rank."""
        if not results:
            return 0.0
        return sum(r.reciprocal_rank for r in results) / len(results)

    @staticmethod
    def faithfulness_score(results: list[EvalResult]) -> float:
        """平均关键词覆盖率."""
        if not results:
            return 0.0
        return sum(r.faithfulness for r in results) / len(results)

    @staticmethod
    def pit_correctness(results: list[EvalResult]) -> float:
        """PIT 正确率（无违规用例的比例）."""
        pit_cases = [r for r in results if r.case.pit_forbidden_date]
        if not pit_cases:
            return 1.0  # 无 PIT 测试用例
        correct = sum(not r.pit_violation for r in pit_cases)
        return correct / len(pit_cases)

    # ── 报告输出 ──────────────────────────────────────────────────────────────

    def print_report(self, results: list[EvalResult]) -> None:
        """打印评估报告到控制台."""
        hit = self.hit_rate_at_k(results)
        m = self.mrr(results)
        faith = self.faithfulness_score(results)
        pit = self.pit_correctness(results)

        print(f"\n{'='*60}")
        print(f"QuantMind KB 评估报告（{len(results)} 用例）")
        print(f"{'='*60}")
        print(f"Hit Rate@{self.top_k}    : {hit:.3f} ({hit*100:.1f}%)")
        print(f"MRR              : {m:.3f}")
        print(f"Faithfulness     : {faith:.3f} ({faith*100:.1f}%)")
        print(f"PIT Correctness  : {pit:.3f} ({pit*100:.1f}%)")
        print(f"{'─'*60}")

        for i, r in enumerate(results):
            status = "✅" if r.hit else "❌"
            pit_flag = "⚠️PIT!" if r.pit_violation else ""
            err_flag = f"[ERR: {r.error[:30]}]" if r.error else ""
            print(
                f"{i+1:2d}. {status} {pit_flag}{err_flag} "
                f"| MRR={r.reciprocal_rank:.2f} "
                f"| Faith={r.faithfulness:.2f} "
                f"| {r.case.description or r.case.query[:30]}"
            )

        print(f"{'='*60}\n")

    def to_dict(self, results: list[EvalResult]) -> dict[str, Any]:
        """将评估结果序列化为字典（用于 HTML 报告）."""
        return {
            "summary": {
                "total_cases": len(results),
                "hit_rate": self.hit_rate_at_k(results),
                "mrr": self.mrr(results),
                "faithfulness": self.faithfulness_score(results),
                "pit_correctness": self.pit_correctness(results),
                "top_k": self.top_k,
            },
            "cases": [
                {
                    "query": r.case.query,
                    "description": r.case.description,
                    "hit": r.hit,
                    "reciprocal_rank": r.reciprocal_rank,
                    "faithfulness": r.faithfulness,
                    "pit_violation": r.pit_violation,
                    "error": r.error,
                    "retrieved_count": len(r.retrieved_texts),
                    "retrieved_snippets": [t[:100] for t in r.retrieved_texts[:3]],
                }
                for r in results
            ],
        }
