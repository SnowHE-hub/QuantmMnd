"""tests/test_llm_reranker.py — LLM Reranker 单元测试（全量 Mock，不发真实 API 请求）.

测试用例
========
- test_output_schema       : 验证 RerankResult 输出字段完整性
- test_batch_split         : N=35 时自动分批调用（batch_size=20）
- test_json_fallback       : LLM 返回非法 JSON 时降级为 LGBM 顺序
- test_grounding_score     : 验证 reasoning_grounding_score 计算正确
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quantmind.models.llm_reranker import (
    LLMListwiseReranker,
    RerankCandidate,
    RerankResult,
)
from quantmind.models.llm_reranker_eval import (
    evaluate_batch,
    reasoning_grounding_score,
    rerank_lift,
)


# ============================================================================
# 测试固件
# ============================================================================


def _make_candidates(n: int = 10, start_ticker: int = 600000) -> list[RerankCandidate]:
    """生成 n 只候选股票，ticker 格式 600XXX.SH."""
    candidates = []
    for i in range(n):
        ticker = f"{start_ticker + i:06d}.SH"
        candidates.append(
            RerankCandidate(
                ticker=ticker,
                lgbm_score=float(n - i),  # 分数降序
                lgbm_rank=i + 1,
                pe_ttm=15.0 + i,
                pb=2.0 + i * 0.1,
                roe_ttm=0.15 + i * 0.01,
                accruals=-0.05 + i * 0.005,
                distance_to_52w_high=0.03 + i * 0.01,
                momentum_6m=0.10 + i * 0.02,
                volatility_3m=0.18 + i * 0.005,
            )
        )
    return candidates


def _make_valid_llm_response(candidates: list[RerankCandidate], top_n: int) -> str:
    """生成合法的 LLM JSON 响应（rankings + portfolio_thesis + risk_warnings）."""
    rankings = [
        {"ticker": c.ticker, "reason": f"ROE={c.roe_ttm*100:.1f}%，估值合理"}
        for c in candidates[:top_n]
    ]
    resp = {
        "rankings": rankings,
        "portfolio_thesis": "组合以高ROE低估值股票为主，动量因子辅助筛选，整体盈利质量良好。",
        "risk_warnings": [
            "行业集中度偏高，消费板块占比超过40%",
            "市场整体估值处于历史高位，回调风险较大",
            "流动性风险：部分成分股换手率偏低",
        ],
    }
    return json.dumps(resp, ensure_ascii=False)


def _make_reranker_with_mock(mock_content: str, batch_size: int = 50) -> LLMListwiseReranker:
    """返回一个注入了 Mock Router 的 LLMListwiseReranker."""
    reranker = LLMListwiseReranker(provider="ollama", model="test-model", batch_size=batch_size)
    mock_resp = MagicMock()
    mock_resp.content = mock_content
    mock_router = MagicMock()
    mock_router.chat.return_value = mock_resp
    reranker._router = mock_router
    return reranker


# ============================================================================
# test_output_schema
# ============================================================================


class TestOutputSchema:
    """验证 RerankResult 输出字段完整性."""

    def test_all_required_fields_present(self):
        """RerankResult 必须包含 rank, ticker, lgbm_rank, reason,
        portfolio_thesis, risk_warnings, is_fallback."""
        required = {
            "rank", "ticker", "lgbm_rank", "reason",
            "portfolio_thesis", "risk_warnings", "is_fallback",
        }
        r = RerankResult(rank=1, ticker="600000.SH", lgbm_rank=3)
        assert required.issubset(set(r.__dataclass_fields__.keys()))

    def test_portfolio_thesis_default_empty(self):
        r = RerankResult(rank=1, ticker="600000.SH", lgbm_rank=1)
        assert r.portfolio_thesis == ""

    def test_risk_warnings_default_empty_list(self):
        r = RerankResult(rank=1, ticker="600000.SH", lgbm_rank=1)
        assert isinstance(r.risk_warnings, list)
        assert len(r.risk_warnings) == 0

    def test_rerank_result_populated(self):
        """通过 mock LLM 调用验证 RerankResult 字段被正确填充."""
        candidates = _make_candidates(n=5)
        reranker = _make_reranker_with_mock(_make_valid_llm_response(candidates, top_n=5))
        results = reranker.rerank(candidates, top_n=5)

        assert len(results) == 5
        for r in results:
            assert isinstance(r, RerankResult)
            assert r.rank >= 1
            assert r.ticker in {c.ticker for c in candidates}
            assert r.lgbm_rank >= 1
            # portfolio_thesis 应被填充（来自 mock LLM）
            assert isinstance(r.portfolio_thesis, str)
            # risk_warnings 应至少 3 条
            assert isinstance(r.risk_warnings, list)
            assert len(r.risk_warnings) >= 3

    def test_all_results_share_portfolio_info(self):
        """所有 RerankResult 应共享相同的 portfolio_thesis 和 risk_warnings."""
        candidates = _make_candidates(n=5)
        reranker = _make_reranker_with_mock(_make_valid_llm_response(candidates, top_n=5))
        results = reranker.rerank(candidates, top_n=5)

        theses = {r.portfolio_thesis for r in results}
        assert len(theses) == 1, "所有结果应有相同的 portfolio_thesis"

        risk_sets = [tuple(r.risk_warnings) for r in results]
        assert len(set(risk_sets)) == 1, "所有结果应有相同的 risk_warnings"

    def test_rerank_candidate_has_shap_values(self):
        """RerankCandidate 应包含 shap_values 字段（默认空 dict）."""
        c = RerankCandidate(ticker="600000.SH", lgbm_score=1.0, lgbm_rank=1)
        assert hasattr(c, "shap_values")
        assert isinstance(c.shap_values, dict)


# ============================================================================
# test_batch_split
# ============================================================================


class TestBatchSplit:
    """N=35 时自动分批调用（batch_size=20）."""

    def test_batch_split_called_multiple_times(self):
        """候选数 35 > batch_size 20，应触发分批调用（多次 LLM 请求）."""
        candidates = _make_candidates(n=35)

        mock_resp = MagicMock()
        mock_router = MagicMock()

        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            # 每批返回 batch 前 5 只的合法 JSON
            msgs = kwargs.get("messages", [])
            # 从 user 消息的表格中提取 ticker
            user_msg = next((m for m in msgs if m.get("role") == "user"), {})
            content = user_msg.get("content", "")
            import re
            found_tickers = re.findall(r"\d{6}\.SH", content)[:10]
            rankings = [{"ticker": t, "reason": "高ROE"} for t in found_tickers[:5]]
            resp_content = json.dumps({
                "rankings": rankings,
                "portfolio_thesis": "批次组合逻辑",
                "risk_warnings": ["风险A", "风险B", "风险C"],
            }, ensure_ascii=False)
            r = MagicMock()
            r.content = resp_content
            return r

        mock_router.chat.side_effect = side_effect

        reranker = LLMListwiseReranker(provider="ollama", model="test", batch_size=20)
        reranker._router = mock_router
        results = reranker.rerank(candidates, top_n=10)

        # 分批调用：35 只 / 20 = 2 批 + 1 次最终合并 = 至少 2 次调用
        assert call_count[0] >= 2, f"Expected ≥2 LLM calls, got {call_count[0]}"
        assert len(results) <= 10

    def test_no_batch_when_within_limit(self):
        """候选数 <= batch_size 时，只调用 1 次 LLM."""
        candidates = _make_candidates(n=10)
        mock_resp = MagicMock()
        mock_resp.content = _make_valid_llm_response(candidates, top_n=5)
        mock_router = MagicMock()
        mock_router.chat.return_value = mock_resp

        reranker = LLMListwiseReranker(provider="ollama", model="test", batch_size=20)
        reranker._router = mock_router
        reranker.rerank(candidates, top_n=5)

        assert mock_router.chat.call_count == 1


# ============================================================================
# test_json_fallback
# ============================================================================


class TestJsonFallback:
    """LLM 返回非法 JSON 时降级处理."""

    def test_invalid_json_falls_back_to_lgbm_order(self):
        """LLM 返回非法 JSON 时，结果应按 LGBM 顺序排列，is_fallback=True."""
        candidates = _make_candidates(n=5)
        reranker = _make_reranker_with_mock("这不是 JSON 格式的文本，完全无效。")
        results = reranker.rerank(candidates, top_n=5)

        assert len(results) == 5
        assert all(r.is_fallback for r in results)
        # 降级后排名应与 LGBM 顺序一致（rank 1 对应 lgbm_rank 1）
        for r in results:
            assert r.rank == r.lgbm_rank

    def test_partial_json_gracefully_handled(self):
        """LLM 返回部分有效 JSON（只有 rankings，缺少 portfolio_thesis），应正常解析。"""
        candidates = _make_candidates(n=5)
        partial_resp = json.dumps([
            {"ticker": c.ticker, "reason": "理由"}
            for c in candidates[:3]
        ], ensure_ascii=False)
        reranker = _make_reranker_with_mock(partial_resp)
        results = reranker.rerank(candidates, top_n=5)

        # 至少 3 只能被解析，is_fallback=False
        assert len(results) == 5
        assert not any(r.is_fallback for r in results)
        # 前 3 名来自 LLM，后 2 名由 fill_missing 补充
        llm_tickers = {c.ticker for c in candidates[:3]}
        assert results[0].ticker in llm_tickers

    def test_llm_exception_falls_back(self):
        """LLM 调用本身抛出异常时，应降级为 LGBM 顺序。"""
        candidates = _make_candidates(n=5)
        reranker = LLMListwiseReranker(provider="ollama", model="test", batch_size=50)
        mock_router = MagicMock()
        mock_router.chat.side_effect = ConnectionError("Ollama service unavailable")
        reranker._router = mock_router

        import warnings
        with warnings.catch_warnings(record=True):
            results = reranker.rerank(candidates, top_n=5)

        assert len(results) == 5
        assert all(r.is_fallback for r in results)

    def test_empty_candidates_returns_empty(self):
        """空候选列表应返回空结果。"""
        reranker = LLMListwiseReranker()
        results = reranker.rerank([], top_n=10)
        assert results == []


# ============================================================================
# test_grounding_score
# ============================================================================


class TestGroundingScore:
    """验证 reasoning_grounding_score 计算正确性."""

    def _make_result(self, ticker: str, reason: str) -> RerankResult:
        return RerankResult(rank=1, ticker=ticker, lgbm_rank=1, reason=reason)

    def _make_candidate(self, ticker: str, **kwargs) -> RerankCandidate:
        return RerankCandidate(
            ticker=ticker, lgbm_score=1.0, lgbm_rank=1, **kwargs
        )

    def test_perfect_grounding(self):
        """reason 中的数字全部来自因子数据，grounding=1.0."""
        c = self._make_candidate("600000.SH", roe_ttm=0.268, pe_ttm=15.5)
        # reason 引用 ROE 26.8% 和 PE 15.5
        r = self._make_result("600000.SH", "ROE_TTM=26.8%，PE_TTM=15.5，估值合理")
        score = reasoning_grounding_score(r, c)
        assert score >= 0.8, f"Expected high grounding score, got {score}"

    def test_zero_grounding(self):
        """reason 中的数字完全不在因子数据中，grounding 应较低。"""
        c = self._make_candidate("600000.SH", roe_ttm=0.10, pe_ttm=20.0)
        # reason 引用完全虚构的数字
        r = self._make_result("600000.SH", "ROE=99.9%，预计涨幅500%")
        score = reasoning_grounding_score(r, c)
        assert score <= 0.2, f"Expected low grounding score, got {score}"

    def test_empty_reason_returns_one(self):
        """空 reason 无幻觉，应返回 1.0。"""
        c = self._make_candidate("600000.SH", roe_ttm=0.15)
        r = self._make_result("600000.SH", "")
        assert reasoning_grounding_score(r, c) == 1.0

    def test_no_numbers_returns_one(self):
        """reason 中无数字，无幻觉，返回 1.0。"""
        c = self._make_candidate("600000.SH", roe_ttm=0.15)
        r = self._make_result("600000.SH", "盈利质量良好，动量强势，估值合理")
        assert reasoning_grounding_score(r, c) == 1.0

    def test_rerank_lift_positive(self):
        """LLM 精排比 LGBM 粗排更贴近真实收益时，lift 应为正。"""
        # 假设真实收益：[0.3, 0.2, 0.1, 0.0, -0.1]
        # LGBM 排名：1,2,3,4,5（基本匹配，但 LLM 更精准）
        lgbm_ranks   = [2, 1, 3, 4, 5]  # LGBM 前两只搞反了
        llm_ranks    = [1, 2, 3, 4, 5]  # LLM 完美匹配
        realized_rets = [0.3, 0.2, 0.1, 0.0, -0.1]
        lgbm_ic, llm_ic, lift = rerank_lift(lgbm_ranks, llm_ranks, realized_rets)
        assert llm_ic >= lgbm_ic, f"Expected llm_ic >= lgbm_ic, got {llm_ic:.4f} vs {lgbm_ic:.4f}"
        assert lift >= 0, f"Expected lift >= 0, got {lift:.4f}"

    def test_evaluate_batch_basic(self):
        """evaluate_batch 应正确计算 avg_grounding 和 parse_success_rate。"""
        candidates = _make_candidates(n=5)
        results = [
            RerankResult(rank=i + 1, ticker=c.ticker, lgbm_rank=c.lgbm_rank,
                         reason=f"ROE={c.roe_ttm*100:.1f}%")
            for i, c in enumerate(candidates)
        ]
        report = evaluate_batch(results, candidates)

        assert 0.0 <= report.avg_grounding <= 1.0
        assert report.n_results == 5
        assert report.n_fallback == 0
        assert report.parse_success_rate == 1.0

    def test_evaluate_batch_with_realized_returns(self):
        """提供实际收益时，evaluate_batch 应计算 rerank_lift。"""
        candidates = _make_candidates(n=5)
        results = [
            RerankResult(rank=i + 1, ticker=c.ticker, lgbm_rank=c.lgbm_rank)
            for i, c in enumerate(candidates)
        ]
        realized = {c.ticker: 0.1 - i * 0.03 for i, c in enumerate(candidates)}
        report = evaluate_batch(results, candidates, realized_returns=realized)

        assert not np.isnan(report.lgbm_rank_ic)
        assert not np.isnan(report.llm_rank_ic)
        assert not np.isnan(report.rerank_lift)
