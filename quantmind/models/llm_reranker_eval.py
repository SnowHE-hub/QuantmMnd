"""quantmind.models.llm_reranker_eval — LLM 重排质量评估工具.

指标
====
- reasoning_grounding_score：reasoning 中引用的数字是否来自输入数据（幻觉检测）
- rerank_lift：LLM 重排相对 LGBM 粗排的 Rank IC 提升
- EvalReport：汇总评估结果的数据类
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = [
    "EvalReport",
    "reasoning_grounding_score",
    "rerank_lift",
    "evaluate_batch",
]


# ============================================================================
# 数据类
# ============================================================================


@dataclass
class EvalReport:
    """单次 LLM 重排的评估结果汇总."""

    # 每只股票的 grounding 分数
    grounding_scores: dict[str, float] = field(default_factory=dict)
    # 整体平均 grounding 分数（0~1，越高越好）
    avg_grounding: float = float("nan")

    # LGBM 粗排 vs 实际收益的 Rank IC
    lgbm_rank_ic: float = float("nan")
    # LLM 精排 vs 实际收益的 Rank IC
    llm_rank_ic: float = float("nan")
    # 提升量（llm_rank_ic - lgbm_rank_ic），正 = LLM 有效
    rerank_lift: float = float("nan")

    # 其他诊断
    n_results: int = 0
    n_fallback: int = 0
    parse_success_rate: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "avg_grounding": round(self.avg_grounding, 4) if not np.isnan(self.avg_grounding) else None,
            "lgbm_rank_ic": round(self.lgbm_rank_ic, 4) if not np.isnan(self.lgbm_rank_ic) else None,
            "llm_rank_ic": round(self.llm_rank_ic, 4) if not np.isnan(self.llm_rank_ic) else None,
            "rerank_lift": round(self.rerank_lift, 4) if not np.isnan(self.rerank_lift) else None,
            "n_results": self.n_results,
            "n_fallback": self.n_fallback,
            "parse_success_rate": round(self.parse_success_rate, 4) if not np.isnan(self.parse_success_rate) else None,
        }


# ============================================================================
# 核心评估函数
# ============================================================================


def _extract_numbers_from_text(text: str) -> list[float]:
    """从文本中提取所有数字（包括百分比、正负号）."""
    # 匹配：-1.23、23.4%、+0.05 等
    raw = re.findall(r"[-+]?\d+\.?\d*%?", text)
    results = []
    for s in raw:
        s_clean = s.rstrip("%")
        try:
            v = float(s_clean)
            if s.endswith("%"):
                v /= 100.0
            results.append(v)
        except ValueError:
            continue
    return results


def _get_candidate_numbers(candidate) -> set[float]:
    """从 RerankCandidate 的因子数据中提取所有数字集合."""
    factor_fields = [
        "pe_ttm", "pb", "roe_ttm", "accruals",
        "distance_to_52w_high", "momentum_6m", "volatility_3m",
    ]
    nums: set[float] = set()
    for f in factor_fields:
        v = getattr(candidate, f, float("nan"))
        if v is not None and not np.isnan(v):
            # 添加原始值及其百分比形式（两种都加，因为 LLM 可能输出任一）
            nums.add(round(v, 4))
            nums.add(round(v * 100, 2))
    return nums


def _number_is_grounded(num: float, ground_truth: set[float], tol: float = 0.02) -> bool:
    """检查 num 是否在 ground_truth 的 tol 误差范围内."""
    for gt in ground_truth:
        if abs(num - gt) <= tol * (abs(gt) + 1e-6):
            return True
    # 绝对误差兜底（适应小数）
    return any(abs(num - gt) < 0.005 for gt in ground_truth)


def reasoning_grounding_score(
    result,
    candidate,
    tol: float = 0.05,
) -> float:
    """计算单支股票 reasoning 的 grounding 分数.

    检查 reasoning 文本中出现的数字是否都能在候选股票的输入数据中找到对应值。

    Args:
        result:    RerankResult（含 reason 字段）
        candidate: RerankCandidate（含因子数据）
        tol:       允许的相对误差（默认 5%）

    Returns:
        0~1 的浮点数：被 grounded 的数字占 reasoning 中总数字数量的比例。
        reasoning 无数字时返回 1.0（无幻觉）。
    """
    reason = getattr(result, "reason", "") or ""
    if not reason.strip():
        return 1.0  # 无 reason，无幻觉

    nums_in_reason = _extract_numbers_from_text(reason)
    if not nums_in_reason:
        return 1.0

    ground_set = _get_candidate_numbers(candidate)
    grounded_count = sum(
        1 for n in nums_in_reason if _number_is_grounded(n, ground_set, tol)
    )
    return grounded_count / len(nums_in_reason)


def rerank_lift(
    lgbm_ranks: list[int],
    llm_ranks: list[int],
    realized_returns: list[float],
) -> tuple[float, float, float]:
    """计算 LLM 重排相比 LGBM 粗排的 Rank IC 提升.

    Args:
        lgbm_ranks:        LGBM 粗排排名（1=最优，越小越好）
        llm_ranks:         LLM 精排排名（1=最优）
        realized_returns:  对应股票的实际收益率

    Returns:
        (lgbm_ic, llm_ic, lift)
        lift = llm_ic - lgbm_ic，正值表示 LLM 有效改善排序
    """
    if len(lgbm_ranks) != len(llm_ranks) != len(realized_returns):
        raise ValueError("All input lists must have the same length.")

    ret_arr = np.array(realized_returns, dtype=float)

    # Spearman 相关（rank vs return），rank 越小应对应 return 越高，故取负
    lgbm_neg = [-r for r in lgbm_ranks]
    llm_neg  = [-r for r in llm_ranks]

    lgbm_ic, _ = stats.spearmanr(lgbm_neg, ret_arr)
    llm_ic,  _ = stats.spearmanr(llm_neg, ret_arr)

    lgbm_ic = float(lgbm_ic) if not np.isnan(lgbm_ic) else float("nan")
    llm_ic  = float(llm_ic)  if not np.isnan(llm_ic)  else float("nan")
    lift = llm_ic - lgbm_ic if not (np.isnan(lgbm_ic) or np.isnan(llm_ic)) else float("nan")

    return lgbm_ic, llm_ic, lift


def evaluate_batch(
    results,
    candidates,
    realized_returns: dict[str, float] | None = None,
) -> EvalReport:
    """对一批 LLM 重排结果进行全面评估.

    Args:
        results:           list[RerankResult]
        candidates:        list[RerankCandidate]
        realized_returns:  ticker → 实际收益率（可选，提供时计算 rerank_lift）

    Returns:
        EvalReport
    """
    if not results:
        return EvalReport()

    cand_map = {c.ticker: c for c in candidates}
    n_fallback = sum(1 for r in results if r.is_fallback)

    # Grounding scores
    grounding_scores: dict[str, float] = {}
    for r in results:
        c = cand_map.get(r.ticker)
        if c is not None:
            grounding_scores[r.ticker] = reasoning_grounding_score(r, c)

    avg_grounding = (
        float(np.mean(list(grounding_scores.values())))
        if grounding_scores else float("nan")
    )

    # Rerank lift（若提供实际收益）
    lgbm_ic = llm_ic = lift = float("nan")
    if realized_returns:
        common = [r for r in results if r.ticker in realized_returns]
        if len(common) >= 3:
            lgbm_rnks = [r.lgbm_rank for r in common]
            llm_rnks  = [r.rank for r in common]
            rets      = [realized_returns[r.ticker] for r in common]
            lgbm_ic, llm_ic, lift = rerank_lift(lgbm_rnks, llm_rnks, rets)

    return EvalReport(
        grounding_scores=grounding_scores,
        avg_grounding=avg_grounding,
        lgbm_rank_ic=lgbm_ic,
        llm_rank_ic=llm_ic,
        rerank_lift=lift,
        n_results=len(results),
        n_fallback=n_fallback,
        parse_success_rate=1.0 - n_fallback / len(results) if results else float("nan"),
    )
