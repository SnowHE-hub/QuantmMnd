"""quantmind.models.llm_reranker — LLM 列表式重排序器.

流程
====
1. LGBM 粗排 → 取 Top-N（通常 50）
2. 计算 SHAP 因子贡献（可选，LightGBM built-in）
3. 将候选股票因子数据格式化为 Prompt 表格
4. 调用本地 Ollama（qwen2.5:7b）或生产 DeepSeek 进行列表重排
5. 解析 LLM 返回的 JSON → RerankResult（含 portfolio_thesis、risk_warnings）
6. 解析失败时自动降级（保持 LGBM 顺序）

输出 JSON 格式（完整版）
======================
{
  "rankings": [{"ticker":"600000.SH","reason":"高ROE与低估值"}, ...],
  "portfolio_thesis": "整体组合以消费龙头为核心...",
  "risk_warnings": ["行业集中风险", "流动性风险", "估值偏高"]
}
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "LLMListwiseReranker",
    "RerankCandidate",
    "RerankResult",
]


# ============================================================================
# 数据容器
# ============================================================================


@dataclass
class RerankCandidate:
    """单支候选股票，含 LGBM 分数、因子数据和 SHAP 贡献（供 LLM Prompt 使用）."""

    ticker: str
    lgbm_score: float
    lgbm_rank: int              # 1 = LGBM 最高分

    # 基本面因子
    pe_ttm: float = float("nan")
    pb: float = float("nan")
    roe_ttm: float = float("nan")
    accruals: float = float("nan")

    # 技术因子
    distance_to_52w_high: float = float("nan")
    momentum_6m: float = float("nan")
    volatility_3m: float = float("nan")

    # SHAP 因子贡献（factor_name → shap_value，由 LGBMRankerModel.explain() 填充）
    shap_values: dict[str, float] = field(default_factory=dict)


@dataclass
class RerankResult:
    """LLM 重排后单支股票的最终结果（含组合级别信息）."""

    rank: int                           # LLM 排名（1 = 最优）
    ticker: str
    lgbm_rank: int                      # 原 LGBM 排名
    reason: str = ""                    # LLM 给出的个股理由

    # 组合级别信息（所有 RerankResult 共享同一份，从第一次 LLM 调用提取）
    portfolio_thesis: str = ""          # 整体组合投资逻辑
    risk_warnings: list[str] = field(default_factory=list)  # 至少 3 条主要风险

    is_fallback: bool = False           # True = JSON 解析失败，已降级为 LGBM 顺序


# ============================================================================
# Prompts
# ============================================================================

_SYSTEM_PROMPT = """\
你是一名专注于 A 股 CSI300 指数成分股的量化研究员。
任务：对候选股票列表进行精细排名，并提供组合级别的投资分析。

评估维度（重要性依次递减）：
1. 盈利质量：ROE_TTM 高、应计利润率（accruals）低（低应计=高现金流质量）
2. 估值：PE_TTM 和 PB 综合判断（避免极高估值泡沫）
3. 动量与技术：distance_to_52w_high 高（接近年高=强势），momentum_6m 正向
4. 风险控制：volatility_3m 低风险优先，但强动量股可适当放宽
5. SHAP 贡献：若提供 SHAP 值，优先考虑正贡献因子集中的股票

安全规则：
- 所有数据已截止于 as_of 日期（PIT 原则），不包含任何未来信息
- 严禁基于训练数据中的股票历史知识做主观判断，只能依据表格中的因子值
- reason 中引用的具体数字必须来自输入数据，不得捏造
- NaN 表示该项数据缺失，缺失维度不作惩罚

输出格式（严格按此 JSON，不加任何说明文字）：
{
  "rankings": [{"ticker":"代码","reason":"个股理由（引用具体数字，30字内）"}, ...],
  "portfolio_thesis": "整体组合投资逻辑（100字内，说明为什么这些股票能形成有效组合）",
  "risk_warnings": ["风险1（具体）", "风险2（具体）", "风险3（具体）"]
}
"""

_USER_PROMPT_TEMPLATE = """\
截止日期（as_of）：{as_of_date}
候选股票数量：{n_candidates}
请从以下 {n_candidates} 只股票中选出并排序前 {top_n} 名（第1名=最优）。

| LGBM排名 | 代码     | PE_TTM  | PB     | ROE_TTM | Accruals | 距52W高% | 动量6M% | 波动3M% |
|---------|---------|---------|--------|---------|----------|---------|---------|---------|
{table_rows}
{shap_section}
请严格按如下格式输出 JSON（只输出 JSON，不要有任何其他文字）：
{{
  "rankings": [{{"ticker":"代码","reason":"理由（引用具体数字，30字内）"}}, ...],
  "portfolio_thesis": "组合投资逻辑（100字内）",
  "risk_warnings": ["风险1", "风险2", "风险3"]
}}
"""


def _fmt(val: float, pct: bool = False, decimals: int = 2) -> str:
    if np.isnan(val):
        return "N/A"
    if pct:
        return f"{val * 100:.{decimals}f}%"
    return f"{val:.{decimals}f}"


def _top2_shap(shap_vals: dict[str, float]) -> str:
    """返回 SHAP 贡献最大的 2 个因子（按绝对值排序）."""
    if not shap_vals:
        return "—"
    sorted_items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
    parts = [f"{k}({v:+.3f})" for k, v in sorted_items]
    return ", ".join(parts)


# ============================================================================
# LLM Listwise Reranker
# ============================================================================


class LLMListwiseReranker:
    """LLM 列表式重排序器：LGBM 粗排 Top-N → LLM 精排 Top-K.

    参数
    ----
    provider:   LLM provider 名称（默认 "ollama"，生产用 "deepseek"）
    model:      模型名称（默认 "qwen2.5:7b"）
    batch_size: 每次 LLM 调用的最大候选数（超出时分批调用）
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:7b",
        batch_size: int = 50,
    ) -> None:
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self._router: Any = None

    def _get_router(self) -> Any:
        if self._router is None:
            from quantmind.core.llm_router import LLMRouter
            self._router = LLMRouter()
        return self._router

    def rerank(
        self,
        candidates: list[RerankCandidate],
        top_n: int = 30,
        as_of_date: str = "",
    ) -> list[RerankResult]:
        """对候选股票进行 LLM 重排，返回 top_n 个结果.

        Args:
            candidates:  已按 LGBM 分数降序排列的候选股票列表
            top_n:       目标输出排名数
            as_of_date:  数据截止日期字符串（用于 Prompt PIT 说明）

        Returns:
            按 LLM 排名升序的 RerankResult 列表（rank=1 最优），
            每个结果均含 portfolio_thesis 和 risk_warnings。
        """
        if not candidates:
            return []

        top_n = min(top_n, len(candidates))

        if len(candidates) <= self.batch_size:
            return self._rerank_batch(candidates, top_n, as_of_date)
        else:
            return self._rerank_multi_batch(candidates, top_n, as_of_date)

    def _rerank_batch(
        self,
        candidates: list[RerankCandidate],
        top_n: int,
        as_of_date: str,
    ) -> list[RerankResult]:
        """单批次 LLM 调用."""
        prompt = self._build_user_prompt(candidates, top_n, as_of_date)
        router = self._get_router()

        try:
            resp = router.chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                provider=self.provider,
                model=self.model,
                temperature=0.05,
                max_tokens=2048,
                fallback=None,
            )
            raw_text = resp.content
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"LLM call failed ({e}), falling back to LGBM order", stacklevel=2)
            return self._fallback_results(candidates, top_n)

        ordered_tickers, is_fallback = self._parse_rankings(raw_text, candidates, top_n)
        reason_map = self._extract_field_map(raw_text, "reason")
        portfolio_thesis = self._extract_portfolio_thesis(raw_text)
        risk_warnings = self._extract_risk_warnings(raw_text)

        ticker_map = {c.ticker: c for c in candidates}
        results = []
        for rank, ticker in enumerate(ordered_tickers[:top_n], start=1):
            cand = ticker_map.get(ticker)
            results.append(
                RerankResult(
                    rank=rank,
                    ticker=ticker,
                    lgbm_rank=cand.lgbm_rank if cand else 999,
                    reason=reason_map.get(ticker, ""),
                    portfolio_thesis=portfolio_thesis,
                    risk_warnings=risk_warnings,
                    is_fallback=is_fallback,
                )
            )
        return results

    def _rerank_multi_batch(
        self,
        candidates: list[RerankCandidate],
        top_n: int,
        as_of_date: str,
    ) -> list[RerankResult]:
        """分批调用：每批取 per_batch_top，合并后再排一轮（保留最终 portfolio 分析）."""
        batches = [
            candidates[i : i + self.batch_size]
            for i in range(0, len(candidates), self.batch_size)
        ]
        per_batch_top = max(top_n // len(batches), 5)

        semifinalists: list[RerankCandidate] = []
        for batch in batches:
            batch_results = self._rerank_batch(batch, per_batch_top, as_of_date)
            batch_ticker_set = {r.ticker for r in batch_results}
            semifinalists += [c for c in batch if c.ticker in batch_ticker_set]

        # 重新编号后做最终排名（此次 portfolio_thesis 和 risk_warnings 取自最终批次）
        for i, c in enumerate(semifinalists):
            c.lgbm_rank = i + 1
        return self._rerank_batch(semifinalists, top_n, as_of_date)

    # ------------------------------------------------------------------
    # Prompt 构造
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        candidates: list[RerankCandidate],
        top_n: int,
        as_of_date: str,
    ) -> str:
        rows = []
        for c in candidates:
            row = (
                f"| {c.lgbm_rank:>7} "
                f"| {c.ticker:<10} "
                f"| {_fmt(c.pe_ttm):>7} "
                f"| {_fmt(c.pb):>6} "
                f"| {_fmt(c.roe_ttm, pct=True):>7} "
                f"| {_fmt(c.accruals, decimals=3):>8} "
                f"| {_fmt(c.distance_to_52w_high, pct=True):>7} "
                f"| {_fmt(c.momentum_6m, pct=True):>7} "
                f"| {_fmt(c.volatility_3m, pct=True):>7} |"
            )
            rows.append(row)

        # SHAP 贡献段（仅当至少一只股票有 shap_values 时附加）
        shap_section = ""
        shap_rows = []
        for c in candidates:
            if c.shap_values:
                shap_rows.append(f"  {c.ticker}: {_top2_shap(c.shap_values)}")
        if shap_rows:
            shap_section = (
                "\n【LGBM SHAP 主导因子（正负贡献最大 Top-2，仅供参考）】\n"
                + "\n".join(shap_rows)
                + "\n"
            )

        return _USER_PROMPT_TEMPLATE.format(
            as_of_date=as_of_date or "未知",
            n_candidates=len(candidates),
            top_n=top_n,
            table_rows="\n".join(rows),
            shap_section=shap_section,
        )

    # ------------------------------------------------------------------
    # 解析工具
    # ------------------------------------------------------------------

    def _parse_rankings(
        self,
        text: str,
        candidates: list[RerankCandidate],
        top_n: int,
    ) -> tuple[list[str], bool]:
        """解析 LLM 输出，提取有序 ticker 列表（三重降级策略）."""
        valid_tickers = {c.ticker for c in candidates}

        # 尝试 1：解析完整结构 {"rankings": [...], ...}
        parsed = self._try_json_parse(text)
        if isinstance(parsed, dict) and "rankings" in parsed:
            tickers = self._extract_tickers_from_list(
                parsed["rankings"], valid_tickers
            )
            if len(tickers) >= max(1, top_n // 2):
                return self._fill_missing(tickers, candidates, top_n), False

        # 尝试 2：纯数组 [{"ticker":...}] 旧格式兼容
        if isinstance(parsed, list):
            tickers = self._extract_tickers_from_list(parsed, valid_tickers)
            if len(tickers) >= max(1, top_n // 2):
                return self._fill_missing(tickers, candidates, top_n), False

        # 尝试 3：正则抽取第一个 JSON 数组
        for m in re.finditer(r"\[.*?\]", text, re.DOTALL):
            sub = self._try_json_parse(m.group(0))
            if sub is not None:
                tickers = self._extract_tickers_from_list(sub, valid_tickers)
                if len(tickers) >= max(1, top_n // 2):
                    return self._fill_missing(tickers, candidates, top_n), False

        # 尝试 4：从文本中直接匹配 6 位 ticker
        found = re.findall(r"\b(\d{6}(?:\.\w+)?)\b", text)
        tickers = list(dict.fromkeys(
            t.split(".")[0] + ("." + t.split(".")[1] if "." in t else "")
            for t in found if t in valid_tickers
        ))
        if len(tickers) >= max(1, top_n // 2):
            return self._fill_missing(tickers, candidates, top_n), False

        warnings.warn("LLM response parse failed, using LGBM order as fallback", stacklevel=3)
        return [c.ticker for c in candidates[:top_n]], True

    def _extract_portfolio_thesis(self, text: str) -> str:
        """从 LLM 输出中提取 portfolio_thesis."""
        parsed = self._try_json_parse(text)
        if isinstance(parsed, dict):
            return str(parsed.get("portfolio_thesis", "")).strip()
        # 正则降级
        m = re.search(r'"portfolio_thesis"\s*:\s*"([^"]*)"', text)
        return m.group(1).strip() if m else ""

    def _extract_risk_warnings(self, text: str) -> list[str]:
        """从 LLM 输出中提取 risk_warnings 列表."""
        parsed = self._try_json_parse(text)
        if isinstance(parsed, dict):
            rw = parsed.get("risk_warnings", [])
            if isinstance(rw, list):
                return [str(r).strip() for r in rw if r]
        # 正则降级：找 "risk_warnings": ["...", "..."]
        m = re.search(r'"risk_warnings"\s*:\s*(\[[^\]]*\])', text, re.DOTALL)
        if m:
            sub = self._try_json_parse(m.group(1))
            if isinstance(sub, list):
                return [str(r).strip() for r in sub if r]
        return []

    def _extract_field_map(self, text: str, field: str) -> dict[str, str]:
        """从 rankings 数组中提取 ticker → field 映射."""
        result: dict[str, str] = {}
        parsed = self._try_json_parse(text)
        rankings = None
        if isinstance(parsed, dict):
            rankings = parsed.get("rankings")
        elif isinstance(parsed, list):
            rankings = parsed
        if rankings is None:
            m = re.search(r'"rankings"\s*:\s*(\[.*?\])', text, re.DOTALL)
            if m:
                rankings = self._try_json_parse(m.group(1))
        if isinstance(rankings, list):
            for item in rankings:
                if isinstance(item, dict):
                    t = str(item.get("ticker", "")).strip()
                    v = str(item.get(field, "")).strip()
                    if t and v:
                        result[t] = v
        return result

    def _try_json_parse(self, text: str) -> Any:
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, AttributeError):
            return None

    def _extract_tickers_from_list(
        self, items: list, valid_tickers: set[str]
    ) -> list[str]:
        tickers = []
        for item in items:
            if isinstance(item, dict):
                t = str(item.get("ticker", "")).strip()
            elif isinstance(item, str):
                t = item.strip()
            else:
                continue
            if t in valid_tickers and t not in tickers:
                tickers.append(t)
        return tickers

    def _fill_missing(
        self,
        tickers: list[str],
        candidates: list[RerankCandidate],
        top_n: int,
    ) -> list[str]:
        result = list(tickers[:top_n])
        seen = set(result)
        for c in candidates:
            if len(result) >= top_n:
                break
            if c.ticker not in seen:
                result.append(c.ticker)
                seen.add(c.ticker)
        return result

    def _fallback_results(
        self, candidates: list[RerankCandidate], top_n: int
    ) -> list[RerankResult]:
        return [
            RerankResult(
                rank=i + 1,
                ticker=c.ticker,
                lgbm_rank=c.lgbm_rank,
                reason="",
                portfolio_thesis="",
                risk_warnings=[],
                is_fallback=True,
            )
            for i, c in enumerate(candidates[:top_n])
        ]
