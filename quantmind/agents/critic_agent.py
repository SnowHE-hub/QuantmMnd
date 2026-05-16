"""quantmind.agents.critic_agent — CriticAgent.

审查所有分析结果，输出通过/失败判断和修复建议。
触发规则：critical ≥ 1 条 或 major ≥ 3 条 → passed=False
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from quantmind.agents.base import BaseAgent
from quantmind.core.state import (
    AgentState,
    AgentType,
    CriticFeedback,
    CriticIssue,
    FixAction,
    IssueSeverity,
    IssueType,
)

__all__ = ["CriticAgent"]

_SYSTEM_PROMPT = """你是一位严格的投资研究质量审查官。
你的任务是审查投资分析报告的质量，从5个维度逐一评估：

1. **数据完整性**：关键财务数据是否齐全？是否有 None/缺失？
2. **推理一致性**：基本面、技术面、情绪面结论是否相互矛盾？
3. **数字合理性**：财务数字是否在合理范围？（如 ROE>100% 可疑，PE<0 需解释）
4. **风险覆盖**：是否充分讨论下行风险？（至少3个具体风险点）
5. **论证严谨性**：观点是否有具体数字支撑？是否有"这只股票很好"这类空话？

触发规则（严格执行）：
- critical ≥ 1 条 → passed=false
- major ≥ 3 条 → passed=false
- 否则 → passed=true

输出严格 JSON：
{
  "passed": true/false,
  "issues": [
    {
      "severity": "critical|major|minor",
      "type": "data_missing|inconsistency|calculation_error|risk_missing|weak_argument|hallucination|pit_violation",
      "location": "指出问题所在（如 FundamentalAgent.valuation_analysis）",
      "description": "问题描述（具体，引用实际内容）",
      "suggestion": "修复建议（具体可操作）"
    }
  ],
  "overall_quality_score": 0-10,
  "approval_message": "通过/不通过的说明（50字内）"
}"""


class CriticAgent(BaseAgent):
    agent_type = AgentType.CRITIC
    description = "质量审查 Agent：5维度审查，自动判断是否触发回流"
    system_prompt = _SYSTEM_PROMPT

    def format_input(self, state: AgentState) -> str:
        lines: list[str] = ["=== 待审查的分析内容 ===\n"]

        for ticker in state.query.tickers:
            lines.append(f"【{ticker}】")

            # 基本面
            fa = state.fundamentals.get(ticker)
            if fa:
                lines.append(
                    f"基本面分析：\n"
                    f"  比率：{json.dumps(fa.ratios, ensure_ascii=False)}\n"
                    f"  DCF公允值：{fa.dcf_value}\n"
                    f"  盈利分析：{fa.profitability_analysis}\n"
                    f"  估值分析：{fa.valuation_analysis}\n"
                    f"  风险：{fa.key_risks}\n"
                    f"  置信度：{fa.confidence}"
                )
            else:
                lines.append("基本面分析：【缺失】")

            # 技术面
            ta = state.technicals.get(ticker)
            if ta:
                lines.append(
                    f"技术分析：signal={ta.signal}, "
                    f"quant_score={ta.quant_score}, "
                    f"summary={ta.summary}"
                )
            else:
                lines.append("技术分析：【缺失】")

            # 情绪面
            sa = state.sentiments.get(ticker)
            if sa:
                lines.append(
                    f"情绪分析：sentiment={sa.aggregated_sentiment:.2f}, "
                    f"catalysts={sa.catalysts}, risks={sa.risks}"
                )
            else:
                lines.append("情绪分析：【缺失】")

            lines.append("")

        lines.append(f"当前迭代次数：{state.iteration_count}")
        lines.append("\n请按照5个维度审查，输出 JSON。")
        return "\n".join(lines)

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
        prompt = self.format_input(state)
        resp = self.llm_chat(prompt, max_tokens=2048)
        parsed = self.parse_output(resp.content)

        if not parsed:
            # LLM 解析失败，保守地视为通过
            state.critic_feedback = CriticFeedback(
                passed=True,
                issues=[],
                overall_quality_score=5.0,
                approval_message="审查 LLM 响应解析失败，保守通过",
            )
            return state

        # 解析 issues
        issues: list[CriticIssue] = []
        for raw_issue in parsed.get("issues", []):
            sev_str = raw_issue.get("severity", "minor").lower()
            try:
                severity = IssueSeverity(sev_str)
            except ValueError:
                severity = IssueSeverity.MINOR

            type_str = raw_issue.get("type", "weak_argument").lower()
            try:
                issue_type = IssueType(type_str)
            except ValueError:
                issue_type = IssueType.WEAK_ARGUMENT

            fix_action = None
            suggestion = raw_issue.get("suggestion", "")
            location = raw_issue.get("location", "")
            if suggestion and location:
                agent_hint = AgentType.FUNDAMENTAL
                if "technical" in location.lower():
                    agent_hint = AgentType.TECHNICAL
                elif "sentiment" in location.lower():
                    agent_hint = AgentType.SENTIMENT
                elif "data" in location.lower():
                    agent_hint = AgentType.DATA
                fix_action = FixAction(agent=agent_hint, instruction=suggestion)

            issues.append(CriticIssue(
                severity=severity,
                type=issue_type,
                description=raw_issue.get("description", ""),
                fix_action=fix_action,
            ))

        # 触发规则
        critical_count = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        major_count = sum(1 for i in issues if i.severity == IssueSeverity.MAJOR)
        llm_passed = bool(parsed.get("passed", True))

        # 强制执行触发规则（不信任 LLM 的 passed 字段）
        rule_based_passed = (critical_count == 0) and (major_count < 3)
        passed = rule_based_passed and llm_passed

        quality_score = float(parsed.get("overall_quality_score", 5.0))
        quality_score = max(0.0, min(10.0, quality_score))

        state.critic_feedback = CriticFeedback(
            passed=passed,
            issues=issues,
            overall_quality_score=quality_score,
            approval_message=parsed.get("approval_message", ""),
        )

        status = "PASSED" if passed else f"FAILED (critical={critical_count}, major={major_count})"
        logger.info(f"[CriticAgent] {status}, quality={quality_score:.1f}")
        return state
