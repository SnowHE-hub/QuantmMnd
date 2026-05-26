"""quantmind/agents/debate_orchestrator.py

6-Agent 对抗辩论编排器（无需 LLM，纯规则置信度聚合）。

架构
----
Round 1 — 独立论点
    5 个分析 Agent（Valuation / Momentum / Quality / Sentiment / Risk）各自运行
    analyze()，输出 AgentSignal；转换为结构化论点：
        {agent_name: {stance, argument, confidence, signal_raw}}

Round 2 — 交叉质疑
    每个 Agent 看到其余 4 个 Agent 的立场，生成反驳（纯规则）：
        - 若本 Agent 立场与多数立场一致：强化论点（confidence 小幅提升）
        - 若本 Agent 立场与多数相反：生成反驳摘要，confidence 折扣

Round 3 — 策略综合
    StrategyAgent 读取全部论点+反驳，计算最终置信度：
        final_confidence = avg_confidence × consistency_factor × conviction_ratio
    输出：{final_confidence, recommendation, holding_period, bull_agents,
           bear_agents, key_debate_point, debate_summary}

置信度公式
----------
    signal_std       = std([s.signal for s in signals])        ∈ [0, 2]
    consistency      = max(0, 1 − signal_std / 1.5)            ∈ [0, 1]
    avg_conf         = mean([s.confidence for s in signals])
    bull_conf        = mean confidence of bull agents
    bear_conf        = mean confidence of bear agents
    conviction_ratio = bull_conf / (bull_conf + bear_conf + ε)  ∈ (0, 1)
    final_confidence = avg_conf × 0.5 + consistency × 0.3 + conviction_ratio × 0.2

    Regime 调节：
        bull  Regime → final_confidence × 1.05 (cap 0.95)
        bear  Regime → final_confidence × 0.90
        neutral       → 不调节

集成方式
--------
>>> from quantmind.agents.debate_orchestrator import DebateOrchestrator
>>> orch = DebateOrchestrator(ticker, as_of, context)
>>> result = orch.run_debate()
>>> # result['final_confidence'] 用于 Layer6+ 过滤
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── 常量 ─────────────────────────────────────────────────────────────────────

_ROUND1_TIMEOUT = 30      # 秒：单个 Agent analyze() 超时
_ROUND2_TIMEOUT = 5       # 秒：单个 rebuttal 生成超时（纯规则，应极快）
_MIN_AGENTS     = 2       # 最少 Agent 数量（不足时返回 fallback）
_STANCE_THRESH  = 0.10    # signal 绝对值 > 0.10 才判定 bull/bear，否则 neutral

# Agent 权重（与 StrategyAgent._AGENT_WEIGHTS 一致）
_AGENT_WEIGHTS: Dict[str, float] = {
    "ValuationAgent": 0.25,
    "MomentumAgent":  0.20,
    "QualityAgent":   0.25,
    "SentimentAgent": 0.15,
    "RiskAgent":      0.15,
}

# Regime 置信度调节因子
_REGIME_FACTOR: Dict[str, float] = {
    "bull":    1.05,
    "neutral": 1.00,
    "bear":    0.90,
}


# ── 数据类 ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentStance:
    """Round 1 输出：单个 Agent 的独立立场."""
    agent_name:  str
    stance:      str        # "bull" | "bear" | "neutral"
    argument:    str        # 一句话论点（中文）
    confidence:  float      # 0–1
    signal_raw:  float      # 原始 signal（−1 to +1）
    evidence:    Dict[str, Any] = field(default_factory=dict)
    warnings:    List[str]  = field(default_factory=list)


@dataclass
class AgentRebuttal:
    """Round 2 输出：单个 Agent 的反驳."""
    agent_name:         str
    original_stance:    str
    rebuttal_target:    str    # 反驳的主要对手立场
    rebuttal_text:      str    # 反驳摘要（中文）
    updated_confidence: float  # 经反驳后调整的 confidence


@dataclass
class DebateResult:
    """Round 3 输出：最终辩论结论."""
    ticker:           str
    final_confidence: float      # 0–1
    recommendation:   str        # "强烈买入" | "买入" | "观察" | "回避"
    holding_period:   str        # "1w" | "2w" | "21d" | "3m"
    bull_agents:      List[str]
    bear_agents:      List[str]
    neutral_agents:   List[str]
    key_debate_point: str        # 核心分歧点（中文）
    debate_summary:   str        # 辩论结论摘要（中文）
    stances:          List[AgentStance]  = field(default_factory=list)
    rebuttals:        List[AgentRebuttal] = field(default_factory=list)
    # 调试信息
    avg_signal:       float = 0.0
    signal_std:       float = 0.0
    consistency:      float = 0.0
    conviction_ratio: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker":           self.ticker,
            "final_confidence": round(self.final_confidence, 4),
            "recommendation":   self.recommendation,
            "holding_period":   self.holding_period,
            "bull_agents":      self.bull_agents,
            "bear_agents":      self.bear_agents,
            "neutral_agents":   self.neutral_agents,
            "key_debate_point": self.key_debate_point,
            "debate_summary":   self.debate_summary,
            "avg_signal":       round(self.avg_signal, 4),
            "signal_std":       round(self.signal_std, 4),
            "consistency":      round(self.consistency, 4),
            "conviction_ratio": round(self.conviction_ratio, 4),
        }


# ── 主类 ──────────────────────────────────────────────────────────────────────

class DebateOrchestrator:
    """6-Agent 对抗辩论编排器。

    Parameters
    ----------
    ticker : str
        股票代码，例如 '600519.SH'。
    as_of : pandas.Timestamp | str
        分析日期。
    context : dict
        从 FeatureBuilder 获取的当日因子快照字典。
    regime : str, optional
        当前宏观 Regime（'bull' | 'neutral' | 'bear'），默认 'neutral'。
    model_version_overrides : dict, optional
        各 Agent 模型版本覆盖，默认均使用 'active'。
    parallel : bool, optional
        Round 1 是否并行执行各 Agent，默认 True。
    timeout : float, optional
        Round 1 单个 Agent 超时秒数，默认 30。

    Examples
    --------
    >>> orch = DebateOrchestrator("600519.SH", as_of, context, regime="bull")
    >>> result = orch.run_debate()
    >>> print(result.final_confidence)   # 0.72
    >>> print(result.recommendation)     # "买入"
    """

    def __init__(
        self,
        ticker:                   str,
        as_of,
        context:                  Dict[str, Any],
        regime:                   str = "neutral",
        model_version_overrides:  Optional[Dict[str, str]] = None,
        parallel:                 bool = True,
        timeout:                  float = _ROUND1_TIMEOUT,
        agent_mode:               str = "fast",
    ) -> None:
        self.ticker  = ticker
        self.as_of   = as_of
        self.context = context
        self.regime  = regime.lower() if regime else "neutral"
        self._overrides   = model_version_overrides or {}
        self._parallel    = parallel
        self._timeout     = timeout
        self._agent_mode  = agent_mode   # 'auto' | 'full' | 'fast'

        # 导入 Agent 类（延迟，避免循环依赖）
        from quantmind.agents.investment_agents import (
            MomentumAgent, QualityAgent, RiskAgent, SentimentAgent, ValuationAgent,
        )
        self._agent_classes = [
            ValuationAgent,
            MomentumAgent,
            QualityAgent,
            SentimentAgent,
            RiskAgent,
        ]

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_debate(self) -> DebateResult:
        """执行完整 3-Round 辩论，返回 DebateResult。"""
        t0 = time.perf_counter()

        # Round 1: 独立论点
        stances = self._round1_independent_stances()
        if len(stances) < _MIN_AGENTS:
            log.warning(
                "DebateOrchestrator[%s]: Round1 仅收到 %d 个立场，使用 fallback",
                self.ticker, len(stances),
            )
            return self._fallback_result(stances)

        # Round 2: 交叉质疑
        rebuttals = self._round2_cross_examination(stances)

        # Round 3: 策略综合
        result = self._round3_synthesis(stances, rebuttals)

        elapsed = time.perf_counter() - t0
        log.info(
            "DebateOrchestrator[%s]: 辩论完成 %.2fs | "
            "conf=%.3f  rec=%s  bull=%s  bear=%s",
            self.ticker, elapsed, result.final_confidence,
            result.recommendation, result.bull_agents, result.bear_agents,
        )
        return result

    # ── Round 1: 独立论点 ───────────────────────────────────────────────────────

    def _round1_independent_stances(self) -> List[AgentStance]:
        """并行（或串行）运行各 Agent，生成独立立场。"""
        if self._parallel:
            return self._run_parallel()
        else:
            return self._run_serial()

    def _run_parallel(self) -> List[AgentStance]:
        stances: List[AgentStance] = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(self._run_single_agent, cls): cls.__name__
                for cls in self._agent_classes
            }
            for fut in as_completed(futures, timeout=self._timeout + 2):
                name = futures[fut]
                try:
                    stance = fut.result(timeout=self._timeout)
                    if stance is not None:
                        stances.append(stance)
                except TimeoutError:
                    log.warning("DebateOrchestrator[%s]: %s Round1 超时", self.ticker, name)
                except Exception as exc:
                    log.warning("DebateOrchestrator[%s]: %s Round1 失败 — %s", self.ticker, name, exc)
        return stances

    def _run_serial(self) -> List[AgentStance]:
        stances: List[AgentStance] = []
        for cls in self._agent_classes:
            try:
                stance = self._run_single_agent(cls)
                if stance is not None:
                    stances.append(stance)
            except Exception as exc:
                log.warning(
                    "DebateOrchestrator[%s]: %s Round1 失败 — %s",
                    self.ticker, cls.__name__, exc,
                )
        return stances

    def _run_single_agent(self, agent_cls) -> Optional[AgentStance]:
        """实例化并运行单个 Agent，转换为 AgentStance。"""
        name  = agent_cls.__name__
        mv    = self._overrides.get(name, "active")
        agent = agent_cls(self.ticker, self.as_of, self.context, model_version=mv)
        sig   = agent.analyze(mode=self._agent_mode)

        # signal → stance
        if sig.signal > _STANCE_THRESH:
            stance = "bull"
        elif sig.signal < -_STANCE_THRESH:
            stance = "bear"
        else:
            stance = "neutral"

        # 生成论点文本：优先使用 summary，拼接核心 evidence
        arg = self._build_argument_text(name, stance, sig)

        return AgentStance(
            agent_name  = name,
            stance      = stance,
            argument    = arg,
            confidence  = float(np.clip(sig.confidence, 0.0, 1.0)),
            signal_raw  = float(sig.signal),
            evidence    = sig.evidence or {},
            warnings    = sig.warnings or [],
        )

    @staticmethod
    def _build_argument_text(agent_name: str, stance: str, sig) -> str:
        """从 AgentSignal 字段构建单句论点摘要（中文）。"""
        stance_zh = {"bull": "看涨", "bear": "看跌", "neutral": "中性"}[stance]
        base = sig.summary if sig.summary else f"{agent_name} 信号={sig.signal:.2f}"

        # 如果 evidence 中有关键指标，附加首条
        if sig.evidence:
            key = next(iter(sig.evidence))
            val = sig.evidence[key]
            if isinstance(val, float):
                base += f"（{key}={val:.3f}）"
            elif val is not None:
                base += f"（{key}={val}）"

        return f"[{stance_zh}] {base}"

    # ── Round 2: 交叉质疑 ───────────────────────────────────────────────────────

    def _round2_cross_examination(self, stances: List[AgentStance]) -> List[AgentRebuttal]:
        """每个 Agent 看到其余立场，生成纯规则反驳。"""
        rebuttals: List[AgentRebuttal] = []
        majority_stance = self._majority_stance(stances)

        for s in stances:
            others = [o for o in stances if o.agent_name != s.agent_name]
            rebuttal = self._generate_rebuttal(s, others, majority_stance)
            rebuttals.append(rebuttal)

        return rebuttals

    @staticmethod
    def _majority_stance(stances: List[AgentStance]) -> str:
        """计算多数立场（bull/bear/neutral）。"""
        counts = {"bull": 0, "bear": 0, "neutral": 0}
        for s in stances:
            counts[s.stance] = counts.get(s.stance, 0) + 1
        return max(counts, key=counts.__getitem__)

    def _generate_rebuttal(
        self,
        self_stance: AgentStance,
        others:      List[AgentStance],
        majority:    str,
    ) -> AgentRebuttal:
        """为单个 Agent 生成反驳（纯规则）。"""
        # 计算对手平均信号
        other_signals  = [o.signal_raw for o in others]
        avg_other_sig  = float(np.mean(other_signals)) if other_signals else 0.0
        signal_gap     = self_stance.signal_raw - avg_other_sig   # 正 = 比多数乐观

        agrees_with_majority = (self_stance.stance == majority)

        if agrees_with_majority:
            # 与多数一致 → 强化论点，confidence 轻微提升
            updated_conf = min(0.95, self_stance.confidence * 1.05)
            minority = [o for o in others if o.stance != majority]
            if minority:
                min_names = "、".join(o.agent_name.replace("Agent", "") for o in minority)
                rebuttal_text = (
                    f"{self_stance.agent_name.replace('Agent','')} 与多数一致（{majority}），"
                    f"认为 {min_names} 的 {minority[0].stance} 观点低估了"
                    f"{'正面催化剂' if majority == 'bull' else '下行风险'}。"
                    f"（信号差 {signal_gap:+.2f}）"
                )
                rebuttal_target = f"反对 {minority[0].stance} 立场"
            else:
                rebuttal_text = (
                    f"{self_stance.agent_name.replace('Agent','')} 与全体一致，"
                    f"无需反驳。"
                )
                rebuttal_target = "无反驳"
        else:
            # 少数派 → 折扣 confidence，但坚持论点
            discount = 0.10 + min(0.15, abs(signal_gap) * 0.1)
            updated_conf = max(0.10, self_stance.confidence - discount)
            majority_agents = [o for o in others if o.stance == majority]
            maj_names = "、".join(o.agent_name.replace("Agent", "") for o in majority_agents[:2])
            rebuttal_text = (
                f"{self_stance.agent_name.replace('Agent','')} 坚持 {self_stance.stance} 立场，"
                f"反驳 {maj_names} 等 {majority} 观点：本 Agent 权重更高的"
                f"{'估值/质量维度' if self_stance.agent_name in ('ValuationAgent','QualityAgent') else '动量/技术维度'}"
                f"显示信号 {self_stance.signal_raw:+.2f}，"
                f"{'正面背离多数' if signal_gap > 0 else '负面背离多数'}。"
            )
            rebuttal_target = f"反对 {majority} 多数立场"

        return AgentRebuttal(
            agent_name          = self_stance.agent_name,
            original_stance     = self_stance.stance,
            rebuttal_target     = rebuttal_target,
            rebuttal_text       = rebuttal_text,
            updated_confidence  = float(np.clip(updated_conf, 0.0, 1.0)),
        )

    # ── Round 3: 策略综合 ──────────────────────────────────────────────────────

    def _round3_synthesis(
        self,
        stances:   List[AgentStance],
        rebuttals: List[AgentRebuttal],
    ) -> DebateResult:
        """StrategyAgent 综合全部论点+反驳，输出最终置信度和建议。"""

        # ── 使用 rebuttal 后的 updated_confidence ─────────────────────────────
        # 构建 name → updated_confidence 映射
        updated_conf_map = {r.agent_name: r.updated_confidence for r in rebuttals}
        signals  = [s.signal_raw  for s in stances]
        confs    = [updated_conf_map.get(s.agent_name, s.confidence) for s in stances]
        weights  = [_AGENT_WEIGHTS.get(s.agent_name, 0.20) for s in stances]

        # 归一化权重
        w_sum = sum(weights) or 1.0
        norm_w = [w / w_sum for w in weights]

        # 加权平均信号
        avg_signal = float(np.dot(norm_w, signals))
        signal_std = float(np.std(signals))

        # ── 置信度组件 ─────────────────────────────────────────────────────────
        # 1. 信号一致性：std 越小 → 越一致
        consistency = float(max(0.0, 1.0 - signal_std / 1.5))

        # 2. 平均置信度（加权）
        avg_conf = float(np.dot(norm_w, confs))

        # 3. bull vs bear conviction 对比
        bull_stances = [s for s in stances if s.stance == "bull"]
        bear_stances = [s for s in stances if s.stance == "bear"]

        bull_conf = float(np.mean([updated_conf_map.get(s.agent_name, s.confidence) for s in bull_stances])) \
                    if bull_stances else 0.0
        bear_conf = float(np.mean([updated_conf_map.get(s.agent_name, s.confidence) for s in bear_stances])) \
                    if bear_stances else 0.0

        # conviction_ratio = bull 方平均置信 / (bull + bear + ε)
        conviction_ratio = bull_conf / (bull_conf + bear_conf + 1e-9)

        # ── 最终置信度 ─────────────────────────────────────────────────────────
        raw_conf = (
            avg_conf         * 0.50
            + consistency    * 0.30
            + conviction_ratio * 0.20
        )

        # Regime 调节
        regime_factor = _REGIME_FACTOR.get(self.regime, 1.0)
        final_confidence = float(np.clip(raw_conf * regime_factor, 0.0, 1.0))

        # ── 推荐信号 → 文字建议 ─────────────────────────────────────────────────
        recommendation = self._signal_to_recommendation(avg_signal, final_confidence)

        # ── 建议持仓期 ─────────────────────────────────────────────────────────
        holding_period = self._suggest_holding(stances, final_confidence)

        # ── 关键分歧点 ─────────────────────────────────────────────────────────
        key_debate_point = self._identify_key_debate(stances, rebuttals)

        # ── 辩论摘要 ──────────────────────────────────────────────────────────
        debate_summary = self._build_summary(
            stances, avg_signal, final_confidence, recommendation, self.regime
        )

        return DebateResult(
            ticker            = self.ticker,
            final_confidence  = final_confidence,
            recommendation    = recommendation,
            holding_period    = holding_period,
            bull_agents       = [s.agent_name for s in bull_stances],
            bear_agents       = [s.agent_name for s in bear_stances],
            neutral_agents    = [s.agent_name for s in stances if s.stance == "neutral"],
            key_debate_point  = key_debate_point,
            debate_summary    = debate_summary,
            stances           = stances,
            rebuttals         = rebuttals,
            avg_signal        = avg_signal,
            signal_std        = signal_std,
            consistency       = consistency,
            conviction_ratio  = conviction_ratio,
        )

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _signal_to_recommendation(avg_signal: float, confidence: float) -> str:
        if avg_signal > 0.4 and confidence > 0.65:
            return "强烈买入"
        elif avg_signal > 0.15:
            return "买入"
        elif avg_signal > -0.10:
            return "观察"
        else:
            return "回避"

    @staticmethod
    def _suggest_holding(stances: List[AgentStance], confidence: float) -> str:
        """根据动量/估值立场和置信度推断建议持仓期。"""
        mom_sig = next(
            (s.signal_raw for s in stances if s.agent_name == "MomentumAgent"), 0.0
        )
        val_sig = next(
            (s.signal_raw for s in stances if s.agent_name == "ValuationAgent"), 0.0
        )
        if mom_sig > 0.4:
            return "1w"
        elif mom_sig > 0.2 and val_sig < 0.2:
            return "2w"
        elif val_sig > 0.3 and confidence > 0.6:
            return "3m"
        else:
            return "21d"

    @staticmethod
    def _identify_key_debate(
        stances:   List[AgentStance],
        rebuttals: List[AgentRebuttal],
    ) -> str:
        """识别核心分歧点：信号差异最大的两个 Agent 的对立描述。"""
        if len(stances) < 2:
            return "无明显分歧"

        # 找信号最高与最低的 Agent
        sorted_s = sorted(stances, key=lambda s: s.signal_raw)
        lowest   = sorted_s[0]
        highest  = sorted_s[-1]

        if abs(highest.signal_raw - lowest.signal_raw) < 0.15:
            return "各方观点基本一致，无重大分歧"

        low_name  = lowest.agent_name.replace("Agent", "")
        high_name = highest.agent_name.replace("Agent", "")
        return (
            f"{high_name}（信号{highest.signal_raw:+.2f}）vs "
            f"{low_name}（信号{lowest.signal_raw:+.2f}）——"
            f"{'估值低估' if 'Valuation' in highest.agent_name else '动量/质量优势'}"
            f"与{'风险/估值压力' if 'Risk' in lowest.agent_name or 'Valuation' in lowest.agent_name else '动量疲弱'}之争"
        )

    @staticmethod
    def _build_summary(
        stances:       List[AgentStance],
        avg_signal:    float,
        final_conf:    float,
        recommendation: str,
        regime:        str,
    ) -> str:
        bull_n    = sum(1 for s in stances if s.stance == "bull")
        bear_n    = sum(1 for s in stances if s.stance == "bear")
        neutral_n = sum(1 for s in stances if s.stance == "neutral")
        total_n   = len(stances)

        regime_zh = {"bull": "牛市", "bear": "熊市", "neutral": "中性市场"}.get(regime, regime)
        return (
            f"共 {total_n} 个 Agent 参与辩论：看涨 {bull_n} / 看跌 {bear_n} / 中性 {neutral_n}。"
            f"加权平均信号 {avg_signal:+.2f}，最终置信度 {final_conf:.2%}。"
            f"当前 Regime={regime_zh}。综合建议：{recommendation}。"
        )

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _fallback_result(self, stances: List[AgentStance]) -> DebateResult:
        """Agent 数量不足时返回保守 fallback。"""
        avg_signal = float(np.mean([s.signal_raw for s in stances])) if stances else 0.0
        return DebateResult(
            ticker            = self.ticker,
            final_confidence  = 0.40,   # 保守值
            recommendation    = "观察",
            holding_period    = "21d",
            bull_agents       = [s.agent_name for s in stances if s.stance == "bull"],
            bear_agents       = [s.agent_name for s in stances if s.stance == "bear"],
            neutral_agents    = [s.agent_name for s in stances if s.stance == "neutral"],
            key_debate_point  = "Agent 数量不足，无法完成完整辩论",
            debate_summary    = f"仅 {len(stances)} 个 Agent 有效，使用保守置信度 0.40",
            stances           = stances,
            rebuttals         = [],
            avg_signal        = avg_signal,
            signal_std        = 0.0,
            consistency       = 0.5,
            conviction_ratio  = 0.5,
        )

    def __repr__(self) -> str:
        return (
            f"DebateOrchestrator(ticker='{self.ticker}', "
            f"regime='{self.regime}', parallel={self._parallel})"
        )


# ── 批量辩论（供 run_30day_sim.py 调用）────────────────────────────────────────

def run_debate_filter(
    candidates_df,
    as_of,
    context_map:          Dict[str, Dict[str, Any]],
    regime:               str = "neutral",
    confidence_threshold: float = 0.60,
    fallback_threshold:   float = 0.50,
    top_n:                int = 10,
    model_version_overrides: Optional[Dict[str, str]] = None,
    parallel_agents:      bool = True,
    parallel_tickers:     bool = False,
    agent_mode:           str = "fast",   # 'auto'/'full'/'fast' — 传给各 Agent
) -> "pd.DataFrame":
    """对候选股票列表跑辩论，添加 debate_confidence 列并过滤。

    Parameters
    ----------
    candidates_df : pd.DataFrame
        index=ticker，已经过 Layer6 选出的候选（通常 ≤15 只）。
    as_of : pd.Timestamp
        分析日期。
    context_map : dict
        {ticker: context_dict}，每只股票的因子快照。
    regime : str
        当前 Regime（'bull'/'neutral'/'bear'）。
    confidence_threshold : float
        一级门槛：>= 此值的股票优先进入最终名单，默认 0.60。
    fallback_threshold : float
        若通过一级门槛的股票 < top_n，降级到此门槛，默认 0.50。
    top_n : int
        最终最多选几只，默认 10。
    parallel_agents : bool
        每只股票内部 Agent 是否并行，默认 True。
    parallel_tickers : bool
        是否跨股票并行（多线程），默认 False（避免资源竞争）。

    Returns
    -------
    pd.DataFrame
        添加 'debate_confidence', 'debate_recommendation', 'debate_holding_period',
        'debate_bull_agents', 'debate_bear_agents', 'debate_summary' 列，
        按 debate_confidence 降序，最多 top_n 行。
    """
    import pandas as pd

    df = candidates_df.copy()
    tickers = df.index.tolist()

    if not tickers:
        return df

    log.info(
        "DebateFilter: 对 %d 只候选股运行辩论 Regime=%s threshold=%.2f",
        len(tickers), regime, confidence_threshold,
    )

    # ── 运行辩论 ─────────────────────────────────────────────────────────────
    results: Dict[str, DebateResult] = {}

    def _run_one(ticker: str) -> tuple:
        ctx = context_map.get(ticker, {})
        orch = DebateOrchestrator(
            ticker=ticker,
            as_of=as_of,
            context=ctx,
            regime=regime,
            model_version_overrides=model_version_overrides,
            parallel=parallel_agents,
            timeout=_ROUND1_TIMEOUT,
            agent_mode=agent_mode,
        )
        res = orch.run_debate()
        return ticker, res

    if parallel_tickers and len(tickers) > 3:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = {pool.submit(_run_one, t): t for t in tickers}
            for fut in as_completed(futs, timeout=120):
                try:
                    tk, res = fut.result(timeout=90)
                    results[tk] = res
                except Exception as exc:
                    log.warning("DebateFilter: %s 辩论失败 — %s", futs[fut], exc)
    else:
        for ticker in tickers:
            try:
                _, res = _run_one(ticker)
                results[ticker] = res
            except Exception as exc:
                log.warning("DebateFilter: %s 辩论失败 — %s", ticker, exc)

    # ── 写回 DataFrame ─────────────────────────────────────────────────────
    df["debate_confidence"]    = df.index.map(lambda t: results[t].final_confidence if t in results else 0.40)
    df["debate_recommendation"] = df.index.map(lambda t: results[t].recommendation   if t in results else "观察")
    df["debate_holding_period"] = df.index.map(lambda t: results[t].holding_period   if t in results else "21d")
    df["debate_bull_agents"]   = df.index.map(lambda t: results[t].bull_agents       if t in results else [])
    df["debate_bear_agents"]   = df.index.map(lambda t: results[t].bear_agents       if t in results else [])
    df["debate_summary"]       = df.index.map(lambda t: results[t].debate_summary    if t in results else "辩论失败")
    df["debate_key_point"]     = df.index.map(lambda t: results[t].key_debate_point  if t in results else "")

    # ── 按置信度排序 + 门槛过滤 ─────────────────────────────────────────────
    df = df.sort_values("debate_confidence", ascending=False)

    above_thresh = df[df["debate_confidence"] >= confidence_threshold]
    if len(above_thresh) >= top_n:
        final = above_thresh.head(top_n)
        log.info(
            "DebateFilter: %d 只通过一级门槛 %.2f，选 Top-%d",
            len(above_thresh), confidence_threshold, top_n,
        )
    else:
        # 降级门槛
        above_fallback = df[df["debate_confidence"] >= fallback_threshold]
        if len(above_fallback) >= top_n:
            final = above_fallback.head(top_n)
            log.info(
                "DebateFilter: 一级门槛仅 %d 只，降级到 %.2f，选 %d 只",
                len(above_thresh), fallback_threshold, len(final),
            )
        else:
            final = df.head(top_n)
            log.info(
                "DebateFilter: 降级门槛仍不足，直接取 Top-%d（置信度最高）",
                top_n,
            )

    log.info(
        "DebateFilter: 最终 %d 只股票入选，avg_confidence=%.3f",
        len(final), final["debate_confidence"].mean(),
    )
    return final


# ── DPO 训练数据构建 ──────────────────────────────────────────────────────────

def build_dpo_pairs(
    realized_pnl_path: str,
    output_path: str,
    top_pct: float = 0.30,
    bot_pct: float = 0.30,
    return_col: str = "actual_return_63d",
    industry_col: str = "industry",
    ticker_col:   str = "ts_code",
    min_pairs:    int = 1,
) -> "pd.DataFrame":
    """从已实现 PnL 数据构建 DPO 训练对。

    规则：
    - chosen  = 同行业内实际收益 top_pct（如 top 30%）的股票
    - rejected = 同行业内实际收益 bottom bot_pct 的股票
    - 同一 (as_of_date, industry) 组内配对

    Parameters
    ----------
    realized_pnl_path : str
        parquet 路径，需要列：ts_code, actual_return_63d, industry, [as_of_date]。
    output_path : str
        输出 parquet 路径。
    top_pct / bot_pct : float
        分位数阈值，默认各 30%。
    return_col : str
        收益率列名，默认 'actual_return_63d'。
    industry_col : str
        行业列名，默认 'industry'。
    ticker_col : str
        股票代码列名，默认 'ts_code'。

    Returns
    -------
    pd.DataFrame
        columns: chosen_ticker, rejected_ticker, chosen_return, rejected_return,
                 industry, [as_of_date], return_diff
    """
    import pandas as pd

    pnl = pd.read_parquet(realized_pnl_path)

    required = {ticker_col, return_col, industry_col}
    missing  = required - set(pnl.columns)
    if missing:
        raise ValueError(f"build_dpo_pairs: 缺少列 {missing}")

    # 若有日期列则按日期分组
    group_cols = [industry_col]
    date_col   = None
    for c in ("as_of_date", "trade_date", "date"):
        if c in pnl.columns:
            date_col = c
            group_cols = [c, industry_col]
            break

    pairs: list = []
    for keys, grp in pnl.groupby(group_cols):
        ret = grp[return_col].dropna()
        if len(ret) < 2:
            continue

        q_high = ret.quantile(1 - top_pct)
        q_low  = ret.quantile(bot_pct)

        chosen_mask   = ret >= q_high
        rejected_mask = ret <= q_low

        chosen_tickers   = grp.loc[ret[chosen_mask].index, ticker_col].tolist()
        rejected_tickers = grp.loc[ret[rejected_mask].index, ticker_col].tolist()

        if not chosen_tickers or not rejected_tickers:
            continue

        # 笛卡尔积，保留有意义的对（chosen_return > rejected_return）
        for c_t in chosen_tickers:
            c_ret = float(ret[grp[ticker_col] == c_t].iloc[0])
            for r_t in rejected_tickers:
                r_ret = float(ret[grp[ticker_col] == r_t].iloc[0])
                if c_ret > r_ret:
                    row: dict = {
                        "chosen_ticker":   c_t,
                        "rejected_ticker": r_t,
                        "chosen_return":   c_ret,
                        "rejected_return": r_ret,
                        "return_diff":     c_ret - r_ret,
                        industry_col:      keys if isinstance(keys, str) else keys[group_cols.index(industry_col)],
                    }
                    if date_col:
                        row[date_col] = keys if isinstance(keys, str) else keys[0]
                    pairs.append(row)

    if not pairs:
        log.warning("build_dpo_pairs: 未能构建任何 DPO 对，返回空 DataFrame")
        return pd.DataFrame(
            columns=["chosen_ticker", "rejected_ticker", "chosen_return",
                     "rejected_return", "return_diff", industry_col]
        )

    result = pd.DataFrame(pairs).drop_duplicates(["chosen_ticker", "rejected_ticker"])
    result = result.sort_values("return_diff", ascending=False).reset_index(drop=True)

    import pathlib
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    log.info(
        "build_dpo_pairs: 生成 %d 对 DPO 训练数据 → %s",
        len(result), output_path,
    )
    return result


__all__ = [
    "DebateOrchestrator",
    "DebateResult",
    "AgentStance",
    "AgentRebuttal",
    "run_debate_filter",
    "build_dpo_pairs",
]
