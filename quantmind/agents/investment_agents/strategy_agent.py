"""quantmind.agents.investment_agents.strategy_agent — 综合策略生成 Agent（调用LLM）."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from quantmind.agents.investment_agents.base_agent import AgentSignal, BaseInvestmentAgent

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PRICE_FILE = _ROOT / "data" / "prices" / "csi300_daily_adj_close.parquet"

# 各 Agent 权重（合计 1.0）
_AGENT_WEIGHTS: dict[str, float] = {
    "ValuationAgent": 0.25,
    "MomentumAgent": 0.20,
    "QualityAgent": 0.25,
    "SentimentAgent": 0.15,
    "RiskAgent": 0.15,
}


@dataclass
class InvestmentStrategy:
    ticker: str
    as_of: str

    # 综合评级
    rating: str               # "积极关注" / "谨慎关注" / "观察" / "回避"
    composite_signal: float   # 加权平均信号
    confidence: float

    # 策略参数（供回测使用）
    entry_price_range: tuple[float, float]
    target_price_1m: float
    target_price_3m: float
    stop_loss_price: float
    position_size: str        # "重仓(5-8%)" / "标准(3-5%)" / "轻仓(1-3%)" / "不建仓"
    holding_horizon: str      # "短期(1个月)" / "中期(3个月)"

    # 报告内容
    investment_thesis: str
    key_risks: list[str] = field(default_factory=list)
    key_catalysts: list[str] = field(default_factory=list)

    # 元数据
    agent_signals: dict = field(default_factory=dict)
    llm_used: bool = False

    @property
    def confidence_score(self) -> float:
        """与面板/HTML 脚本对齐：`confidence` 的别名（0–1）。"""
        return float(self.confidence)


class StrategyAgent(BaseInvestmentAgent):
    """综合策略 Agent — 汇聚 5 个 Agent 信号，调用 LLM 生成投资逻辑."""

    def __init__(
        self,
        ticker: str,
        as_of: str,
        context: dict,
        agent_signals: list[AgentSignal],
        provider: str = "none",
        model: str = "qwen-plus",
    ) -> None:
        super().__init__(ticker, as_of, context)
        self.agent_signals = agent_signals
        self.provider = provider
        self.model = model

    def analyze(self, mode: str = "fast") -> AgentSignal:
        """综合策略分析主入口（mode='auto'/'full' → Ollama ReAct；'fast' → 原逻辑）."""
        use_llm = (mode == "full") or (mode == "auto" and self._ollama_available())
        if use_llm:
            try:
                return self._analyze_with_llm_react()
            except Exception as e:
                logger.warning("[StrategyAgent] Ollama ReAct 失败，降级原逻辑: %s", e)
        # 原逻辑：内部 LLM thesis 生成（基于 Anthropic/本地 LLM，非 ReAct）
        strategy = self.analyze_with_llm()
        return AgentSignal(
            agent_name="StrategyAgent",
            ticker=self.ticker,
            signal=strategy.composite_signal,
            confidence=strategy.confidence,
            summary=f"{strategy.rating}，综合信号{strategy.composite_signal:+.2f}",
            evidence={"rating": strategy.rating, "position_size": strategy.position_size},
            warnings=strategy.key_risks[:3],
        )

    def analyze_with_llm(self) -> InvestmentStrategy:
        """完整策略生成，包含 LLM 投资逻辑."""
        # ── 1. 计算综合信号 ────────────────────────────────────────────────────
        composite, confidence = self._calc_composite_signal()

        # ── 2. 生成评级 ────────────────────────────────────────────────────────
        rating = self._signal_to_rating(composite)

        # ── 3. 获取当前价格并计算策略参数 ─────────────────────────────────────
        current_price = self._get_current_price()
        vol_21d = self._get_vol_21d()

        if current_price is not None and current_price > 0:
            stop_pct = max(0.05, (vol_21d or 0.25) * 1.5)
            exp_ret_1m = composite * 0.05
            exp_ret_3m = composite * 0.12

            entry_low = round(current_price * 0.98, 2)
            entry_high = round(current_price * 1.02, 2)
            target_1m = round(current_price * (1 + exp_ret_1m), 2)
            target_3m = round(current_price * (1 + exp_ret_3m), 2)
            stop_loss = round(current_price * (1 - stop_pct), 2)
        else:
            entry_low = entry_high = target_1m = target_3m = stop_loss = 0.0

        # ── 4. 持仓期限（不依赖 LLM 校准后的置信度）─────────────────────────────────
        holding_horizon = "短期(1个月)" if abs(composite) > 0.5 else "中期(3个月)"

        # ── 5. 收集风险和催化剂 ────────────────────────────────────────────────
        key_risks = self._collect_risks()
        key_catalysts = self._collect_catalysts()

        # ── 6. LLM 生成投资逻辑（可带回结构化 confidence）────────────────────────
        investment_thesis, llm_used, llm_confidence = self._generate_thesis(
            composite, current_price
        )
        if llm_confidence is not None:
            confidence = float(llm_confidence)
        confidence = round(confidence, 3)

        # ── 7. 仓位建议（使用校准后的置信度）────────────────────────────────────
        position_size = self._recommend_position(composite, confidence)

        # ── 8. Agent 信号汇总 ──────────────────────────────────────────────────
        signals_summary = {
            sig.agent_name: {
                "signal": round(sig.signal, 3),
                "confidence": round(sig.confidence, 2),
                "summary": sig.summary,
            }
            for sig in self.agent_signals
        }

        return InvestmentStrategy(
            ticker=self.ticker,
            as_of=self.as_of,
            rating=rating,
            composite_signal=round(composite, 3),
            confidence=round(confidence, 3),
            entry_price_range=(entry_low, entry_high),
            target_price_1m=target_1m,
            target_price_3m=target_3m,
            stop_loss_price=stop_loss,
            position_size=position_size,
            holding_horizon=holding_horizon,
            investment_thesis=investment_thesis,
            key_risks=key_risks,
            key_catalysts=key_catalysts,
            agent_signals=signals_summary,
            llm_used=llm_used,
        )

    # ── 内部方法 ─────────────────────────────────────────────────────────────

    def _calc_composite_signal(self) -> tuple[float, float]:
        """加权计算综合信号和平均置信度."""
        total_weight = 0.0
        weighted_signal = 0.0
        total_confidence = 0.0
        n = 0

        for sig in self.agent_signals:
            w = _AGENT_WEIGHTS.get(sig.agent_name, 0.0)
            effective_w = w * sig.confidence
            weighted_signal += sig.signal * effective_w
            total_weight += effective_w
            total_confidence += sig.confidence
            n += 1

        composite = weighted_signal / total_weight if total_weight > 0 else 0.0
        avg_confidence = total_confidence / n if n > 0 else 0.5
        return self._clamp(composite), float(self._clamp(avg_confidence, 0.0, 1.0))

    def _signal_to_rating(self, composite: float) -> str:
        if composite > 0.4:
            return "积极关注"
        elif composite > 0.1:
            return "谨慎关注"
        elif composite >= -0.1:
            return "观察"
        else:
            return "回避"

    def _recommend_position(self, composite: float, confidence: float) -> str:
        if composite > 0.5 and confidence > 0.7:
            return "重仓(5-8%)"
        elif composite > 0.3:
            return "标准(3-5%)"
        elif composite > 0.1:
            return "轻仓(1-3%)"
        else:
            return "不建仓"

    def _get_current_price(self) -> float | None:
        try:
            df = pd.read_parquet(_PRICE_FILE)
            if self.ticker not in df.columns:
                return None
            series = df[self.ticker].dropna()
            if self.as_of:
                cutoff = pd.Timestamp(self.as_of)
                series = series[series.index <= cutoff]
            return float(series.iloc[-1]) if len(series) > 0 else None
        except Exception as e:
            logger.warning(f"[StrategyAgent] 价格读取失败: {e}")
            return None

    def _get_vol_21d(self) -> float | None:
        """获取 21 日年化波动率."""
        try:
            import numpy as np
            df = pd.read_parquet(_PRICE_FILE)
            if self.ticker not in df.columns:
                return None
            series = df[self.ticker].dropna()
            if self.as_of:
                cutoff = pd.Timestamp(self.as_of)
                series = series[series.index <= cutoff]
            if len(series) < 22:
                return None
            returns = series.pct_change().dropna()
            return float(returns.iloc[-21:].std() * np.sqrt(252))
        except Exception:
            return None

    def _collect_risks(self) -> list[str]:
        risks = []
        for sig in self.agent_signals:
            if sig.agent_name == "RiskAgent":
                risks.extend(sig.warnings[:3])
            elif sig.warnings:
                risks.append(sig.warnings[0])
        return risks[:5]

    def _collect_catalysts(self) -> list[str]:
        catalysts = []
        for sig in self.agent_signals:
            if sig.agent_name == "SentimentAgent":
                pos_words = sig.evidence.get("positive_words", [])
                if pos_words:
                    catalysts.append(f"正面关键词：{', '.join(pos_words[:3])}")
                rating = sig.evidence.get("analyst_rating", "")
                if "买入" in rating or "推荐" in rating:
                    catalysts.append("分析师给出买入评级")
            if sig.agent_name == "MomentumAgent":
                note = sig.evidence.get("rsi_note", "")
                if note:
                    catalysts.append(note)
        return catalysts[:3]

    def _normalize_llm_confidence(self, raw: Any) -> float | None:
        """将 LLM 输出转为 [0,1] 的置信度；无法解析则 None."""
        if raw is None:
            return None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        if v != v:  # NaN
            return None
        # 模型偶尔输出百分制
        if v > 1.0:
            if v <= 100.0:
                v = v / 100.0
            else:
                return None
        if v < 0.0:
            return None
        return float(min(1.0, max(0.0, v)))

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        """从回复中提取单个 JSON 对象（容忍 markdown 围栏与前后杂音）。"""
        s = text.strip()
        if not s:
            return None
        # 去掉 ```json ... ``` / ``` ... ```
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*", "", s, count=1, flags=re.IGNORECASE)
            s = re.sub(r"\s*```\s*$", "", s, count=1)
            s = s.strip()
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(s[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    def _parse_strategy_llm_payload(self, content: str) -> tuple[str | None, float | None]:
        """解析 LLM 返回的 JSON：thesis + confidence（或 confidence_score）。"""
        obj = self._extract_json_object(content)
        if not obj:
            return None, None
        thesis = obj.get("thesis") or obj.get("investment_thesis") or obj.get("analysis")
        if thesis is not None and not isinstance(thesis, str):
            thesis = str(thesis)
        conf = self._normalize_llm_confidence(
            obj.get("confidence", obj.get("confidence_score"))
        )
        if thesis:
            thesis = thesis.strip()[:2000]
        return thesis, conf

    def _generate_thesis(
        self, composite: float, current_price: float | None
    ) -> tuple[str, bool, float | None]:
        """生成投资逻辑；优先 JSON（含 confidence），失败则降级为模板或非结构化正文。"""
        # 构建摘要文本
        signal_lines = []
        for sig in self.agent_signals:
            signal_lines.append(
                f"- {sig.agent_name}: 信号={sig.signal:+.2f}，置信度={sig.confidence:.1%}，{sig.summary}"
            )
        signals_text = "\n".join(signal_lines)

        rating = self._signal_to_rating(composite)
        template = (
            f"综合{len(self.agent_signals)}个维度分析，{self.ticker}综合信号为{composite:+.2f}，"
            f"评级：{rating}。\n"
            f"{signals_text}\n"
            f"{'当前价格：' + str(current_price) + '元。' if current_price else ''}"
            f"建议投资者{'积极关注后续催化剂' if composite > 0 else '保持观望，等待风险释放'}。"
            f"本分析基于量化模型，不构成投资建议。"
        )

        if self.provider == "none":
            return template[:200], False, None

        try:
            from quantmind.agents.llm_client import build_client

            client = build_client(provider=self.provider, model=self.model)
            if client.is_none_provider:
                return template[:200], False, None

            prompt = (
                f"标的：{self.ticker}，截至 {self.as_of}。\n\n"
                f"各维度量化结果：\n{signals_text}\n\n"
                f"加权综合信号：{composite:+.2f}；规则映射评级：{rating}。\n"
                f"{'参考现价（人民币）：' + str(current_price) + ' 元。' if current_price else ''}\n\n"
                "请输出**仅一个合法 JSON 对象**（中文可直接写在字符串里，不要 markdown 围栏），键如下：\n"
                '- "thesis": 字符串，投资逻辑 120～220 字，研究口吻，结尾提示风险；\n'
                '- "confidence": 0 到 1 之间的小数，表示你对上述 thesis（而非原始量化信号）的整体置信度。\n'
                "示例：{\"thesis\":\"……\",\"confidence\":0.62}"
            )

            import threading

            result_holder: list[Any] = [None]

            def _call() -> None:
                try:
                    result_holder[0] = client.chat(
                        system=(
                            "你是卖方研究员。严格遵守用户要求的 JSON 格式输出；"
                            "除了 JSON 不要输出任何其它字符。"
                        ),
                        user=prompt,
                        max_tokens=512,
                        temperature=0.25,
                    )
                except Exception as e:
                    logger.warning(f"[StrategyAgent] LLM 调用失败: {e}")

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout=45.0)

            resp = result_holder[0]
            if resp and getattr(resp, "content", None):
                raw = resp.content.strip()
                thesis_parsed, conf_parsed = self._parse_strategy_llm_payload(raw)
                if thesis_parsed:
                    return thesis_parsed[:2000], True, conf_parsed
                # 非 JSON：仍视为 LLM 成功，退回正文无结构化置信度
                thesis_plain = raw[:200] if raw else template[:200]
                return thesis_plain, True, None

        except Exception as e:
            logger.warning(f"[StrategyAgent] LLM thesis 生成失败，使用模板: {e}")

        return template[:200], False, None

    # ─── Ollama ReAct 路径 ────────────────────────────────────────────────────

    def _analyze_with_llm_react(self) -> AgentSignal:
        """Ollama qwen2.5 ReAct 综合策略分析（读取各 Agent 信号 + Regime）."""
        from quantmind.agents.ollama_client import OllamaReActClient

        tools = [
            {
                "name": "get_all_agent_signals",
                "description": (
                    "运行其他 5 个分析 Agent（Valuation/Momentum/Quality/Sentiment/Risk）"
                    "，获取各自的 signal、confidence 和 summary"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
            {
                "name": "get_regime_context",
                "description": "获取当前 HMM 宏观 Regime（bull/neutral/bear）和各因子权重建议",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "get_position_sizing_advice",
                "description": "根据综合信号和置信度给出仓位建议（Kelly/ATR 方法）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "signal":     {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["signal", "confidence"],
                },
            },
        ]

        def _tool_agent_signals(ticker: str, **kw):
            """串行运行 5 个子 Agent（fast 模式避免递归调用 LLM）."""
            try:
                from quantmind.agents.investment_agents import (
                    ValuationAgent, MomentumAgent, QualityAgent,
                    SentimentAgent, RiskAgent,
                )
                results = {}
                for cls in [ValuationAgent, MomentumAgent, QualityAgent,
                             SentimentAgent, RiskAgent]:
                    try:
                        ag  = cls(ticker, self.as_of, self.context)
                        sig = ag.analyze(mode="fast")   # fast 避免递归
                        results[cls.__name__] = {
                            "signal":     round(sig.signal, 3),
                            "confidence": round(sig.confidence, 3),
                            "summary":    sig.summary[:60],
                        }
                    except Exception as e:
                        results[cls.__name__] = {"error": str(e)[:60]}
                return results
            except Exception as e:
                return {"error": str(e)}

        def _tool_regime(**kw):
            regime_text = self._get_snapshot_text("snapshot_macro")
            composite, confidence = self._calc_composite_signal()
            return {
                "composite_signal": round(composite, 3),
                "composite_confidence": round(confidence, 3),
                "regime_info": regime_text[:300] if regime_text else "Regime 数据不在 context 中",
                "weight_schema": {
                    "ValuationAgent": 0.25, "MomentumAgent": 0.20,
                    "QualityAgent": 0.25, "SentimentAgent": 0.15, "RiskAgent": 0.15,
                },
            }

        def _tool_position(signal: float, confidence: float, **kw):
            abs_sig = abs(signal)
            if abs_sig > 0.5 and confidence > 0.7:
                pos = "重仓(5-8%)"
            elif abs_sig > 0.3 and confidence > 0.6:
                pos = "标准(3-5%)"
            elif abs_sig > 0.15:
                pos = "轻仓(1-3%)"
            else:
                pos = "观察，不建仓"
            return {
                "signal":        signal,
                "confidence":    confidence,
                "position_size": pos,
                "kelly_fraction": round(min(0.08, abs_sig * confidence * 0.1), 3),
            }

        system_prompt = """你是 A 股多因子量化策略综合分析师。

你的任务是综合 5 个专项 Agent 的信号（估值/动量/质量/情绪/风险），
结合当前宏观 Regime，给出最终的综合投资建议。

分析框架：
1. 调用 get_all_agent_signals 获取各维度信号
2. 调用 get_regime_context 获取 Regime 背景
3. 分析各 Agent 信号的一致性：多数 bull 还是 bear？分歧在哪里？
4. 给出综合判断：是否具备 α 机会？仓位建议如何？
5. 调用 get_position_sizing_advice 给出仓位建议

注意：
- RiskAgent 信号为负代表风险高，是否值得持有需综合考量
- 各 Agent 信号高度一致（std 小）→ 置信度高
- Regime=bear 时整体信号需打折，更加谨慎

输出格式：
SIGNAL: <-1到+1，综合看多/看空>
CONFIDENCE: <0到1>
SUMMARY: <3句话综合投资结论，包含评级和仓位建议>
KEY_RISK: <最主要的综合风险>"""

        user_message = f"""请综合分析 {self.ticker} 的投资价值，给出策略建议。

请依次调用：get_all_agent_signals（获取各维度信号）→ get_regime_context（获取宏观背景）
→ get_position_sizing_advice（给出仓位建议）

行业：{self.context.get('industry', 'N/A')}
分析日期：{self.as_of}"""

        client = OllamaReActClient(model="qwen2.5:7b", timeout=90, max_steps=6)
        result = client.chat_with_tools(
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools,
            tool_executors={
                "get_all_agent_signals":     _tool_agent_signals,
                "get_regime_context":        _tool_regime,
                "get_position_sizing_advice": _tool_position,
            },
        )

        if result["fallback"]:
            strategy = self.analyze_with_llm()
            return AgentSignal(
                agent_name="StrategyAgent",
                ticker=self.ticker,
                signal=strategy.composite_signal,
                confidence=strategy.confidence,
                summary=f"{strategy.rating}，综合信号{strategy.composite_signal:+.2f}",
                evidence={"rating": strategy.rating, "method": "fallback_to_template"},
                warnings=strategy.key_risks[:3],
            )

        return AgentSignal(
            agent_name      = "StrategyAgent",
            ticker          = self.ticker,
            signal          = self._clamp(result["signal"]),
            confidence      = max(0.0, min(1.0, result["confidence"])),
            summary         = result["final_answer"][:100],
            evidence        = {"method": "ollama_react", "tools": result["tools_called"]},
            warnings        = [],
            reasoning_trace = result["reasoning_trace"],
            tools_called    = result["tools_called"],
            llm_mode        = True,
        )
