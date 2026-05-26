"""quantmind.agents.investment_agents.valuation_agent — 估值分析 Agent（规则 + LGBM v2 + Ollama ReAct）."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from loguru import logger
from scipy.stats import percentileofscore

from quantmind.agents.investment_agents.base_agent import AgentSignal, BaseInvestmentAgent

# 行业典型 PE/PB 参考（用于工具 fallback，当 panel 不可用时）
_INDUSTRY_PE_TYPICAL: dict[str, tuple[float, float]] = {
    "银行":     (5,  10),   "保险":     (10, 20),
    "证券":     (15, 30),   "房地产":   (8,  20),
    "医药生物": (25, 50),   "食品饮料": (25, 50),
    "白酒":     (30, 60),   "电子":     (25, 45),
    "计算机":   (30, 60),   "传媒":     (20, 40),
    "汽车":     (15, 30),   "钢铁":     (8,  20),
    "化工":     (15, 30),   "电力设备": (20, 40),
    "军工":     (40, 80),   "消费者服务":(20, 40),
    "纺织服饰": (15, 30),   "农林牧渔": (20, 50),
}


class ValuationAgent(BaseInvestmentAgent):
    """估值分析 Agent — 规则层 + 可选 LGBM v2 + Ollama qwen2.5 ReAct."""

    def analyze(self, mode: str = "fast") -> AgentSignal:
        """估值分析主入口.

        Parameters
        ----------
        mode : str
            'auto'  — 有 Ollama → LLM；无 Ollama → 规则
            'full'  — 强制 LLM（Ollama 不可用时降级规则并 WARNING）
            'fast'  — 强制规则（向后兼容默认值）
        """
        use_llm = (mode == "full") or (mode == "auto" and self._ollama_available())

        if use_llm:
            try:
                return self._analyze_with_llm()
            except Exception as e:
                logger.warning(
                    "[ValuationAgent] LLM 分析失败，降级规则: %s", e
                )
        return self._rule_based_analyze()
    # ─── 规则路径 ──────────────────────────────────────────────────────────────

    def _rule_based_analyze(self) -> AgentSignal:
        """规则 + LGBM 分析（原始 analyze() 逻辑，保持向后兼容）."""
        if self._should_use_lgbm():
            try:
                signal = float(self._lgbm_predict())
                signal = self._clamp(signal)
                summary = f"LGBM估值得分={signal:+.2f}（训练分布秩映射）"
                return AgentSignal(
                    agent_name="ValuationAgent",
                    ticker=self.ticker,
                    signal=signal,
                    confidence=0.75,
                    summary=summary[:50],
                    evidence={
                        "method": self._model_record.model_version if self._model_record else "lgbm",
                        "model_version": self._model_record.model_version if self._model_record else "",
                    },
                    warnings=[],
                )
            except Exception as e:
                logger.warning(f"[ValuationAgent/lgbm_v2] 预测失败，降级规则: {e}")

        evidence: dict = {}
        warnings: list[str] = []
        signal = 0.0
        fields_present = 0

        market_text = self._get_snapshot_text("snapshot_latest_market_metrics")
        fi_text = self._get_snapshot_text("snapshot_financial_indicator_summary")
        combined_text = market_text + "\n" + fi_text

        pe_ttm = self._parse_numeric(combined_text, ["pe_ttm", "pe", "市盈率"])
        pb = self._parse_numeric(combined_text, ["pb", "市净率"])
        ps_ttm = self._parse_numeric(combined_text, ["ps_ttm", "ps", "市销率"])
        dividend_yield = self._parse_numeric(combined_text, ["dv_ttm", "dividend_yield", "股息率"])
        roe_ttm = self._parse_numeric(combined_text, ["roe_ttm", "roe"])

        evidence["pe_ttm"] = pe_ttm
        evidence["pb"] = pb
        evidence["ps_ttm"] = ps_ttm
        evidence["dividend_yield"] = dividend_yield
        evidence["roe_ttm"] = roe_ttm

        if pe_ttm is not None:
            fields_present += 1
            if pe_ttm <= 0:
                warnings.append(f"PE={pe_ttm:.1f}（亏损或异常）")
            elif pe_ttm < 15:
                signal += 0.8
            elif pe_ttm <= 30:
                signal += 0.0
            elif pe_ttm <= 50:
                signal -= 0.2
            else:
                signal -= 0.8
                warnings.append(f"PE={pe_ttm:.1f}，估值过高")

        if pb is not None:
            fields_present += 1
            if pb < 0:
                warnings.append(f"PB={pb:.2f}（可能为负净资产）")
            elif pb < 2:
                signal += 0.1
            elif pb > 8:
                signal -= 0.3
                warnings.append(f"PB={pb:.2f}，市净率过高")

        if roe_ttm is not None:
            fields_present += 1
            roe_float = self._pct_to_float(roe_ttm)
            if roe_float is not None:
                if pe_ttm is not None and 0 < pe_ttm <= 25 and roe_float > 0.15:
                    signal += 0.2
                    evidence["roe_bonus"] = f"ROE={roe_float:.1%}，高ROE+合理PE"

                if pe_ttm is not None and pe_ttm > 0 and roe_float > 0:
                    peg = pe_ttm / (roe_float * 100)
                    evidence["peg_approx"] = round(peg, 2)
                    if peg < 1:
                        signal += 0.2
                        evidence["peg_note"] = "PEG<1，成长相对低估"
                    elif peg > 3:
                        signal -= 0.2
                        warnings.append(f"PEG={peg:.1f}，成长溢价过高")

        signal = self._clamp(signal)

        confidence = 0.9 if fields_present >= 3 else (0.6 if fields_present >= 1 else 0.4)

        if pe_ttm is not None and pb is not None:
            summary = (
                f"PE={pe_ttm:.1f}x，PB={pb:.2f}x，估值"
                f"{'偏低' if signal > 0.3 else ('偏高' if signal < -0.3 else '合理')}"
            )
        else:
            summary = "估值数据不足，信号低置信度"

        evidence["method"] = "rules_v1"

        return AgentSignal(
            agent_name="ValuationAgent",
            ticker=self.ticker,
            signal=signal,
            confidence=confidence,
            summary=summary[:50],
            evidence=evidence,
            warnings=warnings,
        )

    def _should_use_lgbm(self) -> bool:
        rec = self._model_record
        if rec is None or rec.model_type != "ml" or not rec.model_version.startswith("lgbm"):
            return False
        if not isinstance(self._ml_model, dict):
            return False
        # v2 bundle 用 "kind" 标识；v3 bundle 用 "version"
        kind = self._ml_model.get("kind") or self._ml_model.get("version") or ""
        return kind.startswith("valuation_lgbm") or kind.startswith("lgbm")

    def _lgbm_predict(self) -> float:
        bundle = self._ml_model
        model = bundle["model"]

        # v3 用 "features"；v2 用 "feature_cols"
        cols: list[str] = bundle.get("features") or bundle.get("feature_cols") or []
        quantiles: dict = bundle.get("feature_quantiles") or {}

        raw_row = [self._valuation_feature(c) for c in cols]
        # 若有分位映射，将原始值转为百分位秩（[0,1]），与训练一致
        ranked_row = []
        for val, col in zip(raw_row, cols):
            if col in quantiles and np.isfinite(val):
                qs = np.asarray(quantiles[col], dtype=float)
                # searchsorted：找到 val 在 quantiles 中的位置 → 近似百分位秩
                idx = float(np.searchsorted(qs, val, side="left"))
                ranked_row.append(idx / len(qs))
            else:
                ranked_row.append(float(val) if np.isfinite(val) else 0.5)
        vec = np.array(ranked_row, dtype=float).reshape(1, -1)

        # v2 有 pred_reference_sorted 做秩映射；v3 直接输出预测值
        ref_arr = bundle.get("pred_reference_sorted")
        best_it = bundle.get("best_iteration") or bundle.get("num_boost_round")

        try:
            if best_it is not None:
                raw = float(model.predict(vec, num_iteration=int(best_it))[0])
            else:
                raw = float(model.predict(vec)[0])
        except TypeError:
            raw = float(model.predict(vec)[0])

        if ref_arr is not None:
            ref = np.asarray(ref_arr, dtype=float)
            if ref.size > 0:
                pct = percentileofscore(ref, raw, kind="rank") / 100.0
                pct = max(0.0, min(1.0, float(pct)))
                return pct * 2.0 - 1.0

        # v3：预测值已是秩百分比 [0,1]（训练时对 label cross-section rank-normalize）
        if 0.0 <= raw <= 1.0:
            return float(raw) * 2.0 - 1.0
        return self._clamp(raw * 10.0)

    def _valuation_feature(self, name: str) -> float:  # noqa: C901
        """特征提取：先查 context dict，再解析快照文本，最后返回中性默认值."""
        ctx = self.context
        # context dict 直接命中
        ctx_val = self._safe_float(ctx.get(name))
        if ctx_val is not None:
            return float(ctx_val)

        market_text = self._get_snapshot_text("snapshot_latest_market_metrics")
        fi_text = self._get_snapshot_text("snapshot_financial_indicator_summary")
        combined = market_text + "\n" + fi_text

        # ── 基础估值乘数 ───────────────────────────────────────────────────────
        if name == "pe_ttm":
            v = self._parse_numeric(combined, ["pe_ttm", "pe", "市盈率"])
            return float(v) if v is not None else 0.0
        if name == "pb":
            v = self._parse_numeric(combined, ["pb", "市净率"])
            return float(v) if v is not None else 0.0
        if name == "ps_ttm":
            v = self._parse_numeric(combined, ["ps_ttm", "ps", "市销率"])
            return float(v) if v is not None else 0.0
        if name == "earnings_yield":
            # 1 / PE
            pe = self._safe_float(self._parse_numeric(combined, ["pe_ttm", "pe", "市盈率"]))
            if pe and pe > 0:
                return float(1.0 / pe)
            return 0.0
        if name == "book_to_market":
            # 1 / PB
            pb = self._safe_float(self._parse_numeric(combined, ["pb", "市净率"]))
            if pb and pb > 0:
                return float(1.0 / pb)
            return 0.0

        # ── 行业相对估值（截面注入；缺失时中性=1） ───────────────────────────
        if name == "sector_pe_ratio":
            return 1.0
        if name == "sector_pb_ratio":
            return 1.0

        # ── 盈利质量 ──────────────────────────────────────────────────────────
        if name in ("roe_ttm", "roe"):
            v = self._parse_numeric(combined, ["roe_ttm", "roe"])
            if v is None:
                return 0.0
            vv = self._pct_to_float(v)
            return float(vv) if vv is not None else float(v)
        if name in ("roa_ttm", "roa"):
            v = self._parse_numeric(combined, ["roa_ttm", "roa"])
            if v is None:
                return 0.0
            vv = self._pct_to_float(v)
            return float(vv) if vv is not None else float(v)
        if name in ("gross_margin", "grossprofit_margin"):
            v = self._parse_numeric(combined, ["gross_margin", "grossprofit_margin", "毛利率"])
            if v is None:
                return 0.0
            vv = self._pct_to_float(v)
            return float(vv) if vv is not None else float(v)
        if name in ("net_margin", "net_profit_margin"):
            v = self._parse_numeric(combined, ["net_margin", "netprofit_margin", "净利率"])
            if v is None:
                return 0.0
            vv = self._pct_to_float(v)
            return float(vv) if vv is not None else float(v)

        # ── 财务健康 ──────────────────────────────────────────────────────────
        if name == "debt_to_assets":
            v = self._parse_numeric(combined, ["debt_to_assets", "资产负债率", "debt_ratio"])
            if v is None:
                return 0.0
            vv = self._pct_to_float(v)
            return float(vv) if vv is not None else float(v)
        if name == "current_ratio":
            v = self._parse_numeric(combined, ["current_ratio", "流动比率"])
            return float(v) if v is not None else 0.0

        # ── 成长 ──────────────────────────────────────────────────────────────
        if name in ("revenue_yoy", "revenue_growth_yoy", "revenue_growth"):
            v = self._parse_numeric(combined, ["revenue_yoy", "or_yoy", "营收同比", "营业收入增长"])
            return float(v) if v is not None else 0.0
        if name in ("net_profit_yoy", "earnings_growth_yoy"):
            v = self._parse_numeric(combined, ["netprofit_yoy", "net_profit_yoy", "净利润同比"])
            return float(v) if v is not None else 0.0
        if name == "earnings_accel_q":
            v = self._parse_numeric(combined, ["earnings_accel_q", "q_op_qoq", "单季利润加速"])
            return float(v) if v is not None else 0.0
        if name == "revenue_accel_q":
            v = self._parse_numeric(combined, ["revenue_accel_q", "q_sales_yoy", "单季营收加速"])
            return float(v) if v is not None else 0.0

        # ── 现金流质量 ────────────────────────────────────────────────────────
        if name == "accruals":
            v = self._parse_numeric(combined, ["accruals", "应计利润", "应计项目"])
            return float(v) if v is not None else 0.0
        if name == "ocf_to_revenue_ttm":
            v = self._parse_numeric(combined, ["ocf_to_revenue_ttm", "经营现金流收入比"])
            return float(v) if v is not None else 0.0
        if name in ("fcf_yield", "cashflow_yield"):
            v = self._parse_numeric(combined, ["fcf_yield", "cashflow_yield", "现金流收益率"])
            return float(v) if v is not None else 0.0

        # ── 规模 ──────────────────────────────────────────────────────────────
        if name == "size_rank":
            return 0.5  # 中性（截面秩，无法实时计算全宇宙）
        if name == "log_market_cap":
            v = self._parse_numeric(combined, ["total_mv", "market_cap", "总市值"])
            if v is not None and float(v) > 0:
                return float(np.log(float(v) + 1))
            return 0.0

        # ── v2 兼容别名 ───────────────────────────────────────────────────────
        if name in ("pe_rank", "pb_rank"):
            return 0.5
        if name == "peg":
            pe_val = self._parse_numeric(combined, ["pe_ttm", "pe", "市盈率"])
            roe_val = self._parse_numeric(combined, ["roe_ttm", "roe"])
            if pe_val is not None and roe_val is not None:
                roe_f = self._pct_to_float(roe_val) or (roe_val / 100)
                if roe_f and roe_f > 0.01 and pe_val > 0:
                    return float(pe_val / (roe_f * 100))
            return 2.0
        if name in ("roe_delta", "roe_yoy"):
            return 0.0
        if name == "roe_stability":
            return 5.0

        return 0.0

    def _parse_numeric(self, text: str, keys: list[str]) -> float | None:
        for key in keys:
            pattern = rf"(?i){re.escape(key)}\s*[:：=]\s*(-?[\d.]+)"
            m = re.search(pattern, text)
            if m:
                return self._safe_float(m.group(1))
        return None

    # ─── LLM ReAct 路径 ────────────────────────────────────────────────────────

    def _analyze_with_llm(self) -> AgentSignal:
        """Ollama qwen2.5 ReAct 估值分析（工具调用 + RAG）."""
        from quantmind.agents.ollama_client import OllamaReActClient

        # ── 提取基础财务数据 ──────────────────────────────────────────────────
        market_text = self._get_snapshot_text("snapshot_latest_market_metrics")
        fi_text     = self._get_snapshot_text("snapshot_financial_indicator_summary")
        combined    = market_text + "\n" + fi_text

        pe_ttm  = self._parse_numeric(combined, ["pe_ttm", "pe", "市盈率"])
        pb      = self._parse_numeric(combined, ["pb", "市净率"])
        ps_ttm  = self._parse_numeric(combined, ["ps_ttm", "ps", "市销率"])
        roe_ttm = self._parse_numeric(combined, ["roe_ttm", "roe"])
        np_yoy  = self._parse_numeric(combined, ["netprofit_yoy", "net_profit_yoy", "净利润同比"])
        rv_yoy  = self._parse_numeric(combined, ["or_yoy", "revenue_yoy", "营收同比"])
        mv      = self._parse_numeric(combined, ["total_mv", "market_cap", "总市值"])
        industry = self.context.get("industry", "N/A")

        # ── 工具定义 ─────────────────────────────────────────────────────────
        tools = [
            {
                "name": "get_industry_peers_valuation",
                "description": (
                    "获取同行业可比公司的 PE/PB/PS 估值分位数，"
                    "判断当前股票估值在行业内的相对位置"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker":  {"type": "string", "description": "股票代码"},
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "如 pe_ttm、pb、ps_ttm",
                        },
                    },
                    "required": ["ticker", "metrics"],
                },
            },
            {
                "name": "get_historical_valuation_band",
                "description": (
                    "获取该股票自身历史 PE/PB 的分位数区间（最近3年），"
                    "判断当前估值处于历史高/中/低位"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker":        {"type": "string"},
                        "metric":        {"type": "string", "description": "pe_ttm 或 pb"},
                        "lookback_years": {"type": "integer", "default": 3},
                    },
                    "required": ["ticker", "metric"],
                },
            },
            {
                "name": "get_analyst_target_price",
                "description": "获取分析师目标价共识和当前价格的比较，计算隐含上行空间",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                    },
                    "required": ["ticker"],
                },
            },
        ]

        # ── 工具执行函数 ──────────────────────────────────────────────────────
        _pe  = pe_ttm
        _pb  = pb
        _ind = industry

        def _tool_get_industry_peers_valuation(
            ticker: str, metrics: list[str], **kw: Any
        ) -> dict:
            return self._tool_industry_peers(ticker, metrics, _pe, _pb, _ind)

        def _tool_get_historical_valuation_band(
            ticker: str, metric: str, lookback_years: int = 3, **kw: Any
        ) -> dict:
            return self._tool_historical_band(ticker, metric, _pe, _pb)

        def _tool_get_analyst_target_price(ticker: str, **kw: Any) -> dict:
            return self._tool_analyst_target(ticker)

        tool_executors = {
            "get_industry_peers_valuation":  _tool_get_industry_peers_valuation,
            "get_historical_valuation_band": _tool_get_historical_valuation_band,
            "get_analyst_target_price":      _tool_get_analyst_target_price,
        }

        # ── 系统提示词 ────────────────────────────────────────────────────────
        system_prompt = """你是一位专业的 A 股量化投资分析师，专注于基本面估值分析。

你的任务是分析给定股票的估值吸引力，输出投资信号。

分析框架（按顺序思考）：
1. 绝对估值：PE/PB/PS 是否合理？与行业和历史均值比较
2. 相对估值：与同行业可比公司相比是否低估？（调用工具）
3. 历史估值：当前估值在历史分位数的位置？（调用工具）
4. 成长匹配：PEG 分析，估值是否与成长性匹配？
5. 综合判断：给出 signal 和 confidence

请先调用工具获取行业对比和历史估值数据，再给出综合判断。

输出格式（严格遵守，最后输出）：
SIGNAL: <数值，-1到+1，正数看多，负数看空>
CONFIDENCE: <数值，0到1>
SUMMARY: <2-3句话的中文分析结论>
KEY_RISK: <最主要的一个风险点>"""

        # ── 用户消息 ──────────────────────────────────────────────────────────
        user_message = f"""请分析 {self.ticker} 的估值吸引力。

当前财务数据：
PE_TTM: {pe_ttm if pe_ttm is not None else 'N/A'}
PB: {pb if pb is not None else 'N/A'}
PS_TTM: {ps_ttm if ps_ttm is not None else 'N/A'}
ROE_TTM: {roe_ttm if roe_ttm is not None else 'N/A'}%
净利润增速(YoY): {np_yoy if np_yoy is not None else 'N/A'}%
营收增速(YoY): {rv_yoy if rv_yoy is not None else 'N/A'}%
行业: {industry}
市值: {mv if mv is not None else 'N/A'}亿元

请先调用 get_industry_peers_valuation 和 get_historical_valuation_band 工具获取行业对比和历史数据，再综合分析给出结论。"""

        # ── ReAct 调用 ────────────────────────────────────────────────────────
        client = OllamaReActClient(model="qwen2.5:7b", timeout=60, max_steps=5)
        result = client.chat_with_tools(
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools,
            tool_executors=tool_executors,
        )

        if result["fallback"]:
            logger.warning("[ValuationAgent] LLM 返回 fallback，降级规则")
            return self._rule_based_analyze()

        signal     = self._clamp(result["signal"])
        confidence = max(0.0, min(1.0, result["confidence"]))

        # 若 LLM 未能解析出有效 signal，使用规则结果作为保底
        if signal == 0.0 and confidence == 0.5:
            rule_sig = self._rule_based_analyze()
            signal     = rule_sig.signal
            confidence = max(rule_sig.confidence, 0.55)  # LLM 分析至少略高置信

        return AgentSignal(
            agent_name      = "ValuationAgent",
            ticker          = self.ticker,
            signal          = signal,
            confidence      = confidence,
            summary         = result["final_answer"][:100],
            evidence        = {
                "method":       "ollama_react",
                "model":        "qwen2.5:7b",
                "tools_called": result["tools_called"],
            },
            warnings        = [],
            reasoning_trace = result["reasoning_trace"],
            tools_called    = result["tools_called"],
            llm_mode        = True,
        )

    # ─── 工具实现 ──────────────────────────────────────────────────────────────

    def _tool_industry_peers(
        self,
        ticker: str,
        metrics: list[str],
        current_pe: float | None,
        current_pb: float | None,
        industry: str,
    ) -> dict[str, Any]:
        """获取同行业可比公司估值分位数（优先读 panel，fallback 行业典型值）."""
        result: dict[str, Any] = {
            "ticker":   ticker,
            "industry": industry,
            "data_source": "fallback_typical",
        }

        # 尝试从 alpha_panel 加载（可能较慢，只取最新截面）
        try:
            import pandas as pd
            from pathlib import Path as _Path
            panel_path = _Path(__file__).resolve().parents[4] / "data" / "panel" / "alpha_panel_v4.parquet"
            if panel_path.exists():
                panel = pd.read_parquet(panel_path)
                latest_as_of = panel.index.get_level_values("as_of").max()
                latest = panel.xs(latest_as_of, level="as_of")
                peers  = latest[latest.get("industry", pd.Series(dtype=str)) == industry]

                for m in metrics:
                    col = {"pe_ttm": "pe_ttm", "pb": "pb", "ps_ttm": "ps_ttm"}.get(m, m)
                    if col in peers.columns:
                        series = peers[col].dropna()
                        if len(series) > 3:
                            cur_val = series.get(ticker, series.median())
                            pct = float((series < cur_val).mean() * 100)
                            result[m] = {
                                "current":             round(float(cur_val), 2),
                                "industry_percentile": round(pct, 1),
                                "industry_median":     round(float(series.median()), 2),
                                "industry_p25":        round(float(series.quantile(0.25)), 2),
                                "industry_p75":        round(float(series.quantile(0.75)), 2),
                            }
                result["data_source"] = "alpha_panel_v4"
                return result
        except Exception as e:
            logger.debug("[ValuationAgent/tool] panel 加载失败: %s，使用典型值", e)

        # Fallback：行业典型 PE 范围
        pe_range = _INDUSTRY_PE_TYPICAL.get(industry, (15, 35))
        pe_mid   = (pe_range[0] + pe_range[1]) / 2
        if "pe_ttm" in metrics and current_pe is not None:
            pct = max(0, min(100, (current_pe - pe_range[0]) / (pe_range[1] - pe_range[0] + 1e-9) * 100))
            result["pe_ttm"] = {
                "current":             round(current_pe, 2),
                "industry_percentile": round(pct, 1),
                "industry_median":     pe_mid,
                "note":                f"{industry} 典型PE区间 {pe_range[0]}-{pe_range[1]}x",
            }
        if "pb" in metrics and current_pb is not None:
            result["pb"] = {
                "current": round(current_pb, 2),
                "note":    "行业 PB 数据使用典型值",
            }
        return result

    def _tool_historical_band(
        self,
        ticker: str,
        metric: str,
        current_pe: float | None,
        current_pb: float | None,
    ) -> dict[str, Any]:
        """获取股票历史估值分位数（从 snapshot 文本解析，或使用合理估算）."""
        cur_val = current_pe if metric == "pe_ttm" else current_pb
        if cur_val is None:
            return {"ticker": ticker, "metric": metric, "error": "当前值不可用"}

        # 从 context 中查找历史高低信息
        all_text = self._get_snapshot_text("snapshot_latest_market_metrics")

        # 尝试解析历史高低
        hi_key = ["52w_high", "52wk_high", "年内最高", "yearly_high"]
        lo_key = ["52w_low",  "52wk_low",  "年内最低", "yearly_low"]
        hist_high = self._parse_numeric(all_text, hi_key)
        hist_low  = self._parse_numeric(all_text, lo_key)

        if hist_high is not None and hist_low is not None and hist_high > hist_low:
            pct = (cur_val - hist_low) / (hist_high - hist_low + 1e-9) * 100
            level = "历史高位(>75%)" if pct > 75 else (
                "历史中位(25-75%)" if pct > 25 else "历史低位(<25%)"
            )
        else:
            pct   = 50.0
            level = "历史分位数据不足"

        return {
            "ticker":          ticker,
            "metric":          metric,
            "current_value":   round(cur_val, 2),
            "hist_percentile": round(pct, 1),
            "position_label":  level,
            "hist_high":       hist_high,
            "hist_low":        hist_low,
            "lookback":        "snapshot 数据，约 1 年",
        }

    def _tool_analyst_target(self, ticker: str) -> dict[str, Any]:
        """获取分析师目标价（从 context 提取或返回提示）."""
        all_text = (
            self._get_snapshot_text("snapshot_latest_market_metrics")
            + self._get_snapshot_text("snapshot_financial_indicator_summary")
        )
        target = self._parse_numeric(all_text, ["target_price", "目标价", "analyst_target"])
        cur_price = self._parse_numeric(all_text, ["close", "current_price", "最新价", "收盘价"])

        if target is not None and cur_price is not None and cur_price > 0:
            upside_pct = (target - cur_price) / cur_price * 100
            return {
                "ticker":         ticker,
                "target_price":   round(target, 2),
                "current_price":  round(cur_price, 2),
                "upside_pct":     round(upside_pct, 1),
                "implied_signal":  "看多" if upside_pct > 10 else ("看空" if upside_pct < -5 else "中性"),
            }
        return {
            "ticker":  ticker,
            "note":    "context 中暂无分析师目标价数据",
            "advice":  "参考行业 PE 均值和公司历史 PE 进行估值判断",
        }
