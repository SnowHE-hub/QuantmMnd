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

    @staticmethod
    def _ctx_val(ctx: dict, *keys: str) -> Any:
        """从 context 中按优先级查找值（同时兼容 snapshot_xxx 和 xxx 两种 key 格式）.

        查找顺序：
          1. 直接 key（如 "pe_ttm"）
          2. 带 snapshot_ 前缀（如 "snapshot_pe_ttm"）
          3. 全部 None → 返回 None

        会过滤掉 0 / None / NaN 值，确保注入的值有实际意义。
        """
        import math
        for k in keys:
            for candidate in (k, f"snapshot_{k}"):
                v = ctx.get(candidate)
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if math.isnan(fv) or fv == 0.0:
                        continue
                    return fv
                except (TypeError, ValueError):
                    # 非数值（如 industry 字符串）直接返回
                    if v:
                        return v
        return None

    def _build_user_message(self, ticker: str, snapshot: dict) -> str:
        """构建发给 LLM 的用户消息，从 snapshot（或 context）中提取财务数据.

        同时兼容 snapshot_pe_ttm 和 pe_ttm 两种 key 格式，
        确保即使 context 只有 snapshot_ 前缀的 key，LLM 也能收到实际数值。

        Parameters
        ----------
        ticker : str
            股票代码
        snapshot : dict
            通常是 ``self.context``（含 snapshot_ 前缀 key 及直接 key）

        Returns
        -------
        str
            格式化的用户消息
        """
        g = lambda *keys: self._ctx_val(snapshot, *keys)  # noqa: E731

        pe_raw    = g("pe_ttm", "snapshot_pe_ttm")
        pb_raw    = g("pb", "snapshot_pb")
        ps_raw    = g("ps_ttm", "snapshot_ps_ttm")
        roe_raw   = g("roe_ttm", "snapshot_roe_ttm")
        np_yoy    = g("net_profit_yoy", "netprofit_yoy", "snapshot_net_profit_yoy")
        rv_yoy    = g("revenue_yoy", "snapshot_revenue_yoy", "or_yoy")
        mv        = g("total_mv", "log_market_cap", "snapshot_total_mv")
        industry  = (
            g("industry", "industry_name", "sw_industry", "exposure_industry")
            or snapshot.get("snapshot_exposure_industry")
            or "N/A"
        )

        def fmt(v, unit: str = "", scale: float = 1.0) -> str:
            if v is None:
                return "N/A"
            try:
                return f"{float(v) * scale:.4f}{unit}"
            except (TypeError, ValueError):
                return str(v)

        # pe_ttm 在 alpha_panel 中是截面因子分（earnings_yield 的秩归一化值），
        # 范围约 0–1；越高表示越便宜（PE 越低）。
        # 为便于 LLM 直觉理解，同时展示原始因子分和"推算 PE"供参考。
        pe_note = ""
        if pe_raw is not None:
            if 0 < float(pe_raw) < 2:
                # 看起来像 earnings_yield 格式（0–1 范围）
                pe_note = f"（因子分={pe_raw:.4f}；factor越高→估值越低）"
            else:
                pe_note = f"（倍数）"

        return (
            f"请分析 {ticker} 的估值吸引力。\n\n"
            f"当前财务数据（来自量化因子面板）：\n"
            f"PE_TTM因子:      {fmt(pe_raw)}{pe_note}\n"
            f"PB:              {fmt(pb_raw)}\n"
            f"PS_TTM:          {fmt(ps_raw)}\n"
            f"ROE_TTM:         {fmt(roe_raw)}\n"
            f"净利润增速(YoY): {fmt(np_yoy, '%')}\n"
            f"营收增速(YoY):   {fmt(rv_yoy, '%')}\n"
            f"市值/规模因子:   {fmt(mv)}\n"
            f"行业:            {industry}\n\n"
            f"注：面板中 pe_ttm 为截面因子分（earnings_yield 排名归一化），"
            f"越高表示该股 PE 越低（越便宜）。\n"
            f"请先调用 get_industry_peers_valuation（指标：pe_ttm,pb,ps_ttm）"
            f"和 get_historical_valuation_band 工具获取行业对比和历史数据，"
            f"再综合分析给出结论。"
        )

    def _analyze_with_llm(self) -> AgentSignal:
        """Ollama qwen2.5 ReAct 估值分析（工具调用 + RAG）.

        数据注入修复：同时兼容 snapshot_xxx 和 xxx 两种 context key 格式，
        确保 pe_ttm 等因子值能正确传入 LLM。
        工具函数修复：使用 level='ticker'（panel index level='ticker'），
        并用 exposure_industry 列做同行业过滤。
        """
        from quantmind.agents.ollama_client import OllamaReActClient

        # ── 用户消息（修复 A：直接从 context 读取，支持 snapshot_ 前缀）─────────
        user_message = self._build_user_message(self.ticker, self.context)

        # ── 工具定义 ─────────────────────────────────────────────────────────
        tools = [
            {
                "name": "get_industry_peers_valuation",
                "description": (
                    "获取同行业可比公司的 pe_ttm/pb/ps_ttm 因子分位数，"
                    "判断当前股票估值在行业内的相对位置"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker":  {"type": "string", "description": "股票代码"},
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "如 ['pe_ttm', 'pb', 'ps_ttm']",
                        },
                    },
                    "required": ["ticker", "metrics"],
                },
            },
            {
                "name": "get_historical_valuation_band",
                "description": (
                    "获取该股票自身历史 pe_ttm/pb 因子分的分位数区间（最近N年），"
                    "判断当前估值处于历史高/中/低位"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker":         {"type": "string"},
                        "metric":         {"type": "string", "description": "pe_ttm 或 pb"},
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
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        ]

        # ── 工具执行函数（修复 B：直接在工具内实现，不再传 _pe/_pb/_ind 闭包变量）
        def _tool_get_industry_peers_valuation(
            ticker: str, metrics: list[str], **kw: Any
        ) -> dict:
            return self._tool_industry_peers(ticker, metrics)

        def _tool_get_historical_valuation_band(
            ticker: str, metric: str, lookback_years: int = 3, **kw: Any
        ) -> dict:
            return self._tool_historical_band(ticker, metric, lookback_years)

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
1. 绝对估值：PE/PB/PS 因子分是否偏高或偏低（因子分越高→估值越低→越便宜）
2. 相对估值：与同行业可比公司相比是否低估？（调用工具）
3. 历史估值：当前估值在历史分位数的位置？（调用工具）
4. 成长匹配：ROE 和 YoY 增速是否支撑当前估值？
5. 综合判断：给出 signal 和 confidence

请先调用工具获取行业对比和历史估值数据，再给出综合判断。

输出格式（严格遵守，最后输出）：
SIGNAL: <数值，-1到+1，正数看多，负数看空>
CONFIDENCE: <数值，0到1>
SUMMARY: <2-3句话的中文分析结论，包含具体数字>
KEY_RISK: <最主要的一个风险点>"""

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
            summary         = result["final_answer"][:150],
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

    # ─── 工具实现（修复 B：level='ticker'，exposure_industry 列）─────────────────

    @staticmethod
    def _load_panel_latest() -> "tuple[Any, Any] | tuple[None, None]":
        """加载 alpha_panel_v4 并返回 (panel, latest_cross_section)，失败返回 (None, None)."""
        try:
            import pandas as pd
            from pathlib import Path as _Path
            panel_path = (
                _Path(__file__).resolve().parents[3]
                / "data" / "panel" / "alpha_panel_v4.parquet"
            )
            if not panel_path.exists():
                return None, None
            panel = pd.read_parquet(panel_path)
            # 取最近有实际数据的截面（跳过全 NaN 截面）
            dates = panel.index.get_level_values("as_of").unique().sort_values()
            for d in reversed(dates):
                cross = panel.xs(d, level="as_of")
                if cross["pe_ttm"].replace(0, float("nan")).notna().sum() > 10:
                    return panel, cross
            # fallback: 最后一个截面
            latest = panel.xs(dates[-1], level="as_of")
            return panel, latest
        except Exception:
            return None, None

    def _tool_industry_peers(
        self,
        ticker: str,
        metrics: list[str],
    ) -> dict[str, Any]:
        """获取同行业可比公司估值分位数（修复：level='ticker'，exposure_industry 列）.

        数据源优先级：
          1. alpha_panel_v4.parquet 最新有效截面
          2. 行业典型 PE 范围 fallback
        """
        import pandas as pd

        panel, latest = self._load_panel_latest()

        # ── panel 路径 ────────────────────────────────────────────────────────
        if latest is not None:
            # 1. 找当前股票所在行业（用 exposure_industry 列）
            industry = None
            try:
                # 修复：使用 level='ticker'
                if ticker in latest.index:
                    row = latest.loc[ticker]
                    for col in ("exposure_industry", "industry", "industry_name", "sw_industry"):
                        val = row.get(col) if isinstance(row, pd.Series) else None
                        if val and str(val) not in ("", "nan", "None"):
                            industry = str(val)
                            break
            except Exception as exc:
                logger.debug("[ValuationAgent/tool] 获取行业失败: %s", exc)

            # 2. 同行业过滤（修复：直接用 exposure_industry 列做 boolean mask）
            if industry:
                for col in ("exposure_industry", "industry", "industry_name", "sw_industry"):
                    if col in latest.columns:
                        peers = latest[latest[col] == industry]
                        if len(peers) >= 3:
                            break
                else:
                    peers = latest  # fallback：全量截面
            else:
                peers = latest

            result: dict[str, Any] = {
                "ticker":      ticker,
                "industry":    industry or "未知",
                "data_source": "alpha_panel_v4",
                "n_peers":     len(peers),
            }

            for m in metrics:
                col = m  # 面板列名与指标名相同
                if col not in peers.columns:
                    result[m] = {"error": f"列 {col!r} 不在面板中"}
                    continue

                vals = peers[col].replace(0, float("nan")).dropna()
                if len(vals) < 3:
                    result[m] = {"error": f"同行业有效样本不足（{len(vals)} 条）"}
                    continue

                # 当前股票值
                try:
                    cur_val = float(latest.loc[ticker, col]) if ticker in latest.index else float("nan")
                except (KeyError, TypeError):
                    cur_val = float("nan")

                if not (cur_val == cur_val) or cur_val == 0:  # NaN 或 0
                    result[m] = {"error": f"{ticker} 的 {col} 数据缺失或为 0"}
                    continue

                pct = float((vals < cur_val).mean() * 100)
                result[m] = {
                    "current":             round(cur_val, 4),
                    "industry_percentile": round(pct, 1),
                    "industry_median":     round(float(vals.median()), 4),
                    "industry_p10":        round(float(vals.quantile(0.10)), 4),
                    "industry_p90":        round(float(vals.quantile(0.90)), 4),
                    "n_peers":             int(len(vals)),
                    "note": (
                        "因子分越高→估值越低→越便宜；"
                        f"当前{pct:.0f}%分位{'（偏高→相对便宜）' if pct >= 60 else '（偏低→相对偏贵）'}"
                    ),
                }
            return result

        # ── Fallback：行业典型 PE 范围 ────────────────────────────────────────
        # 尝试从 context 推断行业
        industry = (
            self._ctx_val(self.context, "industry", "exposure_industry",
                          "industry_name", "sw_industry")
            or "未知"
        )
        current_pe = self._ctx_val(self.context, "pe_ttm", "snapshot_pe_ttm")
        current_pb = self._ctx_val(self.context, "pb", "snapshot_pb")

        pe_range = _INDUSTRY_PE_TYPICAL.get(str(industry), (15, 35))
        pe_mid   = (pe_range[0] + pe_range[1]) / 2
        result = {
            "ticker":      ticker,
            "industry":    str(industry),
            "data_source": "fallback_typical",
        }
        if "pe_ttm" in metrics and current_pe is not None:
            pct = max(0, min(100,
                (float(current_pe) - pe_range[0]) / (pe_range[1] - pe_range[0] + 1e-9) * 100))
            result["pe_ttm"] = {
                "current":             round(float(current_pe), 4),
                "industry_percentile": round(pct, 1),
                "industry_median":     pe_mid,
                "note":                f"{industry} 典型PE区间 {pe_range[0]}-{pe_range[1]}x（典型值估算）",
            }
        if "pb" in metrics and current_pb is not None:
            result["pb"] = {
                "current": round(float(current_pb), 4),
                "note":    "行业 PB 数据暂缺，使用典型值",
            }
        return result

    def _tool_historical_band(
        self,
        ticker: str,
        metric: str,
        lookback_years: int = 3,
    ) -> dict[str, Any]:
        """获取股票历史估值分位数（修复：从 panel 读取历史截面，不再仅依赖 snapshot 文本）."""
        import pandas as pd

        panel, _ = self._load_panel_latest()

        if panel is not None and metric in panel.columns:
            try:
                # 修复：使用 level='ticker'
                hist = panel.xs(ticker, level="ticker")[metric]
                hist = hist.replace(0, float("nan")).dropna()

                if len(hist) >= 4:
                    # 限制 lookback 年数（近似每年 4 个季度）
                    n_quarters = lookback_years * 4
                    hist = hist.iloc[-n_quarters:]

                    cur_val = float(hist.iloc[-1])
                    pct = float((hist < cur_val).mean() * 100)

                    # 历史高位/低位判断
                    if pct >= 75:
                        level = f"历史高位（近{lookback_years}年{pct:.0f}%分位，因子值高→估值低）"
                    elif pct >= 25:
                        level = f"历史中位（近{lookback_years}年{pct:.0f}%分位）"
                    else:
                        level = f"历史低位（近{lookback_years}年{pct:.0f}%分位，因子值低→估值高）"

                    return {
                        "ticker":          ticker,
                        "metric":          metric,
                        "current_value":   round(cur_val, 4),
                        "hist_percentile": round(pct, 1),
                        "position_label":  level,
                        "hist_min":        round(float(hist.min()), 4),
                        "hist_max":        round(float(hist.max()), 4),
                        "hist_median":     round(float(hist.median()), 4),
                        "n_periods":       len(hist),
                        "lookback_years":  lookback_years,
                        "data_source":     "alpha_panel_v4",
                    }
            except KeyError:
                logger.debug("[ValuationAgent/tool] %s 不在 panel 中", ticker)
            except Exception as exc:
                logger.debug("[ValuationAgent/tool] 历史分位计算失败: %s", exc)

        # Fallback：从 context 中读当前值，给出保守估算
        cur_val = self._ctx_val(self.context, metric, f"snapshot_{metric}")
        if cur_val is None:
            return {"ticker": ticker, "metric": metric, "error": "当前值不可用，panel 也无法加载"}

        return {
            "ticker":          ticker,
            "metric":          metric,
            "current_value":   round(float(cur_val), 4),
            "hist_percentile": 50.0,
            "position_label":  "历史分位数据不足（仅单点值）",
            "data_source":     "context_fallback",
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
                "ticker":        ticker,
                "target_price":  round(target, 2),
                "current_price": round(cur_price, 2),
                "upside_pct":    round(upside_pct, 1),
                "implied_signal": "看多" if upside_pct > 10 else ("看空" if upside_pct < -5 else "中性"),
            }
        return {
            "ticker":  ticker,
            "note":    "context 中暂无分析师目标价数据",
            "advice":  "参考行业 PE 因子分位和公司历史估值分位进行判断",
        }
