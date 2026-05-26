"""quantmind.agents.investment_agents.risk_agent — 风险评估 Agent（规则 + GARCH v2）."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from quantmind.agents.investment_agents.base_agent import AgentSignal, BaseInvestmentAgent

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PRICE_FILE = _ROOT / "data" / "raw" / "alpha_prices_panel.parquet"


class RiskAgent(BaseInvestmentAgent):
    """风险评估 Agent — 波动率 + 财务 + 流动性 + 北向资金 + 可选 GARCH v2."""

    def analyze(self, mode: str = "fast") -> AgentSignal:
        """风险分析主入口（mode='auto'/'full' → LLM；'fast' → GARCH/规则）."""
        use_llm = (mode == "full") or (mode == "auto" and self._ollama_available())
        if use_llm:
            try:
                return self._analyze_with_llm()
            except Exception as e:
                logger.warning("[RiskAgent] LLM 失败，降级规则: %s", e)

        return self._rule_based_analyze()

    def _rule_based_analyze(self) -> AgentSignal:
        """原始规则 + GARCH 风险分析逻辑。"""
        evidence: dict = {}
        warnings: list[str] = []
        signal = 0.0

        garch_vol = self._garch_vol()
        evidence["garch_vol"] = round(garch_vol, 6) if garch_vol is not None else None

        # 优先从 pkl 读取预计算的 vol_63d，避免实时读取大价格面板
        vol_63d = self._precomputed_vol_63d()
        if vol_63d is None:
            vol_63d = self._calc_volatility()
        evidence["vol_63d_annualized"] = round(vol_63d, 4) if vol_63d is not None else None

        eff_vol = garch_vol if garch_vol is not None else vol_63d

        # ── 个股 HMM regime extras（v3 bundle 专属） ─────────────────────
        hmm = self._hmm_extras()
        if hmm:
            evidence.update(hmm)
            regime_id = hmm.get("hmm_regime_id", 1)
            if regime_id == 2:
                signal -= 0.25
                warnings.append("HMM 检测到危机状态（高波动区间），风险溢价上升")
            elif regime_id == 0:
                signal += 0.1
                evidence["regime_note"] = "HMM 低波动状态，系统性风险偏低"
            cvar_d = hmm.get("cvar_95_daily")
            if cvar_d is not None and float(cvar_d) > 0.04:
                signal -= 0.15
                warnings.append(f"日度CVaR(95%)={float(cvar_d):.2%}，尾部风险较大")
            # Beta 系统性风险
            beta = hmm.get("beta_vs_csi300")
            if beta is not None:
                if float(beta) > 1.5:
                    signal -= 0.1
                    warnings.append(f"高Beta({float(beta):.2f})，系统性风险暴露偏大")
                elif float(beta) < 0.5:
                    evidence["beta_note"] = f"低Beta({float(beta):.2f})，防御属性强"

        # ── 市场级 HMM（系统性风险感知） ─────────────────────────────────
        mkt = self._market_regime()
        if mkt:
            evidence["market_regime"] = mkt.get("regime_name", "unknown")
            evidence["market_regime_id"] = mkt.get("regime_id", 1)
            mkt_id = mkt.get("regime_id", 1)
            if mkt_id == 2:  # bear_crisis
                signal -= 0.2
                warnings.append(f"市场处于熊市/危机状态（CSI300 HMM），系统性风险高")
            elif mkt_id == 0:  # bull_low_vol
                signal += 0.05
                evidence["market_note"] = "市场低波动牛市状态，系统性尾风险偏低"

        if eff_vol is not None:
            if eff_vol < 0.20:
                signal += 0.3
                evidence["vol_note"] = f"低波动({eff_vol:.1%})，风险可控"
            elif eff_vol <= 0.40:
                signal += 0.0
            else:
                signal -= 0.45
                warnings.append(f"年化波动率≈{eff_vol:.1%}（HMM/GARCH/历史），偏高")

            if garch_vol is not None and vol_63d is not None:
                evidence["vol_blend"] = "hmm_garch_primary_hist_secondary" if hmm else "garch_primary_hist_secondary"

        fi_text = self._get_snapshot_text("snapshot_financial_indicator_summary")
        market_text = self._get_snapshot_text("snapshot_latest_market_metrics")
        combined = fi_text + "\n" + market_text

        debt_to_assets = self._parse_numeric(combined, ["debt_to_assets", "资产负债率"])
        netprofit = self._parse_numeric(combined, ["netprofit", "净利润"])
        market_cap = self._parse_numeric(combined, ["total_mv", "market_cap", "总市值", "circ_mv"])

        if debt_to_assets is not None:
            dta = self._pct_to_float(debt_to_assets)
            evidence["debt_to_assets"] = f"{dta:.1%}" if dta else str(debt_to_assets)
            if dta is not None and dta > 0.70:
                signal -= 0.4
                warnings.append(f"高负债率={dta:.1%}，财务风险较高")

        if netprofit is not None:
            evidence["netprofit"] = netprofit
            if netprofit < 0:
                signal -= 0.5
                warnings.append(f"净利润为负({netprofit:.0f}万)，亏损状态")

        if market_cap is not None:
            evidence["market_cap"] = market_cap
            cap_in_100m = market_cap / 10000
            if cap_in_100m < 50:
                signal -= 0.2
                warnings.append(f"市值≈{cap_in_100m:.0f}亿，小盘流动性风险")
            elif cap_in_100m > 500:
                evidence["size_note"] = f"大盘股({cap_in_100m:.0f}亿)，流动性充足"

        nb_text = self._get_snapshot_text("snapshot_northbound_summary")
        if nb_text:
            hold_ratio = self._parse_numeric(nb_text, ["hold_ratio", "持股比例", "北向持股"])
            if hold_ratio is not None:
                evidence["northbound_hold_ratio"] = hold_ratio
                if any(w in nb_text for w in ["增加", "提升", "净买入", "流入"]):
                    signal += 0.1
                    evidence["northbound_trend"] = "北向净流入"
                elif any(w in nb_text for w in ["减少", "下降", "净卖出", "流出"]):
                    signal -= 0.2
                    warnings.append("北向资金持仓比例下降趋势")

        signal = self._clamp(signal)

        data_count = sum(v is not None for v in [eff_vol, debt_to_assets, market_cap])
        confidence = min(0.9, 0.4 + data_count * 0.15)

        parts = []
        if eff_vol is not None:
            parts.append(f"波动率{eff_vol:.0%}")
        if warnings:
            parts.append(f"{len(warnings)}项风险")
        risk_level = "低风险" if signal > 0.1 else ("高风险" if signal < -0.3 else "中等风险")
        summary = "，".join(parts) + f"，{risk_level}" if parts else risk_level

        return AgentSignal(
            agent_name="RiskAgent",
            ticker=self.ticker,
            signal=round(signal, 3),
            confidence=confidence,
            summary=summary[:50],
            evidence=evidence,
            warnings=warnings,
        )

    def _is_risk_bundle(self) -> bool:
        if not isinstance(self._ml_model, dict):
            return False
        kind = self._ml_model.get("kind") or ""
        return kind in ("risk_garch_v2", "risk_hmm_v3")

    # keep backward compat name
    def _is_garch_bundle(self) -> bool:
        return self._is_risk_bundle()

    def _garch_vol(self) -> float | None:
        """读取训练阶段写入的 21 日前瞻年化波动（garch_v2 / hmm_v3 均支持）."""
        rec = self._model_record
        if (
            rec is None
            or rec.model_type != "ml"
            or not (rec.model_version.startswith("garch") or rec.model_version.startswith("hmm"))
            or not self._is_risk_bundle()
        ):
            return None

        table = self._ml_model.get("vol_by_ticker") or {}
        info = table.get(self.ticker)
        if not isinstance(info, dict):
            return None

        v21 = info.get("ann_vol_forecast_21d")
        if isinstance(v21, (int, float)) and np.isfinite(v21):
            return float(v21)
        lr = info.get("long_run_vol")
        if isinstance(lr, (int, float)) and np.isfinite(lr):
            return float(lr)
        return None

    def _precomputed_vol_63d(self) -> float | None:
        """从 hmm_v3 bundle 读取预计算的 63 日年化波动率."""
        if not self._is_risk_bundle():
            return None
        table = self._ml_model.get("vol_by_ticker") or {}
        info = table.get(self.ticker)
        if not isinstance(info, dict):
            return None
        v = info.get("vol_63d_annualized")
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
        return None

    def _hmm_extras(self) -> dict:
        """从 hmm_v3 bundle 提取 regime 信息、CVaR 和 Beta."""
        if not isinstance(self._ml_model, dict):
            return {}
        if self._ml_model.get("kind") != "risk_hmm_v3":
            return {}
        table = self._ml_model.get("vol_by_ticker") or {}
        info = table.get(self.ticker)
        if not isinstance(info, dict):
            return {}

        regime_labels = self._ml_model.get("regime_labels") or {0: "low_vol", 1: "normal", 2: "crisis"}
        regime_id = info.get("current_regime", 1)
        regime_name = regime_labels.get(regime_id, str(regime_id))
        regime_vols = info.get("regime_vols") or {}

        result: dict = {
            "hmm_regime":        regime_name,
            "hmm_regime_id":     regime_id,
            "hmm_regime_vols":   {regime_labels.get(k, str(k)): v for k, v in regime_vols.items()},
            "cvar_95_daily":     info.get("cvar_95_daily"),
            "cvar_99_daily":     info.get("cvar_99_daily"),
            "ewma_lambda":       info.get("ewma_lambda"),
        }
        # Beta
        for key in ("beta_vs_csi300", "beta_vs_csi500"):
            v = info.get(key)
            if v is not None:
                result[key] = v
        return result

    def _market_regime(self) -> dict:
        """从 hmm_v3 bundle 读取市场级 HMM 状态."""
        if not isinstance(self._ml_model, dict):
            return {}
        if self._ml_model.get("kind") != "risk_hmm_v3":
            return {}
        mkt = self._ml_model.get("market_hmm") or {}
        if not mkt:
            return {}
        regime_id = mkt.get("current_regime", 1)
        labels = mkt.get("regime_labels") or {0: "bull_low_vol", 1: "normal", 2: "bear_crisis"}
        return {
            "regime_id":   regime_id,
            "regime_name": labels.get(regime_id, str(regime_id)),
            "csi300_ret_63d":  mkt.get("csi300_ret_63d"),
            "csi300_vol_21d":  mkt.get("csi300_vol_21d"),
            "recent_probs":    mkt.get("recent_30d_probs"),
        }

    def _calc_volatility(self) -> float | None:
        """从价格面板计算 63 日年化波动率（备用；优先使用预计算的 vol_63d_annualized）."""
        try:
            # 尝试 adj_close，回退 close
            try:
                raw = pd.read_parquet(_PRICE_FILE, filters=[("ts_code", "=", self.ticker)],
                                      columns=["ts_code", "trade_date", "adj_close"])
                price_col = "adj_close"
            except Exception:
                raw = pd.read_parquet(_PRICE_FILE, filters=[("ts_code", "=", self.ticker)],
                                      columns=["ts_code", "trade_date", "close"])
                price_col = "close"
            sub = raw[["trade_date", price_col]].copy()
            if sub.empty:
                return None
            sub["trade_date"] = pd.to_datetime(sub["trade_date"])
            sub = sub.sort_values("trade_date").set_index("trade_date")[price_col].dropna()
            if self.as_of:
                cutoff = pd.Timestamp(self.as_of)
                sub = sub[sub.index <= cutoff]
            if len(sub) < 20:
                return None
            returns = sub.pct_change(fill_method=None).dropna()
            window = returns.iloc[-63:] if len(returns) >= 63 else returns
            return float(window.std() * np.sqrt(252))
        except Exception as e:
            logger.warning(f"[RiskAgent] 波动率计算失败: {e}")
            return None

    def _parse_numeric(self, text: str, keys: list[str]) -> float | None:
        for key in keys:
            pattern = rf"(?i){re.escape(key)}\s*[:：=]\s*(-?[\d.]+)"
            m = re.search(pattern, text)
            if m:
                return self._safe_float(m.group(1))
        return None

    # ─── Ollama ReAct 路径 ────────────────────────────────────────────────────

    def _analyze_with_llm(self) -> AgentSignal:
        """Ollama qwen2.5 ReAct 风险分析（波动率 + 流动性 + Beta）."""
        from quantmind.agents.ollama_client import OllamaReActClient

        mkt_text = self._get_snapshot_text("snapshot_latest_market_metrics")
        fi_text  = self._get_snapshot_text("snapshot_financial_indicator_summary")
        combined = mkt_text + "\n" + fi_text

        tools = [
            {
                "name": "get_volatility_regime",
                "description": "计算股票历史波动率的分位数位置，判断当前波动率处于历史高/中/低位",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
            {
                "name": "get_liquidity_risk",
                "description": "评估股票流动性风险：换手率、成交量稳定性、Amihud 非流动性指标",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
            {
                "name": "get_correlation_with_market",
                "description": "计算股票与沪深300的 Beta 系数和滚动相关性，评估系统性风险敞口",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        ]

        def _tool_vol_regime(ticker: str, **kw):
            # 从 context 读预计算波动率
            vol_63d = self._parse_numeric(combined, ["vol_63d", "volatility_3m", "波动率"])
            garch   = self._parse_numeric(combined, ["garch_vol", "garch_volatility"])
            eff_vol = garch or vol_63d
            level = (
                "高波动 (>0.40)" if eff_vol and eff_vol > 0.40 else
                "中波动 (0.20-0.40)" if eff_vol and eff_vol > 0.20 else
                "低波动 (<0.20)" if eff_vol else "数据不足"
            )
            return {
                "ticker":          ticker,
                "vol_63d":         round(vol_63d, 4) if vol_63d else None,
                "garch_vol":       round(garch, 6) if garch else None,
                "volatility_level": level,
                "risk_flag": "⚠️ 高波动，风险溢价上升" if eff_vol and eff_vol > 0.40 else "✅ 波动率可控",
            }

        def _tool_liquidity(ticker: str, **kw):
            turnover = self._parse_numeric(combined, ["turnover_rate", "换手率", "turn"])
            volume   = self._parse_numeric(combined, ["volume_ratio", "量比"])
            amihud   = self._parse_numeric(combined, ["amihud", "amihud_illiquidity"])
            return {
                "ticker":       ticker,
                "turnover_rate": round(turnover, 2) if turnover else None,
                "volume_ratio":  round(volume, 2) if volume else None,
                "amihud":        round(amihud, 6) if amihud else None,
                "liquidity_flag": (
                    "⚠️ 流动性偏低（换手率<1%）" if turnover and turnover < 1.0 else
                    "⚠️ 流动性过热（换手率>10%）" if turnover and turnover > 10.0 else
                    "✅ 流动性正常"
                ),
            }

        def _tool_beta(ticker: str, **kw):
            beta    = self._parse_numeric(combined, ["beta", "beta_252d"])
            corr    = self._parse_numeric(combined, ["correlation_252d", "market_corr"])
            return {
                "ticker":    ticker,
                "beta":      round(beta, 3) if beta else None,
                "market_corr": round(corr, 3) if corr else None,
                "sys_risk":  (
                    "高系统性风险 (Beta>1.5)" if beta and beta > 1.5 else
                    "中等系统性风险 (0.8-1.5)" if beta and beta > 0.8 else
                    "低系统性风险 (Beta<0.8)" if beta else "Beta 数据不足"
                ),
            }

        system_prompt = """你是一位 A 股量化风险管理专家，专注于个股风险评估。

分析框架（按顺序）：
1. 波动率风险：当前波动率处于历史什么分位？GARCH 模型预测方向？（调用工具）
2. 流动性风险：换手率、成交量是否异常？能否承受正常仓位的进出？（调用工具）
3. 系统性风险：Beta 系数？在市场下跌时的敞口？（调用工具）
4. 财务风险：资产负债率、商誉风险、经营杠杆水平
5. 综合评价：风险溢价是否合理？当前时点的风险性价比

注意：
- 风险 Agent 的 signal 为负 → 看空/回避（风险过高）
- signal 为正 → 风险可控，可以正常持有
- signal 范围 [-1, 0] 区间为主

输出格式：
SIGNAL: <-1到+1，负数表示风险高，建议回避>
CONFIDENCE: <0到1>
SUMMARY: <2-3句话的中文风险评价>
KEY_RISK: <最主要的风险因子>"""

        user_message = f"""请评估 {self.ticker} 的当前风险水平。

请依次调用工具：get_volatility_regime、get_liquidity_risk、get_correlation_with_market。

行业：{self.context.get('industry', 'N/A')}
分析日期：{self.as_of}"""

        client = OllamaReActClient(model="qwen2.5:7b", timeout=60, max_steps=5)
        result = client.chat_with_tools(
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools,
            tool_executors={
                "get_volatility_regime":       _tool_vol_regime,
                "get_liquidity_risk":          _tool_liquidity,
                "get_correlation_with_market": _tool_beta,
            },
        )

        if result["fallback"]:
            return self._rule_based_analyze()

        return AgentSignal(
            agent_name      = "RiskAgent",
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
