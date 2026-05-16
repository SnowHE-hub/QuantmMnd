"""quantmind.agents.investment_agents.valuation_agent — 估值分析 Agent（规则 + LGBM v2）."""

from __future__ import annotations

import re

import numpy as np
from loguru import logger
from scipy.stats import percentileofscore

from quantmind.agents.investment_agents.base_agent import AgentSignal, BaseInvestmentAgent


class ValuationAgent(BaseInvestmentAgent):
    """估值分析 Agent — 规则层 + 可选 LightGBM 截面估值模型."""

    def analyze(self) -> AgentSignal:
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
