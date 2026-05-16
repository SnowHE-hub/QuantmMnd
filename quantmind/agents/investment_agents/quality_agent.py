"""quantmind.agents.investment_agents.quality_agent — 财务质量分析 Agent（规则 + Piotroski F-Score）."""

from __future__ import annotations

import re

from quantmind.agents.investment_agents.base_agent import AgentSignal, BaseInvestmentAgent


class QualityAgent(BaseInvestmentAgent):
    """财务质量分析 Agent — Piotroski v2 或规则基线."""

    def analyze(self) -> AgentSignal:
        rec = self._model_record
        if rec is not None and rec.model_version == "piotroski_v2":
            return self._analyze_piotroski()

        evidence: dict = {}
        warnings: list[str] = []
        signal = 0.0
        fields_present = 0

        fi_text = self._get_snapshot_text("snapshot_financial_indicator_summary")
        market_text = self._get_snapshot_text("snapshot_latest_market_metrics")
        combined = fi_text + "\n" + market_text

        roe_ttm = self._parse_numeric(combined, ["roe_ttm", "roe"])
        grossprofit_margin = self._parse_numeric(
            combined, ["grossprofit_margin", "毛利率", "gross_profit_margin", "gross_margin"]
        )
        netprofit_margin = self._parse_numeric(
            combined, ["netprofit_margin", "净利率", "net_profit_margin"]
        )
        debt_to_assets = self._parse_numeric(combined, ["debt_to_assets", "资产负债率", "debt_ratio"])
        current_ratio = self._parse_numeric(combined, ["current_ratio", "流动比率"])
        netprofit = self._parse_numeric(combined, ["netprofit", "净利润", "net_profit"])

        evidence.update({
            "roe_ttm": roe_ttm,
            "grossprofit_margin": grossprofit_margin,
            "netprofit_margin": netprofit_margin,
            "debt_to_assets": debt_to_assets,
            "current_ratio": current_ratio,
            "method": "rules_v1",
        })

        if roe_ttm is not None:
            fields_present += 1
            roe = self._pct_to_float(roe_ttm)
            if roe is not None:
                if roe > 0.20:
                    signal += 0.6
                    evidence["roe_note"] = f"ROE={roe:.1%}，优秀盈利能力"
                elif roe > 0.15:
                    signal += 0.4
                elif roe > 0.08:
                    signal += 0.1
                elif roe < 0:
                    signal -= 0.5
                    warnings.append(f"ROE={roe:.1%}（亏损）")
                else:
                    signal -= 0.3
                    warnings.append(f"ROE={roe:.1%}，盈利能力偏弱")

        if grossprofit_margin is not None:
            fields_present += 1
            gpm = self._pct_to_float(grossprofit_margin)
            if gpm is not None:
                if gpm > 0.40:
                    signal += 0.3
                    evidence["gpm_note"] = f"毛利率={gpm:.1%}，护城河深"
                elif gpm > 0.20:
                    signal += 0.1
                elif gpm < 0.10:
                    signal -= 0.1
                    warnings.append(f"毛利率={gpm:.1%}，竞争压力大")

        if debt_to_assets is not None:
            fields_present += 1
            dta = self._pct_to_float(debt_to_assets)
            if dta is not None:
                evidence["debt_to_assets_pct"] = f"{dta:.1%}"
                if dta > 0.70:
                    signal -= 0.4
                    warnings.append(f"资产负债率={dta:.1%}，财务杠杆过高")
                elif dta > 0.50:
                    signal -= 0.1
                elif dta < 0.40:
                    signal += 0.2
                    evidence["debt_note"] = f"负债率={dta:.1%}，财务稳健"

        if netprofit is not None and netprofit < 0:
            signal -= 0.2
            warnings.append("净利润为负，亏损状态")

        if current_ratio is not None:
            fields_present += 1
            if current_ratio < 1.0:
                warnings.append(f"流动比率={current_ratio:.2f}，流动性偏低")
            elif current_ratio > 2.0:
                evidence["liquidity"] = "流动比率充足"

        signal = self._clamp(signal)

        confidence = 0.9 if fields_present >= 3 else (0.6 if fields_present >= 1 else 0.4)

        parts = []
        if roe_ttm is not None:
            roe = self._pct_to_float(roe_ttm)
            parts.append(f"ROE={roe:.1%}" if roe else f"ROE={roe_ttm}")
        if grossprofit_margin is not None:
            gpm = self._pct_to_float(grossprofit_margin)
            parts.append(f"毛利率={gpm:.1%}" if gpm else f"毛利率={grossprofit_margin}")
        if debt_to_assets is not None:
            dta = self._pct_to_float(debt_to_assets)
            parts.append(f"负债率={dta:.1%}" if dta else f"负债率={debt_to_assets}")
        quality = "高质量" if signal > 0.5 else ("中等" if signal > 0 else "质量偏弱")
        summary = "，".join(parts[:2]) + f"，财务{quality}" if parts else "财务数据不足"

        return AgentSignal(
            agent_name="QualityAgent",
            ticker=self.ticker,
            signal=signal,
            confidence=confidence,
            summary=summary[:50],
            evidence=evidence,
            warnings=warnings,
        )

    def _analyze_piotroski(self) -> AgentSignal:
        fi_text = self._get_snapshot_text("snapshot_financial_indicator_summary")
        market_text = self._get_snapshot_text("snapshot_latest_market_metrics")
        combined = fi_text + "\n" + market_text

        f_score, denom, signals = self._piotroski_score(combined)
        if denom < 3:
            # 有效信号太少（<3），无法可靠判断，回退到规则打分
            return self._analyze_fallback_rules()
        signal = self._clamp((f_score / denom) * 2.0 - 1.0)

        summary = f"Piotroski F={f_score}/{denom}，质量{'偏强' if signal > 0.2 else ('偏弱' if signal < -0.2 else '中性')}"

        return AgentSignal(
            agent_name="QualityAgent",
            ticker=self.ticker,
            signal=round(signal, 4),
            confidence=0.85 if denom >= 6 else (0.65 if denom >= 3 else 0.40),
            summary=summary[:50],
            evidence={
                "f_score": f_score,
                "f_denominator": denom,
                "signals": signals,
                "method": "piotroski_v2",
            },
            warnings=[],
        )

    def _analyze_fallback_rules(self) -> AgentSignal:
        """有效 Piotroski 信号不足 3 个时，回退到规则基线打分."""
        rec_backup = self._model_record
        self._model_record = None  # type: ignore[assignment]
        try:
            return self.analyze()
        finally:
            self._model_record = rec_backup

    def _piotroski_score(self, combined_text: str) -> tuple[int, int, dict[str, int | None]]:
        """Piotroski F-Score：缺失项跳过且不计入分母."""

        def ctx(keys: list[str]) -> float | None:
            ctx_d = self.context
            for k in keys:
                if isinstance(ctx_d, dict) and ctx_d.get(k) is not None:
                    return self._safe_float(ctx_d[k])
            return None

        def pick(keys: list[str]) -> float | None:
            v = ctx(keys)
            if v is not None:
                return v
            return self._parse_numeric(combined_text, keys)

        signals: dict[str, int | None] = {}
        pts = 0
        denom = 0

        net_profit = pick(["net_profit", "netprofit", "净利润"])
        total_assets = pick(["total_assets", "总资产"])
        roa_direct = pick(["roa"])
        roa = None
        if roa_direct is not None:
            roa = float(roa_direct)
        elif net_profit is not None and total_assets not in (None, 0):
            roa = float(net_profit) / float(total_assets)

        roa_lfy = pick(["roa_lfy", "roa_last_year"])
        roa_yoy_metric = pick(["roa_yoy"])

        cfo = pick(["cfo", "n_cashflow_act", "经营活动现金流", "经营现金流量净额"])
        cfo_ratio = None
        if cfo is not None and total_assets not in (None, 0):
            cfo_ratio = float(cfo) / float(total_assets)

        roe_v = pick(["roe", "roe_ttm"])

        # F1
        if roa is not None:
            denom += 1
            v = 1 if roa > 0 else 0
            signals["F1"] = v
            pts += v

        # F2：cfo_ratio>0；若无 CFO 则用 ROE>0 近似
        if cfo_ratio is not None:
            denom += 1
            v = 1 if cfo_ratio > 0 else 0
            signals["F2"] = v
            pts += v
        else:
            if roe_v is not None:
                roe_f = self._pct_to_float(roe_v)
                if roe_f is None:
                    roe_f = float(roe_v)
                denom += 1
                v = 1 if roe_f > 0 else 0
                signals["F2"] = v
                pts += v
            else:
                signals["F2"] = None

        # F3 ROA YoY
        if roa is not None:
            roa_prev = None
            if roa_lfy is not None:
                roa_prev = float(roa_lfy)
            elif roa_yoy_metric is not None:
                # 若直接给出同比变动（小数），则推算上期
                dy = float(roa_yoy_metric)
                roa_prev = float(roa) - dy
            if roa_prev is not None:
                denom += 1
                v = 1 if float(roa) > float(roa_prev) else 0
                signals["F3"] = v
                pts += v
            else:
                signals["F3"] = None
        else:
            signals["F3"] = None

        # F4 accrual < 0（ROA - CFO/TA）；无 CFO 跳过
        if roa is not None and cfo_ratio is not None:
            denom += 1
            accrual = float(roa) - float(cfo_ratio)
            v = 1 if accrual < 0 else 0
            signals["F4"] = v
            pts += v
        else:
            signals["F4"] = None

        ltd_ratio = pick(["long_term_debt_to_assets", "ltd_to_assets", "长期负债比"])
        ltd_ratio_lfy = pick(["long_term_debt_to_assets_lfy", "ltd_to_assets_lfy"])
        leverage_yoy = pick(["leverage_yoy", "ltd_ratio_yoy"])

        # F5 leverage YoY < 0（长期杠杆下降）
        cur_lev = ltd_ratio
        if cur_lev is not None:
            prev_lev = None
            if ltd_ratio_lfy is not None:
                prev_lev = float(ltd_ratio_lfy)
            elif leverage_yoy is not None:
                prev_lev = float(cur_lev) - float(leverage_yoy)
            if prev_lev is not None:
                denom += 1
                v = 1 if float(cur_lev) < float(prev_lev) else 0
                signals["F5"] = v
                pts += v
            else:
                signals["F5"] = None
        else:
            signals["F5"] = None

        cur_ratio = pick(["current_ratio", "流动比率"])
        cur_ratio_lfy = pick(["current_ratio_lfy"])
        cur_ratio_yoy = pick(["current_ratio_yoy"])

        # F6 流动比率同比上升
        if cur_ratio is not None:
            prev_cr = None
            if cur_ratio_lfy is not None:
                prev_cr = float(cur_ratio_lfy)
            elif cur_ratio_yoy is not None:
                prev_cr = float(cur_ratio) - float(cur_ratio_yoy)
            if prev_cr is not None:
                denom += 1
                v = 1 if float(cur_ratio) > float(prev_cr) else 0
                signals["F6"] = v
                pts += v
            else:
                signals["F6"] = None
        else:
            signals["F6"] = None

        shares_yoy = pick(["shares_yoy", "total_share_yoy", "股本同比"])

        # F7 股本未稀释
        if shares_yoy is not None:
            denom += 1
            v = 1 if float(shares_yoy) <= 0 else 0
            signals["F7"] = v
            pts += v
        else:
            signals["F7"] = None

        gm = pick(["gross_margin", "grossprofit_margin", "毛利率"])
        gm_lfy = pick(["gross_margin_lfy", "grossprofit_margin_lfy"])
        gm_yoy = pick(["gross_margin_yoy"])

        # F8 毛利率同比改善
        if gm is not None:
            gm_f = self._pct_to_float(gm)
            if gm_f is None:
                gm_f = float(gm)
            prev_gm = None
            if gm_lfy is not None:
                pg = self._pct_to_float(gm_lfy)
                prev_gm = float(pg) if pg is not None else float(gm_lfy)
            elif gm_yoy is not None:
                gy = self._pct_to_float(gm_yoy)
                delta = float(gy) if gy is not None else float(gm_yoy)
                prev_gm = float(gm_f) - delta
            if prev_gm is not None:
                denom += 1
                v = 1 if float(gm_f) > float(prev_gm) else 0
                signals["F8"] = v
                pts += v
            else:
                signals["F8"] = None
        else:
            signals["F8"] = None

        ato = pick(["asset_turnover", "总资产周转率"])
        ato_lfy = pick(["asset_turnover_lfy"])
        ato_yoy = pick(["asset_turnover_yoy"])

        # F9 周转率同比上升
        if ato is not None:
            prev_ato = None
            if ato_lfy is not None:
                prev_ato = float(ato_lfy)
            elif ato_yoy is not None:
                prev_ato = float(ato) - float(ato_yoy)
            if prev_ato is not None:
                denom += 1
                v = 1 if float(ato) > float(prev_ato) else 0
                signals["F9"] = v
                pts += v
            else:
                signals["F9"] = None
        else:
            signals["F9"] = None

        return pts, denom, signals

    def _parse_numeric(self, text: str, keys: list[str]) -> float | None:
        for key in keys:
            pattern = rf"(?i){re.escape(key)}\s*[:：=]\s*(-?[\d.]+)"
            m = re.search(pattern, text)
            if m:
                return self._safe_float(m.group(1))
        return None
