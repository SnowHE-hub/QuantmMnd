"""quantmind.agents.investment_agents.momentum_agent — 动量分析 Agent.

版本演化：
  rules_v1:     MA趋势 + RSI 规则（默认）
  lgbm_v2:      LightGBM 动量因子模型
  lstm_v3:      序列 LSTM 涨跌概率
  patchtst_v4:  PatchTST Transformer（1374只 × 7年训练）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from quantmind.agents.investment_agents.base_agent import AgentSignal, BaseInvestmentAgent
from quantmind.models.momentum_lstm import (
    MomentumLSTM,
    _zscore_window,
    build_feature_matrix_for_ticker,
)

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PRICE_FILE = _ROOT / "data" / "prices" / "csi300_daily_adj_close.parquet"
_OHLCV_FILE = _ROOT / "data" / "prices" / "csi300_daily_ohlcv.parquet"
# Alpha 宇宙长表（含 OHLCV，优先级高于旧 CSI300 专属文件）
_LONG_PANEL = _ROOT / "data" / "raw" / "alpha_prices_panel.parquet"
_LONG_PANEL_FALLBACK = _ROOT / "data" / "raw" / "daily_prices_panel.parquet"
_LGBM_V2_PATH = _ROOT / "models" / "agents" / "momentum_lgbm_v2.pkl"

# 动量 LGBM 使用的特征列（与 monthly_train.parquet 对齐）
_MOMENTUM_FEATURES = [
    "momentum_1m", "momentum_3m", "momentum_6m",
    "vol_1m", "vol_3m", "reversal_1w",
    "amihud_1m", "beta_60d", "relative_strength_vs_csi300_60d",
]


def _rsi(series: pd.Series, period: int = 14) -> float:
    """计算最新的 RSI 值（手动实现，不依赖 pandas_ta）."""
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class MomentumAgent(BaseInvestmentAgent):
    """动量分析 Agent — 从日线价格数据计算动量指标.

    自动根据注册表中的激活版本选择分析方法：
    - rules_v1: MA趋势 + RSI 规则
    - lgbm_v2: LightGBM 动量因子打分
    - lstm_v3: LSTM 序列概率（不确定区间回落到规则）
    """

    @classmethod
    def REGISTRY(cls):
        """当前进程的 AgentModelRegistry（与 BaseInvestmentAgent 懒加载一致）."""
        reg = cls._get_registry()
        if reg is None:
            raise RuntimeError("AgentModelRegistry unavailable")
        return reg

    def analyze(self, mode: str = "fast") -> AgentSignal:
        """动量分析主入口.

        mode='auto'/'full' → LLM ReAct；mode='fast' → 原规则/LGBM/LSTM。
        """
        use_llm = (mode == "full") or (mode == "auto" and self._ollama_available())
        if use_llm:
            try:
                return self._analyze_with_llm()
            except Exception as e:
                logger.warning("[MomentumAgent] LLM 失败，降级规则: %s", e)

        reg = self._get_registry()
        active_rec = reg.get_active(self.__class__.__name__) if reg else self._model_record
        rec = active_rec or self._model_record

        if rec is None:
            return self._analyze_rules()

        if rec.model_type == "dl" and self._ml_model is not None:
            if rec.model_version == "patchtst_v4":
                return self._analyze_patchtst()
            return self._analyze_lstm()
        if rec.model_type == "ml" and self._ml_model is not None:
            return self._analyze_lgbm()
        return self._analyze_rules()

    # ── rules_v1：规则分析 ────────────────────────────────────────────────────

    def _analyze_rules(self) -> AgentSignal:
        evidence: dict = {}
        warnings: list[str] = []

        prices = self._load_price_series()
        if prices is None or len(prices) < 65:
            return AgentSignal(
                agent_name="MomentumAgent",
                ticker=self.ticker,
                signal=0.0,
                confidence=0.0,
                summary="价格数据不足，无法计算动量指标",
                evidence={"error": "price data unavailable or insufficient"},
                warnings=["ticker 不在价格面板中或数据不足"],
            )

        ret_5d = float(prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) >= 6 else None
        ret_21d = float(prices.iloc[-1] / prices.iloc[-22] - 1) if len(prices) >= 22 else None
        ret_63d = float(prices.iloc[-1] / prices.iloc[-64] - 1) if len(prices) >= 64 else None

        daily_returns = prices.pct_change().dropna()
        vol_21d = (
            float(daily_returns.iloc[-21:].std() * np.sqrt(252))
            if len(daily_returns) >= 21
            else None
        )
        rsi_14 = _rsi(prices.iloc[-30:]) if len(prices) >= 28 else None

        evidence.update({
            "ret_5d": round(ret_5d, 4) if ret_5d is not None else None,
            "ret_21d": round(ret_21d, 4) if ret_21d is not None else None,
            "ret_63d": round(ret_63d, 4) if ret_63d is not None else None,
            "vol_21d_annualized": round(vol_21d, 4) if vol_21d is not None else None,
            "rsi_14": round(rsi_14, 1) if rsi_14 is not None else None,
            "latest_price": round(float(prices.iloc[-1]), 2),
            "model_version": "rules_v1",
        })

        signal = 0.0
        if ret_21d is not None and ret_63d is not None:
            if ret_21d > 0.05 and ret_63d > 0.10:
                signal += 0.8
            elif ret_21d < -0.10:
                signal -= 0.5
            elif ret_21d > 0.02:
                signal += 0.2
            elif ret_21d < -0.05:
                signal -= 0.2

        if vol_21d is not None and vol_21d > 0.30 and (ret_21d is not None and ret_21d < -0.10):
            signal -= 0.3
            warnings.append(f"高波动({vol_21d:.0%})叠加下跌({ret_21d:.1%})，动量风险较大")

        if ret_5d is not None and ret_5d > 0.05:
            signal -= 0.3
            warnings.append(f"近5日涨幅{ret_5d:.1%}，警惕追高风险")

        if rsi_14 is not None:
            if rsi_14 > 75:
                signal -= 0.2
                warnings.append(f"RSI={rsi_14:.1f}，超买区域")
            elif rsi_14 < 30:
                signal += 0.2
                evidence["rsi_note"] = "超卖区域，或有反转机会"

        signal = self._clamp(signal)
        non_none = sum(v is not None for v in [ret_5d, ret_21d, ret_63d, vol_21d, rsi_14])
        confidence = min(0.9, 0.5 + non_none * 0.08)

        parts = []
        if ret_21d is not None:
            parts.append(f"21日{ret_21d:+.1%}")
        if rsi_14 is not None:
            parts.append(f"RSI={rsi_14:.0f}")
        summary = "，".join(parts) if parts else "动量数据不足"
        trend = "动量良好" if signal > 0.3 else ("动量偏弱" if signal < -0.3 else "动量中性")
        summary = f"{summary}，{trend}"

        return AgentSignal(
            agent_name="MomentumAgent",
            ticker=self.ticker,
            signal=signal,
            confidence=confidence,
            summary=summary[:50],
            evidence=evidence,
            warnings=warnings,
        )

    # ── lgbm_v2：LGBM 动量因子打分 ────────────────────────────────────────────

    def _analyze_lgbm(self) -> AgentSignal:
        """用 LGBM 模型打分，结合规则解释."""
        evidence: dict = {"model_version": "lgbm_v2"}
        warnings: list[str] = []

        features = self._build_momentum_features()
        if features is None:
            logger.debug(f"[MomentumAgent/lgbm_v2] {self.ticker} 特征构建失败，降级")
            return self._analyze_rules()

        try:
            X = np.array([[features.get(f, np.nan) for f in _MOMENTUM_FEATURES]], dtype=np.float32)
            score = float(self._ml_model.predict(X)[0])
            # score 是 forward_return_21d 的预测值，映射到 [-1, 1]
            clipped = float(np.clip(score / 0.10, -1.0, 1.0))  # 10%为满分

            evidence.update({
                "lgbm_score": round(score, 4),
                "momentum_1m": features.get("momentum_1m"),
                "momentum_3m": features.get("momentum_3m"),
                "momentum_6m": features.get("momentum_6m"),
                "vol_1m": features.get("vol_1m"),
            })

            if score > 0.05:
                summary = f"LGBM预测收益{score:+.1%}，动量积极"
                if features.get("momentum_6m", 0) > 0.15:
                    evidence["note"] = "6月动量强劲"
            elif score < -0.05:
                summary = f"LGBM预测收益{score:+.1%}，动量偏弱"
                warnings.append("LGBM预测下行")
            else:
                summary = f"LGBM预测收益{score:+.1%}，动量中性"

            confidence = min(0.9, 0.6 + abs(clipped) * 0.3)
            return AgentSignal(
                agent_name="MomentumAgent",
                ticker=self.ticker,
                signal=clipped,
                confidence=confidence,
                summary=summary[:50],
                evidence=evidence,
                warnings=warnings,
            )
        except Exception as e:
            logger.warning(f"[MomentumAgent/lgbm_v2] 预测失败: {e}，降级")
            return self._analyze_rules()

    def _analyze_lstm(self) -> AgentSignal:
        """LSTM v3：最后 60 日 × 5 特征，概率映射信号；失败则规则兜底。"""
        bundle = self._ml_model
        if not isinstance(bundle, dict) or "state_dict" not in bundle:
            logger.debug("[MomentumAgent/lstm_v3] bundle 无效，降级规则")
            return self._analyze_rules()

        cfg = bundle.get("config") or {}
        seq_len = int(cfg.get("seq_len", 60))
        try:
            close_full = self._adj_close_series()
            if close_full is None or len(close_full) < seq_len + 5:
                return self._analyze_rules()

            cal = close_full.index
            ohl = self._build_ohlc_on_calendar(cal)
            if ohl is None:
                return self._analyze_rules()

            feats = build_feature_matrix_for_ticker(close_full, ohl, cal=cal)
            if feats is None or len(feats) < seq_len:
                return self._analyze_rules()
            win = feats[-seq_len:].astype(np.float64)
            if np.isnan(win).any():
                return self._analyze_rules()
            win_z = _zscore_window(win).astype(np.float32)

            model = MomentumLSTM(
                input_size=int(cfg.get("input_size", 5)),
                hidden_size=int(cfg.get("hidden_size", 64)),
                num_layers=int(cfg.get("num_layers", 2)),
                output_size=1,
                dropout=float(cfg.get("dropout", 0.3)),
            )
            model.load_state_dict(bundle["state_dict"])
            model.eval()
            t_cut = float(cfg.get("prob_threshold", 0.5))
            band = 0.11
            hi = min(0.95, t_cut + band)
            lo = max(0.05, t_cut - band)
            with torch.no_grad():
                x = torch.from_numpy(win_z).unsqueeze(0)
                prob = float(model(x).squeeze().cpu().numpy())

            evidence: dict = {
                "model_version": "lstm_v3",
                "lstm_up_prob": round(prob, 4),
                "lstm_prob_threshold": round(t_cut, 4),
            }
            if prob > hi:
                sig = (prob - 0.5) * 4.0
                summary = f"LSTM上涨概率{prob:.0%}，动量偏多"
                confidence = min(0.9, 0.55 + abs(prob - 0.5))
            elif prob < lo:
                sig = (prob - 0.5) * 4.0
                summary = f"LSTM上涨概率{prob:.0%}，动量偏空"
                confidence = min(0.9, 0.55 + abs(prob - 0.5))
            else:
                logger.debug("[MomentumAgent/lstm_v3] 概率落在校准阈值中性带，降级规则")
                return self._analyze_rules()

            sig = self._clamp(float(sig))
            return AgentSignal(
                agent_name="MomentumAgent",
                ticker=self.ticker,
                signal=sig,
                confidence=float(confidence),
                summary=summary[:50],
                evidence=evidence,
                warnings=[],
            )
        except Exception as e:
            logger.warning(f"[MomentumAgent/lstm_v3] 推理失败: {e}，降级规则")
            return self._analyze_rules()

    # ── patchtst_v4：PatchTST Transformer ─────────────────────────────────────

    def _build_patchtst_features(self, df_ticker: pd.DataFrame, seq_len: int) -> np.ndarray | None:
        """从 alpha_prices_panel 长表单只股票 DataFrame 构建 5 特征矩阵（最后 seq_len 行）."""
        if df_ticker is None or len(df_ticker) < seq_len + 5:
            return None
        d = df_ticker.sort_values("trade_date").tail(seq_len + 40).copy()
        c = pd.to_numeric(d.get("adj_close", d.get("close")), errors="coerce")

        ret      = c.pct_change(fill_method=None).fillna(0.0).values
        ma20     = c.rolling(20, min_periods=10).mean()
        ma_ratio = (c / ma20.replace(0, np.nan)).fillna(1.0).values

        vol    = pd.to_numeric(d.get("vol", pd.Series(1.0, index=d.index)), errors="coerce").fillna(0.0)
        vol_ma = vol.rolling(20, min_periods=10).mean().fillna(1.0)
        vol_ratio = (vol / vol_ma.replace(0, 1.0)).fillna(1.0).values

        hi  = pd.to_numeric(d.get("high",  c), errors="coerce").fillna(0.0).values
        lo  = pd.to_numeric(d.get("low",   c), errors="coerce").fillna(0.0).values
        cl  = pd.to_numeric(d.get("close", c), errors="coerce").fillna(1.0).values
        op  = pd.to_numeric(d.get("open",  c), errors="coerce").fillna(0.0).values
        pre = pd.to_numeric(d.get("pre_close", pd.Series(0.0, index=d.index)), errors="coerce").fillna(0.0).values

        hl_d  = np.where(np.abs(cl) > 1e-8, cl, 1.0)
        high_low = np.nan_to_num((hi - lo) / hl_d, nan=0.0, posinf=0.0)
        gap_d = np.where(pre > 1e-8, pre, 1.0)
        gap   = np.nan_to_num(op / gap_d - 1.0, nan=0.0, posinf=0.0, neginf=0.0)

        feats = np.column_stack([ret, vol_ratio, ma_ratio, high_low, gap]).astype(np.float32)
        if len(feats) < seq_len:
            return None
        return feats[-seq_len:]

    def _analyze_patchtst(self) -> AgentSignal:
        """PatchTST v4：从 alpha_prices_panel 直接读取特征，Transformer 架构。"""
        bundle = self._ml_model
        if not isinstance(bundle, dict) or "state_dict" not in bundle:
            logger.debug("[MomentumAgent/patchtst_v4] bundle 无效，降级规则")
            return self._analyze_rules()

        cfg = bundle.get("config") or {}
        seq_len = int(cfg.get("seq_len", 64))
        try:
            from quantmind.models.momentum_patchtst import PatchTST

            # 直接从 alpha 长表读取单只股票数据
            lp = self._long_panel_ticker_df()
            if lp is None or len(lp) < seq_len + 5:
                logger.debug(f"[MomentumAgent/patchtst_v4] {self.ticker} 数据不足，降级规则")
                return self._analyze_rules()
            if self.as_of:
                lp = lp[lp["trade_date"] <= pd.Timestamp(self.as_of)]

            win = self._build_patchtst_features(lp, seq_len)
            if win is None or np.isnan(win).any():
                return self._analyze_rules()
            win_z = _zscore_window(win.astype(np.float64)).astype(np.float32)

            model = PatchTST(
                n_feats=int(cfg.get("n_feats", 5)),
                seq_len=seq_len,
                patch_len=int(cfg.get("patch_len", 16)),
                stride=int(cfg.get("stride", 8)),
                d_model=int(cfg.get("d_model", 128)),
                n_heads=int(cfg.get("n_heads", 4)),
                n_layers=int(cfg.get("n_layers", 3)),
                dropout=float(cfg.get("dropout", 0.1)),
            )
            model.load_state_dict(bundle["state_dict"])
            model.eval()

            t_cut = float(cfg.get("prob_threshold", 0.5))
            # 中性带 ±0.07（较窄，让更多预测触发 PatchTST 信号）
            band = 0.07
            hi = min(0.95, t_cut + band)
            lo = max(0.05, t_cut - band)

            with torch.no_grad():
                x = torch.from_numpy(win_z).unsqueeze(0)
                prob = float(model(x).squeeze().cpu().numpy())

            evidence: dict = {
                "model_version": "patchtst_v4",
                "patchtst_up_prob": round(prob, 4),
                "prob_threshold": round(t_cut, 4),
                "neutral_band": f"[{lo:.2f},{hi:.2f}]",
            }
            if prob > hi:
                sig = (prob - 0.5) * 4.0
                summary = f"PatchTST上涨概率{prob:.0%}，动量偏多"
                confidence = min(0.9, 0.55 + abs(prob - 0.5))
            elif prob < lo:
                sig = (prob - 0.5) * 4.0
                summary = f"PatchTST上涨概率{prob:.0%}，动量偏空"
                confidence = min(0.9, 0.55 + abs(prob - 0.5))
            else:
                logger.debug(f"[MomentumAgent/patchtst_v4] 中性带({prob:.3f})，降级规则")
                return self._analyze_rules()

            sig = self._clamp(float(sig))
            return AgentSignal(
                agent_name="MomentumAgent",
                ticker=self.ticker,
                signal=sig,
                confidence=float(confidence),
                summary=summary[:50],
                evidence=evidence,
                warnings=[],
            )
        except Exception as e:
            logger.warning(f"[MomentumAgent/patchtst_v4] 推理失败: {e}，降级规则")
            return self._analyze_rules()

    def _build_momentum_features(self) -> dict | None:
        """构建 LGBM 需要的动量特征."""
        prices = self._load_price_series()
        if prices is None or len(prices) < 65:
            return None

        try:
            daily_ret = prices.pct_change().dropna()
            return {
                "momentum_1m": float(prices.iloc[-1] / prices.iloc[-22] - 1) if len(prices) >= 22 else np.nan,
                "momentum_3m": float(prices.iloc[-1] / prices.iloc[-64] - 1) if len(prices) >= 64 else np.nan,
                "momentum_6m": float(prices.iloc[-1] / prices.iloc[-127] - 1) if len(prices) >= 127 else np.nan,
                "vol_1m": float(daily_ret.iloc[-21:].std() * np.sqrt(252)) if len(daily_ret) >= 21 else np.nan,
                "vol_3m": float(daily_ret.iloc[-63:].std() * np.sqrt(252)) if len(daily_ret) >= 63 else np.nan,
                "reversal_1w": float(prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) >= 6 else np.nan,
                "amihud_1m": np.nan,  # 需要成交量数据，暂缺
                "beta_60d": np.nan,
                "relative_strength_vs_csi300_60d": np.nan,
            }
        except Exception:
            return None

    def _long_panel_ticker_df(self) -> pd.DataFrame | None:
        """长表日线面板（alpha 宇宙 1374 只，含 OHLCV）；按 trade_date 升序."""
        for path in (_LONG_PANEL, _LONG_PANEL_FALLBACK):
            if not path.is_file():
                continue
            try:
                df = pd.read_parquet(path, filters=[("ts_code", "=", self.ticker)])
                if df.empty:
                    # 某些 parquet 不支持 pushdown，回退全量 filter
                    df = pd.read_parquet(path)
                    df = df.loc[df["ts_code"].astype(str) == self.ticker].copy()
                if df.empty:
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                return df.sort_values("trade_date")
            except Exception as e:
                logger.debug(f"[MomentumAgent] long panel 读取失败({path.name}): {e}")
        return None

    def _adj_close_series(self) -> pd.Series | None:
        """复权收盘价序列：优先 CSI300 宽表，其次 ``daily_prices_panel`` 长表。"""
        try:
            wide = pd.read_parquet(_PRICE_FILE)
            if self.ticker in wide.columns:
                s = pd.to_numeric(wide[self.ticker], errors="coerce").dropna()
                s.index = pd.to_datetime(s.index)
                s = s.sort_index()
                if self.as_of:
                    s = s[s.index <= pd.Timestamp(self.as_of)]
                return s.astype(float) if len(s) > 0 else None
        except Exception:
            pass

        lp = self._long_panel_ticker_df()
        if lp is None or lp.empty:
            logger.warning(f"[MomentumAgent] {self.ticker} 不在价格面板中")
            return None
        if self.as_of:
            lp = lp[lp["trade_date"] <= pd.Timestamp(self.as_of)]
        s = (
            lp.set_index("trade_date")["adj_close"]
            .astype(float)
            .dropna()
            .sort_index()
        )
        return s if len(s) > 0 else None

    def _build_ohlc_on_calendar(self, cal: pd.DatetimeIndex) -> pd.DataFrame | None:
        """将 OHLCV 对齐到 ``cal``；宽表无该 ticker 时用长表补齐。"""
        try:
            ohlc_long = pd.read_parquet(_OHLCV_FILE).reset_index()
            code_col = "ts_code" if "ts_code" in ohlc_long.columns else "ticker"
            sub = ohlc_long[ohlc_long[code_col] == self.ticker].copy()
            if not sub.empty:
                sub["trade_date"] = pd.to_datetime(sub["trade_date"])
                sub = sub.set_index("trade_date").sort_index()
                ohl = pd.DataFrame(index=cal)
                for f in ("open", "high", "low", "close", "volume", "pre_close"):
                    if f not in sub.columns:
                        ohl[f] = np.nan
                    else:
                        ohl[f] = sub[f].reindex(cal)
                return ohl
        except Exception:
            pass

        lp = self._long_panel_ticker_df()
        if lp is None or lp.empty:
            return None
        if self.as_of:
            lp = lp[lp["trade_date"] <= pd.Timestamp(self.as_of)]
        lp = lp.sort_values("trade_date").set_index("trade_date")
        ohl = pd.DataFrame(index=cal)
        ohl["open"] = lp["open"].reindex(cal) if "open" in lp.columns else np.nan
        ohl["high"] = lp["high"].reindex(cal) if "high" in lp.columns else np.nan
        ohl["low"] = lp["low"].reindex(cal) if "low" in lp.columns else np.nan
        ohl["close"] = lp["close"].reindex(cal) if "close" in lp.columns else np.nan
        ohl["volume"] = lp["vol"].reindex(cal) if "vol" in lp.columns else np.nan
        ohl["pre_close"] = lp["close"].shift(1).reindex(cal) if "close" in lp.columns else np.nan
        return ohl

    def _load_price_series(self) -> pd.Series | None:
        """加载 ticker 的价格序列（PIT 严格）."""
        try:
            return self._adj_close_series()
        except Exception as e:
            logger.warning(f"[MomentumAgent] 价格数据读取失败: {e}")
            return None

    # ─── LLM ReAct 路径 ────────────────────────────────────────────────────────

    def _analyze_with_llm(self) -> AgentSignal:
        """Ollama qwen2.5 ReAct 动量分析（工具调用）."""
        from quantmind.agents.ollama_client import OllamaReActClient

        tech_text = self._get_snapshot_text("snapshot_latest_market_metrics")

        tools = [
            {
                "name": "get_price_momentum_analysis",
                "description": "计算股票多周期价格动量（5/21/63/126/252日），识别趋势强度和方向",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker":  {"type": "string"},
                        "windows": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "时间窗口（交易日数）",
                        },
                    },
                    "required": ["ticker"],
                },
            },
            {
                "name": "get_north_flow_trend",
                "description": "获取北向资金最近 N 日净流入趋势，判断外资态度",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "days":   {"type": "integer", "default": 30},
                    },
                    "required": ["ticker"],
                },
            },
            {
                "name": "get_sector_rotation_context",
                "description": "获取当前市场热点板块和行业轮动方向，判断股票所在行业的资金流向",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

        def _tool_momentum(ticker: str, windows: list | None = None, **kw):
            series = self._load_price_series()
            if series is None or len(series) < 10:
                return {"error": "价格数据不可用"}
            wins = windows or [5, 21, 63, 126, 252]
            result = {}
            for w in wins:
                if len(series) > w:
                    ret = float(series.iloc[-1] / series.iloc[-w - 1] - 1)
                    result[f"ret_{w}d"] = round(ret * 100, 2)
            # RSI(14)
            if len(series) >= 15:
                result["rsi_14"] = round(_rsi(series), 1)
            return result

        def _tool_north_flow(ticker: str, days: int = 30, **kw):
            north_text = self._get_snapshot_text("snapshot_north_flow")
            return {
                "ticker": ticker,
                "days":   days,
                "note":   north_text[:300] if north_text else "北向资金数据不在 context 中",
            }

        def _tool_sector_rotation(**kw):
            regime_text = self._get_snapshot_text("snapshot_macro")
            return {
                "note": regime_text[:300] if regime_text else "当前板块轮动数据不在 context 中",
            }

        system_prompt = """你是一位 A 股技术面和动量分析专家。

分析框架（按顺序）：
1. 多周期动量：5日/21日/63日/126日/252日各期收益率 → 短中长期趋势
2. 趋势判断：短期(5-21d)与中期(63d)是否同向？是否出现反转信号？
3. RSI：是否超买(>70)或超卖(<30)？
4. 北向资金：外资最近是否持续净买入/净卖出？
5. 板块轮动：所在行业当前是否处于资金流入阶段？
6. 综合判断：结合 A股均值回归特性（短期动量负向）+ 长期趋势（正向）

注意：A 股短期（1个月内）存在均值回归，长期（6-12月）趋势有效。

输出格式：
SIGNAL: <-1到+1，正数趋势看多>
CONFIDENCE: <0到1>
SUMMARY: <2-3句话的中文动量判断>
KEY_RISK: <主要风险（如超买、北向流出等）>"""

        user_message = f"""请分析 {self.ticker} 的价格动量和趋势。

请先调用 get_price_momentum_analysis 工具获取多周期动量数据，再综合分析。
行业：{self.context.get('industry', 'N/A')}
分析日期：{self.as_of}"""

        client = OllamaReActClient(model="qwen2.5:7b", timeout=60, max_steps=5)
        result = client.chat_with_tools(
            system_prompt=system_prompt,
            user_message=user_message,
            tools=tools,
            tool_executors={
                "get_price_momentum_analysis": _tool_momentum,
                "get_north_flow_trend":        _tool_north_flow,
                "get_sector_rotation_context": _tool_sector_rotation,
            },
        )

        if result["fallback"]:
            return self._analyze_rules()

        return AgentSignal(
            agent_name      = "MomentumAgent",
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
