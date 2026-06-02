"""app/services/data_service.py — QuantMind 统一数据访问层.

所有前端页面只通过 DataService 读数据，不再直接 read_parquet / json.load。
内部统一：字段对齐、缺失处理、跨表 join、实例级缓存。

设计原则
--------
- 纯 pandas，不引入 DuckDB。
- 每个方法 try/except，数据缺失返回空 DataFrame / None / {}，不抛异常崩溃页面。
- 实例级缓存（self._cache）避免重复读盘；clear_cache() 失效。
- 复用 app/utils/rec_data 的纯函数（load_all_recommendations 等），
  rec_data / sim_data 中重复的 loader 反向委托给本服务（薄包装）。

用法
----
    from app.services.data_service import get_data_service
    svc = get_data_service()
    df = svc.get_recommendations()          # 最新一天真实推荐
    a  = svc.get_agent_analysis("600519.SH") # 六维分析
    st = svc.get_model_status()             # 所有在用模型健康度
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]

# rec_data 的纯函数（不反向 import data_service，无循环依赖）
from app.utils.rec_data import (          # noqa: E402
    load_all_recommendations,
    load_forward_positions,
    load_name_map,
)


class DataService:
    """统一数据访问层（单例可由 get_data_service() 获取）。"""

    def __init__(self, root: Path | str | None = None, enable_cache: bool = True) -> None:
        self._root = Path(root) if root else _ROOT
        self._enable_cache = enable_cache
        self._cache: dict[str, Any] = {}

    # ── 缓存基础设施 ──────────────────────────────────────────────────────────

    def _cached(self, key: str, fn: Callable[[], Any]) -> Any:
        """带实例缓存的惰性求值。"""
        if self._enable_cache and key in self._cache:
            return self._cache[key]
        val = fn()
        if self._enable_cache:
            self._cache[key] = val
        return val

    def clear_cache(self) -> None:
        self._cache.clear()

    # ── 路径辅助 ──────────────────────────────────────────────────────────────

    @property
    def _rec_dir(self) -> Path:
        return self._root / "data" / "recommendations"

    @property
    def _pnl_path(self) -> Path:
        return self._root / "data" / "feedback" / "realized_pnl.parquet"

    @property
    def _fwd_path(self) -> Path:
        return self._root / "data" / "paper_trading" / "forward_positions.json"

    @property
    def _report_dir(self) -> Path:
        return self._root / "reports" / "investment_pipeline"

    # ── 内部：最新快照收盘价（当前价来源）─────────────────────────────────────

    def _latest_snapshot_close(self) -> dict[str, float]:
        """最近一个 snapshot daily_basic 的 {ticker: close}，作当前价。"""
        def _load() -> dict[str, float]:
            snap_root = self._root / "data" / "snapshots"
            if not snap_root.exists():
                return {}
            candidates = sorted(
                snap_root.glob("*/daily_basic.parquet"),
                key=lambda p: p.parent.name,
                reverse=True,
            )
            for p in candidates:
                try:
                    df = pd.read_parquet(p)
                    tcol = "ticker" if "ticker" in df.columns else "ts_code"
                    if "close" not in df.columns:
                        continue
                    return {
                        str(t): float(c)
                        for t, c in zip(df[tcol], df["close"])
                        if pd.notna(c)
                    }
                except Exception:  # noqa: BLE001
                    continue
            return {}
        return self._cached("latest_snapshot_close", _load)

    # ══════════════════════════════════════════════════════════════════════════
    # 推荐
    # ══════════════════════════════════════════════════════════════════════════

    def _all_recs(self) -> list[dict]:
        return self._cached("all_recs", lambda: load_all_recommendations(self._rec_dir))

    def get_recommendation_dates(self) -> list[str]:
        """所有有真实推荐的日期，倒序。"""
        try:
            return [r.get("as_of", "") for r in self._all_recs() if r.get("as_of")]
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] get_recommendation_dates 失败: %s", e)
            return []

    def get_recommendations(self, date: str | None = None) -> pd.DataFrame:
        """真实每日推荐，自动 join 名称/行业/入场价/当前价/浮盈/状态。

        date=None → 最新一天。
        列：ticker, name, industry, entry_price, current_price, pnl_pct,
            lgbm_rank, ensemble_score, rating, status, reason
        """
        cols = ["ticker", "name", "industry", "entry_price", "current_price",
                "pnl_pct", "lgbm_rank", "ensemble_score", "rating", "status", "reason"]
        try:
            recs = self._all_recs()
            if not recs:
                return pd.DataFrame(columns=cols)
            rec = (recs[0] if date is None
                   else next((r for r in recs if r.get("as_of") == date), None))
            if rec is None:
                return pd.DataFrame(columns=cols)

            as_of = rec.get("as_of", "")
            cur_px = self._latest_snapshot_close()
            name_map = self._cached("name_map", lambda: load_name_map(self._root))

            # 状态：OPEN 持仓 / 已结算 / —
            open_set = {
                (str(p.get("as_of")), str(p.get("ticker")))
                for p in self._forward_positions_raw()
                if p.get("status") == "OPEN"
            }
            pnl = self.get_realized_pnl()
            settled_set = set()
            if not pnl.empty:
                settled_set = {
                    (str(a), str(t)) for a, t in
                    zip(pnl.get("as_of_date", []), pnl.get("ticker", []))
                }

            rows = []
            for it in rec.get("top10", []):
                tk = str(it.get("ticker", ""))
                entry = it.get("entry_price")
                cur = cur_px.get(tk)
                pnl_pct = (
                    (cur - entry) / entry
                    if (isinstance(entry, (int, float)) and entry
                        and isinstance(cur, (int, float)))
                    else None
                )
                key = (as_of, tk)
                if key in open_set:
                    status = "持仓中"
                elif key in settled_set:
                    status = "已结算"
                else:
                    status = "—"
                rows.append({
                    "ticker":        tk,
                    "name":          it.get("name") or name_map.get(tk, tk),
                    "industry":      it.get("industry", "") or "—",
                    "entry_price":   entry,
                    "current_price": cur,
                    "pnl_pct":       round(pnl_pct, 4) if pnl_pct is not None else None,
                    "lgbm_rank":     it.get("lgbm_rank"),
                    "ensemble_score": it.get("lgbm_score"),
                    "rating":        it.get("agent_recommendation", ""),
                    "status":        status,
                    "reason":        it.get("reason", ""),
                })
            return pd.DataFrame(rows, columns=cols)
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] get_recommendations 失败: %s", e)
            return pd.DataFrame(columns=cols)

    # ══════════════════════════════════════════════════════════════════════════
    # 6-Agent 六维分析
    # ══════════════════════════════════════════════════════════════════════════

    def _latest_report_date(self) -> str | None:
        try:
            if not self._report_dir.exists():
                return None
            dates = sorted(
                (p.name for p in self._report_dir.iterdir()
                 if p.is_dir() and (p / "strategies.json").exists()),
                reverse=True,
            )
            return dates[0] if dates else None
        except Exception:  # noqa: BLE001
            return None

    def _load_strategies(self, date: str | None) -> list[dict]:
        d = date or self._latest_report_date()
        if not d:
            return []
        def _load() -> list[dict]:
            p = self._report_dir / d / "strategies.json"
            if not p.exists():
                return []
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
            except Exception:  # noqa: BLE001
                return []
        return self._cached(f"strategies::{d}", _load)

    @staticmethod
    def _shape_strategy(s: dict) -> dict:
        """把 strategies.json 单条转为标准六维结构。"""
        raw_agents = s.get("agent_signals", {}) or {}
        agents: dict[str, dict] = {}
        for full_name, sig in raw_agents.items():
            short = full_name.replace("Agent", "")  # ValuationAgent → Valuation
            # 两种格式：
            #   rich（旧）  {Agent: {signal, confidence, summary}}
            #   flat（近期）{Agent: float_signal}
            if isinstance(sig, dict):
                agents[short] = {
                    "signal":     sig.get("signal"),
                    "confidence": sig.get("confidence"),
                    "summary":    sig.get("summary", ""),
                }
            elif isinstance(sig, (int, float)):
                agents[short] = {"signal": float(sig), "confidence": None, "summary": ""}
        return {
            "ticker":           s.get("ticker"),
            "rating":           s.get("rating"),
            "composite_signal": s.get("composite_signal"),
            "confidence":       s.get("confidence"),
            "target_price_1m":  s.get("target_price_1m"),
            "target_price_3m":  s.get("target_price_3m"),
            "stop_loss_price":  s.get("stop_loss_price"),
            "position_size":    s.get("position_size"),
            "holding_horizon":  s.get("holding_horizon"),
            "investment_thesis": s.get("investment_thesis", ""),
            "key_risks":        s.get("key_risks", []),
            "key_catalysts":    s.get("key_catalysts", []),
            "agents":           agents,
        }

    def get_agent_analysis(self, ticker: str, date: str | None = None) -> dict | None:
        """某只股票的六维分析；找不到返回 None。"""
        try:
            strategies = self._load_strategies(date)
            for s in strategies:
                if str(s.get("ticker")) == str(ticker):
                    return self._shape_strategy(s)
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] get_agent_analysis 失败: %s", e)
            return None

    def get_all_agent_analysis(self, date: str | None = None) -> dict[str, dict]:
        """某天所有股票的六维分析，{ticker: analysis}。"""
        try:
            return {
                str(s.get("ticker")): self._shape_strategy(s)
                for s in self._load_strategies(date)
                if s.get("ticker")
            }
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] get_all_agent_analysis 失败: %s", e)
            return {}

    def find_agent_analysis(self, ticker: str) -> tuple[str | None, dict | None]:
        """跨所有报告日期查找某股票最近一次六维分析。

        Returns (report_date, analysis) 或 (None, None)。
        """
        try:
            if not self._report_dir.exists():
                return None, None
            dates = sorted(
                (p.name for p in self._report_dir.iterdir()
                 if p.is_dir() and (p / "strategies.json").exists()),
                reverse=True,
            )
            for d in dates:
                a = self.get_agent_analysis(ticker, d)
                if a is not None:
                    return d, a
            return None, None
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] find_agent_analysis 失败: %s", e)
            return None, None

    def _latest_snapshot_date(self) -> str | None:
        snap_root = self._root / "data" / "snapshots"
        if not snap_root.exists():
            return None
        dates = sorted(
            (p.name for p in snap_root.iterdir()
             if p.is_dir() and (p / "daily_basic.parquet").exists()),
            reverse=True,
        )
        return dates[0] if dates else None

    def compute_agent_analysis_live(
        self, ticker: str, as_of: str | None = None, mode: str = "fast",
    ) -> dict | None:
        """实时运行 6-Agent 辩论（fast 模式现算），返回与 get_agent_analysis 同构的 dict。

        用于股票不在当日推荐池、无落盘分析时的"调用实时分析"。失败返回 None。
        """
        try:
            import datetime as _dt

            from quantmind.agents.debate_orchestrator import DebateOrchestrator
            from quantmind.watchlist.daily_scorer import WatchlistDailyScorer

            as_of = as_of or self._latest_snapshot_date() or _dt.date.today().isoformat()
            scorer = WatchlistDailyScorer()
            factor_info = scorer._get_factor_scores(ticker, as_of)
            raw_factors = factor_info.get("raw_factors", {})

            context: dict[str, Any] = {
                "ticker": ticker, "as_of": as_of, "news": [], "reports": [],
                "snapshot": raw_factors, "lgbm_score": 0.0,
                "composite_score": 0.0, "regime": "neutral",
            }
            for k, v in raw_factors.items():
                context[f"snapshot_{k}"] = v

            regime = self._regime_snapshot().get("current_regime") or "neutral"
            orch = DebateOrchestrator(
                ticker=ticker, as_of=as_of, context=context,
                regime=regime, agent_mode=mode, timeout=30.0,
            )
            dr = orch.run_debate()

            agents: dict[str, dict] = {}
            for s in getattr(dr, "stances", []):
                short = s.agent_name.replace("Agent", "")
                agents[short] = {
                    "signal":     float(getattr(s, "signal_raw", 0.0)),
                    "confidence": float(getattr(s, "confidence", 0.0)),
                    "summary":    getattr(s, "argument", ""),
                }
            return {
                "ticker":           ticker,
                "rating":           dr.recommendation,
                "composite_signal": float(getattr(dr, "avg_signal", 0.0)),
                "confidence":       float(dr.final_confidence),
                "holding_horizon":  dr.holding_period,
                "target_price_1m":  None, "target_price_3m": None,
                "stop_loss_price":  None, "position_size": None,
                "investment_thesis": dr.debate_summary,
                "key_risks":        [dr.key_debate_point] if dr.key_debate_point else [],
                "key_catalysts":    [s.argument for s in getattr(dr, "stances", [])
                                     if getattr(s, "stance", "") == "bull"],
                "agents":           agents,
                "live":             True,
            }
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] compute_agent_analysis_live 失败: %s", e)
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # PnL / 持仓（消除重复 loader）
    # ══════════════════════════════════════════════════════════════════════════

    def get_realized_pnl(self) -> pd.DataFrame:
        """统一的 realized_pnl 读取（替换 sim_data / rec_data 两份）。"""
        def _load() -> pd.DataFrame:
            if not self._pnl_path.exists():
                return pd.DataFrame()
            try:
                df = pd.read_parquet(self._pnl_path)
                for col in ("as_of_date", "entry_date", "exit_date"):
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
                return df
            except Exception as e:  # noqa: BLE001
                log.warning("[DataService] realized_pnl 读取失败: %s", e)
                return pd.DataFrame()
        return self._cached("realized_pnl", _load)

    def _forward_positions_raw(self) -> list[dict]:
        return self._cached("fwd_raw", lambda: load_forward_positions(self._fwd_path))

    def get_forward_positions(self) -> pd.DataFrame:
        """进行中持仓，含 entry_price/current_price/pnl_pct。"""
        cols = ["as_of", "ticker", "name", "predicted_rank", "holding_period",
                "estimated_exit_date", "entry_price", "current_price",
                "pnl_pct", "status"]
        try:
            positions = self._forward_positions_raw()
            if not positions:
                return pd.DataFrame(columns=cols)
            cur_px = self._latest_snapshot_close()
            name_map = self._cached("name_map", lambda: load_name_map(self._root))
            rows = []
            for p in positions:
                tk = str(p.get("ticker", ""))
                entry = p.get("entry_price")
                cur = cur_px.get(tk)
                pnl_pct = (
                    (cur - entry) / entry
                    if (isinstance(entry, (int, float)) and entry
                        and isinstance(cur, (int, float)))
                    else None
                )
                rows.append({
                    "as_of":              p.get("as_of"),
                    "ticker":             tk,
                    "name":               p.get("name") or name_map.get(tk, tk),
                    "predicted_rank":     p.get("predicted_rank"),
                    "holding_period":     p.get("holding_period"),
                    "estimated_exit_date": p.get("estimated_exit_date"),
                    "entry_price":        entry,
                    "current_price":      cur,
                    "pnl_pct":            round(pnl_pct, 4) if pnl_pct is not None else None,
                    "status":             p.get("status", ""),
                })
            return pd.DataFrame(rows, columns=cols)
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] get_forward_positions 失败: %s", e)
            return pd.DataFrame(columns=cols)

    # ══════════════════════════════════════════════════════════════════════════
    # 模型状态（聚合散落三个页面的探测）
    # ══════════════════════════════════════════════════════════════════════════

    def _probe_lgbm(self, rel_path: str) -> dict:
        p = self._root / rel_path
        if not p.exists():
            return {"exists": False}
        try:
            with open(p, "rb") as f:
                m = pickle.load(f)
            feats = getattr(m, "_feature_names", None) or []
            return {
                "exists":     True,
                "direction":  getattr(m, "direction", None),
                "ic_mean":    getattr(m, "ic_mean", None),
                "n_features": len(feats),
                "trained_at": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        except Exception as e:  # noqa: BLE001
            return {"exists": True, "error": str(e)}

    def get_model_status(self) -> dict:
        """所有在用模型的健康状态，一次返回。"""
        def _load() -> dict:
            status: dict[str, Any] = {}
            # LGBM 四个
            for key, rel in (
                ("lgbm_main",  "models/lgbm_v6_main.pkl"),
                ("lgbm_gem",   "models/lgbm_v6_gem.pkl"),
                ("lgbm_star",  "models/lgbm_v6_star.pkl"),
                ("lgbm_alpha", "models/lgbm_v6_alpha.pkl"),
            ):
                status[key] = self._probe_lgbm(rel)

            # FactorCNN
            cnn_p = self._root / "models" / "factor_cnn_v2_augmented.pkl"
            status["factor_cnn"] = (
                {"exists": True,
                 "trained_at": datetime.fromtimestamp(cnn_p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
                if cnn_p.exists() else {"exists": False}
            )

            # 板块路由
            try:
                from quantmind.models.board_router import BoardModelRouter
                rtr = BoardModelRouter()
                routing = {}
                for board, info in rtr.get_routing_status().items():
                    routing[board] = {
                        "is_fallback": info.get("is_fallback"),
                        "direction":   info.get("direction"),
                        "reason":      info.get("reason", ""),
                    }
                status["board_router"] = routing
            except Exception as e:  # noqa: BLE001
                status["board_router"] = {"error": str(e)}

            # HMM regime（从 regime_history 末行，避免重新 fit）
            status["hmm_regime"] = self._regime_snapshot()

            # meta-learner（取最新版本 meta json）
            status["meta_learner"] = self._meta_learner_status()

            return status
        return self._cached("model_status", _load)

    def _regime_snapshot(self) -> dict:
        try:
            p = self._root / "data" / "regime" / "regime_history.parquet"
            if not p.exists():
                return {}
            df = pd.read_parquet(p)
            if df.empty:
                return {}
            last = df.iloc[-1]
            return {
                "current_regime": str(last.get("regime", "")),
                "bull_prob":      float(last.get("bull_prob", 0.0)),
                "neutral_prob":   float(last.get("neutral_prob", 0.0)),
                "bear_prob":      float(last.get("bear_prob", 0.0)),
                "as_of":          str(last.get("date", "")),
            }
        except Exception:  # noqa: BLE001
            return {}

    def _meta_learner_status(self) -> dict:
        try:
            mdir = self._root / "data" / "meta_learner"
            metas = sorted(mdir.glob("meta_learner_v*.meta.json"), reverse=True)
            if not metas:
                return {}
            d = json.loads(metas[0].read_text(encoding="utf-8"))
            return {
                "version":    d.get("version"),
                "cv_auc":     d.get("cv_auc"),
                "train_auc":  d.get("train_auc"),
                "n_samples":  d.get("n_samples"),
                "trained_at": d.get("trained_at"),
            }
        except Exception:  # noqa: BLE001
            return {}

    # ══════════════════════════════════════════════════════════════════════════
    # Regime
    # ══════════════════════════════════════════════════════════════════════════

    def get_regime(self) -> dict:
        """当前 regime + 历史曲线 + 动态权重。"""
        def _load() -> dict:
            out: dict[str, Any] = {"current": self._regime_snapshot()}
            # 历史曲线
            try:
                p = self._root / "data" / "regime" / "regime_history.parquet"
                out["history"] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
            except Exception:  # noqa: BLE001
                out["history"] = pd.DataFrame()
            # 动态权重
            try:
                from quantmind.regime import DynamicWeightManager
                cfg = self._root / "data" / "paper_trading" / "strategy_config_v2.json"
                mgr = DynamicWeightManager(config_path=cfg if cfg.exists() else None)
                regime = out["current"].get("current_regime") or "neutral"
                out["weights"] = mgr.get_weights(regime)
            except Exception:  # noqa: BLE001
                out["weights"] = {}
            return out
        return self._cached("regime", _load)

    # ══════════════════════════════════════════════════════════════════════════
    # 损失信号（周一 cron 在跑，前端没接）
    # ══════════════════════════════════════════════════════════════════════════

    def get_loss_signals(self) -> dict:
        """读 loss_signals_v4/{latest,factor_health,action_plan}.json。"""
        def _load() -> dict:
            base = self._root / "data" / "loss_signals_v4"
            out: dict[str, Any] = {}
            for key, fname in (
                ("latest",       "latest.json"),
                ("factor_health", "factor_health.json"),
                ("action_plan",  "action_plan.json"),
            ):
                p = base / fname
                try:
                    out[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
                except Exception:  # noqa: BLE001
                    out[key] = {}
            return out
        return self._cached("loss_signals", _load)


# ── 模块级单例 ────────────────────────────────────────────────────────────────

_DEFAULT_SERVICE: DataService | None = None


def get_data_service() -> DataService:
    """返回进程级共享的 DataService 单例。"""
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = DataService()
    return _DEFAULT_SERVICE
