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
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]

# 加载 .env（幂等）
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

# rec_data 的纯函数（不反向 import data_service，无循环依赖）
from app.utils.rec_data import (          # noqa: E402
    load_all_recommendations,
    load_forward_positions,
    load_name_map,
)


class DataService:
    """统一数据访问层（单例可由 get_data_service() 获取）。

    DATA_BACKEND 环境变量控制存储后端：
      'parquet'  → 原有文件系统行为（默认，零回归）
      'postgres' → PostgreSQL + MongoDB
    """

    def __init__(
        self,
        root: Path | str | None = None,
        enable_cache: bool = True,
        backend: str | None = None,
    ) -> None:
        self._root = Path(root) if root else _ROOT
        self._enable_cache = enable_cache
        self._backend = backend or os.environ.get("DATA_BACKEND", "parquet")
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
        if self._backend == "postgres":
            return self._mongo_get_recommendation_dates()
        try:
            return [r.get("as_of", "") for r in self._all_recs() if r.get("as_of")]
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] get_recommendation_dates 失败: %s", e)
            return []

    def get_recommendations(self, date: str | None = None) -> pd.DataFrame:  # type: ignore[override]
        if self._backend == "postgres":
            return self._mongo_get_recommendations(date)
        return self._parquet_get_recommendations(date)

    def _parquet_get_recommendations(self, date: str | None = None) -> pd.DataFrame:
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
        if self._backend == "postgres":
            return self._mongo_get_agent_analysis(ticker, date)
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
        if self._backend == "postgres":
            return self._mongo_get_all_agent_analysis(date)
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
        if self._backend == "postgres":
            return self._mongo_find_agent_analysis(ticker)
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
        if self._backend == "postgres":
            return self._cached("realized_pnl_pg", self._pg_get_realized_pnl)
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
        if self._backend == "postgres":
            return self._cached("fwd_raw_mongo", self._mongo_forward_positions_raw)
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

            # FactorCNN（含验证集 IC）
            cnn_p = self._root / "models" / "factor_cnn_v2_augmented.pkl"
            if cnn_p.exists():
                cnn_info: dict[str, Any] = {
                    "exists": True,
                    "trained_at": datetime.fromtimestamp(cnn_p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                }
                try:
                    with open(cnn_p, "rb") as f:
                        cnn_obj = pickle.load(f)
                    if isinstance(cnn_obj, dict):
                        cnn_info["val_ic"] = cnn_obj.get("val_ic_mean")
                        cnn_info["val_icir"] = cnn_obj.get("val_icir")
                except Exception:  # noqa: BLE001
                    pass
                status["factor_cnn"] = cnn_info
            else:
                status["factor_cnn"] = {"exists": False}

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
        if self._backend == "postgres":
            return self._cached("loss_signals_mongo", self._mongo_get_loss_signals)
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

    # ══════════════════════════════════════════════════════════════════════════
    # E3 执行层（simulated_orders）
    # ══════════════════════════════════════════════════════════════════════════

    def get_simulated_orders(
        self,
        status: str = "all",      # 'all' / 'OPEN' / 'CLOSED'
        days: int | None = None,  # 限制最近 N 天的订单（OPEN 看 open_date, CLOSED 看 close_date）
    ) -> pd.DataFrame:
        """读取 simulated_orders 表。"""
        try:
            from sqlalchemy import text
            from app.db.postgres import get_pg_engine

            where = []
            params: dict[str, Any] = {}
            if status != "all":
                where.append("status = :status")
                params["status"] = status
            if days is not None:
                from datetime import date, timedelta
                cutoff = date.today() - timedelta(days=days)
                if status == "CLOSED":
                    where.append("close_date >= :cutoff")
                else:
                    where.append("open_date >= :cutoff")
                params["cutoff"] = cutoff
            where_sql = "WHERE " + " AND ".join(where) if where else ""
            order_sql = "ORDER BY COALESCE(close_date, open_date) DESC, order_id DESC"
            sql = f"SELECT * FROM simulated_orders {where_sql} {order_sql}"

            with get_pg_engine().connect() as conn:
                df = pd.read_sql(text(sql), conn, params=params)
            # 整表为空（无过滤条件）才标记为"后端未填充"，避免把
            # "OPEN 状态恰好没有订单"误判成 PG 空表。
            if df.empty and status == "all" and days is None:
                df = self._mark_pg_empty(df, "simulated_orders")
            return df
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] get_simulated_orders 失败: %s", e)
            return pd.DataFrame()

    def get_execution_stats(self, days: int = 90) -> dict:
        """执行层综合统计。"""
        try:
            df_all = self.get_simulated_orders(status="all")
            if df_all.empty:
                return {
                    "total_orders": 0, "open_orders": 0, "closed_orders": 0,
                    "win_rate": None, "avg_return": None,
                    "exit_reasons": {}, "best_trade": None, "worst_trade": None,
                    "avg_holding_days": None,
                }

            open_df = df_all[df_all["status"] == "OPEN"]
            closed_df = df_all[df_all["status"] == "CLOSED"].copy()

            # 限制 closed 到 N 天
            if not closed_df.empty and days:
                from datetime import date, timedelta
                cutoff = pd.Timestamp(date.today() - timedelta(days=days)).date()
                closed_df["close_date"] = pd.to_datetime(closed_df["close_date"]).dt.date
                closed_df = closed_df[closed_df["close_date"] >= cutoff]

            stats: dict[str, Any] = {
                "total_orders":  int(len(df_all)),
                "open_orders":   int(len(open_df)),
                "closed_orders": int(len(closed_df)),
                "win_rate":      None,
                "avg_return":    None,
                "exit_reasons":  {},
                "best_trade":    None,
                "worst_trade":   None,
                "avg_holding_days": None,
            }

            if not closed_df.empty:
                wins = (closed_df["pnl_pct"] > 0).sum()
                stats["win_rate"] = float(wins) / len(closed_df)
                stats["avg_return"] = float(closed_df["pnl_pct"].mean())
                stats["avg_holding_days"] = (
                    float(closed_df["holding_days"].mean())
                    if "holding_days" in closed_df.columns else None
                )
                stats["exit_reasons"] = closed_df["close_reason"].value_counts().to_dict()
                # best / worst
                best = closed_df.loc[closed_df["pnl_pct"].idxmax()]
                worst = closed_df.loc[closed_df["pnl_pct"].idxmin()]
                stats["best_trade"] = {
                    "ticker": str(best["ticker"]), "name": best.get("name", ""),
                    "pnl_pct": float(best["pnl_pct"]),
                    "open_date": str(best["open_date"]),
                    "close_date": str(best["close_date"]),
                    "close_reason": best.get("close_reason", ""),
                }
                stats["worst_trade"] = {
                    "ticker": str(worst["ticker"]), "name": worst.get("name", ""),
                    "pnl_pct": float(worst["pnl_pct"]),
                    "open_date": str(worst["open_date"]),
                    "close_date": str(worst["close_date"]),
                    "close_reason": worst.get("close_reason", ""),
                }
            return stats
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] get_execution_stats 失败: %s", e)
            return {"error": str(e)}

    def get_execution_vs_hold_comparison(self) -> dict:
        """执行 vs 死扛对比（等权组合 NAV，相同推荐池）。

        - 推荐池: realized_pnl 中所有 (recommend_date, ticker) 对（共 80 笔）
        - 执行组: 按 simulated_orders.pnl_pct（含止损/止盈/到期）等权聚合
        - 死扛组: 用 realized_pnl.actual_return_63d 等权聚合
        - 两组用同一批 ticker → 比较"按规则平仓 vs 死扛到 63 天"的差异

        NAV 算法（避免 cumprod 滚雪球的离谱差异）：
          组合规模归一化：每笔订单分 1/N 资金，单笔贡献 = (1/N) × pnl_pct
          按订单平仓日排序，NAV(t) = 1 + Σ 已平仓订单贡献
          → 两组都是 N 笔订单等权，最终 NAV ≈ 1 + mean(returns)

        返回每个策略的累计净值序列、关键指标。
        """
        try:
            from sqlalchemy import text
            from app.db.postgres import get_pg_engine

            with get_pg_engine().connect() as conn:
                df_orders = pd.read_sql(text(
                    "SELECT ticker, open_date, close_date, open_price, close_price, "
                    "pnl_pct, holding_days, close_reason, status "
                    "FROM simulated_orders WHERE status='CLOSED' "
                    "ORDER BY close_date, order_id"), conn)
                df_pnl = pd.read_sql(text(
                    "SELECT as_of_date, ticker, entry_date, exit_date, "
                    "holding_days, actual_return_63d "
                    "FROM realized_pnl ORDER BY exit_date"), conn)

            if df_orders.empty:
                return {"error": "无 CLOSED 订单数据"}

            # 推荐池规模 N：以 realized_pnl 为准（死扛组的样本量）
            n_total = max(len(df_pnl), len(df_orders), 1)

            # ── 执行组等权 NAV ──
            df_orders["close_date"] = pd.to_datetime(df_orders["close_date"])
            df_orders = df_orders.sort_values("close_date").reset_index(drop=True)
            exec_returns = df_orders["pnl_pct"].fillna(0.0)
            exec_contrib = exec_returns / n_total           # 单笔贡献
            exec_nav = (1.0 + exec_contrib.cumsum()).tolist()
            exec_dates = df_orders["close_date"].dt.strftime("%Y-%m-%d").tolist()

            # ── 死扛组等权 NAV ──
            df_pnl["exit_date"] = pd.to_datetime(df_pnl["exit_date"])
            df_pnl = df_pnl.sort_values("exit_date").reset_index(drop=True)
            hold_returns = df_pnl["actual_return_63d"].fillna(0.0)
            hold_contrib = hold_returns / n_total
            hold_nav = (1.0 + hold_contrib.cumsum()).tolist()
            hold_dates = df_pnl["exit_date"].dt.strftime("%Y-%m-%d").tolist()

            # ── 指标计算（等权组合 → total_return = mean(returns)）──
            def _stats(returns: pd.Series, nav: list[float],
                       holding: pd.Series | None = None) -> dict:
                if returns.empty:
                    return {"n": 0, "win_rate": None, "avg_return": None,
                            "total_return": None, "max_dd": None,
                            "sharpe": None, "avg_holding_days": None}
                # NAV 回撤
                nav_arr = pd.Series(nav)
                roll_max = nav_arr.cummax()
                dd = (nav_arr - roll_max) / roll_max
                avg = returns.mean()
                std = returns.std()
                return {
                    "n":              int(len(returns)),
                    "win_rate":       float((returns > 0).mean()),
                    "avg_return":     float(avg),
                    "total_return":   float(nav_arr.iloc[-1] - 1.0),
                    "max_dd":         float(dd.min()) if not dd.empty else 0.0,
                    "sharpe":         float(avg / std * (252 ** 0.5))
                                       if std and std > 0 else None,
                    "avg_holding_days": (float(holding.mean())
                                          if holding is not None and not holding.empty else None),
                }

            exec_stats = _stats(exec_returns, exec_nav, df_orders["holding_days"])
            hold_stats = _stats(hold_returns, hold_nav, df_pnl.get("holding_days"))

            # 计数（便于 UI/测试断言）
            reason_counts = df_orders["close_reason"].value_counts().to_dict()

            return {
                "n_total":          int(n_total),
                "execute": {
                    "curve": exec_nav, "dates": exec_dates,
                    **exec_stats,
                },
                "hold_to_expiry": {
                    "curve": hold_nav, "dates": hold_dates,
                    **hold_stats,
                },
                "exit_reasons":     reason_counts,
                "exec_stop_count":  int(reason_counts.get("stop_loss", 0) +
                                         reason_counts.get("trailing_stop", 0)),
                "exec_target_count": int(reason_counts.get("target_hit", 0)),
                # 兼容旧 UI（之前用过 execution_nav/hold_nav 字段名）
                "execution_nav":    exec_nav[-1] if exec_nav else None,
                "hold_nav":         hold_nav[-1] if hold_nav else None,
            }
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService] get_execution_vs_hold_comparison 失败: %s", e)
            return {"error": str(e)}

    # ══════════════════════════════════════════════════════════════════════════
    # 数据新鲜度
    # ══════════════════════════════════════════════════════════════════════════

    # (展示标签, 相对路径, 预期最大滞后小时数)
    _FRESHNESS_SPECS = [
        ("每日推荐",       "data/recommendations",                       48),
        ("已结算 PnL",     "data/feedback/realized_pnl.parquet",         24 * 8),
        ("前向持仓",       "data/paper_trading/forward_positions.json",  24 * 8),
        # F-05：指向现役周频面板 v5（原 alpha_panel_v4 已停用）
        ("特征面板(v5)",   "data/panel/alpha_panel_weekly_v5.parquet",   24 * 35),
        ("Regime 历史",    "data/regime/regime_history.parquet",         48),
        ("策略配置",       "data/paper_trading/strategy_config_v2.json", 24 * 35),
        ("损失信号",       "data/loss_signals_v4/latest.json",           24 * 8),
        # F-05：bake-off 产物新鲜度
        ("Bakeoff leaderboard", "data/bakeoff/bucket_leaderboard.csv",   24 * 60),
        ("Bakeoff 预测",   "data/bakeoff/preds",                         24 * 60),
    ]

    def get_data_freshness(self) -> list[dict]:
        """各关键数据文件的最后更新时间 + 距今小时 + 是否超期。"""
        def _load() -> list[dict]:
            import datetime as _dt
            now = _dt.datetime.now()
            out: list[dict] = []
            for label, rel, max_h in self._FRESHNESS_SPECS:
                p = self._root / rel
                if not p.exists():
                    out.append({"label": label, "path": rel, "exists": False,
                                "age_h": None, "mtime": "缺失", "ok": False, "max_h": max_h})
                    continue
                try:
                    if p.is_dir():
                        files = list(p.glob("*"))
                        m = max((f.stat().st_mtime for f in files), default=p.stat().st_mtime)
                    else:
                        m = p.stat().st_mtime
                    age_h = (now - _dt.datetime.fromtimestamp(m)).total_seconds() / 3600
                    out.append({
                        "label": label, "path": rel, "exists": True,
                        "age_h": round(age_h, 1),
                        "mtime": _dt.datetime.fromtimestamp(m).strftime("%Y-%m-%d %H:%M"),
                        "ok": age_h <= max_h, "max_h": max_h,
                    })
                except Exception:  # noqa: BLE001
                    out.append({"label": label, "path": rel, "exists": False,
                                "age_h": None, "mtime": "读取失败", "ok": False, "max_h": max_h})
            return out
        return self._cached("data_freshness", _load)


    # ══════════════════════════════════════════════════════════════════════════
    # DB Backend 实现（DATA_BACKEND=postgres 时走这里）
    # ══════════════════════════════════════════════════════════════════════════

    def _get_pg_engine(self):
        from app.db.postgres import get_pg_engine
        return get_pg_engine()

    def _get_mongo_db(self):
        from app.db.mongo import get_mongo_db
        return get_mongo_db()

    # ── PostgreSQL 实现 ────────────────────────────────────────────────────────

    @staticmethod
    def _mark_pg_empty(df: pd.DataFrame, table: str) -> pd.DataFrame:
        """PG 表为空时：打 warning 并在 DataFrame 上标记 empty_reason。

        目的：让"postgres 后端未填充"这件事**响亮地暴露**，而不是被静默当成
        "用户暂无数据"。前端可读 ``df.attrs.get('empty_reason') == 'pg_table_empty'``
        来显示明显的红色横幅，提示切回 parquet 或运行迁移。
        """
        if df is not None and df.empty:
            log.warning(
                "[DataService/PG] 表 %s 为空，返回空结果 —— postgres 后端未填充。"
                "请切回 DATA_BACKEND=parquet 或运行 scripts/db_migration/02_import_pg.py。",
                table,
            )
            try:
                df.attrs["empty_reason"] = "pg_table_empty"
                df.attrs["empty_table"] = table
            except Exception:  # noqa: BLE001
                pass
        return df

    def _pg_get_realized_pnl(self) -> pd.DataFrame:
        from sqlalchemy import text
        try:
            with self._get_pg_engine().connect() as conn:
                df = pd.read_sql(text("SELECT * FROM realized_pnl ORDER BY as_of_date DESC"), conn)
            # 移除 SERIAL id 列（parquet 版本没有）
            df = df.drop(columns=["id"], errors="ignore")
            for col in ("as_of_date", "entry_date", "exit_date"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            return self._mark_pg_empty(df, "realized_pnl")
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService/PG] get_realized_pnl 失败: %s", e)
            return pd.DataFrame()

    # ── MongoDB 实现 ───────────────────────────────────────────────────────────

    def _mongo_get_recommendation_dates(self) -> list[str]:
        try:
            coll = self._get_mongo_db()["recommendations"]
            docs = list(coll.find({}, {"_id": 1}).sort("_id", -1))
            return [d["_id"] for d in docs]
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService/Mongo] get_recommendation_dates 失败: %s", e)
            return []

    def _mongo_get_recommendations(self, date: str | None = None) -> pd.DataFrame:
        cols = ["ticker", "name", "industry", "entry_price", "current_price",
                "pnl_pct", "lgbm_rank", "ensemble_score", "rating", "status", "reason"]
        try:
            coll = self._get_mongo_db()["recommendations"]
            if date is None:
                doc = coll.find_one(sort=[("_id", -1)])
            else:
                doc = coll.find_one({"_id": date})
            if not doc:
                return pd.DataFrame(columns=cols)

            as_of = doc.get("as_of", doc.get("_id", ""))
            cur_px = self._latest_snapshot_close()
            name_map = self._cached("name_map", lambda: load_name_map(self._root))

            # OPEN 持仓集合（仍从 MongoDB positions 查）
            pos_coll = self._get_mongo_db()["positions"]
            open_set = {
                (str(p.get("as_of")), str(p.get("ticker")))
                for p in pos_coll.find({"status": "OPEN"}, {"as_of": 1, "ticker": 1})
            }

            pnl = self.get_realized_pnl()
            settled_set: set = set()
            if not pnl.empty:
                settled_set = {
                    (str(a), str(t)) for a, t in
                    zip(pnl.get("as_of_date", []), pnl.get("ticker", []))
                }

            rows = []
            for it in doc.get("top10", []):
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
                    "ticker":         tk,
                    "name":           it.get("name") or name_map.get(tk, tk),
                    "industry":       it.get("industry", "") or "—",
                    "entry_price":    entry,
                    "current_price":  cur,
                    "pnl_pct":        round(pnl_pct, 4) if pnl_pct is not None else None,
                    "lgbm_rank":      it.get("lgbm_rank"),
                    "ensemble_score": it.get("lgbm_score"),
                    "rating":         it.get("agent_recommendation", ""),
                    "status":         status,
                    "reason":         it.get("reason", ""),
                })
            return pd.DataFrame(rows, columns=cols)
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService/Mongo] get_recommendations 失败: %s", e)
            return pd.DataFrame(columns=cols)

    def _mongo_forward_positions_raw(self) -> list[dict]:
        try:
            coll = self._get_mongo_db()["positions"]
            return list(coll.find({}, {"_id": 0}))
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService/Mongo] _forward_positions_raw 失败: %s", e)
            return []

    def _mongo_get_agent_analysis(self, ticker: str, date: str | None = None) -> dict | None:
        try:
            coll = self._get_mongo_db()["agent_analysis"]
            if date:
                doc = coll.find_one({"date": date, "ticker": ticker})
            else:
                doc = coll.find_one({"ticker": ticker}, sort=[("date", -1)])
            if not doc:
                return None
            doc.pop("_id", None)
            return self._shape_strategy(doc)
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService/Mongo] get_agent_analysis 失败: %s", e)
            return None

    def _mongo_get_all_agent_analysis(self, date: str | None = None) -> dict[str, dict]:
        try:
            coll = self._get_mongo_db()["agent_analysis"]
            if date is None:
                # 取最近一天
                latest = coll.find_one(sort=[("date", -1)])
                if not latest:
                    return {}
                date = latest["date"]
            docs = list(coll.find({"date": date}, {"_id": 0}))
            return {str(d.get("ticker")): self._shape_strategy(d) for d in docs if d.get("ticker")}
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService/Mongo] get_all_agent_analysis 失败: %s", e)
            return {}

    def _mongo_find_agent_analysis(self, ticker: str) -> tuple[str | None, dict | None]:
        try:
            coll = self._get_mongo_db()["agent_analysis"]
            doc = coll.find_one({"ticker": ticker}, sort=[("date", -1)])
            if not doc:
                return None, None
            date = doc.get("date")
            doc.pop("_id", None)
            return date, self._shape_strategy(doc)
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService/Mongo] find_agent_analysis 失败: %s", e)
            return None, None

    def _mongo_get_loss_signals(self) -> dict:
        try:
            coll = self._get_mongo_db()["loss_signals"]
            doc = coll.find_one(sort=[("run_ts", -1)])
            if not doc:
                return {}
            doc.pop("_id", None)
            # 重组为 {latest:..., factor_health:..., action_plan:...} 结构
            return {
                "latest":        {k: v for k, v in doc.items() if k not in ("action_plan", "factor_health")},
                "factor_health": doc.get("factor_health", {}),
                "action_plan":   doc.get("action_plan", {}),
            }
        except Exception as e:  # noqa: BLE001
            log.warning("[DataService/Mongo] get_loss_signals 失败: %s", e)
            return {}


# ── 模块级单例 ────────────────────────────────────────────────────────────────

_DEFAULT_SERVICE: DataService | None = None


def get_data_service() -> DataService:
    """返回进程级共享的 DataService 单例。"""
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = DataService()
    return _DEFAULT_SERVICE
