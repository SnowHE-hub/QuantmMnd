"""app/db/writers.py — 统一双写接口（parquet + DB）.

控制变量 WRITE_MODE（.env 或环境变量）：
  'parquet_only'  只写 parquet（当前默认，安全模式）
  'dual'          双写（parquet + DB），本阶段目标
  'db_only'       只写 DB（最终模式，待 DB 完全稳定后启用）

设计原则：
  1. 失败隔离 — DB 写失败不影响 parquet 写（业务不中断）
  2. 幂等 — 重复写入相同数据不产生重复记录（upsert/on-conflict）
  3. 日志清晰 — 每次双写打 INFO/WARNING 日志，便于排查
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass

_VALID_MODES = ("parquet_only", "dual", "db_only")


class DataWriter:
    """统一写入接口，支持 parquet / DB / 双写三种模式。"""

    def __init__(self, mode: str | None = None) -> None:
        raw = mode or os.environ.get("WRITE_MODE", "parquet_only")
        self.mode = raw if raw in _VALID_MODES else "parquet_only"

    @property
    def _db_enabled(self) -> bool:
        return self.mode in ("dual", "db_only")

    @property
    def _parquet_enabled(self) -> bool:
        return self.mode in ("parquet_only", "dual")

    # ── 内部：双写调度 ─────────────────────────────────────────────────────────

    def _dual_write(
        self,
        parquet_fn,
        db_fn,
        name: str,
    ) -> None:
        """
        先写 parquet（失败时抛错，中断业务），再写 DB（失败时只 warn，业务继续）。
        """
        if self._parquet_enabled:
            parquet_fn()
            log.info("[Writer/%s] parquet 写入成功", name)

        if self._db_enabled:
            try:
                db_fn()
                log.info("[Writer/%s] DB 写入成功", name)
            except Exception as e:  # noqa: BLE001
                log.warning("[Writer/%s] DB 写入失败（已隔离，parquet 不受影响）: %s", name, e)

    # ── 连接懒加载 ─────────────────────────────────────────────────────────────

    def _pg(self):
        from app.db.postgres import get_pg_engine
        return get_pg_engine()

    def _mongo(self, coll: str):
        from app.db.mongo import get_mongo_db
        return get_mongo_db()[coll]

    # ══════════════════════════════════════════════════════════════════════════
    # 1. 推荐数据（recommendations）
    # ══════════════════════════════════════════════════════════════════════════

    def write_recommendations(
        self,
        date_str: str,
        payload: dict[str, Any],
        *,
        parquet_paths: list[Path] | None = None,
        parquet_content: str | None = None,
    ) -> None:
        """双写推荐 JSON → MongoDB recommendations。

        调用方式（在 daily_update.py::step7_save_json 内）：
            writer.write_recommendations(as_of.isoformat(), payload)
        parquet 写入由调用方自己完成，这里只做 DB 写入（parquet_fn 为 no-op）。
        """
        def _db():
            coll = self._mongo("recommendations")
            # date_str 优先于 payload 内的 as_of（保证 _id 与 as_of 一致）
            doc = {**payload, "as_of": date_str}
            coll.replace_one({"_id": date_str}, {"_id": date_str, **doc}, upsert=True)

        # parquet 已由调用方写完，这里 parquet_fn 是 no-op
        if self._db_enabled:
            try:
                _db()
                log.info("[Writer/recommendations] DB 写入成功 date=%s top10=%d",
                         date_str, len(payload.get("top10", [])))
            except Exception as e:  # noqa: BLE001
                log.warning("[Writer/recommendations] DB 写入失败（已隔离）: %s", e)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. 已实现 PnL（realized_pnl）
    # ══════════════════════════════════════════════════════════════════════════

    def write_realized_pnl(
        self,
        df: pd.DataFrame,
        *,
        full_replace: bool = True,
    ) -> None:
        """双写 realized_pnl DataFrame → PostgreSQL realized_pnl 表。

        full_replace=True（track_realized_pnl.py）：TRUNCATE + INSERT ALL
        full_replace=False（settle_forward_positions.py）：upsert by (as_of_date, ticker)
        """
        if df.empty:
            return

        def _db():
            from sqlalchemy import text
            eng = self._pg()
            df_copy = df.copy()
            # 统一日期格式
            for col in ("as_of_date", "entry_date", "exit_date"):
                if col in df_copy.columns:
                    df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce").dt.date
            # 移除 id 列（SERIAL，DB 自动生成）
            df_copy = df_copy.drop(columns=["id"], errors="ignore")

            if full_replace:
                with eng.begin() as conn:
                    conn.execute(text("TRUNCATE TABLE realized_pnl"))
                df_copy.to_sql("realized_pnl", eng, if_exists="append",
                               index=False, method="multi", chunksize=500)
            else:
                # 增量 upsert：先删已有 (as_of_date, ticker)，再插入
                with eng.begin() as conn:
                    for _, row in df_copy.iterrows():
                        conn.execute(
                            text("DELETE FROM realized_pnl WHERE as_of_date=:d AND ticker=:t"),
                            {"d": str(row["as_of_date"]), "t": str(row["ticker"])},
                        )
                df_copy.to_sql("realized_pnl", eng, if_exists="append",
                               index=False, method="multi", chunksize=500)

        if self._db_enabled:
            try:
                _db()
                log.info("[Writer/realized_pnl] DB 写入成功 rows=%d full_replace=%s",
                         len(df), full_replace)
            except Exception as e:  # noqa: BLE001
                log.warning("[Writer/realized_pnl] DB 写入失败（已隔离）: %s", e)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. 前向持仓（forward_positions）
    # ══════════════════════════════════════════════════════════════════════════

    def write_forward_positions(self, positions: list[dict]) -> None:
        """双写 forward_positions → MongoDB positions collection。"""
        if not positions:
            return

        if self._db_enabled:
            try:
                from pymongo import UpdateOne
                coll = self._mongo("positions")
                ops = [
                    UpdateOne(
                        {"as_of": p.get("as_of"), "ticker": p.get("ticker")},
                        {"$set": p},
                        upsert=True,
                    )
                    for p in positions
                ]
                result = coll.bulk_write(ops, ordered=False)
                log.info("[Writer/positions] DB upserted=%d modified=%d",
                         result.upserted_count, result.modified_count)
            except Exception as e:  # noqa: BLE001
                log.warning("[Writer/positions] DB 写入失败（已隔离）: %s", e)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Loss signals
    # ══════════════════════════════════════════════════════════════════════════

    def write_loss_signals(
        self,
        latest: dict,
        action_plan: dict | None = None,
        factor_health: dict | None = None,
    ) -> None:
        """双写 loss_signals → MongoDB loss_signals collection。"""
        if self._db_enabled:
            try:
                coll = self._mongo("loss_signals")
                run_ts = latest.get("run_ts", "")
                date_id = run_ts[:10] if run_ts else "latest"
                doc: dict[str, Any] = {**latest, "source": "daily_dispatch"}
                if action_plan:
                    doc["action_plan"] = action_plan
                if factor_health:
                    doc["factor_health"] = factor_health
                coll.replace_one({"_id": date_id}, {"_id": date_id, **doc}, upsert=True)
                log.info("[Writer/loss_signals] DB 写入成功 _id=%s", date_id)
            except Exception as e:  # noqa: BLE001
                log.warning("[Writer/loss_signals] DB 写入失败（已隔离）: %s", e)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. 策略配置（strategy_config）
    # ══════════════════════════════════════════════════════════════════════════

    def write_strategy_config(self, config: dict, version: str = "v2") -> None:
        """双写 strategy_config → MongoDB strategy_config collection。"""
        if self._db_enabled:
            try:
                coll = self._mongo("strategy_config")
                coll.replace_one(
                    {"_id": version},
                    {"_id": version, **config},
                    upsert=True,
                )
                log.info("[Writer/strategy_config] DB 写入成功 version=%s", version)
            except Exception as e:  # noqa: BLE001
                log.warning("[Writer/strategy_config] DB 写入失败（已隔离）: %s", e)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. 6-Agent 分析（agent_analysis）
    # ══════════════════════════════════════════════════════════════════════════

    def write_agent_analysis(self, date_str: str, strategies: list[dict]) -> None:
        """双写 strategies.json → MongoDB agent_analysis collection。"""
        if not strategies:
            return

        if self._db_enabled:
            try:
                from pymongo import UpdateOne
                coll = self._mongo("agent_analysis")
                ops = []
                for s in strategies:
                    ticker = s.get("ticker", "")
                    doc_id = f"{date_str}_{ticker}"
                    doc = {"date": date_str, **s}
                    ops.append(UpdateOne({"_id": doc_id}, {"$set": {"_id": doc_id, **doc}}, upsert=True))
                result = coll.bulk_write(ops, ordered=False)
                log.info("[Writer/agent_analysis] DB upserted=%d modified=%d date=%s",
                         result.upserted_count, result.modified_count, date_str)
            except Exception as e:  # noqa: BLE001
                log.warning("[Writer/agent_analysis] DB 写入失败（已隔离）: %s", e)


# ── 进程级单例 ────────────────────────────────────────────────────────────────

_writer: DataWriter | None = None


def get_writer(mode: str | None = None) -> DataWriter:
    """返回进程级共享的 DataWriter 单例（懒初始化）。"""
    global _writer
    if _writer is None or (mode is not None and mode != _writer.mode):
        _writer = DataWriter(mode)
    return _writer
