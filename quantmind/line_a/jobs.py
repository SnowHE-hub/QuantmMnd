"""quantmind.line_a.jobs — 持久任务表（SQLite）.

规则（03_final_plan_v2 第 7.2 节）：
- 任务键 = 快照哈希 + 策略或角色版本 + 证券或案例编号 + 阶段 + 模型产物哈希 + 提示词与 Schema 哈希（由调用方给出字段）；
- 状态：PENDING → RUNNING → SUCCEEDED / RETRYABLE_FAILED / FAILED / SKIPPED_WITH_REASON；
- 产物先写临时文件、校验后原子提交；一个任务键只有**第一份**提交成功的产物进入正式结果，
  重复执行（崩溃恢复、并发）产生的后续结果被丢弃，不替换已有产物；
- RUNNING 超过租约时长视为崩溃，可以重新认领；
- 单写者：任务表只由编排器进程写入（多进程访问时 BEGIN IMMEDIATE 也能保证正确，但不鼓励）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from quantmind.line_a.atomic import atomic_write_bytes, canonical_json, sha256_bytes

PENDING, RUNNING, SUCCEEDED = "PENDING", "RUNNING", "SUCCEEDED"
RETRYABLE_FAILED, FAILED, SKIPPED = "RETRYABLE_FAILED", "FAILED", "SKIPPED_WITH_REASON"
TERMINAL = {SUCCEEDED, FAILED, SKIPPED}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    task_key      TEXT PRIMARY KEY,
    stage         TEXT NOT NULL,
    status        TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    key_fields    TEXT NOT NULL,
    meta          TEXT,
    output_path   TEXT,
    output_sha256 TEXT,
    error         TEXT,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL
);
CREATE INDEX IF NOT EXISTS jobs_stage_status ON jobs(stage, status);
"""


def task_key(**fields: Any) -> str:
    """幂等任务键：字段的确定性 JSON 的 sha256。字段缺失或为 None 也参与哈希（显式写出）。"""
    return sha256_bytes(canonical_json(fields))


class JobStore:
    def __init__(self, path: str | os.PathLike, *, lease_seconds: float = 3600.0, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)

    # -------------------------------------------------------------- 查询
    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT * FROM jobs WHERE task_key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def counts(self, stage: str | None = None) -> dict[str, int]:
        q, args = "SELECT status, COUNT(*) FROM jobs", ()
        if stage:
            q, args = q + " WHERE stage = ?", (stage,)
        return dict(self.db.execute(q + " GROUP BY status", args).fetchall())

    # -------------------------------------------------------------- 状态迁移
    def claim(self, key: str, stage: str, key_fields: dict, meta: dict | None = None) -> bool:
        """认领任务。已经是终态、或者正被别人在租约内运行，返回 False。"""
        now = self.clock()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT status, started_at FROM jobs WHERE task_key = ?", (key,)).fetchone()
            if row is not None:
                status, started = row["status"], row["started_at"] or 0.0
                if status in TERMINAL or (status == RUNNING and now - started < self.lease_seconds):
                    self.db.execute("COMMIT")
                    return False
                self.db.execute("UPDATE jobs SET status = ?, attempts = attempts + 1, started_at = ?, error = NULL "
                                "WHERE task_key = ?", (RUNNING, now, key))
            else:
                self.db.execute("INSERT INTO jobs (task_key, stage, status, attempts, key_fields, meta, created_at, "
                                "started_at) VALUES (?, ?, ?, 1, ?, ?, ?, ?)",
                                (key, stage, RUNNING, canonical_json(key_fields).decode(),
                                 json.dumps(meta or {}, ensure_ascii=False, default=str), now, now))
            self.db.execute("COMMIT")
            return True
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def commit_output(self, key: str, output_path: str | os.PathLike, data: bytes) -> dict:
        """原子写产物并把任务记为 SUCCEEDED。若该任务已经有成功产物，丢弃这次结果并返回已有记录。"""
        output_path = Path(output_path)
        existing = self.get(key)
        if existing and existing["status"] == SUCCEEDED:
            return existing
        sha = atomic_write_bytes(output_path, data)
        now = self.clock()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cur = self.db.execute("UPDATE jobs SET status = ?, output_path = ?, output_sha256 = ?, finished_at = ? "
                                  "WHERE task_key = ? AND status != ?",
                                  (SUCCEEDED, str(output_path), sha, now, key, SUCCEEDED))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        if cur.rowcount == 0:
            raise KeyError(f"task {key} was never claimed")
        return self.get(key)

    def fail(self, key: str, error: str, *, retryable: bool = True) -> None:
        self.db.execute("UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE task_key = ? AND status != ?",
                        (RETRYABLE_FAILED if retryable else FAILED, error[:2000], self.clock(), key, SUCCEEDED))

    def skip(self, key: str, stage: str, key_fields: dict, reason: str) -> None:
        now = self.clock()
        self.db.execute("INSERT INTO jobs (task_key, stage, status, attempts, key_fields, error, created_at, finished_at) "
                        "VALUES (?, ?, ?, 0, ?, ?, ?, ?) ON CONFLICT(task_key) DO UPDATE SET status = excluded.status, "
                        "error = excluded.error, finished_at = excluded.finished_at WHERE jobs.status != ?",
                        (key, stage, SKIPPED, canonical_json(key_fields).decode(), reason, now, now, SUCCEEDED))

    # -------------------------------------------------------------- 便捷入口
    def run(self, stage: str, key_fields: dict, output_path: str | os.PathLike, produce: Callable[[], bytes],
            meta: dict | None = None) -> dict | None:
        """幂等执行：已成功就直接返回已有记录；认领失败（别人在跑）返回 None；否则执行 produce 并提交。
        produce 抛异常时记 RETRYABLE_FAILED 并重新抛出。"""
        key = task_key(stage=stage, **key_fields)
        existing = self.get(key)
        if existing and existing["status"] == SUCCEEDED:
            return existing
        if not self.claim(key, stage, key_fields, meta):
            return None
        try:
            data = produce()
        except Exception as e:  # noqa: BLE001
            self.fail(key, f"{type(e).__name__}: {e}")
            raise
        return self.commit_output(key, output_path, data)
