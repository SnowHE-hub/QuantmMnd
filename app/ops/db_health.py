"""app/ops/db_health.py — 数据库健康检查 + 双写监控 工具函数.

被 app/pages/7_系统控制台.py 和 app/pages/13_系统健康.py 共享调用。

不依赖 Streamlit（纯函数，可在 tests / CLI 复用）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
FAILURE_LOG = LOG_DIR / "db_write_failures.log"
AUDIT_LOG = LOG_DIR / "db_write_audit.log"
ENV_PATH = ROOT / ".env"


# ──────────────────────────────────────────────────────────────────────────────
# 1. 数据库 ping
# ──────────────────────────────────────────────────────────────────────────────

def pg_ping(timeout_sec: float = 3.0) -> dict[str, Any]:
    """PostgreSQL 连通性 + 表统计。"""
    info: dict[str, Any] = {"ok": False}
    try:
        from sqlalchemy import text
        from app.db.postgres import get_pg_engine
        eng = get_pg_engine()
        with eng.connect() as conn:
            ver = conn.execute(text("SELECT version()")).scalar()
            tables = conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
            )).fetchall()
            tbl_counts: dict[str, int] = {}
            for (tname,) in tables:
                try:
                    cnt = conn.execute(text(f'SELECT COUNT(*) FROM "{tname}"')).scalar()
                    tbl_counts[tname] = int(cnt or 0)
                except Exception:  # noqa: BLE001
                    tbl_counts[tname] = -1
            info.update({
                "ok": True,
                "version": str(ver).split(",")[0][:60] if ver else "?",
                "tables": tbl_counts,
                "n_tables": len(tbl_counts),
                "total_rows": sum(v for v in tbl_counts.values() if v > 0),
            })
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return info


def mongo_ping(timeout_sec: float = 3.0) -> dict[str, Any]:
    """MongoDB 连通性 + collection 统计。"""
    info: dict[str, Any] = {"ok": False}
    try:
        from app.db.mongo import get_mongo_db
        db = get_mongo_db()
        # ping
        db.command("ping")
        colls = sorted(db.list_collection_names())
        col_counts: dict[str, int] = {}
        for c in colls:
            try:
                col_counts[c] = db[c].estimated_document_count()
            except Exception:  # noqa: BLE001
                col_counts[c] = -1
        info.update({
            "ok": True,
            "db_name": db.name,
            "collections": col_counts,
            "n_collections": len(col_counts),
            "total_docs": sum(v for v in col_counts.values() if v > 0),
        })
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return info


# ──────────────────────────────────────────────────────────────────────────────
# 2. 双写日志解析
# ──────────────────────────────────────────────────────────────────────────────

def _parse_log_line(line: str) -> dict[str, Any] | None:
    """解析一行日志：TIMESTAMP\\tNAME\\tSTATUS\\tINFO[\\tCTX]"""
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return None
    try:
        ts = datetime.fromisoformat(parts[0])
    except Exception:  # noqa: BLE001
        return None
    return {
        "ts": ts,
        "name": parts[1],
        "status": parts[2],
        "info": parts[3] if len(parts) > 3 else "",
        "ctx": parts[4] if len(parts) > 4 else "",
    }


def read_failures(hours: int = 24) -> list[dict[str, Any]]:
    """读取最近 N 小时的 DB 写入失败记录。"""
    if not FAILURE_LOG.exists():
        return []
    cutoff = datetime.now() - timedelta(hours=hours)
    out: list[dict[str, Any]] = []
    try:
        for line in FAILURE_LOG.read_text(encoding="utf-8").splitlines():
            ev = _parse_log_line(line)
            if ev and ev["ts"] >= cutoff:
                out.append(ev)
    except Exception:  # noqa: BLE001
        return []
    return out


def read_audit(hours: int = 24) -> list[dict[str, Any]]:
    """读取最近 N 小时的 DB 写入成功记录。"""
    if not AUDIT_LOG.exists():
        return []
    cutoff = datetime.now() - timedelta(hours=hours)
    out: list[dict[str, Any]] = []
    try:
        for line in AUDIT_LOG.read_text(encoding="utf-8").splitlines():
            ev = _parse_log_line(line)
            if ev and ev["ts"] >= cutoff:
                out.append(ev)
    except Exception:  # noqa: BLE001
        return []
    return out


def dual_write_stats(days: int = 7) -> dict[str, Any]:
    """统计最近 N 天的双写情况：每日成功/失败数、按 writer 拆分。"""
    success = read_audit(hours=days * 24)
    failures = read_failures(hours=days * 24)

    # 按日期聚合
    by_date: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "fail": 0})
    for ev in success:
        d = ev["ts"].date().isoformat()
        by_date[d]["ok"] += 1
    for ev in failures:
        d = ev["ts"].date().isoformat()
        by_date[d]["fail"] += 1

    # 按 writer 名称聚合
    by_writer: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "fail": 0})
    for ev in success:
        by_writer[ev["name"]]["ok"] += 1
    for ev in failures:
        by_writer[ev["name"]]["fail"] += 1

    daily_rows = [
        {"date": d, "ok": v["ok"], "fail": v["fail"],
         "total": v["ok"] + v["fail"],
         "success_rate": v["ok"] / (v["ok"] + v["fail"]) if (v["ok"] + v["fail"]) > 0 else None}
        for d, v in sorted(by_date.items())
    ]
    writer_rows = [
        {"writer": w, "ok": v["ok"], "fail": v["fail"],
         "total": v["ok"] + v["fail"]}
        for w, v in sorted(by_writer.items(), key=lambda kv: -kv[1]["fail"])
    ]

    return {
        "total_ok": len(success),
        "total_fail": len(failures),
        "success_rate": (
            len(success) / (len(success) + len(failures))
            if (success or failures) else None
        ),
        "by_date": daily_rows,
        "by_writer": writer_rows,
        "last_failure": failures[-1] if failures else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. .env DATA_BACKEND 切换
# ──────────────────────────────────────────────────────────────────────────────

def read_env_value(key: str) -> str | None:
    """直接从 .env 文件读取 key 当前值（不经过 os.environ 缓存）。"""
    if not ENV_PATH.exists():
        return None
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip()
    return None


def write_env_value(key: str, value: str) -> bool:
    """原地修改 .env 中 key 的值（保留其他行）。返回是否成功。"""
    if not ENV_PATH.exists():
        return False
    try:
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
        new_lines = []
        replaced = False
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                k, _, _ = stripped.partition("=")
                if k.strip() == key:
                    new_lines.append(f"{key}={value}")
                    replaced = True
                    continue
            new_lines.append(line)
        if not replaced:
            new_lines.append(f"{key}={value}")
        ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except Exception:  # noqa: BLE001
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 4. Parity 校验调用
# ──────────────────────────────────────────────────────────────────────────────

def run_parity_check(timeout_sec: int = 180) -> dict[str, Any]:
    """运行 tests/test_db_backend_parity.py，解析 pytest 结果。

    Returns: {ok, passed, failed, total, output, failed_tests, duration_sec}
    """
    t0 = datetime.now()
    py = sys.executable
    cmd = [
        py, "-m", "pytest",
        "tests/test_db_backend_parity.py",
        "-q", "--tb=line", "--no-cov",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True,
            timeout=timeout_sec, env={**os.environ, "DATA_BACKEND": "parquet"},
        )
        out = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout {timeout_sec}s", "output": ""}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "output": ""}

    duration = (datetime.now() - t0).total_seconds()

    # 解析 "16 passed in 23.78s" 类输出
    passed = failed = 0
    failed_tests: list[str] = []
    import re
    m_pass = re.search(r"(\d+) passed", out)
    m_fail = re.search(r"(\d+) failed", out)
    if m_pass:
        passed = int(m_pass.group(1))
    if m_fail:
        failed = int(m_fail.group(1))
    # FAILED tests/test_xxx::TestY::test_z
    for m in re.finditer(r"FAILED (tests/[^\s]+)", out):
        failed_tests.append(m.group(1))

    return {
        "ok": failed == 0 and passed > 0,
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "failed_tests": failed_tests,
        "output": out[-3000:],  # 截断末尾
        "duration_sec": round(duration, 1),
        "returncode": proc.returncode,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. 综合状态汇总（健康页面用）
# ──────────────────────────────────────────────────────────────────────────────

def overall_db_status() -> dict[str, Any]:
    """一次性返回所有关键状态。健康页面单独区域用。"""
    fails_24h = read_failures(hours=24)
    return {
        "data_backend":     read_env_value("DATA_BACKEND") or "parquet",
        "write_mode":       read_env_value("WRITE_MODE") or "parquet_only",
        "pg":               pg_ping(),
        "mongo":            mongo_ping(),
        "failures_24h":     len(fails_24h),
        "last_failure":     fails_24h[-1] if fails_24h else None,
    }
