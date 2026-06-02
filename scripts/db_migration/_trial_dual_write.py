"""
双写模式试运行脚本（不调用完整 daily_update，避免 LLM/数据拉取）。
模拟 daily_update.py::step7_save_json 和 track_realized_pnl 写入流程，
验证 WRITE_MODE=dual 下 parquet 和 DB 都被正确写入。

用法:
  WRITE_MODE=dual conda run -n quantmind python scripts/db_migration/_trial_dual_write.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("WRITE_MODE", "dual")

import pandas as pd
from app.db.writers import get_writer
from app.db.mongo import get_mongo_db
from app.db.postgres import get_pg_engine
from sqlalchemy import text

writer = get_writer()
mongo = get_mongo_db()
pg = get_pg_engine()

print(f"WRITE_MODE = {writer.mode}")
print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1：write_recommendations（模拟 step7_save_json）
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1] write_recommendations")

# 用最新存在的 top10.json
rec_dir = ROOT / "data" / "recommendations"
top10_paths = sorted(rec_dir.glob("*/top10.json"), reverse=True)
if top10_paths:
    latest_top10 = top10_paths[0]
    date_str = latest_top10.parent.name
    payload = json.loads(latest_top10.read_text(encoding="utf-8"))
    payload.setdefault("generated_at", "2026-06-02T16:00:00")
    payload.setdefault("step_status", {})

    writer.write_recommendations(date_str, payload)

    doc = mongo["recommendations"].find_one({"_id": date_str})
    if doc:
        top10_count = len(doc.get("top10", []))
        print(f"  ✓ MongoDB recommendations _id={date_str}  top10={top10_count}")
    else:
        print(f"  ✗ MongoDB 中找不到 _id={date_str}")
else:
    print("  SKIP: 无 top10.json 文件")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2：write_realized_pnl（模拟 track_realized_pnl）
# ─────────────────────────────────────────────────────────────────────────────

print("\n[2] write_realized_pnl")

pnl_path = ROOT / "data" / "feedback" / "realized_pnl.parquet"
if pnl_path.exists():
    df = pd.read_parquet(pnl_path)
    writer.write_realized_pnl(df, full_replace=True)

    with pg.connect() as conn:
        db_count = conn.execute(text("SELECT COUNT(*) FROM realized_pnl")).scalar()
    print(f"  ✓ PG realized_pnl: DB={db_count} rows  (parquet={len(df)} rows)")
    assert db_count == len(df), f"行数不一致: DB={db_count}, parquet={len(df)}"
else:
    print("  SKIP: realized_pnl.parquet 不存在")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3：write_forward_positions
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3] write_forward_positions")

fwd_path = ROOT / "data" / "paper_trading" / "forward_positions.json"
if fwd_path.exists():
    fwd_data = json.loads(fwd_path.read_text(encoding="utf-8"))
    positions = fwd_data.get("positions", [])
    writer.write_forward_positions(positions)

    db_count = mongo["positions"].count_documents({})
    print(f"  ✓ MongoDB positions: {db_count} docs  (file={len(positions)} positions)")
else:
    print("  SKIP: forward_positions.json 不存在")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4：write_loss_signals
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4] write_loss_signals")

ls_dir = ROOT / "data" / "loss_signals_v4"
latest_p = ls_dir / "latest.json"
if latest_p.exists():
    latest = json.loads(latest_p.read_text(encoding="utf-8"))
    ap = json.loads((ls_dir / "action_plan.json").read_text()) if (ls_dir / "action_plan.json").exists() else {}
    fh = json.loads((ls_dir / "factor_health.json").read_text()) if (ls_dir / "factor_health.json").exists() else {}
    writer.write_loss_signals(latest, ap, fh)

    run_ts = latest.get("run_ts", "")
    date_id = run_ts[:10] if run_ts else "latest"
    doc = mongo["loss_signals"].find_one({"_id": date_id})
    if doc:
        print(f"  ✓ MongoDB loss_signals _id={date_id}  health={doc.get('overall_health','?')}")
    else:
        print(f"  ✗ MongoDB 中找不到 _id={date_id}")
else:
    print("  SKIP: loss_signals_v4/latest.json 不存在")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5：write_agent_analysis
# ─────────────────────────────────────────────────────────────────────────────

print("\n[5] write_agent_analysis")

pipeline_dir = ROOT / "reports" / "investment_pipeline"
strat_paths = sorted(pipeline_dir.glob("*/strategies.json"), reverse=True) if pipeline_dir.exists() else []
if strat_paths:
    latest_strat = strat_paths[0]
    date_str_a = latest_strat.parent.name
    strategies = json.loads(latest_strat.read_text(encoding="utf-8"))
    if isinstance(strategies, list):
        writer.write_agent_analysis(date_str_a, strategies)
        db_count = mongo["agent_analysis"].count_documents({"date": date_str_a})
        print(f"  ✓ MongoDB agent_analysis date={date_str_a}  docs={db_count}/{len(strategies)}")
else:
    print("  SKIP: 无 strategies.json 文件")


# ─────────────────────────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("试运行完成 ✓")
print(f"MongoDB collections: {mongo.list_collection_names()}")
with pg.connect() as conn:
    tables = conn.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname='public'"
    )).fetchall()
    print(f"PostgreSQL tables: {[r[0] for r in tables]}")
