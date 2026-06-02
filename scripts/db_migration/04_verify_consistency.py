"""
迁移后一致性校验脚本（三层校验）
用法: conda run -n quantmind python scripts/db_migration/04_verify_consistency.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.db.postgres import get_pg_engine
from app.db.mongo import get_mongo_db

engine = get_pg_engine()
mongo_db = get_mongo_db()

PASS_MARK = "✓ PASS"
FAIL_MARK = "✗ FAIL"


# ── PostgreSQL 校验 ─────────────────────────────────────────────────────────

def verify_pg_table(
    parquet_path: str | Path,
    table: str,
    key_cols: list[str],
    reset_index: bool = False,
) -> bool:
    path = ROOT / parquet_path
    if not path.exists():
        print(f"  [{table}] SKIP (parquet 不存在)")
        return True

    df = pd.read_parquet(path)
    if reset_index:
        df = df.reset_index()

    errors = []

    # Layer 1: 行数
    with engine.connect() as conn:
        db_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    diff = abs(len(df) - db_count)
    if diff > max(1, len(df) * 0.001):
        errors.append(f"行数不一致: parquet={len(df)}, DB={db_count}")
    else:
        l1 = f"L1-行数 {len(df):,}≈{db_count:,}"

    # Layer 2: 数值列均值
    num_cols = df.select_dtypes(include=["float64", "float32"]).columns[:5].tolist()
    l2_issues = []
    with engine.connect() as conn:
        for col in num_cols:
            pq_mean = float(df[col].mean(skipna=True))
            try:
                db_mean = conn.execute(text(f'SELECT AVG("{col}") FROM {table}')).scalar()
            except Exception:
                continue
            if db_mean is None:
                continue
            db_mean = float(db_mean)
            if pq_mean != 0 and abs(pq_mean - db_mean) / abs(pq_mean) > 1e-4:
                l2_issues.append(f"{col}: pq={pq_mean:.5f} db={db_mean:.5f}")
    if l2_issues:
        errors.append(f"均值偏差: {', '.join(l2_issues)}")
    else:
        l2 = f"L2-均值 {len(num_cols)} 列 OK"

    # Layer 3: 随机抽样 100 行精确比对
    if not errors and num_cols:
        sample = df.sample(min(100, len(df)), random_state=42)
        l3_issues = []
        with engine.connect() as conn:
            for _, row in sample.iterrows():
                where_parts = []
                params = {}
                for kc in key_cols:
                    val = row[kc]
                    if hasattr(val, "date"):
                        val = val.date() if hasattr(val, "date") else val
                    where_parts.append(f'"{kc}"=:{kc}')
                    params[kc] = str(val) if not isinstance(val, (int, float)) else val
                where = " AND ".join(where_parts)
                db_row = conn.execute(
                    text(f"SELECT * FROM {table} WHERE {where} LIMIT 1"), params
                ).fetchone()
                if db_row is None:
                    l3_issues.append(f"missing {dict(params)}")
                    continue
                db_dict = dict(db_row._mapping)
                for col in num_cols[:3]:
                    pq_val = float(row[col]) if pd.notna(row[col]) else None
                    db_val = db_dict.get(col)
                    if pq_val is not None and db_val is not None:
                        if abs(float(pq_val) - float(db_val)) > 1e-3:
                            l3_issues.append(f"{col}: pq={pq_val:.4f} db={float(db_val):.4f}")
        if l3_issues:
            errors.append(f"抽样不一致({len(l3_issues)}处): {l3_issues[:2]}")
        else:
            l3 = "L3-抽样 OK"

    ok = not errors
    mark = PASS_MARK if ok else FAIL_MARK
    if ok:
        parts = [v for v in [locals().get("l1"), locals().get("l2"), locals().get("l3")] if v]
        detail = " | ".join(parts)
    else:
        detail = " | ".join(errors)
    print(f"  [{table}] {mark} — {detail}")
    return ok


# ── MongoDB 校验 ─────────────────────────────────────────────────────────────

def verify_mongo(collection: str, expected_count: int, check_id: str | None = None) -> bool:
    coll = mongo_db[collection]
    actual = coll.count_documents({})
    ok = actual >= expected_count
    mark = PASS_MARK if ok else FAIL_MARK
    detail = f"docs={actual}"
    if check_id:
        doc = coll.find_one({"_id": check_id})
        if doc is None:
            ok = False
            mark = FAIL_MARK
            detail += f" | _id='{check_id}' 找不到"
    print(f"  [{collection}] {mark} — {detail}")
    return ok


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("QuantMind 迁移一致性校验")
    print("=" * 65)
    results = {}

    print("\n── PostgreSQL 表校验 ──")
    checks = [
        ("data/alpha_universe/alpha_universe.parquet",          "alpha_universe",        ["ts_code"],                    False),
        ("data/raw/alpha_prices_panel.parquet",                 "price_daily",           ["ts_code", "trade_date"],      False),
        ("data/raw/daily_prices_panel.parquet",                 "daily_prices_panel",    ["ts_code", "trade_date"],      False),
        ("data/raw/index_daily_panel.parquet",                  "index_daily",           ["ts_code", "trade_date"],      False),
        ("data/alpha_universe/alpha_daily_basic_combined.parquet","daily_basic",         ["ts_code", "trade_date"],      False),
        ("data/alpha_universe/fina_indicator_combined.parquet", "fina_indicator",        ["ts_code", "ann_date", "end_date"], False),
        ("data/feedback/realized_pnl.parquet",                  "realized_pnl",         ["as_of_date", "ticker"],       False),
        ("data/features/regime_features.parquet",               "regime_features",       ["as_of"],                      True),
    ]
    for pq, tbl, keys, ri in checks:
        results[tbl] = verify_pg_table(pq, tbl, keys, ri)

    # alpha_panel 单独校验（合并文件）
    with engine.connect() as conn:
        ap_count = conn.execute(text("SELECT COUNT(*) FROM alpha_panel")).scalar()
    full_panel_rows = len(pd.read_parquet(ROOT / "data/features/csi300_full_panel.parquet"))
    ok = ap_count >= full_panel_rows
    mark = PASS_MARK if ok else FAIL_MARK
    print(f"  [alpha_panel] {mark} — DB={ap_count:,} (csi300_full_panel={full_panel_rows:,})")
    results["alpha_panel"] = ok

    # nav_curve
    with engine.connect() as conn:
        nav_count = conn.execute(text("SELECT COUNT(*) FROM nav_curve")).scalar()
    sim_nav_files = len(list((ROOT / "data/sim30d/nav").glob("*.json")))
    ok = nav_count >= sim_nav_files
    mark = PASS_MARK if ok else FAIL_MARK
    print(f"  [nav_curve] {mark} — DB={nav_count} (sim30d files={sim_nav_files})")
    results["nav_curve"] = ok

    # snapshot_daily_basic
    with engine.connect() as conn:
        snap_count = conn.execute(text("SELECT COUNT(*) FROM snapshot_daily_basic")).scalar()
    snap_dirs = len(list((ROOT / "data/snapshots").glob("*/daily_basic.parquet")))
    ok = snap_count > 0
    mark = PASS_MARK if ok else FAIL_MARK
    print(f"  [snapshot_daily_basic] {mark} — DB={snap_count:,} ({snap_dirs} snapshot 目录)")
    results["snapshot_daily_basic"] = ok

    print("\n── MongoDB Collection 校验 ──")
    rec_count = len(list((ROOT / "data/recommendations").glob("*/top10.json")))
    results["recommendations"] = verify_mongo("recommendations", rec_count, check_id="2026-06-01")

    pos_data = json.loads((ROOT / "data/paper_trading/forward_positions.json").read_text())
    pos_count = len(pos_data.get("positions", []))
    results["positions"] = verify_mongo("positions", pos_count)

    results["sim_daily"] = verify_mongo(
        "sim_daily",
        len(list((ROOT / "data/sim30d/daily").glob("*.json"))),
        check_id="sim30d_20251009",
    )
    results["loss_signals"] = verify_mongo("loss_signals", 1)
    results["watchlist_scores"] = verify_mongo("watchlist_scores", 1)

    # 汇总
    print("\n" + "=" * 65)
    passed = sum(results.values())
    total = len(results)
    print(f"校验结果：{passed}/{total} 通过")
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"失败项：{', '.join(failed)}")
        sys.exit(1)
    else:
        print("全部通过 ✓")


if __name__ == "__main__":
    main()
