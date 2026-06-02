"""
parquet → PostgreSQL 批量导入脚本
用法: conda run -n quantmind python scripts/db_migration/02_import_pg.py
"""
from __future__ import annotations
import io
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.db.postgres import get_pg_engine

engine = get_pg_engine()


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """把所有 datetime64 列转成 python date（PG DATE 列要求）。"""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.date
    return df


def _copy_import(df: pd.DataFrame, table: str):
    """用 COPY FROM STDIN 导入，比 to_sql 快 10-100x（大表专用）。"""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cols = ",".join(df.columns)
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.copy_expert(
            f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')",
            buf,
        )
        raw.commit()
    finally:
        raw.close()


def import_table(
    parquet_path: str | Path,
    table: str,
    key_cols: list[str],
    method: str = "to_sql",          # 'to_sql' | 'copy'
    chunksize: int = 20_000,
    reset_index: bool = False,
    drop_cols: list[str] | None = None,
    rename: dict | None = None,
) -> int:
    """通用导入函数。返回 DB 最终行数。"""
    path = ROOT / parquet_path
    if not path.exists():
        print(f"  [SKIP] 文件不存在: {path}")
        return 0

    t0 = time.time()
    df = pd.read_parquet(path)
    if reset_index:
        df = df.reset_index()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if rename:
        df = df.rename(columns=rename)
    df = _normalize_dates(df)

    # 把 string[python] 转成 object（psycopg2 不认识 pd.StringDtype）
    for col in df.columns:
        if hasattr(df[col], "dtype") and str(df[col].dtype) == "string":
            df[col] = df[col].astype(object)

    rows_in = len(df)
    print(f"  [{table}] 导入 {rows_in:,} 行，method={method} ...", end="", flush=True)

    if method == "copy":
        # DROP + CREATE TABLE，再 COPY（最快路径）
        df.head(0).to_sql(table, engine, if_exists="replace", index=False)
        _copy_import(df, table)
    else:
        df.to_sql(
            table, engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=chunksize,
        )

    with engine.connect() as conn:
        db_count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

    elapsed = time.time() - t0
    ok = "✓" if abs(rows_in - db_count) <= max(1, rows_in * 0.001) else "✗"
    print(f" {ok} DB={db_count:,} ({elapsed:.1f}s)")
    return db_count


def add_indexes():
    """迁移完成后补建 BRIN / B-Tree 索引。"""
    ddl_list = [
        # price_daily
        "CREATE INDEX IF NOT EXISTS idx_price_daily_date ON price_daily USING BRIN (trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_price_daily_code ON price_daily (ts_code)",
        # daily_prices_panel
        "CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices_panel USING BRIN (trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_daily_prices_code ON daily_prices_panel (ts_code)",
        # index_daily
        "CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily USING BRIN (trade_date)",
        # daily_basic
        "CREATE INDEX IF NOT EXISTS idx_daily_basic_date ON daily_basic USING BRIN (trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_daily_basic_code ON daily_basic (ts_code)",
        # fina_indicator
        "CREATE INDEX IF NOT EXISTS idx_fina_code ON fina_indicator (ts_code)",
        "CREATE INDEX IF NOT EXISTS idx_fina_end_date ON fina_indicator (end_date)",
        # alpha_panel
        "CREATE INDEX IF NOT EXISTS idx_alpha_panel_ticker ON alpha_panel (ticker)",
        "CREATE INDEX IF NOT EXISTS idx_alpha_panel_date ON alpha_panel USING BRIN (as_of)",
        # realized_pnl
        "CREATE INDEX IF NOT EXISTS idx_pnl_date ON realized_pnl (as_of_date)",
        "CREATE INDEX IF NOT EXISTS idx_pnl_ticker ON realized_pnl (ticker)",
        # nav_curve
        "CREATE INDEX IF NOT EXISTS idx_nav_date ON nav_curve USING BRIN (trade_date)",
        # snapshot_daily_basic
        "CREATE INDEX IF NOT EXISTS idx_snap_basic_code ON snapshot_daily_basic (ts_code)",
    ]
    print("\n[INDEXES] 建立索引...")
    with engine.connect() as conn:
        for ddl in ddl_list:
            try:
                conn.execute(text(ddl))
                conn.commit()
                name = ddl.split("EXISTS")[-1].split("ON")[0].strip()
                print(f"  ✓ {name}")
            except Exception as e:
                print(f"  ✗ {ddl[:60]}... → {e}")
    print("[INDEXES] 完成")


def import_alpha_panel():
    """合并所有 alpha_*.parquet + csi300_full_panel → alpha_panel 表。"""
    dfs = []
    features_dir = ROOT / "data" / "features"

    # 1. 历史全量面板
    full_panel = features_dir / "csi300_full_panel.parquet"
    if full_panel.exists():
        df = pd.read_parquet(full_panel).reset_index()
        # 规范化列名：as_of, ticker
        if "as_of" not in df.columns and df.index.name == "as_of":
            df = df.reset_index()
        dfs.append(df)
        print(f"    csi300_full_panel: {df.shape}")

    # 2. 近期每日截面（alpha_YYYY-MM-DD.parquet）
    for p in sorted(features_dir.glob("alpha_20*.parquet")):
        date_str = p.stem.replace("alpha_", "")
        df = pd.read_parquet(p).reset_index()
        if "as_of" not in df.columns:
            df.insert(0, "as_of", date_str)
        if "ticker" not in df.columns and "ts_code" in df.columns:
            df = df.rename(columns={"ts_code": "ticker"})
        dfs.append(df)
        print(f"    {p.name}: {df.shape}")

    if not dfs:
        print("  [SKIP] alpha_panel 无数据文件")
        return 0

    # 合并，填充缺失列（不同版本的 parquet 列集可能不同）
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    # 去重（同一 (as_of, ticker) 保留最新数据）
    combined = combined.drop_duplicates(subset=["as_of", "ticker"], keep="last")
    combined = _normalize_dates(combined)
    for col in combined.columns:
        if hasattr(combined[col], "dtype") and str(combined[col].dtype) == "string":
            combined[col] = combined[col].astype(object)

    rows_in = len(combined)
    print(f"  [alpha_panel] 合并后 {rows_in:,} 行 × {combined.shape[1]} 列，导入...", end="", flush=True)
    t0 = time.time()
    combined.to_sql("alpha_panel", engine, if_exists="replace", index=False, method="multi", chunksize=5000)
    with engine.connect() as conn:
        db_count = conn.execute(text("SELECT COUNT(*) FROM alpha_panel")).scalar()
    print(f" ✓ DB={db_count:,} ({time.time()-t0:.1f}s)")
    return db_count


def import_nav_curve():
    """sim30d/nav/*.json + iteration/*/nav/*.json → nav_curve 表。"""
    rows = []

    def parse_nav_dir(nav_dir: Path, run_id: str):
        for f in sorted(nav_dir.glob("*.json")):
            import json
            data = json.loads(f.read_text(encoding="utf-8"))
            date_str = f.stem  # "20251009"
            rows.append({
                "run_id": run_id,
                "trade_date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                "nav": data.get("nav") or data.get("portfolio_nav"),
                "daily_return": data.get("daily_return") or data.get("portfolio_return"),
                "cumulative_return": data.get("cumulative_return"),
                "benchmark_return": data.get("benchmark_return"),
                "alpha": data.get("alpha"),
                "n_positions": data.get("n_positions"),
            })

    sim_nav = ROOT / "data" / "sim30d" / "nav"
    if sim_nav.exists():
        parse_nav_dir(sim_nav, "sim30d")

    iter_base = ROOT / "data" / "iteration"
    if iter_base.exists():
        for run_dir in iter_base.iterdir():
            nav_dir = run_dir / "nav"
            if nav_dir.is_dir():
                parse_nav_dir(nav_dir, run_dir.name)

    if not rows:
        print("  [SKIP] nav_curve 无数据")
        return 0

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    print(f"  [nav_curve] 导入 {len(df)} 行...", end="", flush=True)
    df.to_sql("nav_curve", engine, if_exists="replace", index=False, method="multi", chunksize=1000)
    with engine.connect() as conn:
        c = conn.execute(text("SELECT COUNT(*) FROM nav_curve")).scalar()
    print(f" ✓ DB={c}")
    return c


def import_snapshot_daily_basic():
    """snapshots/{date}/daily_basic.parquet → snapshot_daily_basic 表。"""
    dfs = []
    snap_root = ROOT / "data" / "snapshots"
    if not snap_root.exists():
        print("  [SKIP] snapshots 目录不存在")
        return 0

    for snap_dir in sorted(snap_root.iterdir()):
        p = snap_dir / "daily_basic.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df.insert(0, "as_of", snap_dir.name)  # e.g. "2026-06-01"
        # 规范 ticker/ts_code 列名
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "ts_code"})
        dfs.append(df)

    if not dfs:
        print("  [SKIP] snapshot_daily_basic 无数据")
        return 0

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined = _normalize_dates(combined)
    combined = combined.drop_duplicates(subset=["as_of", "ts_code"], keep="last")

    # 只保留规划的列（其余列可能字段名不一致）
    keep = ["as_of", "ts_code", "close", "pe_ttm", "pb", "ps_ttm",
            "total_mv", "circ_mv", "turnover_rate", "volume_ratio"]
    keep = [c for c in keep if c in combined.columns]
    combined = combined[keep]

    rows_in = len(combined)
    print(f"  [snapshot_daily_basic] 导入 {rows_in:,} 行...", end="", flush=True)
    t0 = time.time()
    combined.to_sql("snapshot_daily_basic", engine, if_exists="replace", index=False,
                    method="multi", chunksize=5000)
    with engine.connect() as conn:
        c = conn.execute(text("SELECT COUNT(*) FROM snapshot_daily_basic")).scalar()
    print(f" ✓ DB={c:,} ({time.time()-t0:.1f}s)")
    return c


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("QuantMind parquet → PostgreSQL 迁移")
    print("=" * 60)

    results = {}

    # 1. 参考表（小，to_sql 即可）
    print("\n[1] 参考表")
    results["alpha_universe"] = import_table(
        "data/alpha_universe/alpha_universe.parquet",
        "alpha_universe", ["ts_code"],
    )

    # 2. 大时序表（2.27M 行，用 COPY）
    print("\n[2] 大时序表（COPY 模式）")
    results["price_daily"] = import_table(
        "data/raw/alpha_prices_panel.parquet",
        "price_daily", ["ts_code", "trade_date"],
        method="copy",
    )
    results["daily_prices_panel"] = import_table(
        "data/raw/daily_prices_panel.parquet",
        "daily_prices_panel", ["ts_code", "trade_date"],
        method="copy",
    )
    results["index_daily"] = import_table(
        "data/raw/index_daily_panel.parquet",
        "index_daily", ["ts_code", "trade_date"],
    )

    # 3. 基本面表
    print("\n[3] 基本面表")
    results["daily_basic"] = import_table(
        "data/alpha_universe/alpha_daily_basic_combined.parquet",
        "daily_basic", ["ts_code", "trade_date"],
    )
    results["fina_indicator"] = import_table(
        "data/alpha_universe/fina_indicator_combined.parquet",
        "fina_indicator", ["ts_code", "ann_date", "end_date"],
    )

    # 4. Alpha 截面面板
    print("\n[4] Alpha 因子截面")
    print("  [alpha_panel] 合并截面文件...")
    results["alpha_panel"] = import_alpha_panel()

    # 5. Regime 特征
    print("\n[5] Regime + PnL")
    results["regime_features"] = import_table(
        "data/features/regime_features.parquet",
        "regime_features", ["as_of"],
        reset_index=True,
    )
    results["realized_pnl"] = import_table(
        "data/feedback/realized_pnl.parquet",
        "realized_pnl", ["as_of_date", "ticker"],
    )

    # 6. NAV 曲线
    print("\n[6] NAV 曲线")
    results["nav_curve"] = import_nav_curve()

    # 7. Snapshot
    print("\n[7] Snapshot daily basic")
    results["snapshot_daily_basic"] = import_snapshot_daily_basic()

    # 8. 补建索引
    add_indexes()

    # 9. ANALYZE（收集统计信息，让 BRIN 生效）
    print("\n[ANALYZE] 收集统计信息...")
    big_tables = ["price_daily", "daily_prices_panel", "alpha_panel", "daily_basic"]
    with engine.connect() as conn:
        for t in big_tables:
            try:
                conn.execute(text(f"ANALYZE {t}"))
                conn.commit()
                print(f"  ✓ ANALYZE {t}")
            except Exception as e:
                print(f"  ✗ {t}: {e}")

    print("\n" + "=" * 60)
    print("迁移完成，各表行数汇总：")
    for tbl, cnt in results.items():
        print(f"  {tbl:<30} {cnt:>10,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
