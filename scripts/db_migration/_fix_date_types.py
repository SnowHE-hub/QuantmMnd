"""把价格表/PnL 表里 trade_date / entry_date / exit_date 从 text 改成 date 类型。

PG 会自动把 'YYYY-MM-DD' 字符串转 date，不需要重新导入数据。
"""
import sys
sys.path.insert(0, "/home/lenovo/projects/quantmind")
from sqlalchemy import text
from app.db.postgres import get_pg_engine

ALTERS = [
    "ALTER TABLE daily_prices_panel ALTER COLUMN trade_date TYPE date USING trade_date::date",
    "ALTER TABLE price_daily         ALTER COLUMN trade_date TYPE date USING trade_date::date",
    "ALTER TABLE index_daily         ALTER COLUMN trade_date TYPE date USING trade_date::date",
    "ALTER TABLE daily_basic         ALTER COLUMN trade_date TYPE date USING trade_date::date",
    "ALTER TABLE realized_pnl        ALTER COLUMN entry_date  TYPE date USING entry_date::date",
    "ALTER TABLE realized_pnl        ALTER COLUMN exit_date   TYPE date USING exit_date::date",
    "ALTER TABLE fina_indicator      ALTER COLUMN ann_date    TYPE date USING ann_date::date",
    "ALTER TABLE fina_indicator      ALTER COLUMN end_date    TYPE date USING end_date::date",
    "ALTER TABLE alpha_panel         ALTER COLUMN as_of       TYPE date USING as_of::date",
    "ALTER TABLE regime_features     ALTER COLUMN as_of       TYPE date USING as_of::date",
    "ALTER TABLE snapshot_daily_basic ALTER COLUMN as_of      TYPE date USING as_of::date",
]

eng = get_pg_engine()
with eng.begin() as conn:
    for sql in ALTERS:
        try:
            conn.execute(text(sql))
            print(f"  ✓ {sql[:70]}")
        except Exception as e:
            print(f"  ✗ {sql[:70]}\n     {str(e)[:120]}")
print("\nANALYZE 各表...")
with eng.begin() as conn:
    for tbl in ("daily_prices_panel", "price_daily", "index_daily",
                "daily_basic", "realized_pnl", "alpha_panel"):
        try:
            conn.execute(text(f"ANALYZE {tbl}"))
            print(f"  ✓ ANALYZE {tbl}")
        except Exception as e:
            print(f"  ✗ {tbl}: {e}")
