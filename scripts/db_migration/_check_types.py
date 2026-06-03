import sys
sys.path.insert(0, "/home/lenovo/projects/quantmind")
from sqlalchemy import text
from app.db.postgres import get_pg_engine

eng = get_pg_engine()
with eng.connect() as c:
    df = c.execute(text("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_name IN ('daily_prices_panel', 'price_daily', 'realized_pnl')
          AND column_name IN ('trade_date', 'as_of_date', 'entry_date', 'exit_date')
        ORDER BY table_name, column_name
    """)).fetchall()
    for r in df:
        print(r)
