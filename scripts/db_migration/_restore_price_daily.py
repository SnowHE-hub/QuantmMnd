"""一次性恢复 price_daily 表（alpha_prices_panel.parquet --COPY--> PG）。

复刻 02_import_pg.py 对 price_daily 的导入逻辑；导入后 ANALYZE，
让 pg_stat_user_tables.n_live_tup 准确（避免之前误读估算值=0 的坑）。
"""
import io
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.db.postgres import get_pg_engine

NULL = "\\N"
eng = get_pg_engine()

df = pd.read_parquet(ROOT / "data" / "raw" / "alpha_prices_panel.parquet")
for c in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = df[c].dt.date
print(f"读取 alpha_prices_panel.parquet: {df.shape}")

df.head(0).to_sql("price_daily", eng, if_exists="replace", index=False)  # 建空表(schema 来自 dtypes)
buf = io.StringIO()
df.to_csv(buf, index=False, header=False, na_rep=NULL)
buf.seek(0)
raw = eng.raw_connection()
try:
    cur = raw.cursor()
    cur.copy_expert(
        f"COPY price_daily ({','.join(df.columns)}) FROM STDIN WITH (FORMAT CSV, NULL '{NULL}')",
        buf,
    )
    raw.commit()
finally:
    raw.close()

with eng.begin() as conn:
    conn.execute(text("ANALYZE price_daily"))
with eng.connect() as conn:
    real = conn.execute(text("SELECT COUNT(*) FROM price_daily")).scalar()
    est = conn.execute(text(
        "SELECT n_live_tup FROM pg_stat_user_tables WHERE relname='price_daily'")).scalar()
print(f"price_daily 已恢复: COUNT={real}  n_live_tup(已ANALYZE)={est}")
print("--- 当前 PG public 表 ---")
with eng.connect() as conn:
    for (n,) in conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")):
        c = conn.execute(text(f'SELECT COUNT(*) FROM "{n}"')).scalar()
        print(f"  {n:24s} {c:>9}")
