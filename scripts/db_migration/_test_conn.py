"""临时连接测试脚本（迁移完成后可删除）。"""
import sys
sys.path.insert(0, "/home/lenovo/projects/quantmind")

from app.db.postgres import get_pg_engine
from app.db.mongo import get_mongo_db
from sqlalchemy import text

eng = get_pg_engine()
with eng.connect() as c:
    val = c.execute(text("SELECT version()")).scalar()
print(f"PG OK: {val[:40]}")

db = get_mongo_db()
dbs = db.client.list_database_names()
print(f"Mongo OK: db={db.name}, all_dbs={dbs}")
