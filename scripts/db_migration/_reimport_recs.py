"""补充导入 recommendations（含 flat json 格式）。"""
import sys
sys.path.insert(0, "/home/lenovo/projects/quantmind")

import importlib.util, types

# 直接 exec 避免模块名含数字的问题
spec = importlib.util.spec_from_file_location(
    "mongo_import",
    "/home/lenovo/projects/quantmind/scripts/db_migration/03_import_mongo.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.import_recommendations()
print("Done. Checking doc count...")
from app.db.mongo import get_mongo_db
db = get_mongo_db()
print(f"recommendations: {db['recommendations'].count_documents({})} docs")
