"""冒烟测试：验证 ops 模块 + 实际 PG/Mongo ping 工作正常。"""
import sys
sys.path.insert(0, "/home/lenovo/projects/quantmind")

from app.ops.db_health import (
    pg_ping, mongo_ping, read_env_value, write_env_value,
    dual_write_stats, read_failures, read_audit, overall_db_status,
)

print("=" * 55)
print("1. PG ping")
pg = pg_ping()
print(f"   ok={pg['ok']}, tables={pg.get('n_tables')}, rows={pg.get('total_rows')}")
assert pg["ok"], pg.get("error")

print("\n2. Mongo ping")
mg = mongo_ping()
print(f"   ok={mg['ok']}, collections={mg.get('n_collections')}, docs={mg.get('total_docs')}")
assert mg["ok"], mg.get("error")

print("\n3. env 读取")
backend = read_env_value("DATA_BACKEND")
wmode = read_env_value("WRITE_MODE")
print(f"   DATA_BACKEND={backend}, WRITE_MODE={wmode}")

print("\n4. 双写统计（最近 7 天）")
stats = dual_write_stats(days=7)
print(f"   ok={stats['total_ok']}, fail={stats['total_fail']}, "
      f"rate={stats.get('success_rate')}")
print(f"   by_writer: {[r['writer'] for r in stats['by_writer']]}")

print("\n5. 最近 24h 失败")
fails = read_failures(hours=24)
print(f"   {len(fails)} 条")

print("\n6. overall_db_status")
overall = overall_db_status()
print(f"   keys: {sorted(overall.keys())}")
print(f"   backend={overall['data_backend']}, write={overall['write_mode']}, "
      f"fails={overall['failures_24h']}")

print("\n=" * 50)
print("所有冒烟测试通过 ✓")
