"""
生成一些审计日志（成功 + 失败混合），让监控页面有数据可视化。
仅用于开发演示，不影响生产业务。
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
AUDIT = LOG_DIR / "db_write_audit.log"
FAIL = LOG_DIR / "db_write_failures.log"

# 写一些近 7 天的样例（每天每个 writer 1-3 条成功）
now = datetime.now()
writers = ["recommendations", "realized_pnl", "positions",
           "agent_analysis", "loss_signals", "strategy_config"]

with AUDIT.open("a", encoding="utf-8") as f:
    for d in range(7):
        for w in writers:
            for k in range(1 + (d % 3)):
                ts = (now - timedelta(days=d, hours=k)).isoformat(timespec="seconds")
                info = {
                    "recommendations": f"date=2026-{6-d:02d}-01 top10=10",
                    "realized_pnl": "rows=80 full_replace=True",
                    "positions": "upserted=20 modified=0",
                    "agent_analysis": f"date=2026-{6-d:02d}-01 upserted=10",
                    "loss_signals": "_id=2026-06-01",
                    "strategy_config": "version=v2",
                }[w]
                f.write(f"{ts}\t{w}\tOK\t{info}\n")

# 写 2 条历史失败（演示用，时间在 8h 和 3 天前）
with FAIL.open("a", encoding="utf-8") as f:
    f.write(
        f"{(now - timedelta(hours=8)).isoformat(timespec='seconds')}\t"
        f"agent_analysis\tConnectionRefusedError\t"
        f"could not connect to MongoDB\tdate=2026-05-30\n"
    )
    f.write(
        f"{(now - timedelta(days=3)).isoformat(timespec='seconds')}\t"
        f"realized_pnl\tOperationalError\t"
        f"PG temporarily unavailable\trows=80\n"
    )

print(f"已写入 {AUDIT}")
print(f"已写入 {FAIL}")

# 验证
from app.ops.db_health import dual_write_stats, read_failures
stats = dual_write_stats(days=7)
print(f"\n统计：ok={stats['total_ok']}, fail={stats['total_fail']}, "
      f"rate={stats['success_rate']:.4f}")
print(f"by_writer: {[(r['writer'], r['ok'], r['fail']) for r in stats['by_writer']]}")
