"""
生成"双写审计/失败"样例日志，仅供本地演示监控页用。

⚠️ 安全护栏（2026-06-03 加固）：
  - 默认写到 logs/demo/ 下的独立文件，**绝不**碰真实的
    logs/db_write_audit.log / logs/db_write_failures.log
    （监控页和告警据此判断"生产是否真的失败"，被合成数据污染过一次）。
  - 必须显式传 --force-real 才会写真实日志，并打印醒目警告。

用法：
  python scripts/db_migration/_seed_audit_log.py            # 写 logs/demo/（安全）
  python scripts/db_migration/_seed_audit_log.py --force-real  # 写真实日志（危险，需手动确认）
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--force-real", action="store_true",
                    help="写入真实 logs/ 而非 logs/demo/（会污染监控，慎用）")
args = parser.parse_args()

if args.force_real:
    LOG_DIR = ROOT / "logs"
    print("⚠️  --force-real：将写入真实监控日志，这会让监控页出现合成数据！")
else:
    LOG_DIR = ROOT / "logs" / "demo"

LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT = LOG_DIR / "db_write_audit.log"
FAIL = LOG_DIR / "db_write_failures.log"

# 写一些近 7 天的样例（每天每个 writer 1-3 条成功）
now = datetime.now()
writers = ["recommendations", "realized_pnl", "positions",
           "agent_analysis", "loss_signals", "strategy_config"]


def _demo_date(days_ago: int) -> str:
    """近 7 天里一个合法日期（修掉旧版 f'2026-{6-d:02d}-01' 会算出 2026-00-01 的 bug）。"""
    return (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")


with AUDIT.open("a", encoding="utf-8") as f:
    for d in range(7):
        for w in writers:
            for k in range(1 + (d % 3)):
                ts = (now - timedelta(days=d, hours=k)).isoformat(timespec="seconds")
                info = {
                    "recommendations": f"date={_demo_date(d)} top10=10",
                    "realized_pnl": "rows=80 full_replace=True",
                    "positions": "upserted=20 modified=0",
                    "agent_analysis": f"date={_demo_date(d)} upserted=10",
                    "loss_signals": f"_id={_demo_date(d)}",
                    "strategy_config": "version=v2",
                }[w]
                f.write(f"{ts}\t{w}\tOK\t{info}\n")

# 写 2 条历史失败（演示用）
with FAIL.open("a", encoding="utf-8") as f:
    f.write(
        f"{(now - timedelta(hours=8)).isoformat(timespec='seconds')}\t"
        f"agent_analysis\tConnectionRefusedError\t"
        f"could not connect to MongoDB\tdate={_demo_date(4)}\n"
    )
    f.write(
        f"{(now - timedelta(days=3)).isoformat(timespec='seconds')}\t"
        f"realized_pnl\tOperationalError\t"
        f"PG temporarily unavailable\trows=80\n"
    )

print(f"已写入 {AUDIT}")
print(f"已写入 {FAIL}")
if not args.force_real:
    print("（安全模式：写到 logs/demo/，未触碰真实监控日志）")
