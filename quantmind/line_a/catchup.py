"""quantmind.line_a.catchup — 停机补跑检查（03_final_plan_v2 第 3.1、12 节）.

编排器启动时调用：
- 读 Windows 侧监督脚本的心跳（D:\\qm_ops\\logs\\heartbeat.json，WSL 里是 /mnt/d/qm_ops/logs/heartbeat.json）；
- 找出最后一份 data_ready 的快照清单之后、到今天为止漏掉的交易日。
补跑只补数据和研究；漏掉的交易意向按"错过"记录，不按原时间执行。
"""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

HEARTBEAT = Path("/mnt/d/qm_ops/logs/heartbeat.json")


def read_heartbeat(path: Path = HEARTBEAT, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    if not path.exists():
        return {"present": False}
    hb = json.loads(path.read_text(encoding="utf-8-sig"))
    ts = datetime.fromisoformat(hb["utc"].replace("Z", "+00:00"))
    return {"present": True, "utc": hb["utc"], "age_s": (now - ts).total_seconds(), "cron": hb.get("cron"),
            "keepalive_pid": hb.get("keepalive_pid")}


def last_ready_snapshot(snap_root: Path) -> date | None:
    """data/line_a_v2/snapshots/<as_of>/snapshot_*.json 中，notes 含 data_ready=True 的最晚 as_of。"""
    best = None
    for f in Path(snap_root).glob("*/snapshot_*.json"):
        m = json.loads(f.read_text(encoding="utf-8"))
        if "data_ready=True" in m.get("notes", []):
            d = date.fromisoformat(m["as_of"])
            best = d if best is None or d > best else best
    return best


def missed_trading_days(trade_days: list[date], last_done: date | None, today: date) -> list[date]:
    """last_done 之后、today 及以前的交易日（last_done 为 None 时只返回 today 当天，如果它是交易日）。"""
    days = sorted(trade_days)
    if last_done is None:
        return [d for d in days if d == today]
    return [d for d in days if last_done < d <= today]
