"""
json → MongoDB 批量导入脚本
用法: conda run -n quantmind python scripts/db_migration/03_import_mongo.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pymongo import UpdateOne
from app.db.mongo import get_mongo_db

db = get_mongo_db()


def bulk_upsert(collection_name: str, ops: list) -> dict:
    if not ops:
        return {"upserted": 0, "modified": 0}
    coll = db[collection_name]
    result = coll.bulk_write(ops, ordered=False)
    return {"upserted": result.upserted_count, "modified": result.modified_count}


def import_recommendations():
    """recommendations/{date}/top10.json 和 {date}.json → recommendations。
    两种格式都处理，以子目录版本为主（覆盖 flat 版本）。
    """
    seen: dict[str, dict] = {}
    rec_dir = ROOT / "data" / "recommendations"

    # 1. 先处理 flat 文件（优先级低）
    for flat_path in sorted(rec_dir.glob("*.json")):
        date_str = flat_path.stem  # "2024-06-28"
        data = json.loads(flat_path.read_text(encoding="utf-8"))
        if data.get("top10"):
            seen[date_str] = {"as_of": date_str, **data}

    # 2. 再处理子目录版本（覆盖 flat，优先级高）
    for top10_path in sorted(rec_dir.glob("*/top10.json")):
        date_str = top10_path.parent.name
        data = json.loads(top10_path.read_text(encoding="utf-8"))
        seen[date_str] = {"as_of": date_str, **data}

    ops = [
        UpdateOne({"_id": date_str}, {"$set": doc}, upsert=True)
        for date_str, doc in seen.items()
    ]
    r = bulk_upsert("recommendations", ops)
    print(f"  [recommendations] upserted={r['upserted']} modified={r['modified']} total={len(ops)}")


def import_positions():
    """forward_positions.json → positions（每条 position 一个 doc）。"""
    path = ROOT / "data" / "paper_trading" / "forward_positions.json"
    if not path.exists():
        print("  [positions] SKIP: 文件不存在")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    positions = data.get("positions", [])
    ops = []
    for pos in positions:
        key = {"as_of": pos.get("as_of"), "ticker": pos.get("ticker")}
        ops.append(UpdateOne(key, {"$set": pos}, upsert=True))
    r = bulk_upsert("positions", ops)
    print(f"  [positions] upserted={r['upserted']} modified={r['modified']} total={len(ops)}")


def import_agent_analysis():
    """reports/investment_pipeline/{date}/strategies.json → agent_analysis。"""
    ops = []
    pipeline_dir = ROOT / "reports" / "investment_pipeline"
    if not pipeline_dir.exists():
        print("  [agent_analysis] SKIP: 目录不存在")
        return
    for strat_path in sorted(pipeline_dir.glob("*/strategies.json")):
        date_str = strat_path.parent.name
        strategies = json.loads(strat_path.read_text(encoding="utf-8"))
        if not isinstance(strategies, list):
            continue
        for s in strategies:
            ticker = s.get("ticker", "")
            doc_id = f"{date_str}_{ticker}"
            doc = {"date": date_str, **s}
            ops.append(UpdateOne({"_id": doc_id}, {"$set": doc}, upsert=True))
    r = bulk_upsert("agent_analysis", ops)
    print(f"  [agent_analysis] upserted={r['upserted']} modified={r['modified']} total={len(ops)}")


def import_strategy_config():
    """strategy_config_v2.json → strategy_config（单文档，_id='v2'）。"""
    path = ROOT / "data" / "paper_trading" / "strategy_config_v2.json"
    if not path.exists():
        print("  [strategy_config] SKIP: 文件不存在")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    coll = db["strategy_config"]
    coll.replace_one({"_id": "v2"}, {"_id": "v2", **data}, upsert=True)
    print(f"  [strategy_config] upserted (version=v2)")


def import_loss_signals():
    """loss_signals_v4/{latest,action_plan,factor_health,pending_validation}.json
    → loss_signals（每天一个 doc）。
    """
    ops = []
    ls_dir = ROOT / "data" / "loss_signals_v4"
    if not ls_dir.exists():
        print("  [loss_signals] SKIP: 目录不存在")
        return

    # latest.json 用 run_ts 的日期作为 _id
    latest_path = ls_dir / "latest.json"
    if latest_path.exists():
        data = json.loads(latest_path.read_text(encoding="utf-8"))
        run_ts = data.get("run_ts", "")
        date_id = run_ts[:10] if run_ts else "latest"
        ops.append(UpdateOne({"_id": date_id}, {"$set": {**data, "source": "latest"}}, upsert=True))

    # action_plan / factor_health / pending_validation → 子文档合并
    extras = {}
    for fname in ["action_plan.json", "factor_health.json", "pending_validation.json"]:
        p = ls_dir / fname
        if p.exists():
            extras[fname.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))

    if extras and ops:
        ops[-1] = UpdateOne(
            {"_id": ops[-1]._filter["_id"]},
            {"$set": {**extras}},
            upsert=True,
        )

    r = bulk_upsert("loss_signals", ops)
    print(f"  [loss_signals] upserted={r['upserted']} modified={r['modified']}")


def import_sim_daily():
    """sim30d/daily/*.json + iteration/*/daily/*.json → sim_daily。"""
    ops = []

    def process_dir(daily_dir: Path, run_id: str):
        for f in sorted(daily_dir.glob("*.json")):
            date_str = f.stem
            data = json.loads(f.read_text(encoding="utf-8"))
            doc_id = f"{run_id}_{date_str}"
            doc = {"run_id": run_id, "date": date_str, **data}
            ops.append(UpdateOne({"_id": doc_id}, {"$set": doc}, upsert=True))

    sim_daily = ROOT / "data" / "sim30d" / "daily"
    if sim_daily.exists():
        process_dir(sim_daily, "sim30d")

    iter_base = ROOT / "data" / "iteration"
    if iter_base.exists():
        for run_dir in iter_base.iterdir():
            daily_dir = run_dir / "daily"
            if daily_dir.is_dir():
                process_dir(daily_dir, run_dir.name)

    r = bulk_upsert("sim_daily", ops)
    print(f"  [sim_daily] upserted={r['upserted']} modified={r['modified']} total={len(ops)}")


def import_watchlist_scores():
    """watchlist/scores/{date}.json → watchlist_scores。"""
    ops = []
    scores_dir = ROOT / "data" / "watchlist" / "scores"
    if not scores_dir.exists():
        print("  [watchlist_scores] SKIP: 目录不存在")
        return
    for f in sorted(scores_dir.glob("*.json")):
        date_str = f.stem
        data = json.loads(f.read_text(encoding="utf-8"))
        doc = {"date": date_str, **data}
        ops.append(UpdateOne({"_id": date_str}, {"$set": doc}, upsert=True))
    r = bulk_upsert("watchlist_scores", ops)
    print(f"  [watchlist_scores] upserted={r['upserted']} modified={r['modified']} total={len(ops)}")


def create_indexes():
    """建立 MongoDB 查询索引。"""
    idx_plan = {
        "recommendations": [
            [("as_of", 1)],
            [("top10.ticker", 1)],
        ],
        "positions": [
            [("status", 1)],
            [("as_of", 1), ("ticker", 1)],
            [("ticker", 1)],
        ],
        "agent_analysis": [
            [("date", 1), ("ticker", 1)],
            [("ticker", 1), ("date", -1)],
        ],
        "loss_signals": [
            [("run_ts", -1)],
        ],
        "sim_daily": [
            [("run_id", 1), ("date", 1)],
        ],
        "watchlist_scores": [
            [("date", -1)],
        ],
    }
    print("  建立 MongoDB 索引...")
    for coll_name, indexes in idx_plan.items():
        coll = db[coll_name]
        for keys in indexes:
            coll.create_index(keys)
        print(f"    ✓ {coll_name}: {len(indexes)} 个索引")


def main():
    print("=" * 60)
    print("QuantMind json → MongoDB 迁移")
    print("=" * 60)

    print("\n[1] 推荐数据")
    import_recommendations()

    print("\n[2] 持仓")
    import_positions()

    print("\n[3] 6-Agent 分析")
    import_agent_analysis()

    print("\n[4] 策略配置")
    import_strategy_config()

    print("\n[5] Loss signals")
    import_loss_signals()

    print("\n[6] 模拟/回测日报")
    import_sim_daily()

    print("\n[7] 自选股评分")
    import_watchlist_scores()

    print("\n[8] 建立索引")
    create_indexes()

    print("\n" + "=" * 60)
    print("Mongo 迁移完成，各 collection 文档数：")
    for coll_name in ["recommendations", "positions", "agent_analysis",
                       "strategy_config", "loss_signals", "sim_daily", "watchlist_scores"]:
        cnt = db[coll_name].count_documents({})
        print(f"  {coll_name:<25} {cnt:>6,} docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
