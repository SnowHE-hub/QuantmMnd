"""E3 Step 0：盘点现有执行能力，确认 Agent 输出有没有 target_price/stop_loss_price。"""
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/home/lenovo/projects/quantmind")

print("=" * 60)
print("Step 0: 盘点执行能力")
print("=" * 60)

# 1. forward_positions
fp = ROOT / "data" / "paper_trading" / "forward_positions.json"
fwd = json.loads(fp.read_text(encoding="utf-8"))
positions = fwd if isinstance(fwd, list) else fwd.get("positions", [])
print(f"\n[1] forward_positions: {len(positions)} 只")
if positions:
    print(f"    字段: {list(positions[0].keys())}")
    open_count = sum(1 for p in positions if p.get("status") == "OPEN")
    print(f"    OPEN: {open_count}, 其他: {len(positions) - open_count}")
    print(f"    样例: {positions[0]}")

# 2. realized_pnl
pnl_path = ROOT / "data" / "feedback" / "realized_pnl.parquet"
pnl = pd.read_parquet(pnl_path)
print(f"\n[2] realized_pnl: {len(pnl)} 条")
print(f"    字段: {list(pnl.columns)}")
print(f"    exit_reason 分布:")
if "exit_reason" in pnl.columns:
    print(pnl["exit_reason"].value_counts().to_string())
else:
    print("    无 exit_reason 字段 → 历史数据全部按时间到期")
print(f"    holding_days: min={pnl['holding_days'].min()}, "
      f"max={pnl['holding_days'].max()}, mean={pnl['holding_days'].mean():.1f}")
print(f"    actual_return_63d: mean={pnl['actual_return_63d'].mean()*100:.2f}%, "
      f"std={pnl['actual_return_63d'].std()*100:.2f}%")
print(f"    最差: {pnl['actual_return_63d'].min()*100:.2f}%")
print(f"    最佳: {pnl['actual_return_63d'].max()*100:.2f}%")

# 3. Strategy Agent 输出
strat_files = sorted(glob.glob(
    str(ROOT / "reports/investment_pipeline/*/strategies.json")), reverse=True)
print(f"\n[3] strategies.json 文件数: {len(strat_files)}")
if strat_files:
    s_path = strat_files[0]
    print(f"    最新: {s_path}")
    strats = json.loads(Path(s_path).read_text(encoding="utf-8"))
    if isinstance(strats, list) and strats:
        print(f"    含 {len(strats)} 只股票")
        sample = strats[0]
        print(f"    字段: {list(sample.keys())}")
        has_tp = sum(1 for x in strats if x.get("target_price_3m"))
        has_sl = sum(1 for x in strats if x.get("stop_loss_price"))
        has_pos = sum(1 for x in strats if x.get("position_size"))
        has_horizon = sum(1 for x in strats if x.get("holding_horizon"))
        print(f"    target_price_3m: {has_tp}/{len(strats)}")
        print(f"    stop_loss_price: {has_sl}/{len(strats)}")
        print(f"    position_size:   {has_pos}/{len(strats)}")
        print(f"    holding_horizon: {has_horizon}/{len(strats)}")
        # 样例
        print(f"    样例 (ticker={sample.get('ticker')}):")
        for k in ["rating", "composite_signal", "target_price_1m",
                  "target_price_3m", "stop_loss_price",
                  "position_size", "holding_horizon"]:
            print(f"      {k}: {sample.get(k)}")

# 4. 检查推荐里有没有 target/stop_loss
recs_path = ROOT / "data" / "recommendations"
rec_files = sorted(recs_path.glob("*/top10.json"), reverse=True)
print(f"\n[4] recommendations/*/top10.json: {len(rec_files)} 个")
if rec_files:
    latest = json.loads(rec_files[0].read_text(encoding="utf-8"))
    top10 = latest.get("top10", [])
    if top10:
        item = top10[0]
        has_tp = sum(1 for x in top10 if x.get("target_price_3m"))
        has_sl = sum(1 for x in top10 if x.get("stop_loss_price"))
        print(f"    {rec_files[0].parent.name}: {len(top10)} 只")
        print(f"    target_price_3m: {has_tp}, stop_loss_price: {has_sl}")
        print(f"    字段: {list(item.keys())}")

# 5. 当前 PG simulated_orders 是否存在
print(f"\n[5] PG: simulated_orders 表是否已存在?")
import sys
sys.path.insert(0, str(ROOT))
try:
    from sqlalchemy import text
    from app.db.postgres import get_pg_engine
    eng = get_pg_engine()
    with eng.connect() as conn:
        exists = conn.execute(text(
            "SELECT EXISTS (SELECT FROM information_schema.tables "
            "WHERE table_name='simulated_orders')"
        )).scalar()
        print(f"    simulated_orders 存在: {exists}")
        if exists:
            cnt = conn.execute(text("SELECT COUNT(*) FROM simulated_orders")).scalar()
            print(f"    当前行数: {cnt}")
except Exception as e:
    print(f"    查询失败: {e}")
