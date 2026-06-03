"""E3 验收：取执行层关键统计 + 执行 vs 死扛对比，输出汇总。"""
import sys
sys.path.insert(0, "/home/lenovo/projects/quantmind")

import pandas as pd
from sqlalchemy import text
from app.db.postgres import get_pg_engine
from app.services.data_service import get_data_service

eng = get_pg_engine()
svc = get_data_service()

print("=" * 65)
print("E3 验收汇总")
print("=" * 65)

# 1. simulated_orders 表统计
with eng.connect() as conn:
    stat = pd.read_sql(text(
        "SELECT status, COUNT(*) AS n FROM simulated_orders GROUP BY status"
    ), conn)
    reasons = pd.read_sql(text(
        "SELECT close_reason, COUNT(*) AS n, "
        "ROUND(AVG(pnl_pct)::numeric, 4) AS avg_pnl, "
        "ROUND(AVG(holding_days), 1) AS avg_days "
        "FROM simulated_orders WHERE close_reason IS NOT NULL "
        "GROUP BY close_reason ORDER BY n DESC"
    ), conn)

print("\n[1] simulated_orders 状态分布:")
print(stat.to_string(index=False))

print("\n[2] close_reason 分布（含平均收益 + 持仓天数）:")
print(reasons.to_string(index=False))

# 2. 找出"如果按止损执行"的代表性案例
with eng.connect() as conn:
    stop_loss_cases = pd.read_sql(text("""
        SELECT ticker, name, open_date, close_date, open_price,
               close_price, ROUND(pnl_pct::numeric, 4) AS pnl_pct,
               holding_days,
               ROUND(low_price::numeric, 3) AS reached_low,
               ROUND(((low_price - open_price) / open_price * 100)::numeric, 2) AS worst_dd_pct
        FROM simulated_orders
        WHERE close_reason = 'stop_loss'
        ORDER BY pnl_pct ASC LIMIT 5
    """), conn)
    target_hit_cases = pd.read_sql(text("""
        SELECT ticker, name, open_date, close_date, open_price,
               close_price, ROUND(pnl_pct::numeric, 4) AS pnl_pct,
               holding_days
        FROM simulated_orders
        WHERE close_reason = 'target_hit'
        ORDER BY holding_days ASC LIMIT 5
    """), conn)

print("\n[3] 止损触发案例（如果死扛会更惨）:")
if not stop_loss_cases.empty:
    print(stop_loss_cases.to_string(index=False))
else:
    print("  无")

print("\n[4] 止盈触发案例:")
if not target_hit_cases.empty:
    print(target_hit_cases.to_string(index=False))
else:
    print("  无")

# 3. 执行 vs 死扛对比
print("\n[5] 执行 vs 死扛对比:")
cmp = svc.get_execution_vs_hold_comparison()
if "error" in cmp:
    print(f"  错误: {cmp['error']}")
else:
    exec_ = cmp["execute"]
    hold = cmp["hold_to_expiry"]
    def _fmt_sharpe(v):
        return f"{v:.2f}" if v is not None else "—"
    print(f"  按规则执行: n={exec_['n']} 笔, 胜率={exec_['win_rate']*100:.1f}%, "
          f"平均收益={exec_['avg_return']*100:+.2f}%, 累计={exec_['total_return']*100:+.2f}%, "
          f"MaxDD={exec_['max_dd']*100:.2f}%, "
          f"Sharpe={_fmt_sharpe(exec_['sharpe'])}, "
          f"平均天数={exec_['avg_holding_days']:.1f}")
    print(f"  死扛 63 天: n={hold['n']} 笔, 胜率={hold['win_rate']*100:.1f}%, "
          f"平均收益={hold['avg_return']*100:+.2f}%, 累计={hold['total_return']*100:+.2f}%, "
          f"MaxDD={hold['max_dd']*100:.2f}%, "
          f"Sharpe={_fmt_sharpe(hold['sharpe'])}")
    diff_ret = exec_['total_return'] - hold['total_return']
    diff_dd = exec_['max_dd'] - hold['max_dd']
    print(f"\n  💡 结论:")
    print(f"     累计收益差: {diff_ret*100:+.2f}% "
          f"({'执行更高' if diff_ret > 0 else '死扛更高'})")
    print(f"     MaxDD 差异: {diff_dd*100:+.2f}% "
          f"({'执行 DD 更小' if diff_dd > 0 else '死扛 DD 更小'})")
    print(f"     平均持仓: 执行 {exec_['avg_holding_days']:.1f} 天 "
          f"vs 死扛 63 天 — "
          f"执行模式资金周转快 {(63 - exec_['avg_holding_days'])/63*100:.1f}%")
