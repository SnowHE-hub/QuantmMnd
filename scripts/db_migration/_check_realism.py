"""验证修正后的成交价不再整齐，分布有真实差异。"""
import sys
sys.path.insert(0, "/home/lenovo/projects/quantmind")

import pandas as pd
from sqlalchemy import text
from app.db.postgres import get_pg_engine
from app.services.data_service import get_data_service

eng = get_pg_engine()

print("=" * 70)
print("修正后的回填数据分布")
print("=" * 70)

with eng.connect() as conn:
    df = pd.read_sql(text("""
        SELECT close_reason,
               COUNT(*) AS n,
               ROUND(AVG(pnl_pct)::numeric, 4) AS avg_pnl,
               ROUND(STDDEV(pnl_pct)::numeric, 4) AS std_pnl,
               ROUND(MIN(pnl_pct)::numeric, 4) AS min_pnl,
               ROUND(MAX(pnl_pct)::numeric, 4) AS max_pnl
        FROM simulated_orders
        WHERE close_reason IS NOT NULL
        GROUP BY close_reason
        ORDER BY n DESC
    """), conn)
print(df.to_string(index=False))

# 具体看 stop_loss 和 target_hit 的明细
for reason in ("stop_loss", "target_hit", "trailing_stop"):
    print(f"\n[{reason}] 明细:")
    with eng.connect() as conn:
        df = pd.read_sql(text(f"""
            SELECT ticker, name, open_date, close_date,
                   ROUND(open_price::numeric, 3) AS open_px,
                   ROUND(close_price::numeric, 3) AS close_px,
                   ROUND(stop_loss_price::numeric, 3) AS sl_threshold,
                   ROUND(target_price::numeric, 3) AS tgt_threshold,
                   ROUND(pnl_pct::numeric, 4) AS pnl
            FROM simulated_orders
            WHERE close_reason = '{reason}'
            ORDER BY pnl
        """), conn)
    print(df.to_string(index=False) if not df.empty else "  无")

# 新 NAV 对比
print("\n" + "=" * 70)
print("修正后的执行 vs 死扛 NAV 对比")
print("=" * 70)
svc = get_data_service()
cmp = svc.get_execution_vs_hold_comparison()
if "error" in cmp:
    print(f"错误: {cmp['error']}")
else:
    e = cmp["execute"]
    h = cmp["hold_to_expiry"]
    print(f"\n推荐池规模 N = {cmp['n_total']}（每笔分 1/N 资金）")
    print(f"\n按规则执行:")
    print(f"  样本数:    {e['n']}")
    print(f"  胜率:      {e['win_rate']*100:.1f}%")
    print(f"  平均收益:  {e['avg_return']*100:+.2f}%")
    print(f"  组合 NAV:  {1 + e['total_return']:.4f}  (累计 {e['total_return']*100:+.2f}%)")
    print(f"  MaxDD:     {e['max_dd']*100:.2f}%")
    sh = f"{e['sharpe']:.2f}" if e['sharpe'] is not None else "—"
    print(f"  Sharpe:    {sh}")
    print(f"  平均持仓:  {e['avg_holding_days']:.1f} 天")

    print(f"\n死扛 63 天:")
    print(f"  样本数:    {h['n']}")
    print(f"  胜率:      {h['win_rate']*100:.1f}%")
    print(f"  平均收益:  {h['avg_return']*100:+.2f}%")
    print(f"  组合 NAV:  {1 + h['total_return']:.4f}  (累计 {h['total_return']*100:+.2f}%)")
    print(f"  MaxDD:     {h['max_dd']*100:.2f}%")
    sh = f"{h['sharpe']:.2f}" if h['sharpe'] is not None else "—"
    print(f"  Sharpe:    {sh}")

    print(f"\n触发次数: 止损/追踪止损 {cmp['exec_stop_count']}, "
          f"止盈 {cmp['exec_target_count']}, 到期 {cmp['exit_reasons'].get('time_expired', 0)}")

    diff_ret = e['total_return'] - h['total_return']
    diff_dd = e['max_dd'] - h['max_dd']
    print(f"\n💡 结论:")
    print(f"   累计收益差: {diff_ret*100:+.2f}% "
          f"({'执行更高' if diff_ret > 0 else '死扛更高'})")
    print(f"   MaxDD 差:   {diff_dd*100:+.2f}pp "
          f"({'执行 DD 更小' if diff_dd > 0 else '死扛 DD 更小'})")
    if abs(h['total_return']) > 1e-6:
        ratio = abs(e['total_return']) / abs(h['total_return']) if h['total_return'] != 0 else 0
        print(f"   收益比率:   |exec/hold| = {ratio:.2f}x  (合理范围 < 2x)")
