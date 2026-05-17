"""模拟盘归因分析脚本."""
import pandas as pd
import numpy as np
from scipy import stats
import json

df = pd.read_parquet('data/paper_trading/positions.parquet')

print('=== 每期等权组合绩效 ===')
header = '  日期            1w        2w       21d        3m    备注'
print(header)
print('  ' + '-'*65)
for as_of in sorted(df['as_of'].unique()):
    grp = df[df['as_of'] == as_of]
    r1w  = grp['return_1w'].mean()
    r2w  = grp['return_2w'].mean()
    r21d = grp['return_21d'].mean()
    r3m  = grp['return_3m'].mean()
    mark = '<- Q4 2025起' if as_of >= pd.Timestamp('2025-09-30') else ''
    def fmt(v):
        return '%+.2f%%' % (v*100) if pd.notna(v) else '  N/A '
    print('  %-15s %8s %8s %8s %9s  %s' % (
        str(as_of.date()), fmt(r1w), fmt(r2w), fmt(r21d), fmt(r3m), mark))

print('\n=== 持有期 IR + 期胜率 ===')
for hz in ['1w', '2w', '21d', '3m']:
    col = 'return_' + hz
    prets = df.groupby('as_of')[col].mean().dropna()
    if len(prets) == 0:
        continue
    ir = prets.mean() / prets.std() if prets.std() > 0 else 0
    win = (prets > 0).mean()
    q4 = prets[prets.index >= pd.Timestamp('2025-09-30')]
    q4_mean = q4.mean() if len(q4) > 0 else float('nan')
    print('  %s: IR=%+.3f  期胜率=%.1f%%  均值=%+.3f  Q4+均值=%+.3f  (n=%d)' % (
        hz, ir, win*100, prets.mean(), q4_mean, len(prets)))

print('\n=== 精选 Top-3 vs 全部 Top-10 ===')
for hz in ['21d', '3m']:
    col = 'return_' + hz
    top3  = df[df['predicted_rank'] <= 3][col].dropna()
    top10 = df[col].dropna()
    btm5  = df[df['predicted_rank'] >= 6][col].dropna()
    print('  %s: Top3=%+.3f  Top10=%+.3f  Btm5=%+.3f  Top3超额=%+.4f' % (
        hz, top3.mean(), top10.mean(), btm5.mean(), top3.mean()-top10.mean()))

print('\n=== 因子方向（vs 63d 实际收益）===')
rpnl = pd.read_parquet('data/feedback/realized_pnl.parquet')
if 'key_factors' in rpnl.columns:
    kf = rpnl['key_factors'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    kf_df = pd.DataFrame(kf.tolist(), index=rpnl.index)
    rows = []
    for fc in kf_df.columns:
        sub = pd.concat([kf_df[fc], rpnl['actual_return_63d']], axis=1).dropna()
        if len(sub) < 10:
            continue
        ic, p = stats.spearmanr(sub[fc], sub['actual_return_63d'])
        direction = '反转信号' if ic < -0.10 else ('正向信号' if ic > 0.10 else '中性')
        rows.append((fc, ic, p, direction, len(sub)))
    rows.sort(key=lambda x: -abs(x[1]))
    for fc, ic, p, direction, n in rows:
        sig = '**' if p < 0.05 else ('*' if p < 0.10 else '  ')
        print('  %-25s IC=%+.3f  %-8s  p=%.3f  n=%d  %s' % (fc, ic, direction, p, n, sig))

print('\n=== Q4 2025起持仓股票清单 ===')
q4_picks = df[df['as_of'] >= pd.Timestamp('2025-09-30')][
    ['as_of', 'ticker', 'predicted_rank', 'predicted_score', 'return_21d', 'return_3m']
].sort_values(['as_of', 'predicted_rank'])
print(q4_picks.to_string(index=False))

print('\n=== NAV 曲线数据 ===')
for hz in ['1w', '2w', '21d', '3m']:
    nav_path = 'data/paper_trading/nav_curve_%s.parquet' % hz
    try:
        nav_df = pd.read_parquet(nav_path)
        final_nav = nav_df['nav'].iloc[-1]
        max_nav = nav_df['nav'].max()
        min_nav = nav_df['nav'].min()
        drawdown = (min_nav / max_nav - 1)
        total_ret = final_nav - 1
        print('  %s: 累计=%+.2f%%  最大回撤=%.2f%%  最终NAV=%.4f' % (
            hz, total_ret*100, drawdown*100, final_nav))
    except Exception as e:
        print('  %s: 无法加载 (%s)' % (hz, e))
