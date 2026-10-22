"""T-004 重放：v4 标签污染对顺序验证（run_sequential_validation）的影响。

只读：从主 checkout 读代码和面板（不写字节码），输出只写 data/research/t004/。
步骤：
  A. v4 / v4_1 / v4_2 的标签逐截面比较（63d、21d）；
  B. 因子列在 v4 与 v4_1 之间是否逐位相同；
  C. 用同一套代码、同一批特征，在三个面板上重跑 Round 0-7；v4 一遍用来核对能否复现已存的 reports/sequential_val_v4。
"""
import sys, json, importlib.util, logging, time
from pathlib import Path
sys.dont_write_bytecode = True
MAIN = Path('/home/lenovo/projects/quantmind')
OUT = MAIN / 'data/research/t004'
sys.path.insert(0, str(MAIN))
import numpy as np, pandas as pd
from scipy import stats

spec = importlib.util.spec_from_file_location('seqval', MAIN / 'scripts/run_sequential_validation.py')
sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)

OUT.mkdir(parents=True, exist_ok=True)
stored = json.loads((MAIN / 'reports/sequential_val_v4/round_summary.json').read_text(encoding='utf-8'))
feats = stored['feature_cols']
print('features', len(feats), flush=True)
P = {n: pd.read_parquet(MAIN / f'data/panel/alpha_panel_{n}.parquet') for n in ['v4', 'v4_1', 'v4_2']}
for n, p in P.items():
    print(n, p.shape, p.index.names, flush=True)

# A. 标签逐截面比较
rows = []
for other in ['v4_1', 'v4_2']:
    for lab in ['forward_return_63d', 'forward_return_21d']:
        a = P['v4'][lab]; b = P[other][lab]
        j = pd.concat([a.rename('old'), b.rename('new')], axis=1, join='inner').dropna()
        for d, g in j.groupby(level='as_of'):
            if len(g) < 30:
                continue
            err = (g['old'] - g['new']).abs()
            rho = stats.spearmanr(g['old'], g['new'])[0]
            rows.append(dict(vs=other, label=lab, as_of=str(pd.Timestamp(d).date()), n=len(g),
                             share_gt_0p5pct=float((err > 0.005).mean()), share_gt_10pct=float((err > 0.10).mean()),
                             spearman=float(rho)))
A = pd.DataFrame(rows)
A.to_csv(OUT / 'label_diff_by_date.csv', index=False)
for other in ['v4_1', 'v4_2']:
    for lab in ['forward_return_63d', 'forward_return_21d']:
        s = A[(A.vs == other) & (A.label == lab)]
        tot_n = s.n.sum()
        print(f'[A] {other} {lab}: dates {len(s)}, rows {tot_n}, share>10% (row-weighted) {(s.share_gt_10pct*s.n).sum()/tot_n:.4f}, '
              f'dates with spearman<0.99: {s[s.spearman<0.99].as_of.tolist()}', flush=True)

# B. 因子列一致性
common = P['v4'].index.intersection(P['v4_1'].index)
cols = [c for c in P['v4'].columns if c not in ('forward_return_21d', 'forward_return_63d')]
same_cols = [c for c in cols if c in P['v4_1'].columns]
eq = all(P['v4'].loc[common, c].equals(P['v4_1'].loc[common, c]) for c in same_cols)
print('[B] index equal', P['v4'].index.equals(P['v4_1'].index), 'n common', len(common), 'factor cols equal', eq, 'ncols', len(same_cols), flush=True)

# C. 重跑
for n in ['v4', 'v4_1', 'v4_2']:
    t = time.time()
    res = sv.run_validation(P[n], feats, OUT / f'seqval_{n}', 0, 7)
    print(f'[C] {n} done {time.time()-t:.0f}s', flush=True)
    summ = {r['round_id']: r for r in res['ic_trajectory']}
    for rid, r in summ.items():
        print(f'   round {rid} {r["name"]} ic_mean {r["ic_mean"]} {r["champion_decision"]}', flush=True)
    print('   champion round', res['champion_round'], res['champion_ic'], flush=True)
print('stored (reports/sequential_val_v4):', [(r['round_id'], r['ic_mean'], r['champion_decision']) for r in stored['ic_trajectory']], flush=True)
print('ALL DONE', flush=True)
