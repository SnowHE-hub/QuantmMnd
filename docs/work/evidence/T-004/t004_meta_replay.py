"""T-004 · meta-learner：hit 是否依赖 v4 标签；v4 与 v4_1 下 LOQO AUC 是否不同。只读，输出只打印 + 写 data/research/t004/。"""
import sys, importlib.util, json, logging
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
MAIN = Path('/home/lenovo/projects/quantmind')
WT = Path('/home/lenovo/projects/qm-t035')
OUT = MAIN / 'data/research/t004'
logging.disable(logging.CRITICAL)
PNL = MAIN / 'data/feedback/realized_pnl.parquet'

pnl = pd.read_parquet(PNL)
print('rows', len(pnl), 'hit == (actual_rank <= predicted_rank) for all rows:',
      bool((pnl['hit'] == (pnl['actual_rank'] <= pnl['predicted_rank'])).all()))
print('hit independent of panel_return_63d: hit has no NaN', int(pnl['hit'].isna().sum()), '| panel_return_63d NaN rows', int(pnl['panel_return_63d'].isna().sum()))
# 这些 realized_pnl 行里，panel_return_63d 与 v4_1 标签的差
P = {n: pd.read_parquet(MAIN / f'data/panel/alpha_panel_{n}.parquet') for n in ['v4', 'v4_1']}
q = pnl.assign(join_date=pd.to_datetime(pnl['as_of_date']))
q.loc[q['join_date'].dt.strftime('%Y-%m-%d') == '2025-06-28', 'join_date'] = pd.Timestamp('2025-06-30')
for n in ['v4', 'v4_1']:
    s = P[n]['forward_return_63d'].rename(n)
    q = q.merge(s.reset_index().rename(columns={'as_of': 'join_date'}), on=['join_date', 'ticker'], how='left')
d = q.dropna(subset=['v4', 'v4_1'])
d = d.assign(absdiff=(d['v4'] - d['v4_1']).abs())
print('rows with both labels', len(d), '; |v4 - v4_1| > 0.5% rows by quarter:')
print(d.groupby('join_date').agg(n=('absdiff', 'size'), n_bad=('absdiff', lambda x: int((x > 0.005).sum())), max_diff=('absdiff', 'max')).round(3).to_string())

res = {}
for tag, script in [('main_n100', MAIN / 'scripts/train_meta_learner.py'), ('worktree_dedup', WT / 'scripts/train_meta_learner.py')]:
    spec = importlib.util.spec_from_file_location('tml_' + tag, script)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    logging.disable(logging.CRITICAL)
    dfs = {}
    for n in ['v4', 'v4_1']:
        dfs[n] = m.build_training_data(PNL, MAIN / f'data/panel/alpha_panel_{n}.parquet')
    same_X = dfs['v4'][m.AGENT_COLS].equals(dfs['v4_1'][m.AGENT_COLS]); same_y = dfs['v4']['hit'].equals(dfs['v4_1']['hit'])
    print(f'[{tag}] agent proxies equal v4 vs v4_1: {same_X}; hit equal: {same_y}; n={len(dfs["v4"])}')
    for n, df in dfs.items():
        sub = df[m.AGENT_COLS + ['hit', 'join_date']].dropna()
        X = sub[m.AGENT_COLS].values.astype(float); y = sub['hit'].astype(int).values; g = sub['join_date'].values
        cv, aucs = m.loqo_cv_auc(X, y, g, verbose=False)
        qs = [str(pd.Timestamp(x).date()) for x in np.unique(g)]
        res[f'{tag}|{n}'] = dict(cv_auc=cv, aucs=dict(zip(qs, [round(a, 4) for a in aucs])), n=len(sub))
        print(f'[{tag}] panel={n} n={len(sub)} cv_auc={cv:.4f}', dict(zip(qs, [round(a, 2) for a in aucs])))
(OUT / 'meta_learner_replay.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('ALL DONE')
