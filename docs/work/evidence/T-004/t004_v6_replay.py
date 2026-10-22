"""T-004 · LGBM v6（ICIR 0.380）在 v4 / v4_1 / v4_2 标签上的复算与训练 / 测试归因。

只读；输出只写 data/research/t004/。不调用 save，不碰 models/。
- 参数、特征取自 models/lgbm_v6_alpha.meta.json（42 特征，63d，400/31/0.05，早停 50，walk-forward 默认 min_train=10、n_val=2）。
- 2×2：训练 / 验证标签来源 ∈ {v4, v4_1}，评估标签来源 ∈ {v4, v4_1}，把污染对「训练」和「评估」的影响分开。
"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
MAIN = Path('/home/lenovo/projects/quantmind')
OUT = MAIN / 'data/research/t004'
sys.path.insert(0, str(MAIN))
import numpy as np, pandas as pd
import quantmind.models.lgbm_ranker as lr

meta = json.loads((MAIN / 'models/lgbm_v6_alpha.meta.json').read_text(encoding='utf-8'))
feats = meta['feature_cols']; LAB = meta['label_col']
kw = dict(meta['model_params']); kw['n_jobs'] = 4
print('features', len(feats), 'label', LAB, 'params', kw, flush=True)
P = {n: pd.read_parquet(MAIN / f'data/panel/alpha_panel_{n}.parquet') for n in ['v4', 'v4_1', 'v4_2']}
dates = sorted(P['v4'].index.get_level_values('as_of').unique())
folds = lr.WalkForwardSplit(min_train_periods=10, n_val_periods=2).split(dates)
print('folds', len(folds), 'dates', str(dates[0])[:10], str(dates[-1])[:10], flush=True)

def run(train_panel, eval_panels):
    """训练一次，用多个评估面板打分。"""
    res = {k: [] for k in eval_panels}
    for fold in folds:
        Xtr, ytr, gtr = lr.build_lgbm_arrays(train_panel, fold.train_dates, feats, LAB, 5)
        Xv, yv, gv = lr.build_lgbm_arrays(train_panel, fold.val_dates, feats, LAB, 5)
        if len(Xtr) == 0 or len(Xv) == 0:
            continue
        m = lr.LGBMRankerModel(auto_flip=True, **kw)
        m.fit(Xtr, ytr, groups=gtr, X_val=Xv, y_val=yv, groups_val=gv, feature_names=feats)
        for k, ep in eval_panels.items():
            res[k].append(lr._evaluate_fold(m, ep, fold, feats, LAB, 5))
    out = {}
    for k, fr in res.items():
        w = lr.WalkForwardResult(feature_cols=feats, label_col=LAB, n_folds=len(fr), folds=fr)
        if not np.isnan(w.ic_mean) and w.ic_mean < 0:
            w.direction = -1
        out[k] = w
    return out

summary = {}
for tr in ['v4', 'v4_1']:
    r = run(P[tr], {'v4': P['v4'], 'v4_1': P['v4_1']})
    for ev, w in r.items():
        key = f'train={tr}|eval={ev}'
        summary[key] = dict(n_folds=w.n_folds, ic_mean=w.ic_mean, ic_ir=w.ic_ir, direction=w.direction,
                            eff_ic_mean=w.effective_ic_mean, eff_ic_ir=w.effective_ic_ir, win_rate=w.ic_win_rate,
                            ic_by_test_date={str(k.date()): float(v) for k, v in w.ic_series.items()})
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary[key].items() if k != 'ic_by_test_date'}, flush=True)
# v4_2（评估期更长、标签更全；训练和评估都用 v4_2）
r = run(P['v4_2'], {'v4_2': P['v4_2']})
w = r['v4_2']
summary['train=v4_2|eval=v4_2'] = dict(n_folds=w.n_folds, ic_mean=w.ic_mean, ic_ir=w.ic_ir, direction=w.direction,
                                      eff_ic_mean=w.effective_ic_mean, eff_ic_ir=w.effective_ic_ir, win_rate=w.ic_win_rate,
                                      ic_by_test_date={str(k.date()): float(v) for k, v in w.ic_series.items()})
print('train=v4_2|eval=v4_2', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary['train=v4_2|eval=v4_2'].items() if k != 'ic_by_test_date'}, flush=True)
(OUT / 'v6_replay_2x2.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
# 按测试日对比
print('--- IC by test date (eff sign not applied) ---')
a = summary['train=v4|eval=v4']['ic_by_test_date']; b = summary['train=v4|eval=v4_1']['ic_by_test_date']
c = summary['train=v4_1|eval=v4_1']['ic_by_test_date']
print('test_date  v4/v4  v4/v4_1  v4_1/v4_1')
for d in a:
    print(d, round(a[d], 4), round(b.get(d, float('nan')), 4), round(c.get(d, float('nan')), 4))
print('ALL DONE', flush=True)
