import React from 'react';
import { Cpu } from 'lucide-react';
import { ConsensusResponse } from '../../api/analytics';

const MODELS: [string, string][] = [
  ['xgb', 'XGBoost'],
  ['cat', 'CatBoost'],
  ['tabm', 'TabM'],
  ['seqfusion', 'SeqFusion'],
];

// 四个模型的截面百分位（研究池里的排名，0–100，越高越看好）和分歧度。
// 原实现用 CV = σ/|μ|：分数是截面 z 分，均值在 0 附近时 CV 爆炸、符号翻转，没有意义。
export const ModelConsensus: React.FC<{ data: ConsensusResponse | null; title?: string }> = ({ data, title }) => {
  const row = data?.row;
  const tone =
    row?.label === '一致看好' ? 'text-rose-300 border-rose-800 bg-rose-950/40' : row?.label === '模型分歧' ? 'text-purple-300 border-purple-800 bg-purple-950/40' : 'text-slate-300 border-slate-700 bg-slate-900';
  return (
    <div className="p-3.5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-2.5" data-testid="model-consensus">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
        <div className="font-semibold text-white flex items-center gap-1.5">
          <Cpu className="w-3.5 h-3.5 text-purple-400" />
          <span>{title ?? 'F3 四个模型的截面百分位（研究池）'}</span>
        </div>
        {row && <span className={`text-[10px] px-2 py-0.5 rounded border ${tone}`}>{row.label}</span>}
      </div>
      {!data ? (
        <div className="text-slate-500 text-[11px]">读取中…</div>
      ) : !row ? (
        <div className="text-slate-500 text-[11px]">{data.note ?? '不在当日研究池截面里'}</div>
      ) : (
        <>
          <div className="grid grid-cols-5 gap-2 text-center font-mono">
            {[...MODELS, ['ENS', 'ENS 集成'] as [string, string]].map(([k, name]) => {
              const v = row[`${k}_pct`] as number | null;
              return (
                <div key={k} className="p-2 rounded bg-slate-950/60 border border-slate-800">
                  <div className="text-[10px] text-slate-400">{name}</div>
                  <div className="text-xs font-bold text-slate-100 mt-0.5" data-testid={`pct-${k}`}>
                    {v === null ? '—' : v.toFixed(1)}
                  </div>
                  <div className="text-[9px] text-slate-500">z = {typeof row[k] === 'number' ? (row[k] as number).toFixed(2) : '—'}</div>
                </div>
              );
            })}
          </div>
          <div className="text-[11px] text-slate-400">
            分歧度：四个百分位的标准差 {row.dispersion_pp?.toFixed(1)} 个百分点（极差 {row.range_pp?.toFixed(1)}），在 {data.meta.n} 只里排第{' '}
            {row.dispersion_rank_pct?.toFixed(0)} 百分位。
          </div>
          <div className="text-[10px] text-slate-500">
            百分位 = 在当天研究池（PIT 成交额前 1500）里的排名；“一致看好”= ENS 前 20% 且分歧度不高于中位数，“模型分歧”= 分歧度前 20%。
          </div>
        </>
      )}
    </div>
  );
};
