import React, { useState } from 'react';
import { Filter, Star, ArrowRight, Swords } from 'lucide-react';
import { CandidateItem } from '../../api/types';

interface CandidatesFunnelProps {
  candidates: CandidateItem[];
  horizon: '20d' | '63d';
  onChangeHorizon: (h: '20d' | '63d') => void;
  onSelectTicker: (ticker: string) => void;
  onAddToCompare?: (ticker: string) => void;
}

const parseReasons = (reasons: any): string[] => {
  if (!reasons) return [];
  if (Array.isArray(reasons)) return reasons;
  if (typeof reasons === 'string') {
    return reasons
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return [];
};

export const CandidatesFunnel: React.FC<CandidatesFunnelProps> = ({
  candidates,
  horizon,
  onChangeHorizon,
  onSelectTicker,
  onAddToCompare,
}) => {
  const [filterReason, setFilterReason] = useState<string>('ALL');

  // Extract all unique reasons safely
  const allReasons = Array.from(
    new Set(candidates.flatMap((c) => parseReasons(c.reasons)))
  );

  const filtered = candidates.filter((c) => {
    if (filterReason === 'ALL') return true;
    const rList = parseReasons(c.reasons);
    return rList.includes(filterReason);
  });

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-blue-400" />
          <h3 className="text-sm font-bold text-white">F4 量化研究候选池（Research Candidates）</h3>
        </div>

        {/* Horizon Switcher */}
        <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => onChangeHorizon('20d')}
            className={`px-3 py-1 rounded font-medium transition ${
              horizon === '20d'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            20日短周期 (20d)
          </button>
          <button
            onClick={() => onChangeHorizon('63d')}
            className={`px-3 py-1 rounded font-medium transition ${
              horizon === '63d'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            63日季周期 (63d)
          </button>
        </div>
      </div>

      {/* Filter by Reasons */}
      {allReasons.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-slate-500 text-[11px] mr-1">入选理由过滤:</span>
          <button
            onClick={() => setFilterReason('ALL')}
            className={`px-2 py-0.5 rounded text-[11px] transition ${
              filterReason === 'ALL'
                ? 'bg-blue-600/40 text-blue-300 border border-blue-500/50'
                : 'bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-800'
            }`}
          >
            全部 ({candidates.length})
          </button>
          {allReasons.slice(0, 6).map((reason) => (
            <button
              key={reason}
              onClick={() => setFilterReason(reason)}
              className={`px-2 py-0.5 rounded text-[11px] transition ${
                filterReason === reason
                  ? 'bg-cyan-600/40 text-cyan-300 border border-cyan-500/50'
                  : 'bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-800'
              }`}
            >
              {reason}
            </button>
          ))}
        </div>
      )}

      {/* Candidate Table */}
      <div className="border border-slate-800 rounded-lg overflow-x-auto">
        <table className="w-full text-left text-xs divide-y divide-slate-800">
          <thead className="bg-slate-900 text-slate-400 font-medium">
            <tr>
              <th className="px-3.5 py-2">优先级</th>
              <th className="px-3 py-2">代码 / 标的</th>
              <th className="px-3 py-2">入选依据与标签</th>
              <th className="px-3 py-2 text-right">ENS 20d 打分</th>
              <th className="px-3 py-2 text-right">Ridge 打分</th>
              <th className="px-3 py-2 text-right">ENS 63d 打分</th>
              <th className="px-3 py-2 text-center">模型共识状态</th>
              <th className="px-3 py-2 text-center">操作</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 bg-slate-950/40 font-mono">
            {filtered.slice(0, 30).map((c, idx) => {
              const rList = parseReasons(c.reasons);
              const isConsensus =
                c.ens20 !== undefined &&
                c.ridge20 !== undefined &&
                c.ens20 > 1.8 &&
                c.ridge20 > 0.05;

              return (
                <tr
                  key={c.ticker || idx}
                  onClick={() => onSelectTicker(c.ticker)}
                  className="hover:bg-slate-900/60 cursor-pointer transition"
                >
                  <td className="px-3.5 py-2">
                    <span className="font-bold text-cyan-400 flex items-center gap-1">
                      <Star className="w-3 h-3 text-cyan-400 fill-cyan-400/20" />
                      P{c.priority}
                    </span>
                  </td>
                  <td className="px-3 py-2 font-bold text-white group-hover:text-cyan-300">
                    {c.ticker}
                  </td>
                  <td className="px-3 py-2 font-sans">
                    <div className="flex flex-wrap gap-1">
                      {rList.map((r, rIdx) => (
                        <span
                          key={rIdx}
                          className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 border border-slate-700/60"
                        >
                          {r}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-3 py-2 text-right text-slate-200 tabular-nums">
                    {c.ens20 !== undefined && c.ens20 !== null ? Number(c.ens20).toFixed(4) : '—'}
                  </td>
                  <td className="px-3 py-2 text-right text-slate-400 tabular-nums">
                    {c.ridge20 !== undefined && c.ridge20 !== null ? Number(c.ridge20).toFixed(4) : '—'}
                  </td>
                  <td className="px-3 py-2 text-right text-purple-300 tabular-nums">
                    {c.ens63 !== undefined && c.ens63 !== null ? Number(c.ens63).toFixed(4) : '—'}
                  </td>
                  <td className="px-3 py-2 text-center">
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded border font-mono ${
                        isConsensus
                          ? 'bg-cyan-950 text-cyan-300 border-cyan-800/80'
                          : 'bg-slate-900 text-slate-400 border-slate-800'
                      }`}
                    >
                      {isConsensus ? '共识龙头' : '温和共识'}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-center" onClick={(e) => e.stopPropagation()}>
                    <div className="flex items-center justify-center gap-2">
                      <button
                        onClick={() => onSelectTicker(c.ticker)}
                        className="text-blue-400 text-[11px] flex items-center gap-0.5 hover:underline"
                      >
                        <span>穿透</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                      {onAddToCompare && (
                        <button
                          onClick={() => onAddToCompare(c.ticker)}
                          className="p-1 rounded bg-slate-900 hover:bg-slate-800 text-purple-300 border border-purple-800/60 text-[10px] transition"
                          title="加入 PK 竞技场"
                        >
                          <Swords className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
