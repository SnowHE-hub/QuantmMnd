import React, { useState } from 'react';
import { AlertTriangle, ShieldCheck, Tag, ArrowRight } from 'lucide-react';
import { BypassEvent } from '../../api/types';

interface BypassEventsProps {
  events: BypassEvent[];
  onSelectTicker: (ticker: string) => void;
}

const parseStringList = (item: any): string[] => {
  if (!item) return [];
  if (Array.isArray(item)) return item;
  if (typeof item === 'string') return [item];
  return [];
};

export const BypassEvents: React.FC<BypassEventsProps> = ({ events, onSelectTicker }) => {
  const [importanceFilter, setImportanceFilter] = useState<'ALL' | 'major' | 'watch'>('ALL');

  const filtered = events.filter((e) => {
    if (importanceFilter === 'ALL') return true;
    return e.importance === importanceFilter;
  });

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-bold text-white">旁路事件与巨潮公告监测雷达 (MacBERT 分类)</h3>
        </div>

        {/* Importance Filter */}
        <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setImportanceFilter('ALL')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              importanceFilter === 'ALL'
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            全部 ({events.length})
          </button>
          <button
            onClick={() => setImportanceFilter('major')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              importanceFilter === 'major'
                ? 'bg-rose-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            重大事件 (Major)
          </button>
          <button
            onClick={() => setImportanceFilter('watch')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              importanceFilter === 'watch'
                ? 'bg-amber-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            重点关注 (Watch)
          </button>
        </div>
      </div>

      {/* Events Stream List */}
      <div className="space-y-2.5 max-h-96 overflow-y-auto pr-1">
        {filtered.slice(0, 40).map((ev, idx) => {
          const isMajor = ev.importance === 'major';
          const summaries = parseStringList(ev.summaries);
          const evidenceIds = parseStringList(ev.evidence_ids);
          const rulesStr = Array.isArray(ev.rules) ? ev.rules.join(', ') : String(ev.rules || '');

          return (
            <div
              key={idx}
              className={`p-3 rounded-lg border transition text-xs flex flex-col md:flex-row md:items-center justify-between gap-3 ${
                isMajor
                  ? 'bg-rose-950/20 hover:bg-rose-950/30 border-rose-900/40'
                  : 'bg-slate-900/50 hover:bg-slate-900 border-slate-800'
              }`}
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => onSelectTicker(ev.ticker)}
                    className="font-mono font-bold text-blue-400 hover:underline hover:text-blue-300"
                  >
                    {ev.ticker}
                  </button>
                  <span
                    className={`text-[10px] px-1.5 py-0.2 rounded font-bold uppercase ${
                      isMajor
                        ? 'bg-rose-950 text-rose-300 border border-rose-800/80'
                        : 'bg-amber-950 text-amber-300 border border-amber-800/80'
                    }`}
                  >
                    {ev.importance}
                  </span>
                  {rulesStr && (
                    <span className="text-[10px] font-mono text-slate-500">
                      触发规则: {rulesStr}
                    </span>
                  )}
                </div>

                <div className="text-slate-200 space-y-0.5">
                  {summaries.map((s, sIdx) => (
                    <div key={sIdx} className="line-clamp-2">
                      • {s}
                    </div>
                  ))}
                </div>
              </div>

              {/* Evidence IDs */}
              {evidenceIds.length > 0 && (
                <div className="flex flex-wrap gap-1 shrink-0">
                  {evidenceIds.slice(0, 2).map((eId, eIdx) => (
                    <span
                      key={eIdx}
                      className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800/80 text-slate-400 border border-slate-700/60 max-w-[150px] truncate"
                      title={eId}
                    >
                      {eId}
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
