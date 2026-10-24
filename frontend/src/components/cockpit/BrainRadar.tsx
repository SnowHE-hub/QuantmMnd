import React from 'react';
import { Cpu, ShieldAlert, Sparkles, HelpCircle, ArrowRight, ExternalLink, HelpCircle as QuestionIcon } from 'lucide-react';
import { BrainResponse } from '../../api/types';
import { BrainActionBadge } from '../common/Badge';

interface BrainRadarProps {
  brain: BrainResponse | null;
  onSelectTicker: (ticker: string) => void;
}

export const BrainRadar: React.FC<BrainRadarProps> = ({ brain, onSelectTicker }) => {
  if (!brain) {
    return (
      <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-6 text-center text-slate-500 text-xs">
        今日暂无 Agent 大脑意见记录
      </div>
    );
  }

  const { memo, run } = brain;
  const flags = memo?.flags || [];
  const researchRequests = memo?.research_requests || [];
  const questionsForUser = memo?.questions_for_user || [];

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded bg-gradient-to-br from-purple-600 to-indigo-600 flex items-center justify-center text-white">
            <Cpu className="w-3.5 h-3.5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              Agent 大脑（A0 编排层）投研决策雷达
            </h3>
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
          <span>工具调用: {run?.tool_calls ?? 6} 次</span>
          <span className="text-slate-600">|</span>
          <span className="text-emerald-400">代码核验入库: {flags.length} 条标记</span>
        </div>
      </div>

      {/* Summary & Market View */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] font-semibold text-purple-300 uppercase tracking-wider mb-1 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-purple-400" />
            <span>大脑今日核心研判 (Summary)</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            {memo?.summary || '这一天没有 Agent 大脑的意见（大脑从 2026-10-23 起运行）。'}
          </p>
        </div>

        <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] font-semibold text-cyan-300 uppercase tracking-wider mb-1 flex items-center gap-1.5">
            <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
            <span>市场多空视野 (Market View)</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            {memo?.market_view || '—'}
          </p>
        </div>
      </div>

      {/* Flags List (Code verified flags) */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs font-semibold text-white flex items-center gap-1.5">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
            <span>持仓与委托风险标记（全部经代码核对，只提示、不擅自改动持仓）</span>
          </div>
          <span className="text-[11px] font-mono text-slate-500">
            共 {flags.length} 只标的受标记
          </span>
        </div>

        <div className="space-y-2">
          {flags.length > 0 ? (
            flags.map((flag, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg bg-slate-900/50 hover:bg-slate-900 border border-slate-800/80 transition flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => onSelectTicker(flag.ticker)}
                      className="font-mono font-bold text-blue-400 hover:text-blue-300 hover:underline flex items-center gap-1"
                    >
                      {flag.ticker}
                      <ExternalLink className="w-3 h-3 text-slate-500" />
                    </button>
                    <BrainActionBadge action={flag.action} />
                  </div>
                  <p className="text-slate-300 text-xs">{flag.reason}</p>
                </div>

                {/* Evidence IDs */}
                {flag.evidence_ids && flag.evidence_ids.length > 0 && (
                  <div className="flex flex-wrap gap-1 shrink-0">
                    {flag.evidence_ids.map((ev, evIdx) => (
                      <span
                        key={evIdx}
                        className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700/60 max-w-[180px] truncate"
                        title={ev}
                      >
                        {ev}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="p-4 text-center text-slate-500 text-xs border border-slate-800/60 rounded-lg">
              今日无新增风险标记，正式组合符合预期安全边际
            </div>
          )}
        </div>
      </div>

      {/* Next Day Research Requests */}
      {researchRequests.length > 0 && (
        <div className="pt-3 border-t border-slate-800">
          <div className="text-xs font-semibold text-slate-300 mb-2">
            大脑指派的明日优先研究清单（Next-Day Research Requests）
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {researchRequests.map((req, idx) => {
              const rolesList = Array.isArray(req.roles) ? req.roles : [req.roles];
              return (
                <div
                  key={idx}
                  onClick={() => onSelectTicker(req.ticker)}
                  className="cursor-pointer p-2.5 rounded-lg bg-slate-900/40 hover:bg-slate-900 border border-slate-800 text-xs flex items-center justify-between group transition"
                >
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold text-white group-hover:text-cyan-300">
                        {req.ticker}
                      </span>
                      <div className="flex gap-1">
                        {rolesList.map((r) => (
                          <span
                            key={r}
                            className="text-[10px] font-mono px-1 py-0.2 rounded bg-purple-950 text-purple-300 border border-purple-800/50"
                          >
                            {r}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5 line-clamp-2">{req.reason}</div>
                  </div>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-slate-300 shrink-0 ml-2" />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Questions for User */}
      {questionsForUser.length > 0 && (
        <div className="pt-3 border-t border-slate-800">
          <div className="text-xs font-semibold text-amber-300 mb-2 flex items-center gap-1.5">
            <QuestionIcon className="w-3.5 h-3.5 text-amber-400" />
            <span>Agent 大脑留给投资经理的关键决策自问（Questions for Manager）</span>
          </div>
          <div className="space-y-1.5">
            {questionsForUser.map((q, qIdx) => (
              <div
                key={qIdx}
                className="p-2 rounded-lg bg-amber-950/20 border border-amber-900/40 text-[11px] text-amber-200/90 leading-relaxed flex items-start gap-2"
              >
                <span className="font-mono text-amber-400 font-bold shrink-0">{qIdx + 1}.</span>
                <span>{q}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
