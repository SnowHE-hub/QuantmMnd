import React, { useState } from 'react';
import { ShieldCheck, CheckCircle2, AlertTriangle, XCircle, Search, HelpCircle, Layers, FileCheck, ArrowRight, Code2 } from 'lucide-react';
import { ClaimItem, CheckStatus } from '../../api/types';
import { CheckBadge } from '../common/Badge';
import { DualPaneEvidenceInspector } from './DualPaneEvidenceInspector';

interface ClaimsInspectorProps {
  asOf: string;
  claims: ClaimItem[];
  ticker: string;
}

export const ClaimsInspector: React.FC<ClaimsInspectorProps> = ({ claims, ticker, asOf }) => {
  const [filterCheck, setFilterCheck] = useState<string>('ALL');
  const [filterRole, setFilterRole] = useState<string>('ALL');
  const [search, setSearch] = useState<string>('');
  const [inspectedClaim, setInspectedClaim] = useState<ClaimItem | null>(null);

  const roles = ['ALL', 'A1', 'A2', 'A3', 'A4', 'A6'];

  const checkCounts = {
    verified: claims.filter((c) => c.check === 'verified').length,
    minor_mismatch: claims.filter((c) => c.check === 'minor_mismatch').length,
    mismatch: claims.filter((c) => c.check === 'mismatch').length,
    unverifiable: claims.filter(
      (c) => c.check === 'unverifiable' || c.check === 'missing_evidence' || c.check === 'weak_basis'
    ).length,
  };

  const filtered = claims.filter((c) => {
    const role = c.ref?.split(':')[0] || 'A1';
    const matchRole = filterRole === 'ALL' || role === filterRole;
    const matchCheck = filterCheck === 'ALL' || c.check === filterCheck;
    const matchSearch =
      search === '' ||
      c.text?.toLowerCase().includes(search.toLowerCase()) ||
      c.ref?.toLowerCase().includes(search.toLowerCase()) ||
      c.formula?.toLowerCase().includes(search.toLowerCase());
    return matchRole && matchCheck && matchSearch;
  });

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Title & Audit Summary */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <FileCheck className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-bold text-white">
              {ticker} 智能体事实声明透视镜（F7 代码级逐条核验）
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            每个角色的结论分解为最小声明，数字由代码核对原始数据，防幻觉核心底座
          </p>
        </div>

        {/* Verification Summary Counts */}
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/60">
            已核对: {checkCounts.verified}
          </span>
          {checkCounts.minor_mismatch > 0 && (
            <span className="px-2 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-800/60">
              微小差异: {checkCounts.minor_mismatch}
            </span>
          )}
          {checkCounts.mismatch > 0 && (
            <span className="px-2 py-0.5 rounded bg-rose-950/80 text-rose-300 border border-rose-800/60">
              不匹配: {checkCounts.mismatch}
            </span>
          )}
          <span className="text-slate-400 font-bold">总声明: {claims.length} 条</span>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
        {/* Role Pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-slate-500 text-[11px] mr-1">角色:</span>
          {roles.map((r) => (
            <button
              key={r}
              onClick={() => setFilterRole(r)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono transition ${
                filterRole === r
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-900 text-slate-400 border border-slate-800 hover:bg-slate-800'
              }`}
            >
              {r === 'ALL'
                ? '全部角色'
                : r === 'A1'
                ? 'A1 财务'
                : r === 'A2'
                ? 'A2 公告'
                : r === 'A3'
                ? 'A3 治理'
                : r === 'A4'
                ? 'A4 资金'
                : 'A6 审查'}
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 w-full sm:w-56">
          <Search className="w-3.5 h-3.5 text-slate-400 mr-2 shrink-0" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索声明内容、公式..."
            className="bg-transparent text-white placeholder-slate-500 focus:outline-none w-full"
          />
        </div>
      </div>

      {/* Active Dual-Pane Evidence Inspector */}
      {inspectedClaim && (
        <DualPaneEvidenceInspector
          claim={inspectedClaim}
          onClose={() => setInspectedClaim(null)}
          ticker={ticker}
          asOf={asOf}
        />
      )}

      {/* Claims List */}
      <div className="space-y-2.5 max-h-[500px] overflow-y-auto pr-1">
        {filtered.length > 0 ? (
          filtered.map((claim, idx) => {
            const role = claim.ref?.split(':')[0] || 'A1';
            const isSelected = inspectedClaim?.ref === claim.ref;

            return (
              <div
                key={idx}
                onClick={() => setInspectedClaim(isSelected ? null : claim)}
                className={`p-3 rounded-lg border text-xs space-y-1.5 transition cursor-pointer ${
                  isSelected
                    ? 'bg-slate-900 border-cyan-500/80 shadow-md ring-1 ring-cyan-500/30'
                    : 'bg-slate-900/50 hover:bg-slate-900 border-slate-800'
                }`}
              >
                {/* Meta line */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold px-1.5 py-0.2 rounded bg-slate-800 text-cyan-300">
                      {claim.ref}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-purple-950/80 text-purple-300 border border-purple-800/40">
                      {claim.type}
                    </span>
                    {claim.period && (
                      <span className="text-[10px] text-slate-500 font-mono">
                        期间: {claim.period}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-cyan-400 flex items-center gap-1 hover:underline">
                      <Code2 className="w-3 h-3" />
                      <span>{isSelected ? '收起溯源' : '双栏溯源'}</span>
                    </span>
                    <CheckBadge status={claim.check} />
                  </div>
                </div>

                {/* Claim Text */}
                <div className="text-slate-200 text-xs leading-relaxed font-normal">
                  {claim.text}
                </div>

                {/* Formula (if computed) */}
                {claim.formula && (
                  <div className="p-1.5 rounded bg-slate-950 border border-slate-800 text-[11px] font-mono text-cyan-300 flex items-center gap-2">
                    <span className="text-slate-500">计算公式:</span>
                    <span>{claim.formula}</span>
                    {claim.inputs && claim.inputs.length > 0 && (
                      <span className="text-slate-500">
                        (依据: {claim.inputs.join(', ')})
                      </span>
                    )}
                  </div>
                )}

                {/* A6 Reviewer verdict */}
                {claim.a6 && (
                  <div className="text-[11px] text-slate-400 bg-slate-950/60 p-2 rounded border border-slate-800/60">
                    <span className="font-bold text-slate-300 mr-1.5">[A6审查意见]:</span>
                    {claim.a6}
                  </div>
                )}

                {/* Evidence IDs */}
                {claim.evidence && claim.evidence.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1 pt-1 border-t border-slate-800/40">
                    <span className="text-[10px] text-slate-500 mr-1">底层证据:</span>
                    {claim.evidence.map((ev: any, evIdx: number) => {
                      const evId = typeof ev === 'string' ? ev : ev.evidence_id || JSON.stringify(ev);
                      return (
                        <span
                          key={evIdx}
                          className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800/80 text-slate-400 border border-slate-700/60 max-w-[240px] truncate"
                          title={evId}
                        >
                          {evId}
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })
        ) : (
          <div className="p-8 text-center text-slate-500 text-xs border border-slate-800/60 rounded-lg">
            该标的在此截面下未进入深度研究池，或无匹配的声明记录
          </div>
        )}
      </div>
    </div>
  );
};
