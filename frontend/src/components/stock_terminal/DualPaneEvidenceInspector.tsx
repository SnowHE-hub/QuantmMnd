import React, { useEffect, useState } from 'react';
import { X, FileCheck2, Code2, Database, Search, Loader2, AlertTriangle } from 'lucide-react';
import { ClaimItem } from '../../api/types';
import { CheckBadge } from '../common/Badge';
import { analyticsApi, ClaimTrace } from '../../api/analytics';

interface DualPaneEvidenceInspectorProps {
  claim: ClaimItem | null;
  onClose: () => void;
  ticker: string;
  asOf: string;
}

const fmt = (v: unknown): string => {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 6 });
  return String(v);
};

// 声明核验双栏：左边是 Agent 写的声明，右边是后端 /api/claim 返回的真实核验记录
// （所引记录的字段原值、校验器在运行时和现在各自的结论与备注、按公式重算的结果）。这里不生成任何"示意代码"。
export const DualPaneEvidenceInspector: React.FC<DualPaneEvidenceInspectorProps> = ({ claim, onClose, ticker, asOf }) => {
  const [trace, setTrace] = useState<ClaimTrace | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!claim) return;
    let alive = true;
    setTrace(null);
    setError(null);
    analyticsApi
      .claim(asOf, ticker, claim.ref)
      .then((t) => alive && setTrace(t))
      .catch((e) => alive && setError(String(e.message || e)));
    return () => {
      alive = false;
    };
  }, [claim, ticker, asOf]);

  if (!claim) return null;
  const changed = trace?.check_at_run && trace?.check_now && trace.check_at_run.status !== trace.check_now.status;

  return (
    <div className="p-4 rounded-xl bg-slate-950 border border-cyan-800/80 shadow-2xl space-y-4" data-testid="claim-trace">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-cyan-950 border border-cyan-800 flex items-center justify-center text-cyan-400">
            <FileCheck2 className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm font-bold text-white">声明核验溯源 ({claim.ref})</span>
              <CheckBadge status={claim.check} />
            </div>
            <p className="text-[11px] text-slate-400">
              左：Agent 写的声明 · 右：校验器（{trace?.verifier ?? 'verifier.py'}）对照所引记录的真实结果
            </p>
          </div>
        </div>
        <button onClick={onClose} className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <div className="p-3.5 rounded-lg bg-slate-900/70 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-cyan-300 flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5" />
              Agent 声明
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
              {trace?.claim.claim_type ?? claim.type}
              {trace?.model ? ` · ${trace.model}` : ''}
            </span>
          </div>
          <div className="p-2.5 rounded bg-slate-950/80 border border-slate-800/80 text-slate-200 leading-relaxed">“{claim.text}”</div>
          <div className="font-mono text-[11px] space-y-1 px-1">
            <div className="flex justify-between">
              <span className="text-slate-400">声明数值</span>
              <span className="text-white">
                {fmt(claim.value)} {claim.unit || ''}
              </span>
            </div>
            {claim.period && (
              <div className="flex justify-between">
                <span className="text-slate-400">报告期</span>
                <span className="text-slate-300">{claim.period}</span>
              </div>
            )}
            {trace?.claim.formula && (
              <div className="flex justify-between">
                <span className="text-slate-400">公式</span>
                <span className="text-cyan-300">{trace.claim.formula}</span>
              </div>
            )}
          </div>
          {trace && trace.inputs.length > 0 && (
            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">输入声明</div>
              {trace.inputs.map((i) => (
                <div key={i.local_id} className="flex justify-between gap-2 font-mono text-[11px] bg-slate-950/60 rounded px-2 py-1">
                  <span className="text-slate-400 shrink-0">{i.local_id}</span>
                  <span className="text-slate-300 truncate" title={i.text}>
                    {i.text}
                  </span>
                  <span className="text-white shrink-0">
                    {fmt(i.value)} {i.unit || ''}
                  </span>
                </div>
              ))}
            </div>
          )}
          {claim.a6 && (
            <div className="text-[11px] text-slate-400 bg-slate-950/40 p-2 rounded border border-slate-800/60">
              <span className="text-purple-300 font-semibold mr-1">A6 审查：</span>
              {claim.a6}
            </div>
          )}
        </div>

        <div className="p-3.5 rounded-lg bg-slate-900/70 border border-slate-800 space-y-2.5">
          <div className="font-semibold text-emerald-400 flex items-center gap-1.5">
            <Code2 className="w-3.5 h-3.5" />
            代码校验记录
          </div>
          {!trace && !error && (
            <div className="flex items-center gap-2 text-slate-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> 读取运行记录…
            </div>
          )}
          {error && (
            <div className="p-2 rounded bg-rose-950/40 border border-rose-800 text-rose-300 flex gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>没有找到这条声明的运行记录：{error}</span>
            </div>
          )}
          {trace && (
            <>
              <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <div className="text-slate-500 text-[10px]">运行时结论</div>
                  <div className="text-white">{trace.check_at_run?.status ?? '—'}</div>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <div className="text-slate-500 text-[10px]">现在的校验器重跑</div>
                  <div className={changed ? 'text-amber-300' : 'text-white'}>{trace.check_now?.status ?? '—'}</div>
                </div>
              </div>
              {changed && (
                <div className="text-[11px] text-amber-300">校验器版本更新后结论变了（运行时 → 现在），以现在的为准。</div>
              )}
              {(trace.check_now?.notes?.length ?? 0) > 0 && (
                <div className="space-y-1">
                  <div className="text-[10px] text-slate-500">校验器备注</div>
                  {trace.check_now!.notes.map((n, i) => (
                    <div key={i} className="font-mono text-[11px] text-rose-300 bg-slate-950 rounded px-2 py-1 break-all">
                      {n}
                    </div>
                  ))}
                </div>
              )}
              {trace.recompute && (
                <div className="p-2 rounded bg-[#070b14] border border-slate-800 font-mono text-[11px] space-y-0.5">
                  <div className="text-slate-500 text-[10px]">按公式重算（verifier._eval，输入为各输入声明的数值）</div>
                  <div className="text-emerald-300">{trace.recompute.formula}</div>
                  {trace.recompute.error ? (
                    <div className="text-rose-300">{trace.recompute.error}</div>
                  ) : (
                    <>
                      <div>
                        重算 = <span className="text-white">{fmt(trace.recompute.value)}</span>，声明 ={' '}
                        <span className="text-white">{fmt(trace.recompute.stated)}</span>
                      </div>
                      <div>
                        相对差 ={' '}
                        <span className="text-white">
                          {trace.recompute.rel_diff === null || trace.recompute.rel_diff === undefined
                            ? '—'
                            : trace.recompute.rel_diff.toExponential(2)}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              )}
              <div className="space-y-1.5">
                <div className="text-[10px] text-slate-500 flex items-center gap-1">
                  <Database className="w-3 h-3" /> 所引证据（{trace.cited.length} 条）
                </div>
                {trace.cited.length === 0 && <div className="text-slate-500 text-[11px]">这条声明没有直接引用记录（computed / inference 靠输入声明）。</div>}
                {trace.cited.map((c, i) => (
                  <div key={i} className="p-2 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] space-y-0.5">
                    <div className="text-cyan-300 break-all">{c.ref.evidence_id}</div>
                    {!c.found ? (
                      <div className="text-rose-300">本次运行的工具结果里没有这个证据编号</div>
                    ) : (
                      <>
                        {c.ref.field && (
                          <div>
                            字段 {c.ref.field}：记录值 <span className="text-white">{fmt(c.record_value)}</span> {c.record_unit || ''}
                            {c.ref.value !== null && c.ref.value !== undefined && (
                              <>
                                {' '}
                                · 声明写的 <span className="text-white">{fmt(c.ref.value)}</span> {c.ref.unit || ''}
                              </>
                            )}
                          </div>
                        )}
                        {c.ref.quote && <div className="text-slate-300 font-sans">引文：“{c.ref.quote}”</div>}
                        {typeof c.record?.title === 'string' && <div className="text-slate-400 font-sans">{c.record.title}</div>}
                        {typeof c.record?.text_excerpt === 'string' && (
                          <div className="text-slate-500 font-sans line-clamp-3">{c.record.text_excerpt}</div>
                        )}
                      </>
                    )}
                  </div>
                ))}
              </div>
              <div className="text-[10px] text-slate-500 font-mono">运行文件 {trace.run_file}</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
