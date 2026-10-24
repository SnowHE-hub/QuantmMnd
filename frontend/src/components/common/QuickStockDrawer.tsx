import React, { useState, useEffect } from 'react';
import {
  X,
  TrendingUp,
  ShieldAlert,
  FileCheck2,
  Cpu,
  ArrowRight,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  Layers,
  Sparkles,
  Swords,
} from 'lucide-react';
import { quantmindApi } from '../../api/client';
import { adaptRiskCard, adaptClaimItem } from '../../api/adapters';
import { analyticsApi, ConsensusResponse } from '../../api/analytics';
import { ModelConsensus } from './ModelConsensus';
import { CheckBadge, BrainActionBadge } from './Badge';

interface QuickStockDrawerProps {
  ticker: string | null;
  selectedDate: string;
  onClose: () => void;
  onOpenFullTerminal: (ticker: string) => void;
  onAddToCompare?: (ticker: string) => void;
}

export const QuickStockDrawer: React.FC<QuickStockDrawerProps> = ({
  ticker,
  selectedDate,
  onClose,
  onOpenFullTerminal,
  onAddToCompare,
}) => {
  const [loading, setLoading] = useState(false);
  const [riskCard, setRiskCard] = useState<any>(null);
  const [claims, setClaims] = useState<any[]>([]);
  const [brainFlag, setBrainFlag] = useState<any>(null);
  const [bypassEv, setBypassEv] = useState<any>(null);
  const [consensus, setConsensus] = useState<ConsensusResponse | null>(null);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);

    Promise.allSettled([
      // 1. Risk card
      quantmindApi.getRiskCards(selectedDate, ticker).then((res) => (res && res.length > 0 ? adaptRiskCard(res[0]) : null)),
      // 2. Claims
      quantmindApi.getResearchTicker(selectedDate, ticker).then((res) => (res.claims || []).map(adaptClaimItem)),
      // 3. Brain flag
      quantmindApi.getBrain(selectedDate).then((b) => b?.memo?.flags?.find((f) => f.ticker === ticker) || null),
      // 4. Bypass event
      quantmindApi.getBypass(selectedDate).then((list) => list?.find((e) => e.ticker === ticker) || null),
      // 5. Model scores
      analyticsApi.consensus(selectedDate, ticker),
    ]).then(([rc, cl, bf, be, sc]) => {
      if (rc.status === 'fulfilled') setRiskCard(rc.value);
      if (cl.status === 'fulfilled') setClaims(cl.value || []);
      if (bf.status === 'fulfilled') setBrainFlag(bf.value);
      if (be.status === 'fulfilled') setBypassEv(be.value);
      setConsensus(sc.status === 'fulfilled' ? sc.value : null);
      setLoading(false);
    });
  }, [ticker, selectedDate]);

  if (!ticker) return null;

  // Claim check stats
  const totalClaims = claims.length;
  const verifiedCount = claims.filter((c) => c.check === 'verified').length;
  const mismatchCount = claims.filter((c) => c.check === 'mismatch').length;
  const minorCount = claims.filter((c) => c.check === 'minor_mismatch').length;
  const unverifiableCount = totalClaims - verifiedCount - mismatchCount - minorCount;

  const verifiedPct = totalClaims > 0 ? ((verifiedCount / totalClaims) * 100).toFixed(0) : '0';
  const mismatchPct = totalClaims > 0 ? ((mismatchCount / totalClaims) * 100).toFixed(0) : '0';

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end bg-black/60 backdrop-blur-xs animate-in fade-in duration-200">
      <div
        className="w-full max-w-xl bg-[#0c1220] border-l border-slate-800 shadow-2xl h-full flex flex-col justify-between"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Header */}
        <div className="p-4 border-b border-slate-800 bg-slate-900/60 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400 font-mono font-bold text-xs">
              QM
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-base font-bold text-white">{ticker}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-blue-950 text-blue-300 border border-blue-800/60 font-mono">
                  {selectedDate}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">个股决策归因与因果履历速览抽屉</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {onAddToCompare && (
              <button
                onClick={() => onAddToCompare(ticker)}
                className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-purple-300 border border-purple-800/80 text-xs font-medium flex items-center gap-1 transition"
                title="加入同屏 PK 竞技场"
              >
                <Swords className="w-3 h-3" />
                <span>加入PK</span>
              </button>
            )}
            <button
              onClick={() => {
                onOpenFullTerminal(ticker);
                onClose();
              }}
              className="px-2.5 py-1 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium flex items-center gap-1 transition shadow-sm"
            >
              <span>进入深度终端</span>
              <ExternalLink className="w-3 h-3" />
            </button>
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="p-5 overflow-y-auto space-y-4 flex-1 text-xs">
          {loading ? (
            <div className="py-20 text-center text-xs text-blue-400 font-mono animate-pulse">
              正在全链路穿透 {ticker} 的量化打分、风险卡与智能体证据...
            </div>
          ) : (
            <>
              {/* 1. Agent Brain & Bypass Alert Bar (if any) */}
              {(brainFlag || bypassEv) && (
                <div className="p-3 rounded-xl bg-amber-950/30 border border-amber-800/60 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-amber-300 flex items-center gap-1.5">
                      <ShieldAlert className="w-4 h-4 text-amber-400" />
                      <span>红队预警与大脑标记</span>
                    </span>
                    {brainFlag && <BrainActionBadge action={brainFlag.action} />}
                  </div>
                  {brainFlag && (
                    <p className="text-[11px] text-amber-200/90 leading-relaxed">
                      {brainFlag.reason}
                    </p>
                  )}
                  {bypassEv && (
                    <div className="text-[10px] text-slate-400 font-mono pt-1 border-t border-amber-900/40">
                      触发旁路规则: {bypassEv.rules?.join(', ')} ({bypassEv.importance} 级别)
                    </div>
                  )}
                </div>
              )}

              {/* 2. L4: 四个模型的截面百分位（后端 /api/consensus） */}
              <ModelConsensus data={consensus} title="阶段 L4: F3 四个模型的截面百分位（研究池）" />

              {/* 3. L5: Risk Card Snapshot */}
              <div className="p-3.5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-2.5">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
                  <div className="font-semibold text-white flex items-center gap-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                    <span>阶段 L5: F5c 风险与执行特征</span>
                  </div>
                  <span className="text-[11px] font-mono text-slate-400">
                    行业: {riskCard?.industry_tushare || '—'}
                  </span>
                </div>

                {riskCard ? (
                  <div className="grid grid-cols-3 gap-2 font-mono text-[11px]">
                    <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                      <div className="text-[10px] text-slate-400 font-sans">市场 Beta (252)</div>
                      <div
                        className={`text-xs font-bold mt-0.5 ${
                          riskCard.beta_252 > 2.0 ? 'text-rose-400' : 'text-slate-200'
                        }`}
                      >
                        {riskCard.beta_252 !== undefined ? riskCard.beta_252.toFixed(2) : '—'}
                      </div>
                    </div>
                    <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                      <div className="text-[10px] text-slate-400 font-sans">60日年化波动</div>
                      <div className="text-xs font-bold text-slate-200 mt-0.5">
                        {riskCard.vol_60_ann !== undefined
                          ? `${(riskCard.vol_60_ann * 100).toFixed(1)}%`
                          : '—'}
                      </div>
                    </div>
                    <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                      <div className="text-[10px] text-slate-400 font-sans">HAR次日预测</div>
                      <div className="text-xs font-bold text-amber-400 mt-0.5">
                        {riskCard.har_next_day_vol_ann !== undefined
                          ? `${(riskCard.har_next_day_vol_ann * 100).toFixed(1)}%`
                          : '—'}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-slate-500 italic py-1">未收录该标的风险执行卡</div>
                )}
              </div>

              {/* 4. L8 (F7): Code Verification Pass Rate Bar */}
              <div className="p-3.5 rounded-xl bg-slate-900/50 border border-slate-800 space-y-2.5">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
                  <div className="font-semibold text-white flex items-center gap-1.5">
                    <FileCheck2 className="w-3.5 h-3.5 text-emerald-400" />
                    <span>阶段 L8: F7 代码核验声明履历</span>
                  </div>
                  <span className="font-mono text-xs text-slate-300">
                    {totalClaims} 条事实声明
                  </span>
                </div>

                {totalClaims > 0 ? (
                  <div className="space-y-2">
                    {/* Visual Progress Ratio Bar */}
                    <div className="h-3 rounded-full bg-slate-800 overflow-hidden flex">
                      <div
                        style={{ width: `${verifiedPct}%` }}
                        className="bg-emerald-500 h-full"
                        title={`已核对: ${verifiedCount} 条 (${verifiedPct}%)`}
                      />
                      <div
                        style={{ width: `${totalClaims > 0 ? (minorCount / totalClaims) * 100 : 0}%` }}
                        className="bg-amber-400 h-full"
                        title={`微小差异: ${minorCount} 条`}
                      />
                      <div
                        style={{ width: `${mismatchPct}%` }}
                        className="bg-rose-500 h-full"
                        title={`不匹配: ${mismatchCount} 条 (${mismatchPct}%)`}
                      />
                    </div>

                    <div className="flex items-center justify-between text-[11px] font-mono">
                      <span className="text-emerald-400 font-bold">已核实: {verifiedCount} 条 ({verifiedPct}%)</span>
                      {minorCount > 0 && <span className="text-amber-400">小差异: {minorCount} 条</span>}
                      {mismatchCount > 0 && <span className="text-rose-400 font-bold">不匹配: {mismatchCount} 条</span>}
                      <span className="text-slate-400">无证据: {unverifiableCount} 条</span>
                    </div>

                    {/* Sample Claims List */}
                    <div className="space-y-1.5 mt-2 max-h-48 overflow-y-auto pr-1">
                      {claims.slice(0, 4).map((c, idx) => (
                        <div
                          key={idx}
                          className="p-2 rounded bg-slate-950 border border-slate-800/80 text-[11px] space-y-1"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-mono font-bold text-cyan-400">{c.ref}</span>
                            <CheckBadge status={c.check} />
                          </div>
                          <p className="text-slate-300 line-clamp-2">{c.text}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-slate-500 italic py-2">
                    该标的在本期未触发 A1-A4 专职智能体深度研究（可从操作中心手动触发）
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Footer Actions */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between">
          <span className="text-[11px] text-slate-500 font-mono">快捷键: Esc 退出</span>
          <button
            onClick={() => {
              onOpenFullTerminal(ticker);
              onClose();
            }}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs flex items-center gap-1.5 transition shadow-sm"
          >
            <span>进入个股深度透视终端</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
