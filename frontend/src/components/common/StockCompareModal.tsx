import React, { useState, useEffect } from 'react';
import {
  X,
  Swords,
  TrendingUp,
  ShieldAlert,
  FileCheck2,
  Cpu,
  BarChart2,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  ExternalLink,
} from 'lucide-react';
import { EChart as ReactECharts, EChartHandle } from './EChart';
import { quantmindApi } from '../../api/client';
import { adaptRiskCard, adaptClaimItem, safeNumber } from '../../api/adapters';
import { analyticsApi, ConsensusResponse } from '../../api/analytics';
import { RiskCardItem, ClaimItem, StockPriceRow } from '../../api/types';

interface StockCompareModalProps {
  isOpen: boolean;
  onClose: () => void;
  tickerA: string;
  tickerB: string;
  selectedDate: string;
  onSelectTicker: (ticker: string) => void;
}

interface StockProfile {
  ticker: string;
  riskCard: RiskCardItem | null;
  claims: ClaimItem[];
  scores: ConsensusResponse | null;
  prices: StockPriceRow[];
  loading: boolean;
}

export const StockCompareModal: React.FC<StockCompareModalProps> = ({
  isOpen,
  onClose,
  tickerA,
  tickerB,
  selectedDate,
  onSelectTicker,
}) => {
  const [profileA, setProfileA] = useState<StockProfile>({
    ticker: tickerA,
    riskCard: null,
    claims: [],
    scores: null,
    prices: [],
    loading: true,
  });

  const [profileB, setProfileB] = useState<StockProfile>({
    ticker: tickerB,
    riskCard: null,
    claims: [],
    scores: null,
    prices: [],
    loading: true,
  });

  useEffect(() => {
    if (!isOpen || !tickerA) return;

    const loadStock = async (ticker: string, setProfile: (p: any) => void) => {
      setProfile((prev: any) => ({ ...prev, ticker, loading: true }));
      try {
        const [rcRes, clRes, scRes, spRes] = await Promise.allSettled([
          quantmindApi.getRiskCards(selectedDate, ticker).then((res) => (res && res.length > 0 ? adaptRiskCard(res[0]) : null)),
          quantmindApi.getResearchTicker(selectedDate, ticker).then((res) => (res.claims || []).map(adaptClaimItem)),
          analyticsApi.consensus(selectedDate, ticker),
          quantmindApi.getStockPrices(ticker, '2026-08-01', selectedDate),
        ]);

        setProfile({
          ticker,
          riskCard: rcRes.status === 'fulfilled' ? rcRes.value : null,
          claims: clRes.status === 'fulfilled' ? clRes.value || [] : [],
          scores: scRes.status === 'fulfilled' ? scRes.value : null,
          prices: spRes.status === 'fulfilled' ? spRes.value || [] : [],
          loading: false,
        });
      } catch (e) {
        console.warn(`Error loading comparison for ${ticker}:`, e);
        setProfile((prev: any) => ({ ...prev, loading: false }));
      }
    };

    loadStock(tickerA, setProfileA);
    if (tickerB) {
      loadStock(tickerB, setProfileB);
    }
  }, [isOpen, tickerA, tickerB, selectedDate]);

  if (!isOpen) return null;

  // 模型对比用截面百分位（0–100）：原实现直接画 z 分、上限取 max(…, 4)，负分被压到圆心，雷达图会变形
  const rowA = profileA.scores?.row || null;
  const rowB = profileB.scores?.row || null;
  const conA = { badgeLabel: rowA?.label ?? '无打分', badgeClass: 'bg-slate-800 text-slate-300 border-slate-700', description: rowA ? `ENS 百分位 ${rowA.ENS_pct?.toFixed(1)}，分歧度 ${rowA.dispersion_pp?.toFixed(1)} 个百分点` : '不在研究池截面' };
  const conB = { badgeLabel: rowB?.label ?? '无打分', badgeClass: 'bg-slate-800 text-slate-300 border-slate-700', description: rowB ? `ENS 百分位 ${rowB.ENS_pct?.toFixed(1)}，分歧度 ${rowB.dispersion_pp?.toFixed(1)} 个百分点` : '不在研究池截面' };

  // Claim verified percentages
  const claimsA = profileA.claims;
  const verifiedA = claimsA.filter((c) => c.check === 'verified').length;
  const pctA = claimsA.length > 0 ? ((verifiedA / claimsA.length) * 100).toFixed(0) : '0';

  const claimsB = profileB.claims;
  const verifiedB = claimsB.filter((c) => c.check === 'verified').length;
  const pctB = claimsB.length > 0 ? ((verifiedB / claimsB.length) * 100).toFixed(0) : '0';

  // Model Radar Option
  const modelLabels = ['XGBoost', 'CatBoost', 'TabM', 'SeqFusion', 'ENS综合'];
  const pctVals = (r: typeof rowA) => (r ? ['xgb_pct', 'cat_pct', 'tabm_pct', 'seqfusion_pct', 'ENS_pct'].map((k) => safeNumber(r[k])) : [0, 0, 0, 0, 0]);

  const radarOption = {
    tooltip: { trigger: 'item' },
    legend: {
      data: [tickerA, tickerB || '待选标的'],
      textStyle: { color: '#94a3b8', fontSize: 11 },
      top: 5,
    },
    radar: {
      indicator: modelLabels.map((name) => ({ name, max: 100, min: 0 })),
      radius: '65%',
      splitNumber: 4,
      axisName: { color: '#cbd5e1', fontSize: 11 },
      splitLine: { lineStyle: { color: '#334155' } },
      splitArea: { show: false },
      axisLine: { lineStyle: { color: '#475569' } },
    },
    series: [
      {
        type: 'radar',
        data: [
          {
            value: pctVals(rowA),
            name: tickerA,
            itemStyle: { color: '#06b6d4' },
            areaStyle: { color: 'rgba(6, 182, 212, 0.25)' },
          },
          ...(tickerB
            ? [
                {
                  value: pctVals(rowB),
                  name: tickerB,
                  itemStyle: { color: '#a855f7' },
                  areaStyle: { color: 'rgba(168, 85, 247, 0.25)' },
                },
              ]
            : []),
        ],
      },
    ],
  };

  // Automated PK Verdict
  const ensA = safeNumber(rowA?.ENS_pct);
  const ensB = safeNumber(rowB?.ENS_pct);
  const volA = safeNumber(profileA.riskCard?.vol_60_ann);
  const volB = safeNumber(profileB.riskCard?.vol_60_ann);

  let verdictHeadline = '多维量化竞技场对比综述';
  let verdictDetail = '';

  if (ensA > ensB && volA <= volB) {
    verdictHeadline = `标的 ${tickerA} 在 Alpha 预期与风险控制上全面占优`;
    verdictDetail = `${tickerA} ENS 研究池百分位 (${ensA.toFixed(1)} vs ${ensB.toFixed(1)}) 更高，且 60 日年化波动 (${(volA * 100).toFixed(1)}% vs ${(volB * 100).toFixed(1)}%) 更低。（打分是截面排序信号，不是收益预测；两者不能直接换算成夏普）`;
  } else if (ensA > ensB) {
    verdictHeadline = `标的 ${tickerA} 进攻性更强，标的 ${tickerB} 防御属性更优`;
    verdictDetail = `${tickerA} ENS 研究池百分位更高 (${ensA.toFixed(1)} vs ${ensB.toFixed(1)})，但 ${tickerB} 波动率 (${(volB * 100).toFixed(1)}% vs ${(volA * 100).toFixed(1)}%) 显著更低。`;
  } else {
    verdictHeadline = `标的 ${tickerB} 在本期模型打分中处于优势`;
    verdictDetail = `${tickerB} ENS 研究池百分位领先 (${ensB.toFixed(1)} vs ${ensA.toFixed(1)})，F7 声明核验通过率为 ${pctB}%。`;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-5xl bg-[#0c1220] border border-slate-700 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Header */}
        <div className="p-4 bg-gradient-to-r from-blue-950/70 via-slate-900 to-purple-950/70 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-600 to-purple-600 flex items-center justify-center text-white shadow-md">
              <Swords className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                双股同屏 PK 竞技场 (Side-by-Side Arena)
              </h3>
              <p className="text-[11px] text-slate-400 font-mono">
                截面: {selectedDate} · 全要素模型打分、风险特征与智能体代码核验并排裁决
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-5 overflow-y-auto space-y-5 text-xs flex-1">
          {/* Top Automated PK Verdict Banner */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-cyan-950/40 via-slate-900 to-purple-950/40 border border-slate-700/80 space-y-1.5">
            <div className="flex items-center gap-2 text-cyan-300 font-bold text-sm">
              <CheckCircle2 className="w-4 h-4 text-cyan-400" />
              <span>{verdictHeadline}</span>
            </div>
            <p className="text-slate-300 leading-relaxed text-xs">{verdictDetail}</p>
          </div>

          {/* Side-by-Side Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Stock A Card */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-cyan-800/60 space-y-3.5 relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 to-blue-500" />
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-lg font-bold text-cyan-300">{tickerA}</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded border ${conA.badgeClass}`}>
                      {conA.badgeLabel}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    {profileA.riskCard?.industry_tushare || '行业未知'} · {conA.description}
                  </div>
                </div>
                <button
                  onClick={() => {
                    onSelectTicker(tickerA);
                    onClose();
                  }}
                  className="px-2.5 py-1 rounded bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-800 text-xs font-mono flex items-center gap-1 transition"
                >
                  <span>透视终端</span>
                  <ExternalLink className="w-3 h-3" />
                </button>
              </div>

              {/* Metric Highlights */}
              <div className="grid grid-cols-3 gap-2 text-center font-mono text-[11px]">
                <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] text-slate-400">ENS 20d 研究池百分位</div>
                  <div className="text-sm font-bold text-cyan-400 mt-0.5">
                    {rowA ? ensA.toFixed(1) : "—"}
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] text-slate-400">Beta (252)</div>
                  <div className="text-sm font-bold text-white mt-0.5">
                    {profileA.riskCard?.beta_252 !== undefined ? profileA.riskCard.beta_252.toFixed(2) : '—'}
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
                  <div className="text-[10px] text-slate-400">60日年化波动</div>
                  <div className="text-sm font-bold text-amber-400 mt-0.5">
                    {volA > 0 ? `${(volA * 100).toFixed(1)}%` : '—'}
                  </div>
                </div>
              </div>

              {/* F7 Verification Bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-400">F7 代码核验通过率:</span>
                  <span className="text-emerald-400 font-bold font-mono">{pctA}% ({verifiedA}/{claimsA.length} 条)</span>
                </div>
                <div className="h-2 rounded-full bg-slate-800 overflow-hidden flex">
                  <div style={{ width: `${pctA}%` }} className="bg-emerald-500 h-full" />
                </div>
              </div>
            </div>

            {/* Stock B Card */}
            <div className="p-4 rounded-xl bg-slate-900/60 border border-purple-800/60 space-y-3.5 relative overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-purple-500 to-pink-500" />
              {tickerB ? (
                <>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-lg font-bold text-purple-300">{tickerB}</span>
                        <span className={`text-[10px] px-2 py-0.5 rounded border ${conB.badgeClass}`}>
                          {conB.badgeLabel}
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {profileB.riskCard?.industry_tushare || '行业未知'} · {conB.description}
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        onSelectTicker(tickerB);
                        onClose();
                      }}
                      className="px-2.5 py-1 rounded bg-purple-950 hover:bg-purple-900 text-purple-300 border border-purple-800 text-xs font-mono flex items-center gap-1 transition"
                    >
                      <span>透视终端</span>
                      <ExternalLink className="w-3 h-3" />
                    </button>
                  </div>

                  {/* Metric Highlights */}
                  <div className="grid grid-cols-3 gap-2 text-center font-mono text-[11px]">
                    <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
                      <div className="text-[10px] text-slate-400">ENS 20d 研究池百分位</div>
                      <div className="text-sm font-bold text-purple-400 mt-0.5">
                        {rowB ? ensB.toFixed(1) : "—"}
                      </div>
                    </div>
                    <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
                      <div className="text-[10px] text-slate-400">Beta (252)</div>
                      <div className="text-sm font-bold text-white mt-0.5">
                        {profileB.riskCard?.beta_252 !== undefined ? profileB.riskCard.beta_252.toFixed(2) : '—'}
                      </div>
                    </div>
                    <div className="p-2 rounded bg-slate-950/70 border border-slate-800">
                      <div className="text-[10px] text-slate-400">60日年化波动</div>
                      <div className="text-sm font-bold text-amber-400 mt-0.5">
                        {volB > 0 ? `${(volB * 100).toFixed(1)}%` : '—'}
                      </div>
                    </div>
                  </div>

                  {/* F7 Verification Bar */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-slate-400">F7 代码核验通过率:</span>
                      <span className="text-emerald-400 font-bold font-mono">{pctB}% ({verifiedB}/{claimsB.length} 条)</span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-800 overflow-hidden flex">
                      <div style={{ width: `${pctB}%` }} className="bg-emerald-500 h-full" />
                    </div>
                  </div>
                </>
              ) : (
                <div className="py-12 text-center text-slate-500 italic">
                  请在全站点击任意股票卡片上的“加入 PK”选择对照标的
                </div>
              )}
            </div>
          </div>

          {/* Model Radar & Comparison Chart */}
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <span className="font-semibold text-white flex items-center gap-2">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <span>L4 四族量化算法（XGB / CatBoost / TabM / SeqFusion）雷达对比</span>
              </span>
              <span className="text-[11px] text-slate-400 font-mono">
                青色: {tickerA} · 紫色: {tickerB || '未选'}
              </span>
            </div>
            <div className="h-64 w-full">
              <ReactECharts option={radarOption} style={{ height: '100%', width: '100%' }} />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between">
          <span className="text-[11px] text-slate-500 font-mono">快捷键: Esc 退出竞技场</span>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs transition"
          >
            完成裁决并返回
          </button>
        </div>
      </div>
    </div>
  );
};
