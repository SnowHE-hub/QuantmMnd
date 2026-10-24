import React from 'react';
import { X, Sparkles, CheckCircle2, AlertTriangle, ShieldAlert, ArrowRight, Gauge, DollarSign } from 'lucide-react';
import { FormalReport, BrainResponse, MarketState, HealthInfo } from '../../api/types';

interface DailyRitualModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedDate: string;
  health: HealthInfo | null;
  formal: FormalReport | null;
  brain: BrainResponse | null;
  market: MarketState | null;
  onSelectTicker: (ticker: string) => void;
  onNavigateTab: (tab: any) => void;
  onOpenBriefingExporter?: () => void;
}

export const DailyRitualModal: React.FC<DailyRitualModalProps> = ({
  isOpen,
  onClose,
  selectedDate,
  health,
  formal,
  brain,
  market,
  onSelectTicker,
  onNavigateTab,
  onOpenBriefingExporter,
}) => {
  if (!isOpen) return null;

  const orders = formal?.tiers?.aum_1e5?.orders_next_open || [];
  const flags = brain?.memo?.flags || [];
  const skippedCount = orders.filter((o) => o.est_shares === 0).length;
  // 市场描述全部来自当天的 m_market（原来是写死的 10-23 文案）
  const hs = (market?.trend as any)?.['沪深300'];
  const vol = (market?.volatility_hs300 as any) || {};
  const p = (v: any, d = 1) => (typeof v === 'number' && Number.isFinite(v) ? `${v > 0 ? '+' : ''}${(v * 100).toFixed(d)}%` : '—');
  const weak = hs && hs.vs_ma20 < 0 && hs.vs_ma60 < 0;
  const volUp = typeof vol.next_day_vol_ann === 'number' && typeof vol.realized_vol_20d_ann === 'number' && vol.next_day_vol_ann > vol.realized_vol_20d_ann;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-2xl bg-[#0c1220] border border-slate-700 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="p-4 bg-gradient-to-r from-purple-900/40 via-blue-900/40 to-slate-900/40 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center text-white shadow-md">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                每日收盘 3 分钟投研复盘向导
              </h3>
              <p className="text-[11px] text-purple-300/90 font-mono">
                截面: {selectedDate} · 4步闭环复盘系统产出与执行计划
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Scrollable Ritual Steps */}
        <div className="p-5 overflow-y-auto space-y-4 text-xs">
          {/* Step 1: Pipeline & Data Status */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-200 flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-blue-600/30 text-blue-400 flex items-center justify-center text-[10px] font-mono">
                  1
                </span>
                <span>今日收盘流水线与数据就绪状态</span>
              </span>
              <span className="text-emerald-400 font-mono flex items-center gap-1 font-bold">
                <CheckCircle2 className="w-3.5 h-3.5" /> ALL DONE
              </span>
            </div>
            <p className="text-slate-300 leading-relaxed text-[11px]">
              Tushare 湖表、巨潮公告与卖方预测已完成时点增量合并，GPU 前瞻构建器顺利生成日特征宽表，模型打分与前瞻影子账本均已入库（最新数据日期: {health?.latest_run || selectedDate}）。
            </p>
          </div>

          {/* Step 2: Market Regime View */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-200 flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-purple-600/30 text-purple-400 flex items-center justify-center text-[10px] font-mono">
                  2
                </span>
                <span>大盘宏观与微观波动晴雨表</span>
              </span>
              <span className="text-amber-400 font-mono text-[11px]">{hs ? (weak ? '指数在 MA20、MA60 之下' : '指数未同时跌破 MA20 / MA60') : '无市场数据'}{volUp ? ' · 预测波动高于近 20 日' : ''}</span>
            </div>
            <p className="text-slate-300 leading-relaxed text-[11px]">
              沪深300 相对 MA20 {p(hs?.vs_ma20)}、相对 MA60 {p(hs?.vs_ma60)}，近 20 日 {p(hs?.ret_20d)}；HAR 次日波动预测（年化）{p(vol.next_day_vol_ann, 1).replace('+', '')}，近 20 日已实现 {p(vol.realized_vol_20d_ann, 1).replace('+', '')}（3 年分位 {typeof vol.realized_vol_20d_pct3y === 'number' ? (vol.realized_vol_20d_pct3y * 100).toFixed(0) : '—'}%）。
            </p>
          </div>

          {/* Step 3: Formal Recommendation & Orders */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-200 flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-emerald-600/30 text-emerald-400 flex items-center justify-center text-[10px] font-mono">
                  3
                </span>
                <span>三档账户净值与明日开盘委托买卖</span>
              </span>
              <span className="text-cyan-400 font-mono text-[11px]">
                {orders.length} 笔委托（{skippedCount} 笔留现金）
              </span>
            </div>
            <p className="text-slate-300 leading-relaxed text-[11px]">
              10 万档{formal?.mode === 'preview' ? '（建仓预览，首次调仓 ' + (formal?.next_rebalance || '—') + '）' : ''}共 {orders.length} 笔委托，其中 {skippedCount} 笔买不起一个最小申报单位，按正式规则跳过、现金保留。
            </p>
          </div>

          {/* Step 4: Red Team Risk Warnings */}
          <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-200 flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-rose-600/30 text-rose-400 flex items-center justify-center text-[10px] font-mono">
                  4
                </span>
                <span>Agent 大脑与红队风控重点标的</span>
              </span>
              <span className="text-rose-400 font-mono text-[11px]">
                {flags.length} 只标的受标记
              </span>
            </div>

            <div className="space-y-1.5 pt-1">
              {flags.map((f, idx) => (
                <div
                  key={idx}
                  onClick={() => {
                    onSelectTicker(f.ticker);
                    onClose();
                  }}
                  className="p-2 rounded-lg bg-slate-950 border border-slate-800 hover:border-blue-500 cursor-pointer flex items-center justify-between transition"
                >
                  <div>
                    <span className="font-mono font-bold text-cyan-300 mr-2">{f.ticker}</span>
                    <span className="text-slate-300 text-[11px]">{f.reason.slice(0, 60)}...</span>
                  </div>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-500 shrink-0 ml-2" />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 bg-slate-950 border-t border-slate-800 flex items-center justify-between">
          <span className="text-[11px] text-slate-500">3 分钟复盘完毕 · 把握全局因果</span>
          <div className="flex items-center gap-2">
            {onOpenBriefingExporter && (
              <button
                onClick={() => {
                  onClose();
                  onOpenBriefingExporter();
                }}
                className="px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs transition"
              >
                导出今日执行简报
              </button>
            )}
            <button
              onClick={() => {
                onNavigateTab('cockpit');
                onClose();
              }}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs transition shadow-sm"
            >
              进入投研工作台执行
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
