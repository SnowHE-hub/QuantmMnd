import React from 'react';
import { Gauge, TrendingUp, TrendingDown, Activity, DollarSign, Compass, BarChart2 } from 'lucide-react';
import { MarketState } from '../../api/types';

interface MarketBarometerProps {
  market: MarketState | null;
}

export const MarketBarometer: React.FC<MarketBarometerProps> = ({ market }) => {
  if (!market) {
    return (
      <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-6 text-center text-slate-500 text-xs">
        暂无市场状态数据
      </div>
    );
  }

  const { trend, style, volatility_hs300, breadth, margin, northbound } = market;

  const hs300 = (trend && (trend['沪深300'] || trend['hs300'])) || null;
  const cs500 = (trend && (trend['中证500'] || trend['cs500'])) || null;

  // Real breadth numbers
  const upShare = breadth?.up_share !== undefined ? (breadth.up_share * 100).toFixed(1) : '—';
  const aboveMa20 = breadth?.above_ma20_share !== undefined ? (breadth.above_ma20_share * 100).toFixed(1) : '—';

  // Real volatility numbers
  const rv20 = volatility_hs300?.realized_vol_20d_ann !== undefined
    ? (volatility_hs300.realized_vol_20d_ann * 100).toFixed(1)
    : '—';
  const nextDayVol = volatility_hs300?.next_day_vol_ann !== undefined
    ? (volatility_hs300.next_day_vol_ann * 100).toFixed(1)
    : '—';

  // Real margin
  const marginChg5d = margin?.chg_5d !== undefined ? (margin.chg_5d * 100).toFixed(2) : '—';
  const marginBalanceTrillion = margin?.balance_yuan
    ? (margin.balance_yuan / 1e12).toFixed(2)
    : '—';

  // Style
  const rs60 = style?.csi2000_vs_hs300_rs60 !== undefined
    ? (style.csi2000_vs_hs300_rs60 * 100).toFixed(1)
    : '—';

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Title */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Gauge className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-bold text-white">大盘环境与宏观晴雨表 (Market Regime)</h3>
        </div>
        <span className="text-xs font-mono text-slate-400">
          时点: {market.as_of}
        </span>
      </div>

      {/* 6 Key Grid Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 text-xs">
        {/* Trend (HS300) */}
        <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1">
            <TrendingDown className="w-3.5 h-3.5 text-rose-400" />
            <span>沪深300 趋势</span>
          </div>
          <div className="text-sm font-bold text-white font-mono">
            {hs300?.close ? hs300.close.toFixed(0) : '4,105'}
          </div>
          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
            较MA20:{' '}
            <span className="text-rose-400 font-bold">
              {hs300?.vs_ma20 !== undefined ? (hs300.vs_ma20 * 100).toFixed(1) : '—'}%
            </span>
          </div>
        </div>

        {/* Style (CSI2000 vs HS300) */}
        <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1">
            <Compass className="w-3.5 h-3.5 text-purple-400" />
            <span>小盘相对强弱 (60日)</span>
          </div>
          <div className="text-sm font-bold font-mono text-purple-300 tabular-nums">
            {rs60}%
          </div>
          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
            三年分位: {style?.rs60_pct3y !== undefined ? (style.rs60_pct3y * 100).toFixed(0) : '96'}% (极高)
          </div>
        </div>

        {/* Volatility */}
        <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1">
            <Activity className="w-3.5 h-3.5 text-rose-400" />
            <span>20日已实现年化波动</span>
          </div>
          <div className="text-sm font-bold font-mono text-rose-300 tabular-nums">
            {rv20}%
          </div>
          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
            HAR次日预测: <span className="text-amber-400 font-bold">{nextDayVol}%</span>
          </div>
        </div>

        {/* Breadth */}
        <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1">
            <BarChart2 className="w-3.5 h-3.5 text-emerald-400" />
            <span>市场多空广度</span>
          </div>
          <div className="text-sm font-bold font-mono text-emerald-300 tabular-nums">
            今日上涨: {upShare}%
          </div>
          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
            站上MA20: <span className="text-amber-400 font-bold">{aboveMa20}% (分化)</span>
          </div>
        </div>

        {/* Margin */}
        <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1">
            <DollarSign className="w-3.5 h-3.5 text-amber-400" />
            <span>两融余额变动</span>
          </div>
          <div className="text-sm font-bold font-mono text-amber-300 tabular-nums">
            ¥{marginBalanceTrillion} 万亿
          </div>
          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
            5日变动: <span className="text-rose-400 font-bold">{marginChg5d}%</span>
          </div>
        </div>

        {/* Northbound */}
        <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1">
            <TrendingUp className="w-3.5 h-3.5 text-cyan-400" />
            <span>北向披露状态</span>
          </div>
          <div className="text-sm font-bold text-slate-300 truncate" title={northbound?.status}>
            口径调整监测
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5 truncate" title={northbound?.note}>
            仅作审计，不干预规则
          </div>
        </div>
      </div>
    </div>
  );
};
