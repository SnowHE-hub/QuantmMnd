import React from 'react';
import { ShieldAlert, Activity, AlertOctagon, TrendingUp, AlertTriangle } from 'lucide-react';
import { RiskCardItem } from '../../api/types';

interface RiskCardRadarProps {
  riskCard: any;
  ticker: string;
}

export const RiskCardRadar: React.FC<RiskCardRadarProps> = ({ riskCard, ticker }) => {
  if (!riskCard) {
    return (
      <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 text-center text-slate-500 text-xs">
        该标的在此截面下未命中风险执行卡
      </div>
    );
  }

  const metrics = [
    {
      label: '市场 Beta (252日)',
      value: riskCard.beta_252 !== undefined ? Number(riskCard.beta_252).toFixed(2) : '—',
      desc: riskCard.beta_252 && riskCard.beta_252 > 1.3 ? '高 Beta (高弹性标的)' : '稳健 Beta',
      warn: Boolean(riskCard.beta_252 && riskCard.beta_252 > 1.5),
    },
    {
      label: '历史年化波动率 (60日)',
      value:
        riskCard.vol_60_ann !== undefined
          ? `${(riskCard.vol_60_ann * 100).toFixed(1)}%`
          : '—',
      desc: '过去 60 交易日已实现年化波动',
      warn: Boolean(riskCard.vol_60_ann && riskCard.vol_60_ann > 0.45),
    },
    {
      label: 'HAR 次日波动率预测',
      value:
        riskCard.har_next_day_vol_ann !== undefined
          ? `${(riskCard.har_next_day_vol_ann * 100).toFixed(1)}%`
          : '—',
      desc: '时序模型预测的明日开盘波动',
      warn: Boolean(riskCard.har_next_day_vol_ann && riskCard.har_next_day_vol_ann > 0.5),
    },
    {
      label: '动量指标 (20日 / 60日 / 250日)',
      value: `${riskCard.mom_20 !== undefined ? (riskCard.mom_20 * 100).toFixed(1) : '—'}% / ${
        riskCard.mom_60 !== undefined ? (riskCard.mom_60 * 100).toFixed(1) : '—'
      }% / ${riskCard.mom_250 !== undefined ? (riskCard.mom_250 * 100).toFixed(1) : '—'}%`,
      desc: '短 / 中 / 长期价格趋势动量',
      warn: Boolean(riskCard.mom_20 && riskCard.mom_20 < -0.3),
    },
    {
      label: '跳空开盘幅度 (p95)',
      value:
        riskCard.gap_p95_250 !== undefined
          ? `${(riskCard.gap_p95_250 * 100).toFixed(2)}%`
          : riskCard.gap_freq_60 !== undefined
          ? `${(riskCard.gap_freq_60 * 100).toFixed(1)}%`
          : '—',
      desc: '过去250日跳空95分位幅度',
      warn: Boolean(riskCard.gap_p95_250 && riskCard.gap_p95_250 > 0.05),
    },
    {
      label: '涨停 / 跌停收盘次数 (250日)',
      value: `${riskCard.limit_up_close_250 ?? '—'} 次 / ${
        riskCard.limit_down_close_250 ?? '—'
      } 次`,
      desc: '极端行情流动性锁定统计',
      warn: Boolean((riskCard.limit_down_close_250 ?? 0) >= 2),
    },
    {
      label: '异动偏离累计 (3日 / 10日)',
      value: `${
        riskCard.av_3d_cum_dev !== undefined
          ? `${(riskCard.av_3d_cum_dev * 100).toFixed(1)}%`
          : '—'
      } / ${
        riskCard.av_10d_cum_dev !== undefined
          ? `${(riskCard.av_10d_cum_dev * 100).toFixed(1)}%`
          : '—'
      }`,
      desc: '交易所异常波动距离监控',
      warn: Boolean(riskCard.av_3d_cum_dev && Math.abs(riskCard.av_3d_cum_dev) > 0.15),
    },
    {
      label: '除权除息与规则状态',
      value: riskCard.ex_dividend_next20d ? '未来20日除权' : '无除权计划',
      desc: riskCard.rules_status || '—',
      warn: Boolean(riskCard.ex_dividend_next20d),
    },
  ];

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-rose-400" />
          <h3 className="text-sm font-bold text-white">F5c 风险与执行卡指标 (Risk & Execution Cards)</h3>
        </div>
        <span className="text-xs font-mono text-cyan-300">{ticker}</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        {metrics.map((m, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-lg border transition ${
              m.warn
                ? 'bg-rose-950/20 border-rose-800/60'
                : 'bg-slate-900/60 border-slate-800'
            }`}
          >
            <div className="text-[11px] text-slate-400 font-medium mb-1">{m.label}</div>
            <div
              className={`text-sm font-bold font-mono tabular-nums ${
                m.warn ? 'text-rose-400' : 'text-slate-100'
              }`}
            >
              {m.value}
            </div>
            <div className="text-[10px] text-slate-500 mt-1 truncate" title={m.desc}>
              {m.desc}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
