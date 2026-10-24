import React, { useMemo, useRef, useState } from 'react';
import { EChart as ReactECharts, EChartHandle } from '../common/EChart';
import { DailyRow, d10, DOWN, UP } from '../../api/ledger';

type EChartsLike = {
  getZr: () => { on: (ev: string, fn: (e: { offsetX: number; offsetY: number }) => void) => void; off: (ev: string) => void };
  containPixel: (finder: object, px: number[]) => boolean;
  convertFromPixel: (finder: object, px: number[]) => number[] | number;
};

interface Props {
  daily: DailyRow[];
  aum: number;
  notional: boolean; // 影子账本没有金额，按名义本金折算
  selectedDate: string;
  onSelectDate: (d: string) => void;
}

const BENCH: [keyof DailyRow, string, string][] = [
  ['bench_hs300', '沪深300', '#94a3b8'],
  ['bench_csi2000', '中证2000', '#a78bfa'],
  ['bench_pool_ew', '研究池等权', '#f59e0b'],
];

// 资金曲线 + 回撤：上下两个图区共用时间轴；调仓日打图钉；点击某一天，下面的持仓和流水回放到那一天
export const EquityCurve: React.FC<Props> = ({ daily, aum, notional, selectedDate, onSelectDate }) => {
  const [mode, setMode] = useState<'yuan' | 'pct'>('yuan');
  const dates = useMemo(() => daily.map((r) => d10(r.date)), [daily]);
  // 画布点击回调只注册一次，用 ref 拿到最新的日期序列和回调（切换账户后不会用到旧数据）
  const datesRef = useRef(dates);
  const onSelectRef = useRef(onSelectDate);
  datesRef.current = dates;
  onSelectRef.current = onSelectDate;
  // 曲线不画数据点，系列的 click 只有点中图钉才触发；改为监听整个画布，把像素位置换算成日期
  const onReady = (chart: unknown) => {
    const c = chart as EChartsLike;
    const h = (e: { offsetX: number; offsetY: number }) => {
      const px = [e.offsetX, e.offsetY];
      const g = c.containPixel({ gridIndex: 0 }, px) ? 0 : c.containPixel({ gridIndex: 1 }, px) ? 1 : -1;
      if (g < 0) return;
      const p = c.convertFromPixel({ gridIndex: g }, px); // [类目序号, y 值]；用 xAxisIndex 作 finder 会返回 null
      const i = Math.round(Array.isArray(p) ? p[0] : p);
      if (i >= 0 && i < datesRef.current.length) onSelectRef.current(datesRef.current[i]);
    };
    const zr = c.getZr();
    zr.on('click', h);
    return () => zr.off('click');
  };
  const option = useMemo(() => {
    const val = (r: number) => (mode === 'yuan' ? aum * (1 + r) : r * 100);
    const acct = daily.map((r) => val(r.cum_ret));
    const pins = daily
      .map((r, i) => ({ r, i }))
      .filter(({ r }) => r.is_rebalance)
      .map(({ r, i }) => ({ coord: [dates[i], acct[i]], value: '调', itemStyle: { color: '#0ea5e9' }, name: d10(r.date) }));
    const sel = dates.indexOf(selectedDate);
    return {
      animation: false,
      backgroundColor: 'transparent',
      legend: { top: 0, textStyle: { color: '#cbd5e1', fontSize: 11 }, data: ['本账户', ...BENCH.map((b) => b[1])] },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(15,23,42,0.95)',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc', fontSize: 11 },
        valueFormatter: (v: number) =>
          !Number.isFinite(v) ? '—' : mode === 'yuan' ? `¥${v.toLocaleString('zh-CN', { maximumFractionDigits: 0 })}` : `${v.toFixed(2)}%`,
      },
      axisPointer: { link: [{ xAxisIndex: 'all' }] },
      grid: [
        { left: 70, right: 20, top: 30, height: '58%' },
        { left: 70, right: 20, top: '76%', height: '16%' },
      ],
      xAxis: [
        { type: 'category', data: dates, axisLabel: { color: '#64748b', fontSize: 10 }, axisLine: { lineStyle: { color: '#334155' } } },
        { type: 'category', data: dates, gridIndex: 1, axisLabel: { show: false }, axisLine: { lineStyle: { color: '#334155' } } },
      ],
      yAxis: [
        {
          type: 'value',
          scale: true,
          axisLabel: { color: '#64748b', fontSize: 10, formatter: (v: number) => (mode === 'yuan' ? `${(v / 1e4).toFixed(1)}万` : `${v.toFixed(0)}%`) },
          splitLine: { lineStyle: { color: '#1e293b' } },
        },
        { type: 'value', gridIndex: 1, max: 0, axisLabel: { color: '#64748b', fontSize: 10, formatter: (v: number) => `${v.toFixed(0)}%` }, splitLine: { show: false } },
      ],
      series: [
        {
          name: '本账户',
          type: 'line',
          data: acct,
          symbol: 'none',
          lineStyle: { width: 2, color: '#38bdf8' },
          markPoint: { symbol: 'pin', symbolSize: 26, label: { color: '#fff', fontSize: 9 }, data: pins },
          markLine:
            sel >= 0
              ? { symbol: 'none', silent: true, lineStyle: { color: '#f8fafc', type: 'dashed' }, label: { show: false }, data: [{ xAxis: selectedDate }] }
              : undefined,
        },
        ...BENCH.filter(([k]) => daily.some((r) => typeof r[k] === 'number')).map(([k, name, color]) => ({
          name,
          type: 'line',
          data: daily.map((r) => (typeof r[k] === 'number' ? val(r[k] as number) : null)),
          symbol: 'none',
          lineStyle: { width: 1.5, color, type: 'dashed' },
        })),
        {
          name: '回撤',
          type: 'line',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: daily.map((r) => r.drawdown * 100),
          symbol: 'none',
          lineStyle: { width: 1, color: DOWN },
          areaStyle: { color: 'rgba(5,150,105,0.25)' },
        },
      ],
    };
  }, [daily, dates, mode, aum, selectedDate]);

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-4 space-y-2" data-testid="equity-curve">
      <div className="flex items-center justify-between">
        <div className="text-sm font-bold text-white">
          资金曲线与回撤 <span className="text-[11px] text-slate-400 font-normal">· 蓝色图钉 = 调仓决策日 · 点击任一天回放当天的持仓与成交</span>
        </div>
        <div className="flex gap-1 text-xs">
          {(['yuan', 'pct'] as const).map((m) => (
            <button
              key={m}
              data-testid={`curve-mode-${m}`}
              onClick={() => setMode(m)}
              className={`px-2 py-1 rounded ${mode === m ? 'bg-sky-700 text-white' : 'bg-slate-900 text-slate-300 border border-slate-700'}`}
            >
              {m === 'yuan' ? (notional ? '金额（名义 100 万）' : '金额 ¥') : '收益率 %'}
            </button>
          ))}
        </div>
      </div>
      <ReactECharts option={option} style={{ height: 380 }} notMerge onReady={onReady} />
      <div className="text-[10px] text-slate-500">
        基准都按同一起点归一：沪深300、中证2000 用指数收盘；研究池等权是模拟器内部同口径的基准。回撤按本账户自己的历史高点计算。
        <span className="ml-1" style={{ color: UP }}>红 = 盈利</span> / <span style={{ color: DOWN }}>绿 = 亏损</span>。
      </div>
    </div>
  );
};
