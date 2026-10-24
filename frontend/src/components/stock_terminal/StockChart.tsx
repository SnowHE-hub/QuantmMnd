import React from 'react';
import { EChart as ReactECharts, EChartHandle } from '../common/EChart';
import { StockPriceRow } from '../../api/types';

interface StockChartProps {
  ticker: string;
  prices: StockPriceRow[];
}

export const StockChart: React.FC<StockChartProps> = ({ ticker, prices }) => {
  if (!prices || prices.length === 0) {
    return (
      <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-8 text-center text-slate-500 text-xs">
        暂无日线行情数据（可选择其他标的或调整日期）
      </div>
    );
  }

  // Calculate moving averages
  const dates = prices.map((p) => p.trade_date?.slice(0, 10));
  // ECharts candlestick expects [open, close, lowest, highest]
  const ohlcData = prices.map((p) => [p.open, p.close, p.low, p.high]);
  const volumes = prices.map((p, idx) => [idx, p.vol, p.close >= p.open ? 1 : -1]);

  const calcMA = (dayCount: number) => {
    const result = [];
    for (let i = 0; i < prices.length; i++) {
      if (i < dayCount - 1) {
        result.push('-');
        continue;
      }
      let sum = 0;
      for (let j = 0; j < dayCount; j++) {
        sum += prices[i - j].close;
      }
      result.push((sum / dayCount).toFixed(2));
    }
    return result;
  };

  const ma5 = calcMA(5);
  const ma20 = calcMA(20);

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross', lineStyle: { color: '#475569' } },
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: '#334155',
      textStyle: { color: '#f8fafc', fontSize: 11 },
    },
    axisPointer: {
      link: [{ xAxisIndex: 'all' }],
    },
    grid: [
      { left: '4%', right: '3%', top: '10%', height: '56%' },
      { left: '4%', right: '3%', top: '72%', height: '20%' },
    ],
    xAxis: [
      {
        type: 'category',
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
        splitLine: { show: false },
      },
      {
        type: 'category',
        gridIndex: 1,
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { show: false },
        splitLine: { show: false },
      },
    ],
    yAxis: [
      {
        scale: true,
        splitArea: { show: false },
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#94a3b8', fontSize: 10 },
        splitLine: { lineStyle: { color: '#1e293b' } },
      },
      {
        scale: true,
        gridIndex: 1,
        splitNumber: 2,
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#64748b', fontSize: 9 },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: 'K线 (OHLC)',
        type: 'candlestick',
        data: ohlcData,
        itemStyle: {
          color: '#ef4444', // Red for up (A-share convention)
          color0: '#10b981', // Green for down (A-share convention)
          borderColor: '#ef4444',
          borderColor0: '#10b981',
        },
      },
      {
        name: 'MA5',
        type: 'line',
        data: ma5,
        smooth: true,
        lineStyle: { opacity: 0.8, width: 1.2, color: '#f59e0b' },
        showSymbol: false,
      },
      {
        name: 'MA20',
        type: 'line',
        data: ma20,
        smooth: true,
        lineStyle: { opacity: 0.8, width: 1.2, color: '#06b6d4' },
        showSymbol: false,
      },
      {
        name: '成交量',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: volumes.map((v) => ({
          value: v[1],
          itemStyle: { color: v[2] === 1 ? '#ef4444' : '#10b981' },
        })),
      },
    ],
  };

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-4 shadow-lg">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-mono font-bold text-white text-sm">{ticker}</span>
          <span className="text-xs text-slate-400">历史日频 K 线图与成交量</span>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono">
          <span className="flex items-center gap-1 text-amber-400">
            <span className="w-2 h-0.5 bg-amber-400" /> MA5
          </span>
          <span className="flex items-center gap-1 text-cyan-400">
            <span className="w-2 h-0.5 bg-cyan-400" /> MA20
          </span>
          <span className="text-slate-500">{prices.length} 交易日</span>
        </div>
      </div>
      <div className="h-72 w-full">
        <ReactECharts option={option} style={{ height: '100%', width: '100%' }} />
      </div>
    </div>
  );
};
