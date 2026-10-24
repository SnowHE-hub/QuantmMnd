import React, { useState, useEffect } from 'react';
import { EChart as ReactECharts, EChartHandle } from '../common/EChart';
import { Award, TrendingUp, DollarSign, Layers, BookOpen } from 'lucide-react';
import { BacktestT052Response, BacktestNavRow } from '../../api/types';
import { quantmindApi } from '../../api/client';
import { MarkdownViewer } from '../common/MarkdownViewer';

interface BacktestT052Props {
  t052Data: BacktestT052Response | null;
}

export const BacktestT052: React.FC<BacktestT052Props> = ({ t052Data }) => {
  const keys = t052Data?.primary?.keys || [
    'W6_ENS20_n10_s4_b2.0_open_100000_main+chinext',
    'W3_ENS20_n10_s4_b2.0_open_100000_main+chinext',
    'W6_ENS20_n30_s4_b2.0_open_1000000_all',
    'W6_ENS20_n100_s4_b2.0_open_10000000_all',
  ];

  const [selectedKey, setSelectedKey] = useState<string>(
    keys[0] || 'W6_ENS20_n10_s4_b2.0_open_100000_main+chinext'
  );
  const [navRows, setNavRows] = useState<BacktestNavRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [showReport, setShowReport] = useState(false);

  useEffect(() => {
    if (!selectedKey) return;
    setLoading(true);
    // Replace + with %2B for safe URL path
    const safeKey = selectedKey.replace(/\+/g, '%2B');
    quantmindApi
      .getBacktestNav(safeKey)
      .then((rows) => setNavRows(rows || []))
      .catch((err) => console.warn('Failed to load nav for', selectedKey, err))
      .finally(() => setLoading(false));
  }, [selectedKey]);

  // Build chart option
  const dates = navRows.map((r) => r.date?.slice(0, 10));
  const equities = navRows.map((r) => r.equity);
  const benchmarks = navRows.map((r) => r.bench_pool_ew);

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: '#334155',
      textStyle: { color: '#f8fafc', fontSize: 11 },
    },
    legend: {
      data: ['策略组合资产', '前1500等权基准'],
      textStyle: { color: '#94a3b8', fontSize: 11 },
      top: 5,
    },
    grid: { left: '4%', right: '3%', top: '15%', bottom: '10%' },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8', fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      scale: true,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8', fontSize: 10 },
      splitLine: { lineStyle: { color: '#1e293b' } },
    },
    series: [
      {
        name: '策略组合资产',
        type: 'line',
        data: equities,
        smooth: true,
        lineStyle: { width: 2, color: '#3b82f6' },
        itemStyle: { color: '#3b82f6' },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(59, 130, 246, 0.3)' },
              { offset: 1, color: 'rgba(59, 130, 246, 0.0)' },
            ],
          },
        },
      },
      {
        name: '前1500等权基准',
        type: 'line',
        data: benchmarks,
        smooth: true,
        lineStyle: { width: 1.5, type: 'dashed', color: '#94a3b8' },
        itemStyle: { color: '#94a3b8' },
      },
    ],
  };

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-amber-400" />
            <h3 className="text-sm font-bold text-white">
              T-052 模拟盘网格回测实验成果实验室
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            4,038 组网格超参数模拟盘（W3/W6 调仓周期、10万/100万/1000万三档规模、不同板块权限）
          </p>
        </div>

        <button
          onClick={() => setShowReport(!showReport)}
          className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 flex items-center gap-1.5 transition"
        >
          <BookOpen className="w-3.5 h-3.5 text-blue-400" />
          <span>{showReport ? '收起实验报告' : '精读完整评测报告'}</span>
        </button>
      </div>

      {/* Grid Key Selector */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-slate-500 text-[11px]">参数配置组合:</span>
        {keys.map((k) => (
          <button
            key={k}
            onClick={() => setSelectedKey(k)}
            className={`px-2.5 py-1 rounded text-xs font-mono transition ${
              selectedKey === k
                ? 'bg-blue-600 text-white font-bold shadow-sm shadow-blue-500/20'
                : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
            }`}
          >
            {k.replace('W6_ENS20_', '').replace('W3_ENS20_', '')}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="h-72 w-full pt-2">
        {loading ? (
          <div className="h-full flex items-center justify-center text-xs text-blue-400 animate-pulse font-mono">
            加载净值序列中...
          </div>
        ) : (
          <ReactECharts option={option} style={{ height: '100%', width: '100%' }} />
        )}
      </div>

      {/* Markdown Report Drawer/Section */}
      {showReport && t052Data?.report_markdown && (
        <div className="mt-4 pt-4 border-t border-slate-800">
          <div className="text-xs font-semibold text-white mb-2 flex items-center gap-1.5">
            <BookOpen className="w-4 h-4 text-cyan-400" />
            <span>T-052 模拟盘网格评测报告正文</span>
          </div>
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 max-h-96 overflow-y-auto">
            <MarkdownViewer content={t052Data.report_markdown} />
          </div>
        </div>
      )}
    </div>
  );
};
