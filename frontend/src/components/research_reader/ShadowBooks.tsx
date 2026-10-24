import React, { useState, useEffect } from 'react';
import { EChart as ReactECharts, EChartHandle } from '../common/EChart';
import { BookMarked, TrendingUp, Layers, CheckCircle2, ArrowRight } from 'lucide-react';
import { BooksOverview, BookNavRow } from '../../api/types';
import { quantmindApi } from '../../api/client';

interface ShadowBooksProps {
  booksOverview: BooksOverview | null;
  selectedDate: string;
}

export const ShadowBooks: React.FC<ShadowBooksProps> = ({ booksOverview, selectedDate }) => {
  const books = booksOverview?.books || {};
  const bookKeys = Object.keys(books);

  const [selectedBook, setSelectedBook] = useState<string>(bookKeys[0] || 'ridge_native_20d__all_boards');
  const [navRows, setNavRows] = useState<BookNavRow[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedBook) return;
    setLoading(true);
    quantmindApi
      .getBookDetail(selectedDate, selectedBook, 'nav')
      .then((rows) => setNavRows(rows || []))
      .catch((err) => console.warn('Failed to load book detail for', selectedBook, err))
      .finally(() => setLoading(false));
  }, [selectedBook, selectedDate]);

  // Chart
  const dates = navRows.map((r) => r.date?.slice(0, 10));
  const navNets = navRows.map((r) => r.nav_net);
  const excesses = navRows.map((r) => r.excess_net);

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: '#334155',
      textStyle: { color: '#f8fafc', fontSize: 11 },
    },
    legend: {
      data: ['策略净值 (NAV Net)', '超额收益 (Excess Net)'],
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
        name: '策略净值 (NAV Net)',
        type: 'line',
        data: navNets,
        smooth: true,
        lineStyle: { width: 2, color: '#10b981' },
        itemStyle: { color: '#10b981' },
      },
      {
        name: '超额收益 (Excess Net)',
        type: 'line',
        data: excesses,
        smooth: true,
        lineStyle: { width: 1.5, color: '#06b6d4' },
        itemStyle: { color: '#06b6d4' },
      },
    ],
  };

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Title */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <BookMarked className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-white">
            前瞻影子账本（Shadow Books）多臂策略容量评测
          </h3>
        </div>
        <span className="text-xs font-mono text-slate-400">
          共跟踪 {bookKeys.length} 个策略臂
        </span>
      </div>

      {/* Books Summary Table */}
      <div className="border border-slate-800 rounded-lg overflow-x-auto">
        <table className="w-full text-left text-xs divide-y divide-slate-800">
          <thead className="bg-slate-900 text-slate-400 font-medium">
            <tr>
              <th className="px-3.5 py-2">策略臂标识 (Arm & Board)</th>
              <th className="px-3 py-2 text-right">最新净值 (NAV)</th>
              <th className="px-3 py-2 text-right">超额收益 (Excess)</th>
              <th className="px-3 py-2 text-right">累计收益率</th>
              <th className="px-3 py-2 text-right">持仓标的数</th>
              <th className="px-3 py-2 text-center">状态</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 bg-slate-950/40 font-mono">
            {bookKeys.map((k) => {
              const b = books[k];
              const isSelected = selectedBook === k;
              return (
                <tr
                  key={k}
                  onClick={() => setSelectedBook(k)}
                  className={`cursor-pointer transition ${
                    isSelected ? 'bg-blue-950/40 text-white font-bold' : 'hover:bg-slate-900/60 text-slate-300'
                  }`}
                >
                  <td className="px-3.5 py-2">
                    <span className="text-cyan-300">{k}</span>
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {b?.nav_net !== undefined ? b.nav_net.toFixed(4) : '—'}
                  </td>
                  <td className="px-3 py-2 text-right text-emerald-400 tabular-nums">
                    {b?.excess_net !== undefined ? `${(b.excess_net * 100).toFixed(2)}%` : '—'}
                  </td>
                  <td className="px-3 py-2 text-right text-slate-200 tabular-nums">
                    {b?.cum_return_net !== undefined ? `${(b.cum_return_net * 100).toFixed(2)}%` : '—'}
                  </td>
                  <td className="px-3 py-2 text-right text-slate-400 tabular-nums">
                    {b?.n_positions ?? 30} 只
                  </td>
                  <td className="px-3 py-2 text-center">
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded ${
                        isSelected
                          ? 'bg-blue-600 text-white font-bold'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {isSelected ? '选定查看' : '点击分析'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Selected Book NAV Curve */}
      <div className="pt-2">
        <div className="text-xs font-semibold text-slate-300 mb-1 flex items-center justify-between">
          <span>{selectedBook} 实时前瞻净值轨迹</span>
          {loading && <span className="text-cyan-400 animate-pulse font-mono">加载中...</span>}
        </div>
        <div className="h-64 w-full">
          <ReactECharts option={option} style={{ height: '100%', width: '100%' }} />
        </div>
      </div>
    </div>
  );
};
