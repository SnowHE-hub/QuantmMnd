import React from 'react';
import { DollarSign, Wallet, ArrowUpRight, TrendingUp, ShieldCheck, AlertCircle, PieChart, Layers, BarChart3 } from 'lucide-react';
import { FormalReport, TierData } from '../../api/types';

interface FormalPortfolioProps {
  formal: FormalReport | null;
  onSelectTicker: (ticker: string) => void;
  selectedTierKey: string;
  onSelectTierKey: (key: string) => void;
}

export const FormalPortfolio: React.FC<FormalPortfolioProps> = ({
  formal,
  onSelectTicker,
  selectedTierKey,
  onSelectTierKey,
}) => {
  if (!formal) {
    return (
      <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-8 text-center space-y-3">
        <div className="w-10 h-10 rounded-full bg-blue-950/80 border border-blue-800 flex items-center justify-center mx-auto text-blue-400">
          <Layers className="w-5 h-5" />
        </div>
        <div className="text-sm font-bold text-white">此历史截面处于影子研发阶段</div>
        <p className="text-xs text-slate-400 max-w-lg mx-auto">
          正式荐股日报从 2026-10-23 起生成（10-24 用户批准，go_live 2026-10-26，首次调仓 2026-10-28）；这一天没有正式日报。您可在此截面查阅下方的量化研究候选池（F4）、打分排行及前瞻影子账本（Books）。
        </p>
      </div>
    );
  }

  const tiers = formal.tiers || {};
  const currentTier: TierData | undefined = tiers[selectedTierKey];

  const tierOptions = [
    { key: 'aum_1e5', label: '10 万元档', aumStr: '100k RMB', desc: '小资金池 (主板+创业板精选 10 只)' },
    { key: 'aum_1e6', label: '100 万元档', aumStr: '1M RMB', desc: '中资金池 (30 只配置)' },
    { key: 'aum_1e7', label: '1000 万元档', aumStr: '10M RMB', desc: '机构资金池 (100 只分散)' },
  ];

  // Calculate board distribution
  const holdings = currentTier?.holdings || [];
  const boardCounts: Record<string, number> = {};
  holdings.forEach((h) => {
    const b = h.board || '主板';
    boardCounts[b] = (boardCounts[b] || 0) + 1;
  });

  // Calculate cash ratio
  const totalEquity = currentTier?.equity || currentTier?.aum || 0;
  const cashAmount = currentTier?.cash || 0;
  const cashPct = totalEquity > 0 ? ((cashAmount / totalEquity) * 100).toFixed(1) : '100.0';
  const investedPct = (100 - Number(cashPct)).toFixed(1);

  // Max holding weight
  const maxWeight = holdings.length > 0 ? Math.max(...holdings.map((h) => h.weight || 0)) * 100 : 0;

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Header & Tier Switcher */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <h3 className="text-base font-bold text-white">正式荐股纸面账户持仓透视与组合暴露诊断</h3>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/60">
              {formal.mode === 'preview' ? '首次调仓预览 (Preview)' : '实盘模拟 (Live)'}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            下个调仓日: <span className="font-mono text-cyan-300">{formal.next_rebalance || '—'}</span>
            {formal.today_is_rebalance && (
              <span className="ml-2 font-bold text-amber-400">（今日为调仓日）</span>
            )}
          </p>
        </div>

        {/* Tier Buttons */}
        <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800">
          {tierOptions.map((opt) => (
            <button
              key={opt.key}
              onClick={() => onSelectTierKey(opt.key)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition ${
                selectedTierKey === opt.key
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {currentTier ? (
        <div className="space-y-4">
          {/* Key Metric Highlights */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium mb-1">总资产 (Equity)</div>
              <div className="text-base font-bold font-mono text-white tabular-nums">
                ¥{Number(currentTier.equity || currentTier.aum).toLocaleString()}
              </div>
              <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                基准规模: ¥{Number(currentTier.aum).toLocaleString()}
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium mb-1">现金留存 (Cash)</div>
              <div className="text-base font-bold font-mono text-cyan-300 tabular-nums">
                ¥{Number(currentTier.cash || 0).toLocaleString()}
              </div>
              <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                占比: {cashPct}% (留存防御)
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium mb-1">成立以来收益率</div>
              <div
                className={`text-base font-bold font-mono tabular-nums ${
                  (currentTier.return_since_start || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {((currentTier.return_since_start || 0) * 100).toFixed(2)}%
              </div>
              <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                等权基准: {((currentTier.pool_ew_since_start || 0) * 100).toFixed(2)}%
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div className="text-[11px] text-slate-400 font-medium mb-1">持仓标的 / 板块权限</div>
              <div className="text-base font-bold font-mono text-white">
                {currentTier.holdings?.length || 0} / {currentTier.top_n} 只
              </div>
              <div className="text-[10px] text-slate-400 font-mono mt-0.5 truncate" title={Array.isArray(currentTier.boards) ? currentTier.boards.join(', ') : currentTier.boards}>
                权限: {Array.isArray(currentTier.boards) ? currentTier.boards.join(' + ') : currentTier.boards || '全市场'}
              </div>
            </div>
          </div>

          {/* Portfolio Health & Exposure Diagnostics Card */}
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-3 text-xs">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                <BarChart3 className="w-4 h-4 text-cyan-400" />
                <span>组合多维暴露与健康度诊断 (Portfolio Exposure Health)</span>
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                单票最大权重: {maxWeight.toFixed(1)}% {maxWeight > 20 ? '(高集中)' : '(分散合理)'}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Asset Allocation Bar */}
              <div>
                <div className="flex justify-between text-[11px] mb-1 font-mono">
                  <span className="text-slate-400">资产仓位配置:</span>
                  <span className="text-cyan-300">股票仓位 {investedPct}% | 现金 {cashPct}%</span>
                </div>
                <div className="h-2.5 rounded-full bg-slate-800 overflow-hidden flex">
                  <div style={{ width: `${investedPct}%` }} className="bg-blue-600 h-full" />
                  <div style={{ width: `${cashPct}%` }} className="bg-cyan-500/80 h-full" />
                </div>
              </div>

              {/* Board Breakdown */}
              <div>
                <div className="flex justify-between text-[11px] mb-1 font-mono">
                  <span className="text-slate-400">板块配置分布:</span>
                  <span className="text-slate-300">
                    {Object.keys(boardCounts).length > 0
                      ? Object.entries(boardCounts).map(([k, v]) => `${k} ${v}只`).join(' · ')
                      : '现金准备中，明日开盘建仓'}
                  </span>
                </div>
                <div className="text-[10px] text-slate-500 font-mono">
                  严格执行 T+1 可交易过滤与流动性时点前 1500 强支持域约束
                </div>
              </div>
            </div>
          </div>

          {/* Holdings Table */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="text-xs font-semibold text-white flex items-center gap-1.5">
                <Wallet className="w-3.5 h-3.5 text-blue-400" />
                <span>当前账户持仓明细（点击任意行可呼出右侧速览抽屉）</span>
              </div>
              <span className="text-[11px] text-slate-400 font-mono">
                共 {currentTier.holdings?.length || 0} 只持仓
              </span>
            </div>

            <div className="border border-slate-800 rounded-lg overflow-x-auto">
              <table className="w-full text-left text-xs divide-y divide-slate-800">
                <thead className="bg-slate-900 text-slate-400 font-medium">
                  <tr>
                    <th className="px-3.5 py-2">代码 / 名称</th>
                    <th className="px-3 py-2">板块</th>
                    <th className="px-3 py-2 text-right">持仓股数</th>
                    <th className="px-3 py-2 text-right">最新市价</th>
                    <th className="px-3 py-2 text-right">持仓市值</th>
                    <th className="px-3 py-2 text-right">组合权重</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-950/40 font-mono">
                  {currentTier.holdings && currentTier.holdings.length > 0 ? (
                    currentTier.holdings.map((h) => (
                      <tr
                        key={h.ticker}
                        onClick={() => onSelectTicker(h.ticker)}
                        className="hover:bg-blue-950/30 cursor-pointer transition"
                      >
                        <td className="px-3.5 py-2">
                          <div className="font-bold text-white group-hover:text-cyan-300">
                            {h.name}
                          </div>
                          <div className="text-[10px] text-cyan-400">{h.ticker}</div>
                        </td>
                        <td className="px-3 py-2 text-slate-400 text-[11px]">{h.board}</td>
                        <td className="px-3 py-2 text-right text-slate-200 tabular-nums">
                          {Number(h.shares).toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-right text-slate-200 tabular-nums">
                          ¥{Number(h.price).toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right font-medium text-slate-100 tabular-nums">
                          ¥{Number(h.value).toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-right text-cyan-300 font-bold tabular-nums">
                          {(h.weight * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="text-center py-6 text-slate-500 italic">
                        首次调仓前，账户处于现金准备状态，明日开盘将执行初次建仓
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-slate-500 text-xs">未找到该资金档数据</div>
      )}
    </div>
  );
};
