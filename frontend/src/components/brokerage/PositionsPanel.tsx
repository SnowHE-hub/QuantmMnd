import React, { useState } from 'react';
import { Lock } from 'lucide-react';
import { d10, pctS, PositionRow, RoundTrip, tone, yuan } from '../../api/ledger';

interface Props {
  date: string;
  positions: PositionRow[];
  weightsOnly: boolean;
  roundTrips: RoundTrip[];
  onPickTicker: (t: string) => void;
}

// 持仓明细（某一天收盘后）与已清仓回合战绩
export const PositionsPanel: React.FC<Props> = ({ date, positions, weightsOnly, roundTrips, onPickTicker }) => {
  const [view, setView] = useState<'hold' | 'closed'>('hold');
  const sorted = [...roundTrips].sort((a, b) => b.net_pnl - a.net_pnl);
  const best = sorted.slice(0, 5);
  const worst = [...sorted].reverse().slice(0, 5);

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-4 space-y-3" data-testid="positions-panel">
      <div className="flex items-center justify-between">
        <div className="flex gap-1 text-xs">
          <button
            data-testid="tab-hold"
            onClick={() => setView('hold')}
            className={`px-3 py-1 rounded ${view === 'hold' ? 'bg-sky-700 text-white' : 'bg-slate-900 text-slate-300 border border-slate-700'}`}
          >
            当前持仓（{positions.length}）
          </button>
          <button
            data-testid="tab-closed"
            onClick={() => setView('closed')}
            disabled={weightsOnly}
            className={`px-3 py-1 rounded ${view === 'closed' ? 'bg-sky-700 text-white' : 'bg-slate-900 text-slate-300 border border-slate-700'} disabled:opacity-40`}
          >
            已清仓回合（{roundTrips.length}）
          </button>
        </div>
        <div className="text-[11px] text-slate-400 font-mono" data-testid="positions-date">
          {view === 'hold' ? `${date} 收盘后` : '全部已完成的买入 → 卖出回合（扣双边费用、含期间分红）'}
        </div>
      </div>

      {view === 'hold' &&
        (weightsOnly ? (
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <div className="text-[11px] text-amber-300 mb-2">影子账本按目标权重记账，没有股数、成本和逐笔成交（研究口径）；下面是最近一次决策的目标与权重。</div>
            <table className="w-full text-xs font-mono">
              <thead className="text-slate-500 sticky top-0 bg-[#0c1220]">
                <tr>
                  <th className="text-left py-1">代码 / 名称</th>
                  <th className="text-left">板块</th>
                  <th className="text-right">目标权重</th>
                  <th className="text-right">决策日</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr key={p.ticker} className="border-t border-slate-800">
                    <td className="py-1 text-cyan-300">
                      {p.ticker} <span className="text-slate-400 font-sans">{p.name}</span>
                    </td>
                    <td className="font-sans text-slate-400">{p.board}</td>
                    <td className="text-right">{(p.weight * 100).toFixed(2)}%</td>
                    <td className="text-right text-slate-400">{d10(p.as_of)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : positions.length === 0 ? (
          <div className="text-center text-slate-500 text-xs py-6">这一天收盘后没有持仓（全部是现金）。</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono" data-testid="holdings-table">
              <thead className="text-slate-500 bg-slate-900">
                <tr>
                  <th className="text-left px-2 py-1.5">代码 / 名称</th>
                  <th className="text-right">持仓 / 可用</th>
                  <th className="text-right">成本价 / 摊薄</th>
                  <th className="text-right">收盘价</th>
                  <th className="text-right">当日涨跌</th>
                  <th className="text-right">当日盈亏</th>
                  <th className="text-right">浮动盈亏</th>
                  <th className="text-right">市值 / 仓位</th>
                  <th className="text-right px-2">建仓 / 天数</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr key={p.ticker} className="border-t border-slate-800 hover:bg-slate-900/70 cursor-pointer" onClick={() => onPickTicker(p.ticker)} data-testid={`hold-${p.ticker}`}>
                    <td className="px-2 py-1.5">
                      <div className="text-cyan-300 font-bold">{p.ticker}</div>
                      <div className="font-sans text-slate-400 text-[11px]">
                        {p.name} <span className="px-1 rounded bg-slate-800 text-[10px]">{p.board}</span>
                      </div>
                    </td>
                    <td className="text-right">
                      <div data-testid={`shares-${p.ticker}`}>{p.shares?.toLocaleString()}</div>
                      <div className="text-[11px] text-slate-400 flex items-center justify-end gap-1">
                        {p.bought_today && <Lock className="w-3 h-3 text-amber-400" />}
                        {p.available?.toLocaleString()}
                      </div>
                    </td>
                    <td className="text-right">
                      <div>{p.avg_cost?.toFixed(3)}</div>
                      <div className="text-[11px] text-slate-400">{p.diluted_cost?.toFixed(3)}</div>
                    </td>
                    <td className="text-right">{p.close?.toFixed(2)}</td>
                    <td className={`text-right ${tone(p.day_ret)}`}>{pctS(p.day_ret)}</td>
                    <td className={`text-right ${tone(p.day_pnl)}`}>{yuan(p.day_pnl, true)}</td>
                    <td className={`text-right ${tone(p.unrealized)}`}>
                      <div>{yuan(p.unrealized, true)}</div>
                      <div className="text-[11px]">{pctS(p.unrealized_pct)}</div>
                    </td>
                    <td className="text-right">
                      <div data-testid={`value-${p.ticker}`}>{yuan(p.value)}</div>
                      <div className="text-[11px] text-slate-400">{(p.weight * 100).toFixed(1)}%</div>
                    </td>
                    <td className="text-right px-2">
                      <div>{d10(p.open_date)}</div>
                      <div className="text-[11px] text-slate-400">{p.days_held} 天</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-[10px] text-slate-500 mt-2">
              成本价 = 买入成交额 + 买入费用（佣金、过户费、模拟冲击成本）后的移动加权均价；摊薄成本再扣掉持有期间收到的现金分红。
              <Lock className="w-3 h-3 text-amber-400 inline mx-1" />= 当天买入、T+1 前不可卖的部分不计入“可用”。点击一行看 K 线买卖点。
            </div>
          </div>
        ))}

      {view === 'closed' && (
        <div className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[
              ['最赚钱 Top 5', best],
              ['最亏钱 Top 5', worst],
            ].map(([title, rows]) => (
              <div key={title as string} className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                <div className="text-xs font-semibold text-slate-200 mb-1">{title as string}</div>
                {(rows as RoundTrip[]).map((r) => (
                  <div key={r.ticker + r.open_date} className="flex justify-between text-[11px] font-mono py-0.5 cursor-pointer hover:bg-slate-900" onClick={() => onPickTicker(r.ticker)}>
                    <span className="text-cyan-300">
                      {r.ticker} <span className="font-sans text-slate-400">{r.name}</span>
                    </span>
                    <span className={tone(r.net_pnl)}>
                      {yuan(r.net_pnl, true)} ({pctS(r.return, 1)})
                    </span>
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div className="overflow-x-auto max-h-80 overflow-y-auto">
            <table className="w-full text-xs font-mono" data-testid="roundtrip-table">
              <thead className="text-slate-500 sticky top-0 bg-[#0c1220]">
                <tr>
                  <th className="text-left py-1">代码 / 名称</th>
                  <th className="text-right">买入日 / 均价</th>
                  <th className="text-right">卖出日 / 均价</th>
                  <th className="text-right">持有天数</th>
                  <th className="text-right">净盈亏（扣费）</th>
                  <th className="text-right">收益率</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => (
                  <tr key={r.ticker + r.open_date} className="border-t border-slate-800 hover:bg-slate-900/60 cursor-pointer" onClick={() => onPickTicker(r.ticker)}>
                    <td className="py-1 text-cyan-300">
                      {r.ticker} <span className="font-sans text-slate-400">{r.name}</span>
                      {r.writeoff && <span className="ml-1 text-amber-400 font-sans">（清算）</span>}
                    </td>
                    <td className="text-right">
                      {d10(r.open_date)} / {r.avg_buy?.toFixed(2)}
                    </td>
                    <td className="text-right">
                      {d10(r.close_date)} / {r.avg_sell?.toFixed(2)}
                    </td>
                    <td className="text-right">{r.days_held}</td>
                    <td className={`text-right ${tone(r.net_pnl)}`}>{yuan(r.net_pnl, true)}</td>
                    <td className={`text-right ${tone(r.return)}`}>{pctS(r.return)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
