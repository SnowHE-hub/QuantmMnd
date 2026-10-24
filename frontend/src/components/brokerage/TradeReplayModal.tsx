import React, { useEffect, useState } from 'react';
import { EChart as ReactECharts, EChartHandle } from '../common/EChart';
import { X, Loader2 } from 'lucide-react';
import { d10, DOWN, ledgerApi, pctS, tone, UP, yuan } from '../../api/ledger';

interface Props {
  account: string;
  ticker: string | null;
  onClose: () => void;
}

type Data = Awaited<ReturnType<typeof ledgerApi.ticker>>;

// 个股交易轨迹复盘：未复权日 K（成交价同口径）+ B / S 旗标 + 每个回合的区间盈亏
export const TradeReplayModal: React.FC<Props> = ({ account, ticker, onClose }) => {
  const [data, setData] = useState<Data | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!ticker) return;
    setData(null);
    setErr(null);
    ledgerApi
      .ticker(account, ticker)
      .then(setData)
      .catch((e) => setErr(String(e.message || e)));
  }, [account, ticker]);
  if (!ticker) return null;

  const px = data?.prices ?? [];
  const dates = px.map((p) => d10(p.trade_date));
  const marks = (data?.trades ?? []).map((t) => {
    const buy = t.side === 'buy';
    return {
      name: `${buy ? '买入' : '卖出'} ${t.shares} 股 @ ${t.price.toFixed(2)}`,
      coord: [d10(t.date), t.price],
      value: buy ? 'B' : 'S',
      symbol: 'pin',
      symbolSize: 34,
      symbolRotate: buy ? 180 : 0,
      symbolOffset: [0, buy ? 18 : -18],
      itemStyle: { color: buy ? UP : '#0284c7' },
      label: { color: '#fff', fontWeight: 'bold', offset: [0, buy ? 4 : 0] },
    };
  });
  const option = {
    animation: false,
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' }, backgroundColor: 'rgba(15,23,42,0.95)', borderColor: '#334155', textStyle: { color: '#f8fafc', fontSize: 11 } },
    grid: { left: 56, right: 16, top: 20, bottom: 40 },
    xAxis: { type: 'category', data: dates, axisLabel: { color: '#64748b', fontSize: 10 } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: '#64748b', fontSize: 10 }, splitLine: { lineStyle: { color: '#1e293b' } } },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 14, bottom: 8 }],
    series: [
      {
        name: ticker,
        type: 'candlestick',
        data: px.map((p) => [p.open, p.close, p.low, p.high]),
        itemStyle: { color: UP, color0: DOWN, borderColor: UP, borderColor0: DOWN },
        markPoint: { data: marks },
      },
    ],
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4" onClick={onClose}>
      <div className="w-full max-w-5xl bg-[#0c1220] border border-slate-700 rounded-2xl p-4 space-y-3" onClick={(e) => e.stopPropagation()} data-testid="replay-modal">
        <div className="flex items-center justify-between">
          <div className="text-sm font-bold text-white">
            {ticker} 交易轨迹复盘 <span className="text-[11px] text-slate-400 font-normal">· B = 买入成交 · S = 卖出成交（价格为实际成交价）</span>
          </div>
          <button onClick={onClose} className="p-1 rounded hover:bg-slate-800 text-slate-400">
            <X className="w-4 h-4" />
          </button>
        </div>
        {err && <div className="text-rose-300 text-xs">{err}</div>}
        {!data && !err && <Loader2 className="w-5 h-5 animate-spin text-slate-400" />}
        {data && (
          <>
            <ReactECharts option={option} style={{ height: 360 }} notMerge />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="space-y-1" data-testid="replay-trades">
                <div className="text-slate-400">成交（{data.trades.length} 笔）</div>
                {data.trades.map((t, i) => (
                  <div key={i} className="flex justify-between font-mono text-[11px] border-t border-slate-800 py-0.5">
                    <span className={t.side === 'buy' ? 'text-rose-300' : 'text-sky-300'}>
                      {d10(t.date)} {t.side === 'buy' ? 'B 买入' : 'S 卖出'} {t.shares} 股 @ {t.price.toFixed(2)}
                    </span>
                    <span className="text-slate-400">费 {t.fee_total.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <div className="space-y-1">
                <div className="text-slate-400">回合</div>
                {data.round_trips.length === 0 && <div className="text-slate-500">还在持有中，没有完成的回合。</div>}
                {data.round_trips.map((r, i) => (
                  <div key={i} className="flex justify-between font-mono text-[11px] border-t border-slate-800 py-0.5">
                    <span>
                      {d10(r.open_date)} → {d10(r.close_date)}（{r.days_held} 天）{r.avg_buy?.toFixed(2)} → {r.avg_sell?.toFixed(2)}
                    </span>
                    <span className={tone(r.net_pnl)}>
                      {yuan(r.net_pnl, true)} {pctS(r.return)}
                    </span>
                  </div>
                ))}
                {data.actions.length > 0 && (
                  <div className="text-[11px] text-amber-300 pt-1">
                    持有期间分红送转 {data.actions.length} 次，合计现金 {yuan(data.actions.reduce((s, a) => s + a.cash_dividend, 0))}
                  </div>
                )}
              </div>
            </div>
            <div className="text-[10px] text-slate-500">{data.note}</div>
          </>
        )}
      </div>
    </div>
  );
};
