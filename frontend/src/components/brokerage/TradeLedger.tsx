import React, { useMemo, useState } from 'react';
import { ActionRow, d10, RejectRow, tone, TradeRow, yuan } from '../../api/ledger';

interface Props {
  trades: TradeRow[];
  rejects: RejectRow[];
  actions: ActionRow[];
  selectedDate: string;
  onPickTicker: (t: string) => void;
}

type Row =
  | { kind: 'trade'; date: string; t: TradeRow }
  | { kind: 'reject'; date: string; r: RejectRow }
  | { kind: 'action'; date: string; a: ActionRow };

const SIDE: Record<string, [string, string]> = {
  buy: ['证券买入', 'bg-rose-950/80 text-rose-300 border-rose-800'],
  sell: ['证券卖出', 'bg-sky-950/80 text-sky-300 border-sky-800'],
  writeoff: ['清算卖出', 'bg-sky-950/80 text-sky-300 border-sky-800'],
};

// 交割单：成交（红买蓝卖）、未成交（灰：涨停 / 停牌等；橙：买不起一个申报单位）、分红送转
export const TradeLedger: React.FC<Props> = ({ trades, rejects, actions, selectedDate, onPickTicker }) => {
  const [onlyDay, setOnlyDay] = useState(false);
  const rows = useMemo(() => {
    const all: Row[] = [
      ...trades.map((t) => ({ kind: 'trade' as const, date: d10(t.date), t })),
      ...rejects.map((r) => ({ kind: 'reject' as const, date: d10(r.date), r })),
      ...actions.map((a) => ({ kind: 'action' as const, date: d10(a.date), a })),
    ];
    return all.filter((x) => !onlyDay || x.date === selectedDate).sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
  }, [trades, rejects, actions, onlyDay, selectedDate]);
  const tot = trades.reduce(
    (s, t) => ({ c: s.c + t.commission, tr: s.tr + t.transfer, st: s.st + t.stamp, sl: s.sl + t.slippage, dt: s.dt + t.dividend_tax }),
    { c: 0, tr: 0, st: 0, sl: 0, dt: 0 }
  );

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-4 space-y-3" data-testid="trade-ledger">
      <div className="flex items-center justify-between">
        <div className="text-sm font-bold text-white">交割单（{trades.length} 笔成交 · {rejects.length} 笔未成交 · {actions.length} 次分红送转）</div>
        <label className="text-xs text-slate-300 flex items-center gap-1.5">
          <input type="checkbox" checked={onlyDay} onChange={() => setOnlyDay(!onlyDay)} data-testid="ledger-only-day" />
          只看 {selectedDate}
        </label>
      </div>
      <div className="text-[11px] font-mono text-slate-400">
        累计费用：佣金 {yuan(tot.c)} · 过户费 {yuan(tot.tr)} · 印花税 {yuan(tot.st)} · 红利税 {yuan(tot.dt)} · 模拟冲击成本 {yuan(tot.sl)}
      </div>
      <div className="overflow-x-auto max-h-[420px] overflow-y-auto">
        <table className="w-full text-xs font-mono">
          <thead className="text-slate-500 sticky top-0 bg-[#0c1220]">
            <tr>
              <th className="text-left py-1">日期</th>
              <th className="text-left">业务</th>
              <th className="text-left">代码 / 名称</th>
              <th className="text-right">成交价</th>
              <th className="text-right">股数</th>
              <th className="text-right">成交金额</th>
              <th className="text-right">佣金 / 过户 / 印花</th>
              <th className="text-right">冲击 / 红利税</th>
              <th className="text-right">发生金额</th>
              <th className="text-right">资金余额</th>
              <th className="text-right">平仓盈亏</th>
              <th className="text-left pl-3">原因</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={12} className="text-center text-slate-500 py-6 font-sans">
                  {onlyDay ? '这一天没有成交、未成交或分红记录。' : '还没有任何交易。'}
                </td>
              </tr>
            )}
            {rows.map((x, i) => {
              if (x.kind === 'trade') {
                const t = x.t;
                const [label, cls] = SIDE[t.side];
                return (
                  <tr key={i} className="border-t border-slate-800 hover:bg-slate-900/60 cursor-pointer" onClick={() => onPickTicker(t.ticker)} data-testid="ledger-trade">
                    <td className="py-1">{x.date}</td>
                    <td>
                      <span className={`px-1.5 py-0.5 rounded border text-[10px] font-sans ${cls}`}>{label}</span>
                    </td>
                    <td className="text-cyan-300">
                      {t.ticker} <span className="font-sans text-slate-400">{t.name}</span>
                    </td>
                    <td className="text-right">{t.price.toFixed(2)}</td>
                    <td className="text-right">{t.shares.toLocaleString()}</td>
                    <td className="text-right">{yuan(t.gross)}</td>
                    <td className="text-right text-slate-400">
                      {t.commission.toFixed(2)} / {t.transfer.toFixed(2)} / {t.stamp.toFixed(2)}
                    </td>
                    <td className="text-right text-slate-400">
                      {t.slippage.toFixed(2)} / {t.dividend_tax.toFixed(2)}
                    </td>
                    <td className={`text-right ${t.net_cash < 0 ? 'text-slate-200' : 'text-slate-200'}`}>{yuan(t.net_cash, true)}</td>
                    <td className="text-right">{yuan(t.cash_after)}</td>
                    <td className={`text-right ${tone(t.realized_pnl)}`}>{t.realized_pnl === null ? '' : yuan(t.realized_pnl, true)}</td>
                    <td className="pl-3 font-sans text-slate-400 whitespace-nowrap">{t.reason}</td>
                  </tr>
                );
              }
              if (x.kind === 'reject') {
                const r = x.r;
                const odd = r.reason === 'cannot_afford_one_lot';
                return (
                  <tr key={i} className="border-t border-slate-800 text-slate-500" data-testid="ledger-reject">
                    <td className="py-1">{x.date}</td>
                    <td>
                      <span className={`px-1.5 py-0.5 rounded border text-[10px] font-sans ${odd ? 'bg-amber-950/60 text-amber-300 border-amber-800' : 'bg-slate-800 text-slate-300 border-slate-700'}`}>
                        {odd ? '零股跳过' : '未成交'}
                      </span>
                    </td>
                    <td>
                      {r.ticker} <span className="font-sans">{r.name}</span>
                    </td>
                    <td colSpan={8} className="text-right font-sans">
                      {r.side === 'buy' ? '买入' : '卖出'}委托：{r.reason_cn}
                    </td>
                    <td />
                  </tr>
                );
              }
              const a = x.a;
              return (
                <tr key={i} className="border-t border-slate-800 text-amber-200/80" data-testid="ledger-action">
                  <td className="py-1">{x.date}</td>
                  <td>
                    <span className="px-1.5 py-0.5 rounded border text-[10px] font-sans bg-amber-950/40 border-amber-900">红利 / 送转</span>
                  </td>
                  <td>{a.ticker}</td>
                  <td colSpan={5} className="text-right font-sans">
                    {a.shares_after !== a.shares_before ? `送转后 ${a.shares_before} → ${a.shares_after} 股；` : ''}
                    现金红利入账
                  </td>
                  <td className="text-right">{yuan(a.cash_dividend, true)}</td>
                  <td colSpan={3} />
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="text-[10px] text-slate-500">
        成交价 = 次日开盘价（未复权）；佣金万 2.5、最低 5 元；过户费 0.001%；卖出印花税 0.05%；红利税按持有期（≤1 月 20%、≤1 年 10%）在卖出时扣。
        “模拟冲击成本”是按流动性分档估算的滑点，不是券商收费，但已从现金里扣除。
      </div>
    </div>
  );
};
