import React, { useMemo, useState } from 'react';
import { DailyRow, d10, LedgerMeta, pctS, yuan } from '../../api/ledger';

interface Props {
  daily: DailyRow[];
  monthly: LedgerMeta['monthly'];
  aum: number;
  notional: boolean;
  selectedDate: string;
  onSelectDate: (d: string) => void;
}

// 每日盈亏日历（同花顺 / 富途式）：格子里是当天赚亏的金额和收益率，红绿深浅表示幅度；右边是月度汇总与胜率
export const PnlCalendar: React.FC<Props> = ({ daily, monthly, aum, notional, selectedDate, onSelectDate }) => {
  const months = useMemo(() => [...new Set(daily.map((r) => d10(r.date).slice(0, 7)))], [daily]);
  const [month, setMonth] = useState<string>(months[months.length - 1]);
  const cur = months.includes(month) ? month : months[months.length - 1];
  const byDay = useMemo(() => {
    const m: Record<string, DailyRow> = {};
    daily.forEach((r) => (m[d10(r.date)] = r));
    return m;
  }, [daily]);
  // 影子账本没有金额：按名义本金折算当日盈亏
  const pnlOf = (r: DailyRow) => (typeof r.daily_pnl === 'number' ? r.daily_pnl : (r.daily_ret_net ?? 0) * (notional ? 1e6 : aum));
  const retOf = (r: DailyRow) => (typeof r.daily_ret === 'number' ? r.daily_ret : r.daily_ret_net ?? 0);
  const maxAbs = Math.max(1e-9, ...daily.map((r) => Math.abs(retOf(r))));

  const [y, mo] = cur.split('-').map(Number);
  const first = new Date(y, mo - 1, 1);
  const lead = (first.getDay() + 6) % 7; // 周一开头
  const nDays = new Date(y, mo, 0).getDate();
  const cells: (string | null)[] = [...Array(lead).fill(null), ...Array.from({ length: nDays }, (_, i) => `${cur}-${String(i + 1).padStart(2, '0')}`)];

  const mrows = (monthly && monthly.length
    ? monthly
    : months.map((m) => {
        const rs = daily.filter((r) => d10(r.date).startsWith(m));
        const p = rs.map(pnlOf);
        return { month: m, pnl: p.reduce((s, v) => s + v, 0), days: rs.length, win_days: p.filter((v) => v > 0).length, win_rate: p.filter((v) => v > 0).length / Math.max(1, rs.length), start_equity: 0, end_equity: 0 };
      })) as NonNullable<LedgerMeta['monthly']>;
  const mInfo = mrows.find((m) => m.month === cur);

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-4 space-y-3" data-testid="pnl-calendar">
      <div className="flex items-center justify-between">
        <div className="text-sm font-bold text-white">每日盈亏日历</div>
        <div className="flex gap-1 flex-wrap justify-end">
          {months.map((m) => (
            <button
              key={m}
              onClick={() => setMonth(m)}
              className={`px-2 py-0.5 rounded text-[11px] font-mono ${m === cur ? 'bg-sky-700 text-white' : 'bg-slate-900 text-slate-400 border border-slate-800'}`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_240px] gap-4">
        <div>
          <div className="grid grid-cols-7 gap-1 text-center text-[10px] text-slate-500 mb-1">
            {['一', '二', '三', '四', '五', '六', '日'].map((w) => (
              <div key={w}>周{w}</div>
            ))}
          </div>
          <div className="grid grid-cols-7 gap-1">
            {cells.map((d, i) => {
              const r = d ? byDay[d] : undefined;
              if (!d) return <div key={i} />;
              if (!r)
                return (
                  <div key={d} className="h-14 rounded bg-slate-950/40 border border-slate-900 p-1 text-[10px] text-slate-700">
                    {Number(d.slice(8))}
                  </div>
                );
              const ret = retOf(r);
              const a = Math.min(1, Math.abs(ret) / maxAbs);
              const bg = ret > 0 ? `rgba(225,29,72,${0.12 + 0.6 * a})` : ret < 0 ? `rgba(5,150,105,${0.12 + 0.6 * a})` : 'rgba(51,65,85,0.3)';
              const p = pnlOf(r);
              return (
                <button
                  key={d}
                  data-testid={`cal-${d}`}
                  data-pnl={p}
                  onClick={() => onSelectDate(d)}
                  className={`h-14 rounded p-1 text-left border ${d === selectedDate ? 'border-white' : 'border-transparent'} hover:border-slate-400`}
                  style={{ background: bg }}
                  title={`${d} 当日盈亏 ${yuan(p, true)}（${pctS(ret)}）${r.is_rebalance ? ' · 调仓决策日' : ''}`}
                >
                  <div className="flex justify-between text-[10px] text-slate-200">
                    <span>{Number(d.slice(8))}</span>
                    {r.is_rebalance && <span className="text-sky-300">调</span>}
                  </div>
                  <div className="text-[11px] font-mono font-bold text-white leading-tight">{yuan(p, true, 0)}</div>
                  <div className="text-[10px] font-mono text-slate-100/90">{pctS(ret)}</div>
                </button>
              );
            })}
          </div>
        </div>
        <div className="space-y-2 text-xs">
          {mInfo && (
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1 font-mono" data-testid="month-summary">
              <div className="text-slate-400 font-sans">{cur} 月度汇总</div>
              <div className={`text-lg font-bold ${mInfo.pnl > 0 ? 'text-rose-400' : mInfo.pnl < 0 ? 'text-emerald-400' : 'text-white'}`}>{yuan(mInfo.pnl, true)}</div>
              <div className="text-slate-300">
                交易日 {mInfo.days} · 盈利 {mInfo.win_days} 天 · 胜率 {(mInfo.win_rate * 100).toFixed(0)}%
              </div>
            </div>
          )}
          <div className="max-h-56 overflow-y-auto">
            <table className="w-full font-mono text-[11px]">
              <tbody>
                {mrows.map((m) => (
                  <tr key={m.month} className="border-t border-slate-800 cursor-pointer hover:bg-slate-900" onClick={() => setMonth(m.month)}>
                    <td className="py-1 text-slate-400">{m.month}</td>
                    <td className={`text-right ${m.pnl > 0 ? 'text-rose-400' : m.pnl < 0 ? 'text-emerald-400' : ''}`}>{yuan(m.pnl, true, 0)}</td>
                    <td className="text-right text-slate-400">{(m.win_rate * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {notional && <div className="text-[10px] text-slate-500">影子账本按权重记账，金额按名义本金 100 万折算。</div>}
        </div>
      </div>
    </div>
  );
};
