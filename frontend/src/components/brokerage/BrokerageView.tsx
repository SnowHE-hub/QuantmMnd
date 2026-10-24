import React, { useEffect, useMemo, useState } from 'react';
import { Wallet, ShieldCheck, AlertTriangle, Loader2 } from 'lucide-react';
import { AccountListItem, d10, DailyRow, LedgerMeta, ledgerApi, pctS, RoundTrip, tone, yuan } from '../../api/ledger';
import { EquityCurve } from './EquityCurve';
import { PnlCalendar } from './PnlCalendar';
import { PositionsPanel } from './PositionsPanel';
import { TradeLedger } from './TradeLedger';
import { TradeReplayModal } from './TradeReplayModal';

const GROUPS: [AccountListItem['kind'], string][] = [
  ['t052', '2026 全程模拟（T-052 历史回放）'],
  ['formal', 'L10 正式荐股跟踪账户'],
  ['shadow', 'L9 前瞻影子账本（权重记账）'],
];
const NOTIONAL = 1e6;

type Pos = Awaited<ReturnType<typeof ledgerApi.positions>>;
type Trades = Awaited<ReturnType<typeof ledgerApi.trades>>;

// 模拟持仓与交易账本：账户切换 → 资产总览 → 资金曲线 / 盈亏日历 → 持仓 / 回合 → 交割单 → 个股买卖点复盘。
// 所有数字来自 /api/ledger（后端按股数逐笔记账并做会计核对），这里不做任何估算。
export const BrokerageView: React.FC = () => {
  const [accounts, setAccounts] = useState<AccountListItem[]>([]);
  const [account, setAccount] = useState<string>('t052_W6_100000');
  const [data, setData] = useState<{ meta: LedgerMeta; daily: DailyRow[]; round_trips: RoundTrip[] } | null>(null);
  const [trades, setTrades] = useState<Trades | null>(null);
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [pos, setPos] = useState<Pos | null>(null);
  const [replay, setReplay] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    ledgerApi
      .accounts()
      .then((a) => {
        setAccounts(a);
        if (!a.some((x) => x.account === account) && a.length) setAccount(a[0].account);
      })
      .catch((e) => setErr(String(e.message || e)));
  }, []);

  useEffect(() => {
    let alive = true;
    setData(null);
    setTrades(null);
    setPos(null);
    Promise.all([ledgerApi.account(account), ledgerApi.trades(account)])
      .then(([d, t]) => {
        if (!alive) return;
        setData(d);
        setTrades(t);
        setSelectedDate(d10(d.daily[d.daily.length - 1]?.date));
      })
      .catch((e) => alive && setErr(String(e.message || e)));
    return () => {
      alive = false;
    };
  }, [account]);

  useEffect(() => {
    if (!selectedDate) return;
    let alive = true;
    ledgerApi
      .positions(account, selectedDate)
      .then((p) => alive && setPos(p))
      .catch(() => alive && setPos(null));
    return () => {
      alive = false;
    };
  }, [account, selectedDate]);

  const meta = data?.meta;
  const isShadow = meta?.kind === 'shadow';
  const aum = meta?.aum ?? NOTIONAL;
  const day = pos?.day;
  const s = meta?.summary;
  const grouped = useMemo(() => GROUPS.map(([k, label]) => [label, accounts.filter((a) => a.kind === k)] as const), [accounts]);

  return (
    <div className="space-y-4" data-testid="brokerage-view">
      {/* 账户切换 */}
      <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row md:items-center gap-3 justify-between">
        <div className="flex items-center gap-2">
          <Wallet className="w-5 h-5 text-sky-400" />
          <div>
            <div className="text-sm font-bold text-white">模拟持仓与交易账本</div>
            <div className="text-[11px] text-slate-400">按股数逐笔记账：整手、最低 5 元佣金、过户费、印花税、T+1、涨停 / 停牌不可成交、分红与红利税</div>
          </div>
        </div>
        <select
          data-testid="account-select"
          value={account}
          onChange={(e) => setAccount(e.target.value)}
          className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white min-w-[320px]"
        >
          {grouped.map(([label, list]) => (
            <optgroup key={label} label={label}>
              {list.map((a) => (
                <option key={a.account} value={a.account}>
                  {a.title}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>

      {err && <div className="text-rose-300 text-xs">{err}</div>}
      {!data && !err && (
        <div className="flex items-center gap-2 text-slate-400 text-xs">
          <Loader2 className="w-4 h-4 animate-spin" /> 读取账本…
        </div>
      )}

      {data && meta && s && (
        <>
          <div className="text-[11px] text-slate-400 flex items-start gap-2">
            {meta.checks?.passed ? (
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            )}
            <span data-testid="account-desc">
              {meta.description} 期间 {meta.start} → {meta.end}。
              {meta.kind !== 'shadow' && (
                <span data-testid="checks">
                  会计核对{meta.checks?.passed ? '通过' : '未通过'}：总资产 = 现金 + 持仓市值（最大误差 {Number(meta.checks.equity_eq_cash_plus_mv_max_abs ?? 0).toExponential(1)} 元）；
                  总盈亏 = 已实现 + 未实现 + 分红（最大误差 {Number(meta.checks.pnl_decomposition_max_abs ?? 0).toExponential(1)} 元）
                  {meta.checks.t052_nav_reproduced_max_abs !== undefined && `；与 T-052 存档净值逐日一致（最大误差 ${Number(meta.checks.t052_nav_reproduced_max_abs).toFixed(2)} 元）`}。
                </span>
              )}
            </span>
          </div>

          {meta.kind === 'formal' && meta.mode === 'preview' && (
            <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-800/60 text-xs text-amber-200 space-y-2" data-testid="formal-preview">
              <div>
                正式账户 go_live {meta.go_live}，首次调仓 {meta.next_rebalance ?? '—'}。在那之前账户全是现金，下面是按 {meta.end} 收盘价估算的建仓预览委托（尚未成交，不计入账本）。
                从首次调仓起，每天收盘后用同一个模拟器重放，成交、持仓、盈亏会逐日累积到这里。
              </div>
              <table className="w-full font-mono text-[11px]">
                <tbody>
                  {(meta.preview_orders ?? []).map((o) => (
                    <tr key={o.ticker} className="border-t border-amber-900/40">
                      <td className="py-0.5">{o.side === 'buy' ? '预览买入' : '预览卖出'}</td>
                      <td>
                        {o.ticker} <span className="font-sans">{o.name}</span>
                      </td>
                      <td className="font-sans">{o.board}</td>
                      <td className="text-right">{o.est_shares.toLocaleString()} 股</td>
                      <td className="text-right">@ {o.ref_close.toFixed(2)}</td>
                      <td className="text-right">{o.est_shares ? yuan(o.est_value) : '买不起一个申报单位，跳过'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* 资产总览 */}
          <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-4 space-y-3" data-testid="account-header">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <div className="text-[11px] text-slate-400">总资产（{d10(day?.date) || meta.end}）</div>
                <div className="text-2xl font-bold font-mono text-white" data-testid="hdr-equity">
                  {isShadow ? yuan(NOTIONAL * (1 + (day?.cum_ret ?? s.total_ret))) : yuan(day?.equity ?? s.equity)}
                </div>
                <div className="text-[11px] text-slate-500 font-mono">本金 {yuan(aum)}{isShadow ? '（名义）' : ''}</div>
              </div>
              <div>
                <div className="text-[11px] text-slate-400">累计盈亏</div>
                <div className={`text-2xl font-bold font-mono ${tone(day?.cum_ret ?? s.total_ret)}`} data-testid="hdr-total-pnl">
                  {yuan(aum * (day?.cum_ret ?? s.total_ret), true)}
                </div>
                <div className={`text-sm font-mono ${tone(day?.cum_ret ?? s.total_ret)}`} data-testid="hdr-total-ret">
                  {pctS(day?.cum_ret ?? s.total_ret)}
                </div>
              </div>
              <div className="space-y-0.5 text-[11px] font-mono" data-testid="hdr-alpha">
                <div className="text-slate-400 font-sans">同期基准与超额</div>
                {(
                  [
                    ['沪深300', day?.bench_hs300 ?? s.bench.hs300],
                    ['中证2000', day?.bench_csi2000 ?? s.bench.csi2000],
                    ['研究池等权', day?.bench_pool_ew ?? s.bench.pool_ew],
                  ] as [string, number | undefined][]
                ).map(([n, b]) => (
                  <div key={n} className="flex justify-between gap-2">
                    <span className="text-slate-400">{n} {pctS(b)}</span>
                    <span className={tone((day?.cum_ret ?? s.total_ret) - (b ?? 0))}>超额 {b === undefined ? '—' : pctS((day?.cum_ret ?? s.total_ret) - b)}</span>
                  </div>
                ))}
              </div>
              <div>
                <div className="text-[11px] text-slate-400">最大回撤</div>
                <div className="text-xl font-bold font-mono text-emerald-400">{pctS(s.max_drawdown)}</div>
                {!isShadow && <div className="text-[11px] text-slate-500 font-mono">成交 {s.n_trades} 笔 · 回合 {s.n_round_trips}</div>}
              </div>
            </div>
            {!isShadow && day && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3 text-xs font-mono border-t border-slate-800 pt-3">
                  <Cell label="当日盈亏" value={yuan(day.daily_pnl, true)} cls={tone(day.daily_pnl)} testid="hdr-daily" />
                  <Cell label="持仓浮动盈亏" value={yuan(day.unrealized, true)} cls={tone(day.unrealized)} testid="hdr-unrealized" />
                  <Cell label="已清仓落袋盈亏" value={yuan(day.realized_cum, true)} cls={tone(day.realized_cum)} testid="hdr-realized" />
                  <Cell label="累计现金分红" value={yuan(day.dividends_cum, true)} cls={tone(day.dividends_cum)} />
                  <Cell label="持仓市值" value={yuan(day.market_value)} testid="hdr-mv" />
                  <Cell label="可用现金" value={yuan(day.cash)} testid="hdr-cash" />
                  <Cell label="累计交易成本" value={yuan(day.fees_cum)} testid="hdr-fees" />
                </div>
                <div>
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>仓位 {(((day.market_value ?? 0) / (day.equity ?? 1)) * 100).toFixed(1)}%</span>
                    <span>{day.n_pos} 只持仓</span>
                  </div>
                  <div className="h-2 rounded bg-slate-800 overflow-hidden">
                    <div className="h-full bg-sky-500" style={{ width: `${Math.min(100, ((day.market_value ?? 0) / (day.equity ?? 1)) * 100)}%` }} />
                  </div>
                </div>
              </>
            )}
          </div>

          <EquityCurve daily={data.daily} aum={aum} notional={isShadow} selectedDate={selectedDate} onSelectDate={setSelectedDate} />
          <PnlCalendar daily={data.daily} monthly={meta.monthly} aum={aum} notional={isShadow} selectedDate={selectedDate} onSelectDate={setSelectedDate} />
          <PositionsPanel
            date={pos?.date ?? selectedDate}
            positions={pos?.positions ?? []}
            weightsOnly={Boolean(pos?.weights_only)}
            roundTrips={data.round_trips}
            onPickTicker={setReplay}
          />
          {!isShadow && trades && (
            <TradeLedger trades={trades.trades} rejects={trades.rejects} actions={trades.actions} selectedDate={selectedDate} onPickTicker={setReplay} />
          )}
        </>
      )}
      <TradeReplayModal account={account} ticker={replay} onClose={() => setReplay(null)} />
    </div>
  );
};

const Cell: React.FC<{ label: string; value: string; cls?: string; testid?: string }> = ({ label, value, cls, testid }) => (
  <div>
    <div className="text-[10px] text-slate-400 font-sans">{label}</div>
    <div className={`font-bold ${cls ?? 'text-white'}`} data-testid={testid}>
      {value}
    </div>
  </div>
);
