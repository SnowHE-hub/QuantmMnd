// 模拟账户账本接口（后端 quantmind/line_a/ledger.py 生成、api_server.py 只读提供）。
// 所有金额、成本、盈亏都由后端按股数逐笔记账并做会计核对；前端只做格式化和展示。

const API = '/api';
async function get<T>(url: string): Promise<T> {
  const r = await fetch(API + url);
  if (!r.ok) throw new Error(`API ${r.status} at ${url}`);
  return r.json() as Promise<T>;
}

export type AccountKind = 't052' | 'formal' | 'shadow';

export interface AccountSummary {
  equity?: number;
  cash?: number;
  market_value?: number;
  total_pnl?: number;
  total_ret: number;
  realized?: number;
  unrealized?: number;
  dividends?: number;
  daily_pnl?: number;
  fees?: Record<'commission' | 'transfer' | 'stamp' | 'slippage' | 'dividend_tax', number>;
  fees_total?: number;
  max_drawdown: number;
  cum_cost?: number;
  bench: Record<string, number>;
  n_trades?: number;
  n_rejects?: number;
  n_round_trips?: number;
}

export interface AccountListItem {
  account: string;
  kind: AccountKind;
  title: string;
  description: string;
  start: string;
  end: string;
  aum?: number;
  mode?: string;
  tier?: string;
  summary: AccountSummary;
  checks_passed: boolean;
}

export interface LedgerMeta extends AccountListItem {
  monthly?: { month: string; pnl: number; days: number; win_days: number; win_rate: number; start_equity: number; end_equity: number }[];
  rebalance_dates: string[];
  pending_orders?: { side: string; ticker: string; name: string; value: number; decided: string }[];
  preview_orders?: { side: string; ticker: string; name: string; board: string; est_shares: number; ref_close: number; est_value: number }[];
  checks: Record<string, number | boolean | string>;
  rules?: Record<string, unknown>;
  next_rebalance?: string;
  go_live?: string;
}

export interface DailyRow {
  date: string;
  equity?: number;
  cash: number;
  market_value?: number;
  n_pos?: number;
  daily_pnl?: number;
  daily_ret?: number;
  daily_ret_net?: number;
  nav?: number;
  cum_ret: number;
  drawdown: number;
  realized_cum?: number;
  unrealized?: number;
  dividends_cum?: number;
  fees_cum?: number;
  bench_pool_ew?: number;
  bench_hs300?: number;
  bench_csi2000?: number;
  is_rebalance: boolean;
  n_trades?: number;
}

export interface PositionRow {
  date: string;
  ticker: string;
  name: string;
  board: string;
  shares?: number;
  available?: number;
  bought_today?: boolean;
  avg_cost?: number;
  diluted_cost?: number;
  close?: number;
  pre_close?: number;
  day_ret?: number;
  day_pnl?: number;
  value?: number;
  weight: number;
  unrealized?: number;
  unrealized_pct?: number;
  open_date?: string;
  days_held?: number;
  as_of?: string;
}

export interface TradeRow {
  date: string;
  ticker: string;
  name: string;
  board: string;
  side: 'buy' | 'sell' | 'writeoff';
  shares: number;
  price: number;
  gross: number;
  commission: number;
  transfer: number;
  stamp: number;
  slippage: number;
  dividend_tax: number;
  fee_total: number;
  net_cash: number;
  cash_after: number;
  realized_pnl: number | null;
  avg_cost?: number | null;
  reason: string;
  decided?: string;
}

export interface RejectRow {
  date: string;
  ticker: string;
  name: string;
  side: string;
  reason: string;
  reason_cn: string;
}

export interface ActionRow {
  date: string;
  ticker: string;
  cash_dividend: number;
  shares_before: number;
  shares_after: number;
}

export interface RoundTrip {
  ticker: string;
  name: string;
  board: string;
  open_date: string;
  close_date: string;
  avg_buy: number;
  avg_sell: number;
  buy_shares: number;
  days_held: number;
  net_pnl: number;
  return: number;
  dividends: number;
  writeoff: boolean;
}

export const ledgerApi = {
  accounts: () => get<AccountListItem[]>('/ledger/accounts'),
  account: (a: string) => get<{ meta: LedgerMeta; daily: DailyRow[]; round_trips: RoundTrip[] }>(`/ledger/${a}`),
  positions: (a: string, date?: string) =>
    get<{ date: string; positions: PositionRow[]; trades: TradeRow[]; rejects: RejectRow[]; actions: ActionRow[]; day: DailyRow; weights_only: boolean }>(
      `/ledger/${a}/positions${date ? `?date=${date}` : ''}`
    ),
  trades: (a: string) => get<{ trades: TradeRow[]; rejects: RejectRow[]; actions: ActionRow[] }>(`/ledger/${a}/trades`),
  ticker: (a: string, t: string) =>
    get<{ prices: { trade_date: string; open: number; high: number; low: number; close: number; vol: number }[]; trades: TradeRow[]; round_trips: RoundTrip[]; actions: ActionRow[]; note: string }>(
      `/ledger/${a}/ticker/${t}`
    ),
};

// ---------------------------------------------------------------- 格式化（A 股：赚红亏绿）
export const UP = '#e11d48';
export const DOWN = '#059669';
export const d10 = (s: string | undefined | null) => (s ? String(s).slice(0, 10) : '—');
export const yuan = (v: number | null | undefined, signed = false, digits = 2) =>
  v === null || v === undefined || !Number.isFinite(v)
    ? '—'
    : `${signed && v > 0 ? '+' : v < 0 ? '−' : ''}¥${Math.abs(v).toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
export const pctS = (v: number | null | undefined, digits = 2, signed = true) =>
  v === null || v === undefined || !Number.isFinite(v) ? '—' : `${signed && v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v * 100).toFixed(digits)}%`;
export const tone = (v: number | null | undefined) =>
  v === null || v === undefined || !Number.isFinite(v) || v === 0 ? 'text-slate-200' : v > 0 ? 'text-rose-400' : 'text-emerald-400';
