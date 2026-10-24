// 后端聚合接口（quantmind/line_a/api_analytics.py）的类型、请求与 A 股执行规则。
// 原来前端自己算的事后收益、红队命中率、模型共识、声明核验溯源、漏斗数量，全部改由后端按系统口径计算；
// 前端只负责渲染，不再自己拼数字。

const API = '/api';

async function get<T>(url: string): Promise<T> {
  const res = await fetch(`${API}${url}`);
  if (!res.ok) throw new Error(`API ${res.status} at ${url}: ${(await res.text()).slice(0, 200)}`);
  return res.json() as Promise<T>;
}

export interface ForwardRow {
  ticker: string;
  status: 'filled' | 'suspended_at_entry' | 'limit_up_at_open' | 'pending';
  entry_price_raw: number | null;
  [k: string]: number | string | null; // ret_{h} / excess_pool_{h} / excess_csi2000_{h}
}

export interface ForwardBench {
  exit_date: string | null;
  pool_ew?: number | null;
  hs300?: number | null;
  csi1000?: number | null;
  csi2000?: number | null;
}

export interface ForwardResponse {
  meta: {
    as_of: string;
    entry_date: string | null;
    last_bar: string;
    convention: string;
    horizons: number[];
    benchmark_names?: Record<string, string>;
    note?: string;
  };
  rows: ForwardRow[];
  benchmarks: Record<string, ForwardBench>;
}

export interface GroupStats {
  n: number;
  mean: number | null;
  share_below_0: number | null;
}

export interface RedTeamResponse {
  meta: ForwardResponse['meta'] & {
    source: string;
    horizon: number;
    rules: Record<string, string>;
    rule_version: string;
    n_subjects: number;
    flagged_share: number;
    definition: string;
  };
  benchmarks: Record<string, ForwardBench>;
  rows: (ForwardRow & { rules: string[] })[];
  by_group: Record<
    'absolute' | 'excess_pool' | 'excess_csi2000',
    { flagged: GroupStats; unflagged: GroupStats; hit_rate: number | null; base_rate: number | null; lift: number | null; mean_gap: number | null }
  >;
  by_rule: Record<string, GroupStats>;
  missing_inputs?: string[];
  warning: string | null;
}

export interface ConsensusRow {
  ticker: string;
  xgb_pct: number | null;
  cat_pct: number | null;
  tabm_pct: number | null;
  seqfusion_pct: number | null;
  ENS_pct: number | null;
  dispersion_pp: number | null;
  range_pp: number | null;
  dispersion_rank_pct: number | null;
  label: string;
  [k: string]: number | string | null;
}

export interface ConsensusResponse {
  meta: { as_of: string; n: number; universe: string; score_scale: string; dispersion: string; spearman: Record<string, Record<string, number>> };
  row?: ConsensusRow | null;
  note?: string;
}

export interface ClaimTrace {
  as_of: string;
  ticker: string;
  ref: string;
  run_file: string;
  model: string;
  claim: {
    local_id: string;
    claim_type: string;
    text: string;
    value: number | null;
    unit: string | null;
    formula?: string | null;
    inputs?: string[];
    evidence?: { evidence_id: string; field?: string | null; value?: number | null; unit?: string | null; quote?: string | null }[];
  };
  check_at_run: { status: string; notes: string[] } | null;
  check_now: { status: string; notes: string[] } | null;
  cited: {
    ref: { evidence_id: string; field?: string | null; value?: number | null; unit?: string | null; quote?: string | null };
    found: boolean;
    record: Record<string, unknown> | null;
    record_value: unknown;
    record_unit: string | null;
  }[];
  inputs: { local_id: string; claim_type: string; text: string; value: number | null; unit: string | null; check: string | null }[];
  recompute: { formula: string; env: Record<string, number | null>; value?: number; stated?: number | null; rel_diff?: number | null; error?: string } | null;
  verifier: string;
}

export interface FunnelLayer {
  layer: string;
  name: string;
  n: number;
  detail: Record<string, number | null>;
}

export interface ForwardIcResponse {
  horizon: number;
  model: string;
  rows: { as_of: string; ic: number | null; n: number; top10_minus_pool?: number; exit_date?: string; note?: string }[];
  mean_ic: number | null;
  n_dates: number;
  note: string;
}

export interface ExecutionRules {
  execution: { commission_rate: number; min_commission: number; transfer_rate: number; participation: number; price: string };
  stamp_duty_sell: number;
  lot: Record<'MAIN' | 'CHINEXT' | 'STAR' | 'BSE', { min: number; step: number }>;
  tiers: Record<string, { aum: number; top_n: number; boards: string[] }>;
  source: string;
}

export const analyticsApi = {
  forward: (d: string, opts: { tickers?: string[]; source?: 'formal' | 'f4' | 'f3top'; horizons?: number[]; top?: number } = {}) => {
    const q = new URLSearchParams();
    if (opts.tickers?.length) q.set('tickers', opts.tickers.join(','));
    if (opts.source) q.set('source', opts.source);
    if (opts.horizons) q.set('horizons', opts.horizons.join(','));
    if (opts.top) q.set('top', String(opts.top));
    return get<ForwardResponse>(`/forward/${d}?${q}`);
  },
  redteam: (d: string, source: 'formal' | 'f4' | 'f3top' = 'f4', horizon = 5, version: 'v1' | 'v2' = 'v2') =>
    get<RedTeamResponse>(`/redteam/${d}?source=${source}&horizon=${horizon}&version=${version}`),
  consensus: (d: string, ticker: string) => get<ConsensusResponse>(`/consensus/${d}?ticker=${encodeURIComponent(ticker)}`),
  claim: (d: string, ticker: string, ref: string) =>
    get<ClaimTrace>(`/claim/${d}/${encodeURIComponent(ticker)}/${encodeURIComponent(ref)}`),
  funnel: (d: string) => get<{ as_of: string; layers: FunnelLayer[] }>(`/funnel/${d}`),
  forwardIc: (horizon = 5) => get<ForwardIcResponse>(`/forward_ic?horizon=${horizon}`),
  executionRules: () => get<ExecutionRules>('/rules/execution'),
};

// ---------------------------------------------------------------- A 股执行规则（与 sim_account.py 一致）
export type Board = 'MAIN' | 'CHINEXT' | 'STAR' | 'BSE';

export const boardOf = (ticker: string): Board => {
  const t = String(ticker);
  if (t.startsWith('688') || t.startsWith('689')) return 'STAR';
  if (t.startsWith('300') || t.startsWith('301')) return 'CHINEXT';
  if (t.endsWith('.BJ') || t.startsWith('92') || t.startsWith('8') || t.startsWith('43')) return 'BSE';
  return 'MAIN';
};

/** 申报数量合法化：不足最小申报单位记 0；超过的部分按步长取整（科创板 200 股起、1 股递增；北交所 100 股起、1 股递增）。 */
export const legalShares = (shares: number, board: Board, lot: ExecutionRules['lot']): number => {
  const { min, step } = lot[board] ?? { min: 100, step: 100 };
  if (shares < min) return 0;
  return min + Math.floor((shares - min) / step + 1e-9) * step;
};

/** 下一档合法数量（加减按钮用）：在 0 和最小单位之间跳，之上按步长；科创板等 1 股步长的按 100 股为一步，便于操作。 */
export const stepShares = (shares: number, dir: 1 | -1, board: Board, lot: ExecutionRules['lot']): number => {
  const { min, step } = lot[board] ?? { min: 100, step: 100 };
  const ui = Math.max(step, 100);
  if (dir > 0) return shares < min ? min : legalShares(shares + ui, board, lot);
  if (shares <= min) return 0;
  return legalShares(Math.max(min, shares - ui), board, lot);
};

export interface TradeCost {
  commission: number;
  transfer: number;
  stamp: number;
  total: number;
}

/** 单笔费用：佣金（最低 5 元）、过户费、卖出印花税；滑点不在这里估计（正式日报里是事后实际值）。 */
export const tradeCost = (gross: number, side: 'buy' | 'sell', r: ExecutionRules): TradeCost => {
  if (gross <= 0) return { commission: 0, transfer: 0, stamp: 0, total: 0 };
  const commission = Math.max(gross * r.execution.commission_rate, r.execution.min_commission);
  const transfer = gross * r.execution.transfer_rate;
  const stamp = side === 'sell' ? gross * r.stamp_duty_sell : 0;
  return { commission, transfer, stamp, total: commission + transfer + stamp };
};

export const pct = (v: number | null | undefined, digits = 2, signed = true): string =>
  v === null || v === undefined || !Number.isFinite(v) ? '—' : `${signed && v > 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`;
