// QuantMind Defensive Runtime Adapter Layer
// Ensures 100% resilient data shapes and prevents any undefined/null/type mismatches

import {
  FormalReport,
  BrainResponse,
  MarketState,
  CandidateItem,
  BypassEvent,
  RiskCardItem,
  ClaimItem,
  OrderItem,
  HoldingItem,
  TierData,
} from './types';

export const safeArray = <T>(val: any): T[] => {
  if (!val) return [];
  if (Array.isArray(val)) return val;
  return [];
};

export const safeStringList = (val: any): string[] => {
  if (!val) return [];
  if (Array.isArray(val)) {
    return val.map((x) => String(x || '').trim()).filter(Boolean);
  }
  if (typeof val === 'string') {
    return val
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return [];
};

export const safeNumber = (val: any, fallback: number = 0): number => {
  if (val === null || val === undefined || isNaN(Number(val))) return fallback;
  return Number(val);
};

export const safeString = (val: any, fallback: string = ''): string => {
  if (val === null || val === undefined) return fallback;
  return String(val);
};

// 1. Adapt Candidate Item
export const adaptCandidateItem = (raw: any): CandidateItem => {
  return {
    ticker: safeString(raw?.ticker, '000000.SZ'),
    horizon: raw?.horizon === '63d' ? '63d' : '20d',
    reasons: safeStringList(raw?.reasons),
    priority: safeNumber(raw?.priority, 99),
    ens20: raw?.ens20 !== null && raw?.ens20 !== undefined ? safeNumber(raw.ens20) : undefined,
    ridge20: raw?.ridge20 !== null && raw?.ridge20 !== undefined ? safeNumber(raw.ridge20) : undefined,
    ens63: raw?.ens63 !== null && raw?.ens63 !== undefined ? safeNumber(raw.ens63) : undefined,
    ridge63: raw?.ridge63 !== null && raw?.ridge63 !== undefined ? safeNumber(raw.ridge63) : undefined,
    name: raw?.name,
    board: raw?.board,
  };
};

// 2. Adapt Bypass Event
export const adaptBypassEvent = (raw: any): BypassEvent => {
  return {
    ticker: safeString(raw?.ticker, '000000.SZ'),
    importance: raw?.importance === 'major' ? 'major' : 'watch',
    summaries: safeStringList(raw?.summaries),
    evidence_ids: safeStringList(raw?.evidence_ids),
    rules: safeStringList(raw?.rules),
    rule_version: raw?.rule_version,
  };
};

// 3. Adapt Risk Card
export const adaptRiskCard = (raw: any): RiskCardItem => {
  if (!raw) return { ticker: '' };
  return {
    ticker: safeString(raw.ticker),
    beta_252: raw.beta_252 !== undefined ? safeNumber(raw.beta_252) : undefined,
    vol_60_ann: raw.vol_60_ann !== undefined ? safeNumber(raw.vol_60_ann) : undefined,
    mom_20: raw.mom_20 !== undefined ? safeNumber(raw.mom_20) : undefined,
    mom_60: raw.mom_60 !== undefined ? safeNumber(raw.mom_60) : undefined,
    mom_250: raw.mom_250 !== undefined ? safeNumber(raw.mom_250) : undefined,
    adv20_yuan: raw.adv20_yuan !== undefined ? safeNumber(raw.adv20_yuan) : undefined,
    gap_p95_250: raw.gap_p95_250 !== undefined ? safeNumber(raw.gap_p95_250) : undefined,
    // 250 日口径；不再混用 60 日字段（两个窗口不能互相替代）
    limit_up_close_250: raw.limit_up_close_250 !== undefined && raw.limit_up_close_250 !== null ? safeNumber(raw.limit_up_close_250) : undefined,
    limit_down_close_250: raw.limit_down_close_250 !== undefined && raw.limit_down_close_250 !== null ? safeNumber(raw.limit_down_close_250) : undefined,
    industry_tushare: raw.industry_tushare ?? undefined,
    gap_freq_60: raw.gap_freq_60 !== undefined ? safeNumber(raw.gap_freq_60) : undefined,
    har_next_day_vol_ann: raw.har_next_day_vol_ann !== undefined ? safeNumber(raw.har_next_day_vol_ann) : undefined,
    av_3d_cum_dev: raw.av_3d_cum_dev !== undefined ? safeNumber(raw.av_3d_cum_dev) : undefined,
    av_10d_cum_dev: raw.av_10d_cum_dev !== undefined ? safeNumber(raw.av_10d_cum_dev) : undefined,
    ex_dividend_next20d: safeString(raw.ex_dividend_next20d),
    rules_status: safeString(raw.rules_status),
  };
};

// 4. Adapt Market State
export const adaptMarketState = (raw: any): MarketState => {
  if (!raw) return { as_of: '' };
  return {
    as_of: safeString(raw.as_of),
    trend: raw.trend || {},
    style: raw.style || {},
    volatility_hs300: raw.volatility_hs300 || {},
    breadth: raw.breadth || {},
    margin: raw.margin || {},
    northbound: raw.northbound || {},
  };
};

// 5. Adapt Formal Tier
export const adaptTierData = (raw: any): TierData => {
  const holdings: HoldingItem[] = safeArray<any>(raw?.holdings).map((h) => ({
    ticker: safeString(h?.ticker),
    name: safeString(h?.name),
    board: safeString(h?.board),
    shares: safeNumber(h?.shares),
    price: safeNumber(h?.price),
    value: safeNumber(h?.value),
    weight: safeNumber(h?.weight),
  }));

  const orders: OrderItem[] = safeArray<any>(raw?.orders_next_open).map((o) => ({
    side: o?.side === 'sell' ? 'sell' : 'buy',
    ticker: safeString(o?.ticker),
    name: safeString(o?.name),
    board: safeString(o?.board),
    est_shares: safeNumber(o?.est_shares),
    ref_close: safeNumber(o?.ref_close),
    est_value: safeNumber(o?.est_value),
    rank_in_target: safeNumber(o?.rank_in_target),
    decided: safeString(o?.decided),
  }));

  return {
    aum: safeNumber(raw?.aum),
    top_n: safeNumber(raw?.top_n, 10),
    boards: Array.isArray(raw?.boards) ? raw.boards.join(' + ') : safeString(raw?.boards, '全市场'),
    equity: safeNumber(raw?.equity, raw?.aum || 0),
    cash: safeNumber(raw?.cash, raw?.aum || 0),
    return_since_start: safeNumber(raw?.return_since_start, 0),
    pool_ew_since_start: safeNumber(raw?.pool_ew_since_start, 0),
    holdings,
    orders_next_open: orders,
    costs: {
      commission: safeNumber(raw?.costs?.commission),
      transfer: safeNumber(raw?.costs?.transfer),
      stamp: safeNumber(raw?.costs?.stamp),
      slippage: safeNumber(raw?.costs?.slippage),
      dividend_tax: safeNumber(raw?.costs?.dividend_tax),
    },
    targets_latest: safeStringList(raw?.targets_latest),
  };
};

// 6. Adapt Formal Report
export const adaptFormalReport = (raw: any): FormalReport | null => {
  if (!raw || !raw.tiers) return null;
  const tiers: Record<string, TierData> = {};
  for (const k of Object.keys(raw.tiers)) {
    tiers[k] = adaptTierData(raw.tiers[k]);
  }

  return {
    as_of: safeString(raw.as_of),
    mode: safeString(raw.mode, 'preview'),
    approval: {
      research_report_approved: Boolean(raw.approval?.research_report_approved),
      capital_gate: Boolean(raw.approval?.capital_gate),
      approved_on: safeString(raw.approval?.approved_on),
      evidence_at_approval: safeString(raw.approval?.evidence_at_approval),
    },
    kill_switch: Boolean(raw.kill_switch),
    go_live: Boolean(raw.go_live),
    rebalance_dates: safeStringList(raw.rebalance_dates),
    today_is_rebalance: Boolean(raw.today_is_rebalance),
    next_rebalance: safeString(raw.next_rebalance),
    tiers,
  };
};

// 7. Adapt Claim Item
export const adaptClaimItem = (raw: any): ClaimItem => {
  return {
    ticker: safeString(raw?.ticker),
    ref: safeString(raw?.ref, 'A1:c0'),
    type: raw?.type || 'reported_fact',
    text: safeString(raw?.text),
    value: raw?.value !== null && raw?.value !== undefined ? safeNumber(raw.value) : null,
    unit: raw?.unit || null,
    period: raw?.period || null,
    check: raw?.check || 'unverifiable',
    a6: raw?.a6 || undefined,
    formula: raw?.formula || undefined,
    inputs: safeStringList(raw?.inputs),
    evidence: safeArray(raw?.evidence),
  };
};

// 模型共识（原 calculateModelConsensus，CV = σ/|μ|）与事后收益（原 calculateForwardReturn，不复权价格、无基准）
// 已移到后端：/api/consensus（截面百分位与分歧度）、/api/forward（T+1 开盘复权入场、研究池等权与指数基准）。
// 见 src/api/analytics.ts 与 quantmind/line_a/api_analytics.py。
