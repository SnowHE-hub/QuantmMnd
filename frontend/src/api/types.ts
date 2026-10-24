// QuantMind TypeScript Type Definitions for all 19 Read-Only Endpoints and Catalog

export interface HealthInfo {
  ok: boolean;
  data_root: string;
  latest_run: string;
}

export interface AvailableDates {
  runs: string[];
  books: string[];
  research: string[];
  formal: string[];
  brain?: string[];
  claims?: string[];
}

export interface HoldingItem {
  ticker: string;
  name: string;
  board: string;
  shares: number;
  price: number;
  value: number;
  weight: number;
}

export interface OrderItem {
  side: 'buy' | 'sell';
  ticker: string;
  name: string;
  board: string;
  est_shares: number;
  ref_close: number;
  est_value: number;
  rank_in_target: number;
  decided: string;
}

export interface TierData {
  aum: number;
  top_n: number;
  boards: string;
  equity: number;
  cash: number;
  return_since_start: number;
  pool_ew_since_start: number;
  holdings: HoldingItem[];
  orders_next_open: OrderItem[];
  costs: {
    commission: number;
    transfer: number;
    stamp: number;
    slippage: number;
    dividend_tax: number;
  };
  targets_latest: string[];
}

export interface FormalReport {
  as_of: string;
  mode: string;
  approval: {
    research_report_approved: boolean;
    capital_gate: boolean;
    approved_on: string;
    evidence_at_approval: string;
  };
  kill_switch: boolean;
  go_live: boolean;
  rebalance_dates: string[];
  today_is_rebalance: boolean;
  next_rebalance: string;
  tiers: {
    aum_1e5?: TierData;
    aum_1e6?: TierData;
    aum_1e7?: TierData;
    [key: string]: TierData | undefined;
  };
}

export interface BrainFlag {
  ticker: string;
  action: 'caution' | 'avoid_new_buy' | 'review_holding';
  reason: string;
  evidence_ids: string[];
}

export interface ResearchRequest {
  ticker: string;
  roles: Array<'A1' | 'A2' | 'A3' | 'A4'>;
  reason: string;
}

export interface BrainMemo {
  as_of: string;
  summary: string;
  market_view: string;
  flags: BrainFlag[];
  research_requests: ResearchRequest[];
  questions_for_user: string[];
}

export interface BrainResponse {
  memo: BrainMemo;
  run: {
    status?: string;
    cost_usd?: number;
    total_tokens?: number;
    tool_calls?: number;
    flags_dropped?: any[];
    [key: string]: any;
  };
}

export interface MarketIndexTrend {
  last_date?: string;
  close?: number;
  ret_5d?: number;
  ret_20d?: number;
  ret_60d?: number;
  vs_ma20?: number;
  vs_ma60?: number;
  [key: string]: any;
}

export interface MarketState {
  as_of: string;
  trend?: Record<string, MarketIndexTrend | any>;
  style?: {
    csi2000_vs_hs300_rs20?: number;
    csi2000_vs_hs300_rs60?: number;
    rs60_pct3y?: number;
    amount_ratio_20d?: number;
    amount_ratio_pct3y?: number;
    [key: string]: any;
  };
  volatility_hs300?: {
    next_day_vol_ann?: number;
    realized_vol_20d_ann?: number;
    realized_vol_20d_pct3y?: number;
    coef?: number[];
    n_fit?: number;
    [key: string]: any;
  };
  breadth?: {
    n?: number;
    up_share?: number;
    above_ma20_share?: number;
    above_ma60_share?: number;
    new_high_20d?: number;
    new_low_20d?: number;
    [key: string]: any;
  };
  margin?: {
    last_date?: string;
    balance_yuan?: number;
    chg_5d?: number;
    chg_20d?: number;
    pct3y?: number;
    [key: string]: any;
  };
  northbound?: {
    status?: string;
    note?: string;
    last_date?: string;
    north_money?: number;
    [key: string]: any;
  };
  [key: string]: any;
}

export interface CandidateItem {
  ticker: string;
  horizon: '20d' | '63d';
  reasons: string | string[];
  priority: number;
  ens20?: number;
  ridge20?: number;
  ens63?: number;
  ridge63?: number;
  name?: string;
  board?: string;
  [key: string]: any;
}

export interface ScoreItem {
  ticker: string;
  ENS?: number;
  score?: number;
  xgb?: number;
  cat?: number;
  tabm?: number;
  seqfusion?: number;
  ridge?: number;
  name?: string;
  [key: string]: any;
}

export interface BypassEvent {
  ticker: string;
  importance: 'major' | 'watch';
  summaries: string | string[];
  evidence_ids: string | string[];
  rules: string | string[];
  [key: string]: any;
}

export interface RiskCardItem {
  ticker: string;
  beta_252?: number;
  vol_60_ann?: number;
  mom_20?: number;
  mom_60?: number;
  har_next_day_vol_ann?: number;
  gap_freq_60?: number;
  limit_up_count_60?: number;
  limit_down_count_60?: number;
  abnormal_vol_dist?: number;
  is_ex_dividend?: boolean;
  [key: string]: any;
}

export interface ResearchSummary {
  ticker: string;
  name?: string;
  one_line?: string;
  verdict?: string;
  role_summaries?: Record<string, string>;
  a6_verdicts?: Record<string, number>;
  [key: string]: any;
}

export interface ResearchResponse {
  status: any;
  summaries: ResearchSummary[];
  cases: any[];
  report_markdown?: string | null;
}

export type CheckStatus =
  | 'verified'
  | 'minor_mismatch'
  | 'mismatch'
  | 'missing_evidence'
  | 'unverifiable'
  | 'weak_basis';

export interface ClaimItem {
  ticker: string;
  ref: string;
  type: 'reported_fact' | 'computed' | 'inference' | 'counter_evidence' | 'open_question';
  text: string;
  value: number | null;
  unit: string | null;
  period: string | null;
  check: CheckStatus;
  a6?: string;
  formula?: string;
  inputs?: string[];
  evidence?: any[];
  [key: string]: any;
}

export interface ResearchTickerResponse {
  claims: ClaimItem[];
  risk_flags: any[];
}

export interface BookSummaryItem {
  nav_net: number;
  nav_gross?: number;
  excess_net: number;
  cum_return_net: number;
  turnover_daily?: number;
  n_positions?: number;
  cash?: number;
  [key: string]: any;
}

export interface BooksOverview {
  as_of: string;
  grid?: string;
  challengers?: string[];
  books: Record<string, BookSummaryItem>;
  [key: string]: any;
}

export interface BookNavRow {
  date: string;
  nav_net: number;
  nav_gross?: number;
  nav_bench?: number;
  cash?: number;
  n_positions?: number;
  cum_cost?: number;
  daily_ret_net?: number;
  excess_net?: number;
  cum_return_net?: number;
}

export interface StockPriceRow {
  ts_code: string;
  trade_date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  pre_close: number;
  pct_chg: number;
  vol: number;
  amount: number;
  adj_factor: number;
}

export interface AnnouncementItem {
  date?: string;
  ann_date?: string;
  title: string;
  category?: string;
  type?: string;
  stage?: string;
  importance?: string;
  evidence_id?: string;
  [key: string]: any;
}

export interface AnnouncementsResponse {
  announcements?: AnnouncementItem[];
  [key: string]: any;
}

export interface BacktestT052Response {
  primary: {
    production_arm?: string;
    keys?: string[];
    [key: string]: any;
  };
  report_markdown?: string | null;
}

export interface BacktestNavRow {
  date: string;
  equity: number;
  cash: number;
  invested: number;
  n_pos: number;
  bench_pool_ew: number;
  max_weight: number;
}

export interface ExperimentSummary {
  id: string;
  title: string;
  bytes: number;
  verdict_lines: string[];
}

export interface CatalogPatternSample {
  pattern: string;
  file_count: number;
  sample_file: string;
  n_rows?: number;
  n_cols?: number;
  date_column?: string;
  date_min?: string;
  date_max?: string;
  columns?: Array<{
    name: string;
    type: string;
  }>;
}

export interface CatalogEntry {
  id: string;
  category: string;
  title: string;
  path: string;
  abs_roots?: string[];
  paths?: string[];
  size_bytes: number;
  n_files: number;
  file_types: string[];
  patterns: string[];
  coverage?: {
    source?: string;
    min_date?: string;
    max_date?: string;
    [key: string]: any;
  };
  schema?: {
    patterns?: CatalogPatternSample[];
  };
  purpose: string;
  producer: string;
  consumers: string;
  confidence: string;
  regenerable: string;
  notes: string;
}

export interface CatalogData {
  generated_at: string;
  generator: string;
  generated_from: string;
  roots: Record<string, string>;
  totals: {
    size_bytes: number;
    size_human: string;
    n_files: number;
    n_dirs: number;
    n_catalog_entries: number;
  };
  categories: string[];
  entries: CatalogEntry[];
  top_roots?: any[];
  top_dirs?: any[];
  top_files?: any[];
}
