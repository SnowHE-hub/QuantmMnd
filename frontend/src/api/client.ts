// QuantMind API Client - 100% Real API integration with error handling and fallback
import {
  HealthInfo,
  AvailableDates,
  FormalReport,
  BrainResponse,
  MarketState,
  CandidateItem,
  ScoreItem,
  BypassEvent,
  RiskCardItem,
  ResearchResponse,
  ResearchTickerResponse,
  BooksOverview,
  BookNavRow,
  StockPriceRow,
  AnnouncementsResponse,
  BacktestT052Response,
  BacktestNavRow,
  ExperimentSummary,
  CatalogData,
} from './types';

const API_BASE = '/api';

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText} at ${url}`);
  }
  return res.json() as Promise<T>;
}

async function fetchText(url: string): Promise<string> {
  const res = await fetch(`${API_BASE}${url}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${res.statusText} at ${url}`);
  }
  return res.text();
}

export const quantmindApi = {
  // 1. Health
  getHealth: () => fetchJson<HealthInfo>('/health'),

  // 2. Dates
  getDates: () => fetchJson<AvailableDates>('/dates'),

  // 3. Formal Report (JSON)
  getFormal: (d: string) => fetchJson<FormalReport>(`/formal/${d}`),

  // 4. Formal Report (Markdown)
  getFormalMarkdown: (d: string) => fetchText(`/formal/${d}/markdown`),

  // 5. Agent Brain
  getBrain: (d: string) => fetchJson<BrainResponse>(`/brain/${d}`),

  // 6. Market State
  getMarket: (d: string) => fetchJson<MarketState>(`/market/${d}`),

  // 7. Research Candidates
  getCandidates: (d: string, horizon: '20d' | '63d' = '20d', top: number = 50) =>
    fetchJson<CandidateItem[]>(`/candidates/${d}?horizon=${horizon}&top=${top}`),

  // 8. Scores
  getScores: (d: string, model: string = 'ens_20d', top: number = 100) =>
    fetchJson<ScoreItem[]>(`/scores/${d}?model=${model}&top=${top}`),

  // 9. Bypass Events
  getBypass: (d: string, importance?: 'major' | 'watch') => {
    const q = importance ? `?importance=${importance}` : '';
    return fetchJson<BypassEvent[]>(`/bypass/${d}${q}`);
  },

  // 10. Risk Cards
  getRiskCards: (d: string, ticker?: string) => {
    const q = ticker ? `?ticker=${encodeURIComponent(ticker)}` : '';
    return fetchJson<RiskCardItem[]>(`/risk_cards/${d}${q}`);
  },

  // 11. Research Overview
  getResearch: (d: string) => fetchJson<ResearchResponse>(`/research/${d}`),

  // 12. Research for Ticker
  getResearchTicker: (d: string, ticker: string) =>
    fetchJson<ResearchTickerResponse>(`/research/${d}/${encodeURIComponent(ticker)}`),

  // 13. Books Summary
  getBooks: (d: string) => fetchJson<BooksOverview>(`/books/${d}`),

  // 14. Book Details (NAV, targets, etc.)
  getBookDetail: (d: string, book: string, kind: 'nav' | 'targets' | 'capacity' | 'rejected' = 'nav') =>
    fetchJson<BookNavRow[]>(`/books/${d}/${encodeURIComponent(book)}/${kind}`),

  // 15. Stock Prices (OHLCV)
  getStockPrices: (ticker: string, start: string = '2026-01-01', end?: string) => {
    const q = end ? `?start=${start}&end=${end}` : `?start=${start}`;
    return fetchJson<StockPriceRow[]>(`/stock/${encodeURIComponent(ticker)}/prices${q}`);
  },

  // 16. Stock Announcements
  getStockAnnouncements: (ticker: string, asOf: string, lookbackDays: number = 90) =>
    fetchJson<AnnouncementsResponse>(
      `/stock/${encodeURIComponent(ticker)}/announcements?as_of=${asOf}&lookback_days=${lookbackDays}`
    ),

  // 17. Backtest T052 Summary
  getBacktestT052: () => fetchJson<BacktestT052Response>('/backtest/t052'),

  // 18. Backtest NAV
  getBacktestNav: (key: string) =>
    fetchJson<BacktestNavRow[]>(`/backtest/t052/nav/${encodeURIComponent(key)}`),

  // 19. Experiments List & Detail
  getExperiments: () => fetchJson<ExperimentSummary[]>('/experiments'),
  getExperimentDetail: (tid: string) => fetchText(`/experiments/${encodeURIComponent(tid)}`),

  // 20. Data Catalog (with fallback)
  getCatalog: async (): Promise<CatalogData> => {
    try {
      return await fetchJson<CatalogData>('/catalog');
    } catch (e) {
      console.warn('API /catalog fallback to bundled data_catalog.json', e);
      // 离线兜底：打包进单独的 chunk，只有接口不可用时才加载（原来直接打进主包，约 2 MB）
      const local = await import('../data/catalog.json');
      return (local.default ?? local) as unknown as CatalogData;
    }
  },
};
