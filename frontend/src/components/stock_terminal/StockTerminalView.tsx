import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, Sparkles, AlertCircle } from 'lucide-react';
import { StockChart } from './StockChart';
import { ClaimsInspector } from './ClaimsInspector';
import { RiskCardRadar } from './RiskCardRadar';
import { AnnouncementsTimeline } from './AnnouncementsTimeline';
import {
  StockPriceRow,
  ClaimItem,
  RiskCardItem,
  AnnouncementItem,
} from '../../api/types';
import { quantmindApi } from '../../api/client';

interface StockTerminalViewProps {
  currentTicker: string;
  onSelectTicker: (t: string) => void;
  selectedDate: string;
}

export const StockTerminalView: React.FC<StockTerminalViewProps> = ({
  currentTicker,
  onSelectTicker,
  selectedDate,
}) => {
  const [tickerInput, setTickerInput] = useState(currentTicker);
  const [prices, setPrices] = useState<StockPriceRow[]>([]);
  const [claims, setClaims] = useState<ClaimItem[]>([]);
  const [riskCard, setRiskCard] = useState<RiskCardItem | null>(null);
  const [announcements, setAnnouncements] = useState<AnnouncementItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const quickTickers = [
    '688383.SH',
    '600000.SH',
    '000001.SZ',
    '600519.SH',
    '300750.SZ',
    '688041.SH',
  ];

  const loadTickerData = async (ticker: string) => {
    setLoading(true);
    setError(null);
    try {
      // 1. Load prices
      try {
        const p = await quantmindApi.getStockPrices(ticker, '2026-06-01');
        setPrices(p || []);
      } catch (e) {
        console.warn('Failed to load prices for', ticker, e);
        setPrices([]);
      }

      // 2. Load research claims
      try {
        const r = await quantmindApi.getResearchTicker(selectedDate, ticker);
        setClaims(r.claims || []);
      } catch (e) {
        console.warn('Failed to load research claims for', ticker, e);
        setClaims([]);
      }

      // 3. Load risk card
      try {
        const rc = await quantmindApi.getRiskCards(selectedDate, ticker);
        setRiskCard(rc && rc.length > 0 ? rc[0] : null);
      } catch (e) {
        console.warn('Failed to load risk card for', ticker, e);
        setRiskCard(null);
      }

      // 4. Load announcements
      try {
        const ann = await quantmindApi.getStockAnnouncements(ticker, selectedDate, 90);
        setAnnouncements(ann?.announcements || []);
      } catch (e) {
        console.warn('Failed to load announcements for', ticker, e);
        setAnnouncements([]);
      }
    } catch (err: any) {
      setError(err.message || '加载个股数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setTickerInput(currentTicker);
    loadTickerData(currentTicker);
  }, [currentTicker, selectedDate]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanTicker = tickerInput.trim().toUpperCase();
    if (cleanTicker) {
      onSelectTicker(cleanTicker);
    }
  };

  return (
    <div className="space-y-4">
      {/* Search Header Bar */}
      <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-4 shadow-lg flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400">
            <TrendingUp className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">个股全景透视交互终端 (Stock Terminal)</h2>
            <p className="text-xs text-slate-400">
              行情 K 线 · 智能体研报声明 · 代码级检验 · 风险执行卡 · 时点公告链
            </p>
          </div>
        </div>

        {/* Ticker Search Form */}
        <form onSubmit={handleSubmit} className="flex items-center gap-2 w-full sm:w-auto">
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs w-full sm:w-60">
            <Search className="w-3.5 h-3.5 text-slate-400 mr-2 shrink-0" />
            <input
              type="text"
              value={tickerInput}
              onChange={(e) => setTickerInput(e.target.value)}
              placeholder="输入代码 如 688383.SH"
              className="bg-transparent text-white font-mono placeholder-slate-500 focus:outline-none w-full"
            />
          </div>
          <button
            type="submit"
            className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs shadow-sm transition shrink-0"
          >
            穿透分析
          </button>
        </form>
      </div>

      {/* Quick Select Pills */}
      <div className="flex items-center gap-2 text-xs">
        <span className="text-slate-500 text-[11px]">快捷直达:</span>
        <div className="flex flex-wrap gap-1.5">
          {quickTickers.map((t) => (
            <button
              key={t}
              onClick={() => onSelectTicker(t)}
              className={`px-2 py-0.5 rounded font-mono text-xs transition ${
                currentTicker === t
                  ? 'bg-blue-600 text-white font-bold'
                  : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="p-4 text-center text-xs text-blue-400 font-mono animate-pulse">
          正在穿透加载 {currentTicker} 的全景数据 (K 线 / F7 声明 / 风险卡 / 公告)...
        </div>
      )}

      {/* 1. Candlestick Chart */}
      <StockChart ticker={currentTicker} prices={prices} />

      {/* 2. Risk Card Metrics */}
      <RiskCardRadar riskCard={riskCard} ticker={currentTicker} />

      {/* 3. Claims Inspector (A1-A6) */}
      <ClaimsInspector claims={claims} ticker={currentTicker} asOf={selectedDate} />

      {/* 4. Announcements Timeline */}
      <AnnouncementsTimeline announcements={announcements} ticker={currentTicker} />
    </div>
  );
};
