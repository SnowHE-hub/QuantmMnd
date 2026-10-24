import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Header } from './components/common/Header';
import { Sidebar, TabKey } from './components/common/Sidebar';
import { CommandPalette } from './components/common/CommandPalette';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { QuickStockDrawer } from './components/common/QuickStockDrawer';
import { DailyRitualModal } from './components/common/DailyRitualModal';
import { StockCompareModal } from './components/common/StockCompareModal';
import { DailyBriefingExporter } from './components/common/DailyBriefingExporter';
import { ShowcaseTourModal } from './components/common/ShowcaseTourModal';
import { HealthDoctorBanner } from './components/common/HealthDoctorBanner';
import { TopologyView } from './components/topology/TopologyView';
import { DataObservatoryView } from './components/data_observatory/DataObservatoryView';
import { CockpitView } from './components/cockpit/CockpitView';
import { StockTerminalView } from './components/stock_terminal/StockTerminalView';
import { ResearchReaderView } from './components/research_reader/ResearchReaderView';
import { RunbookView } from './components/runbook/RunbookView';
import { BrokerageView } from './components/brokerage/BrokerageView';
import { Swords, X } from 'lucide-react';
import {
  HealthInfo,
  AvailableDates,
  CatalogData,
  FormalReport,
  BrainResponse,
  MarketState,
  CandidateItem,
  BypassEvent,
  RiskCardItem,
  BacktestT052Response,
  BooksOverview,
  ExperimentSummary,
} from './api/types';
import { quantmindApi } from './api/client';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabKey>('cockpit');
  const [selectedDate, setSelectedDate] = useState<string>('');  // 从 /api/health 的 latest_run 取，不写死日期
  const [selectedTicker, setSelectedTicker] = useState<string>('688383.SH');
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [selectedExpId, setSelectedExpId] = useState<string>('T-052');
  const [horizon, setHorizon] = useState<'20d' | '63d'>('20d');

  // Modals & Panels
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState<boolean>(false);
  const [isDailyRitualOpen, setIsDailyRitualOpen] = useState<boolean>(false);
  const [isShowcaseTourOpen, setIsShowcaseTourOpen] = useState<boolean>(false);
  const [isBriefingExporterOpen, setIsBriefingExporterOpen] = useState<boolean>(false);
  const [isCompareModalOpen, setIsCompareModalOpen] = useState<boolean>(false);
  const [comparePair, setComparePair] = useState<string[]>(['688383.SH', '688048.SH']);

  const [quickDrawerTicker, setQuickDrawerTicker] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Data states
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [dates, setDates] = useState<AvailableDates | null>(null);
  const [catalog, setCatalog] = useState<CatalogData | null>(null);
  const [formal, setFormal] = useState<FormalReport | null>(null);
  const [brain, setBrain] = useState<BrainResponse | null>(null);
  const [market, setMarket] = useState<MarketState | null>(null);
  const [candidates, setCandidates] = useState<CandidateItem[]>([]);
  const [bypassEvents, setBypassEvents] = useState<BypassEvent[]>([]);
  const [riskCards, setRiskCards] = useState<RiskCardItem[]>([]);
  const [t052Data, setT052Data] = useState<BacktestT052Response | null>(null);
  const [booksOverview, setBooksOverview] = useState<BooksOverview | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const datesRef = useRef<AvailableDates | null>(null);   // 只请求这一天确实存在的产物（正式日报、大脑、账本不是每天都有）

  // Load global metadata (Health, Dates, Catalog, Experiments, T052)
  const loadGlobalMeta = useCallback(async () => {
    try {
      const [h, d, cat, exp, t52] = await Promise.all([
        quantmindApi.getHealth().catch(() => null),
        quantmindApi.getDates().catch(() => null),
        quantmindApi.getCatalog().catch(() => null),
        quantmindApi.getExperiments().catch(() => []),
        quantmindApi.getBacktestT052().catch(() => null),
      ]);

      if (h) setHealth(h);
      if (d) {
        datesRef.current = d;
        setDates(d);
      }
      if (cat) setCatalog(cat);
      setExperiments(exp || []);
      setT052Data(t52);

      const runs = d?.runs || [];
      if (h?.latest_run) setSelectedDate(h.latest_run);
      else if (runs.length) setSelectedDate(runs[runs.length - 1]);
    } catch (e) {
      console.warn('Error loading global metadata:', e);
    }
  }, []);

  // Load date-specific data
  const loadDateData = useCallback(async (d: string, h: '20d' | '63d') => {
    setIsLoading(true);
    try {
      const av = datesRef.current;
      const has = (k: keyof AvailableDates) => !av || !av[k] || av[k]!.includes(d);
      const skip = Promise.reject(new Error('not available for this date'));
      skip.catch(() => undefined);
      const [f, b, m, c, by, bk, rc] = await Promise.allSettled([
        has('formal') ? quantmindApi.getFormal(d) : skip,
        has('brain') ? quantmindApi.getBrain(d) : skip,
        quantmindApi.getMarket(d),
        quantmindApi.getCandidates(d, h, 50),
        quantmindApi.getBypass(d),
        has('books') ? quantmindApi.getBooks(d) : skip,
        quantmindApi.getRiskCards(d),
      ]);

      setFormal(f.status === 'fulfilled' ? f.value : null);
      setBrain(b.status === 'fulfilled' ? b.value : null);
      setMarket(m.status === 'fulfilled' ? m.value : null);
      setCandidates(c.status === 'fulfilled' && c.value ? c.value : []);
      setBypassEvents(by.status === 'fulfilled' && by.value ? by.value : []);
      setBooksOverview(bk.status === 'fulfilled' ? bk.value : null);
      setRiskCards(rc.status === 'fulfilled' && rc.value ? rc.value : []);
    } catch (e) {
      console.warn('Error loading date data:', e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadGlobalMeta();
  }, [loadGlobalMeta]);

  useEffect(() => {
    if (selectedDate) loadDateData(selectedDate, horizon);
  }, [selectedDate, horizon, loadDateData]);

  const latestDate = dates?.runs?.[dates.runs.length - 1] || health?.latest_run || selectedDate;

  const handleRefresh = () => {
    loadGlobalMeta();
    loadDateData(selectedDate, horizon);
  };

  // Inspecting ticker opens the Quick Stock Drawer
  const handleInspectTicker = (ticker: string) => {
    setQuickDrawerTicker(ticker);
  };

  // Navigating directly into full Stock Terminal
  const handleOpenFullTerminal = (ticker: string) => {
    setSelectedTicker(ticker);
    setActiveTab('stock_terminal');
    setQuickDrawerTicker(null);
  };

  // Add stock to side-by-side comparison pair
  const handleAddToCompare = (ticker: string) => {
    setComparePair((prev) => {
      if (prev.includes(ticker)) return prev;
      if (prev.length >= 2) return [prev[1], ticker];
      return [...prev, ticker];
    });
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col font-sans">
      {/* 1. Top Global Header (Time Machine & Rituals) */}
      <Header
        health={health}
        dates={dates}
        selectedDate={selectedDate}
        onSelectDate={setSelectedDate}
        onOpenCommandPalette={() => setIsCommandPaletteOpen(true)}
        onOpenDailyRitual={() => setIsDailyRitualOpen(true)}
        onOpenShowcaseTour={() => setIsShowcaseTourOpen(true)}
        onOpenBriefingExporter={() => setIsBriefingExporterOpen(true)}
        onRefresh={handleRefresh}
        isLoading={isLoading}
      />

      {/* 2. Proactive Health Doctor Banner */}
      <HealthDoctorBanner
        catalog={catalog}
        health={health}
        selectedDate={selectedDate}
        onNavigateTab={setActiveTab}
      />

      {/* 3. Main Workspace Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Navigation Sidebar */}
        <Sidebar
          activeTab={activeTab}
          onTabChange={setActiveTab}
          catalogCount={catalog?.entries?.length || catalog?.totals?.n_catalog_entries || 0}
          taskCardsCount={experiments.length}
        />

        {/* Central Display Pane */}
        <main className="flex-1 p-5 overflow-y-auto bg-[#070b14]">
          <div className="max-w-[1700px] mx-auto">
            <ErrorBoundary fallbackTitle="当前页面加载遇到异常，请点击重置">
              {activeTab === 'topology' && <TopologyView selectedDate={selectedDate} />}

              {activeTab === 'data_observatory' && (
                <DataObservatoryView
                  catalog={catalog}
                  selectedEntryId={selectedDatasetId}
                  onSelectEntryId={setSelectedDatasetId}
                />
              )}

              {activeTab === 'cockpit' && (
                <CockpitView
                  formal={formal}
                  brain={brain}
                  market={market}
                  candidates={candidates}
                  bypassEvents={bypassEvents}
                  riskCards={riskCards}
                  horizon={horizon}
                  onChangeHorizon={setHorizon}
                  onSelectTicker={handleInspectTicker}
                  onAddToCompare={handleAddToCompare}
                  selectedDate={selectedDate}
                  latestDate={latestDate}
                />
              )}

              {activeTab === 'brokerage' && <BrokerageView />}

              {activeTab === 'stock_terminal' && (
                <StockTerminalView
                  currentTicker={selectedTicker}
                  onSelectTicker={setSelectedTicker}
                  selectedDate={selectedDate}
                />
              )}

              {activeTab === 'research_reader' && (
                <ResearchReaderView
                  selectedDate={selectedDate}
                  t052Data={t052Data}
                  booksOverview={booksOverview}
                  experiments={experiments}
                  selectedExpId={selectedExpId}
                  onSelectExpId={setSelectedExpId}
                />
              )}

              {activeTab === 'runbook' && (
                <RunbookView
                  selectedDate={selectedDate}
                  onNavigateTab={setActiveTab}
                  onSelectTicker={handleInspectTicker}
                />
              )}
            </ErrorBoundary>
          </div>
        </main>
      </div>

      {/* 4. Global Slide-Over Quick Stock Drawer */}
      <QuickStockDrawer
        ticker={quickDrawerTicker}
        selectedDate={selectedDate}
        onClose={() => setQuickDrawerTicker(null)}
        onOpenFullTerminal={handleOpenFullTerminal}
        onAddToCompare={handleAddToCompare}
      />

      {/* 5. 3-Minute Daily Ritual Wizard Modal */}
      <DailyRitualModal
        isOpen={isDailyRitualOpen}
        onClose={() => setIsDailyRitualOpen(false)}
        selectedDate={selectedDate}
        health={health}
        formal={formal}
        brain={brain}
        market={market}
        onSelectTicker={(t) => {
          setIsDailyRitualOpen(false);
          handleInspectTicker(t);
        }}
        onNavigateTab={(tab) => {
          setIsDailyRitualOpen(false);
          setActiveTab(tab);
        }}
        onOpenBriefingExporter={() => setIsBriefingExporterOpen(true)}
      />

      {/* 6. Stock A vs Stock B Side-by-Side Arena */}
      <StockCompareModal
        isOpen={isCompareModalOpen}
        onClose={() => setIsCompareModalOpen(false)}
        tickerA={comparePair[0] || '688383.SH'}
        tickerB={comparePair[1] || ''}
        selectedDate={selectedDate}
        onSelectTicker={handleInspectTicker}
      />

      {/* 7. One-Click Daily Briefing Exporter */}
      <DailyBriefingExporter
        isOpen={isBriefingExporterOpen}
        onClose={() => setIsBriefingExporterOpen(false)}
        selectedDate={selectedDate}
        formal={formal}
        brain={brain}
        market={market}
        health={health}
      />

      {/* 8. Interactive Showcase / Defense Tour */}
      <ShowcaseTourModal
        isOpen={isShowcaseTourOpen}
        onClose={() => setIsShowcaseTourOpen(false)}
        onNavigateTab={setActiveTab}
      />

      {/* 9. Global Command Palette (Ctrl+K) */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        catalog={catalog?.entries || []}
        experiments={experiments}
        onSelectTicker={handleOpenFullTerminal}
        onSelectDataset={(id) => {
          setSelectedDatasetId(id);
          setActiveTab('data_observatory');
        }}
        onSelectExperiment={(id) => {
          setSelectedExpId(id);
          setActiveTab('research_reader');
        }}
        onNavigateTab={setActiveTab}
      />

      {/* 10. Floating Compare PK Dock (Bottom-Right) */}
      {comparePair.length > 0 && (
        <div className="fixed bottom-5 right-5 z-40 bg-gradient-to-r from-purple-950 via-slate-900 to-indigo-950 border border-purple-600/80 rounded-2xl shadow-2xl p-2.5 px-3.5 flex items-center gap-3 animate-in fade-in slide-in-from-bottom-2">
          <div className="flex items-center gap-2 text-xs font-mono">
            <Swords className="w-4 h-4 text-purple-300" />
            <span className="text-white font-bold hidden sm:inline">PK 对决:</span>
            <span className="px-2 py-0.5 rounded bg-black/50 text-cyan-300 border border-cyan-800">
              {comparePair[0]}
            </span>
            {comparePair[1] ? (
              <>
                <span className="text-slate-400">vs</span>
                <span className="px-2 py-0.5 rounded bg-black/50 text-purple-300 border border-purple-800">
                  {comparePair[1]}
                </span>
              </>
            ) : (
              <span className="text-slate-500 text-[10px] hidden md:inline">(点击标的"加入PK")</span>
            )}
          </div>
          <button
            onClick={() => setIsCompareModalOpen(true)}
            className="px-3 py-1 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white text-xs font-semibold shadow transition"
          >
            开启对决
          </button>
          <button
            onClick={() => setComparePair([])}
            className="text-slate-500 hover:text-white p-0.5 transition"
            title="清空对比栏"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
};
