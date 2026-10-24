import React, { useState, useEffect } from 'react';
import { Search, X, Database, TrendingUp, BookOpen, Layers, ArrowRight } from 'lucide-react';
import { CatalogEntry, ExperimentSummary } from '../../api/types';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  catalog: CatalogEntry[];
  experiments: ExperimentSummary[];
  onSelectTicker: (ticker: string) => void;
  onSelectDataset: (id: string) => void;
  onSelectExperiment: (id: string) => void;
  onNavigateTab: (tab: any) => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  catalog,
  experiments,
  onSelectTicker,
  onSelectDataset,
  onSelectExperiment,
  onNavigateTab,
}) => {
  const [query, setQuery] = useState('');

  // Close on Escape or click backdrop
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (isOpen) onClose();
      }
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const q = query.trim().toLowerCase();

  // Filter datasets
  const matchedDatasets = catalog
    .filter(
      (d) =>
        d.title?.toLowerCase().includes(q) ||
        d.path?.toLowerCase().includes(q) ||
        d.id?.toLowerCase().includes(q) ||
        d.category?.toLowerCase().includes(q)
    )
    .slice(0, 5);

  // Filter experiments
  const matchedExperiments = experiments
    .filter(
      (e) =>
        e.id?.toLowerCase().includes(q) ||
        e.title?.toLowerCase().includes(q)
    )
    .slice(0, 5);

  // Common quick ticker recommendations
  const commonTickers = [
    { ticker: '688383.SH', name: '新益昌' },
    { ticker: '600000.SH', name: '浦发银行' },
    { ticker: '000001.SZ', name: '平安银行' },
    { ticker: '600519.SH', name: '贵州茅台' },
    { ticker: '300750.SZ', name: '宁德时代' },
    { ticker: '688041.SH', name: '海光信息' },
  ].filter(
    (t) =>
      t.ticker.toLowerCase().includes(q) ||
      t.name.toLowerCase().includes(q)
  );

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-2xl bg-[#0c1220] border border-slate-700/80 rounded-xl shadow-2xl overflow-hidden flex flex-col">
        {/* Search Input Bar */}
        <div className="flex items-center px-4 py-3 border-b border-slate-800 bg-slate-900/60">
          <Search className="w-4 h-4 text-slate-400 mr-2.5 shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索股票代码 (如 688383.SH)、数据集 (如 lake, panel)、任务卡 (如 T-052)..."
            className="w-full bg-transparent text-sm text-white placeholder-slate-500 focus:outline-none"
            autoFocus
          />
          <button
            onClick={onClose}
            className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div className="max-h-[60vh] overflow-y-auto p-3 space-y-4">
          {/* Quick Tickers */}
          <div>
            <div className="px-2 py-1 text-[11px] font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
              <TrendingUp className="w-3 h-3 text-cyan-400" />
              <span>个股快速穿透</span>
            </div>
            <div className="grid grid-cols-2 gap-1.5 mt-1">
              {commonTickers.map((t) => (
                <button
                  key={t.ticker}
                  onClick={() => {
                    onSelectTicker(t.ticker);
                    onNavigateTab('stock_terminal');
                    onClose();
                  }}
                  className="flex items-center justify-between p-2 rounded-lg bg-slate-900/50 hover:bg-blue-900/30 border border-slate-800/80 hover:border-blue-700/60 text-left transition"
                >
                  <div>
                    <span className="font-mono text-xs font-semibold text-white mr-2">
                      {t.ticker}
                    </span>
                    <span className="text-xs text-slate-400">{t.name}</span>
                  </div>
                  <ArrowRight className="w-3 h-3 text-slate-500" />
                </button>
              ))}
            </div>
          </div>

          {/* Datasets */}
          {matchedDatasets.length > 0 && (
            <div>
              <div className="px-2 py-1 text-[11px] font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                <Database className="w-3 h-3 text-blue-400" />
                <span>数据集 ({catalog.length} 个盘点资产)</span>
              </div>
              <div className="space-y-1 mt-1">
                {matchedDatasets.map((d) => (
                  <button
                    key={d.id}
                    onClick={() => {
                      onSelectDataset(d.id);
                      onNavigateTab('data_observatory');
                      onClose();
                    }}
                    className="w-full flex items-center justify-between p-2 rounded-lg hover:bg-slate-900 text-left transition"
                  >
                    <div>
                      <div className="text-xs font-medium text-slate-200">{d.title}</div>
                      <div className="text-[11px] text-slate-500 font-mono">
                        {d.path} · {(d.size_bytes / (1024 * 1024)).toFixed(1)} MB
                      </div>
                    </div>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                      {d.category}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Experiment Task Cards */}
          {matchedExperiments.length > 0 && (
            <div>
              <div className="px-2 py-1 text-[11px] font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
                <BookOpen className="w-3 h-3 text-emerald-400" />
                <span>实验任务卡 ({experiments.length} 张任务卡)</span>
              </div>
              <div className="space-y-1 mt-1">
                {matchedExperiments.map((e) => (
                  <button
                    key={e.id}
                    onClick={() => {
                      onSelectExperiment(e.id);
                      onNavigateTab('research_reader');
                      onClose();
                    }}
                    className="w-full flex items-center justify-between p-2 rounded-lg hover:bg-slate-900 text-left transition"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-bold text-emerald-400 bg-emerald-950/60 px-1.5 py-0.5 rounded border border-emerald-800/40">
                        {e.id}
                      </span>
                      <span className="text-xs text-slate-300 truncate max-w-md">{e.title}</span>
                    </div>
                    <span className="text-[10px] text-slate-500 font-mono">
                      {(e.bytes / 1024).toFixed(1)} KB
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer Shortcut Helper */}
        <div className="px-4 py-2 bg-slate-950 border-t border-slate-800 text-[11px] text-slate-500 flex items-center justify-between">
          <span>使用 ↑ ↓ 导航，Enter 确认，Esc 退出</span>
          <span className="font-mono">QuantMind 深度索引</span>
        </div>
      </div>
    </div>
  );
};
