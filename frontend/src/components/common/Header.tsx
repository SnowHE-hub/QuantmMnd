import React, { useEffect } from 'react';
import {
  Activity,
  Calendar,
  Search,
  RefreshCw,
  Database,
  Terminal,
  ChevronLeft,
  ChevronRight,
  Clock,
  Sparkles,
  ShieldCheck,
} from 'lucide-react';
import { HealthInfo, AvailableDates } from '../../api/types';

interface HeaderProps {
  health: HealthInfo | null;
  dates: AvailableDates | null;
  selectedDate: string;
  onSelectDate: (d: string) => void;
  onOpenCommandPalette: () => void;
  onRefresh: () => void;
  isLoading: boolean;
  onOpenDailyRitual: () => void;
  onOpenShowcaseTour?: () => void;
  onOpenBriefingExporter?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  dates,
  selectedDate,
  onSelectDate,
  onOpenCommandPalette,
  onRefresh,
  isLoading,
  onOpenDailyRitual,
  onOpenShowcaseTour,
  onOpenBriefingExporter,
}) => {
  const allDates = dates?.runs?.length ? dates.runs : selectedDate ? [selectedDate] : [];
  const currentIndex = allDates.indexOf(selectedDate);

  // Keyboard navigation for time machine: [ for prev day, ] for next day
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input or textarea
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }
      if (e.key === '[' && currentIndex > 0) {
        onSelectDate(allDates[currentIndex - 1]);
      } else if (e.key === ']' && currentIndex < allDates.length - 1 && currentIndex !== -1) {
        onSelectDate(allDates[currentIndex + 1]);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [allDates, currentIndex, onSelectDate]);

  const handlePrevDay = () => {
    if (currentIndex > 0) {
      onSelectDate(allDates[currentIndex - 1]);
    }
  };

  const handleNextDay = () => {
    if (currentIndex < allDates.length - 1 && currentIndex !== -1) {
      onSelectDate(allDates[currentIndex + 1]);
    }
  };

  // Check what products exist for this date
  const hasFormal = dates?.formal?.includes(selectedDate);
  const hasBooks = dates?.books?.includes(selectedDate);
  const hasResearch = dates?.research?.includes(selectedDate);

  return (
    <header className="h-14 border-b border-slate-800 bg-[#0c1220]/95 backdrop-blur sticky top-0 z-40 px-4 flex items-center justify-between">
      {/* Brand & System Title */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-cyan-500 shadow-md shadow-blue-500/20">
          <Terminal className="w-4 h-4 text-white" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="font-bold text-base tracking-wide text-white">QuantMind</span>
            <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-blue-900/60 text-blue-300 border border-blue-700/50">
              Observatory v2.0
            </span>
          </div>
          <p className="text-[11px] text-slate-400">AI Agent 多模态量化投资系统 · 全景指挥舱</p>
        </div>
      </div>

      {/* Middle: Global Time-Travel Machine */}
      <div className="flex items-center gap-2.5">
        {/* Time-Machine Container */}
        <div className="flex items-center gap-1 bg-slate-900/90 border border-slate-700/80 rounded-lg p-0.5 text-xs">
          {/* Prev Day Button */}
          <button
            onClick={handlePrevDay}
            disabled={currentIndex <= 0}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white disabled:opacity-30 disabled:hover:bg-transparent transition"
            title="前一交易日快捷键: ["
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          {/* Date Selector */}
          <div className="flex items-center gap-1.5 px-2">
            <Calendar className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-slate-400 text-[11px] hidden sm:inline">时间机器:</span>
            <select
              value={selectedDate}
              onChange={(e) => onSelectDate(e.target.value)}
              className="bg-transparent text-white font-mono font-bold focus:outline-none cursor-pointer"
            >
              {allDates.map((d) => (
                <option key={d} value={d} className="bg-slate-900 text-white font-mono">
                  {d} {d === health?.latest_run ? '(最新产出)' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Next Day Button */}
          <button
            onClick={handleNextDay}
            disabled={currentIndex >= allDates.length - 1 || currentIndex === -1}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white disabled:opacity-30 disabled:hover:bg-transparent transition"
            title="后一交易日快捷键: ]"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Date Coverage Badges */}
        <div className="hidden md:flex items-center gap-1 text-[10px] font-mono">
          <span
            className={`px-1.5 py-0.5 rounded border ${
              hasFormal
                ? 'bg-emerald-950 text-emerald-400 border-emerald-800/60'
                : 'bg-slate-900 text-slate-600 border-slate-800'
            }`}
            title={hasFormal ? '含正式荐股日报' : '无正式日报'}
          >
            正式
          </span>
          <span
            className={`px-1.5 py-0.5 rounded border ${
              hasResearch
                ? 'bg-purple-950 text-purple-300 border-purple-800/60'
                : 'bg-slate-900 text-slate-600 border-slate-800'
            }`}
            title={hasResearch ? '含 6-Agent 影子研究' : '无智能体研究'}
          >
            研究
          </span>
          <span
            className={`px-1.5 py-0.5 rounded border ${
              hasBooks
                ? 'bg-blue-950 text-blue-300 border-blue-800/60'
                : 'bg-slate-900 text-slate-600 border-slate-800'
            }`}
            title={hasBooks ? '含前瞻影子账本' : '无影子账本'}
          >
            账本
          </span>
        </div>

        {/* 3-Minute Daily Ritual Trigger Button */}
        <button
          onClick={onOpenDailyRitual}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-gradient-to-r from-purple-600/30 to-blue-600/30 hover:from-purple-600/40 hover:to-blue-600/40 border border-purple-500/40 text-purple-200 text-xs font-medium transition shadow-sm"
          title="今日复盘向导"
        >
          <Sparkles className="w-3.5 h-3.5 text-purple-300" />
          <span className="hidden xl:inline">3分钟复盘向导</span>
        </button>

        {/* Showcase Tour Button */}
        {onOpenShowcaseTour && (
          <button
            onClick={onOpenShowcaseTour}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-gradient-to-r from-amber-600/30 to-purple-600/30 hover:from-amber-600/40 hover:to-purple-600/40 border border-amber-500/40 text-amber-200 text-xs font-medium transition shadow-sm"
            title="系统全景高光巡航路演模式"
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-300" />
            <span className="hidden xl:inline">全景路演</span>
          </button>
        )}

        {/* Daily Briefing Exporter Button */}
        {onOpenBriefingExporter && (
          <button
            onClick={onOpenBriefingExporter}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs font-medium transition shadow-sm"
            title="导出今日投研风控执行单 (Markdown / PDF)"
          >
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden xl:inline">投研执行单</span>
          </button>
        )}

        {/* Command Palette Trigger */}
        <button
          onClick={onOpenCommandPalette}
          className="flex items-center gap-1.5 bg-slate-900/80 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200 rounded-lg px-2.5 py-1 text-xs transition"
          title="全局检索 (Ctrl+K)"
        >
          <Search className="w-3.5 h-3.5" />
          <kbd className="hidden sm:inline bg-slate-800 text-slate-300 border border-slate-700 px-1 rounded text-[10px] font-mono">
            Ctrl+K
          </kbd>
        </button>
      </div>

      {/* Right: Health, Data Root & Refresh */}
      <div className="flex items-center gap-3">
        {/* Status Indicator */}
        <div className="flex items-center gap-1.5 text-xs">
          <span className="relative flex h-2 w-2">
            <span
              className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                health?.ok ? 'bg-emerald-400' : 'bg-rose-400'
              }`}
            />
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                health?.ok ? 'bg-emerald-500' : 'bg-rose-500'
              }`}
            />
          </span>
          <span className={`font-medium ${health?.ok ? 'text-emerald-400' : 'text-rose-400'}`}>
            {health?.ok ? 'WSL 8780 在线' : '离线'}
          </span>
        </div>

        {/* Refresh Button */}
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition disabled:opacity-50"
          title="刷新数据"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-blue-400' : ''}`} />
        </button>
      </div>
    </header>
  );
};
