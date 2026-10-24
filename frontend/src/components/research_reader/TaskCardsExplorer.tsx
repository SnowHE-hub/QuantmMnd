import React, { useState, useEffect } from 'react';
import { BookOpen, Search, CheckCircle2, ChevronRight, FileText, ArrowLeft } from 'lucide-react';
import { ExperimentSummary } from '../../api/types';
import { quantmindApi } from '../../api/client';
import { MarkdownViewer } from '../common/MarkdownViewer';

interface TaskCardsExplorerProps {
  experiments: ExperimentSummary[];
  selectedExpId?: string;
  onSelectExpId?: (id: string) => void;
}

export const TaskCardsExplorer: React.FC<TaskCardsExplorerProps> = ({
  experiments,
  selectedExpId,
  onSelectExpId,
}) => {
  const [activeId, setActiveId] = useState<string>(selectedExpId || (experiments[0]?.id || 'T-052'));
  const [activeContent, setActiveContent] = useState<string>('');
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedExpId) {
      setActiveId(selectedExpId);
    }
  }, [selectedExpId]);

  useEffect(() => {
    if (!activeId) return;
    setLoading(true);
    quantmindApi
      .getExperimentDetail(activeId)
      .then((txt) => setActiveContent(txt || ''))
      .catch((err) => setActiveContent(`加载任务卡 ${activeId} 失败: ${err.message}`))
      .finally(() => setLoading(false));
  }, [activeId]);

  const handleSelect = (id: string) => {
    setActiveId(id);
    if (onSelectExpId) onSelectExpId(id);
  };

  const filtered = experiments.filter(
    (e) =>
      e.id?.toLowerCase().includes(search.toLowerCase()) ||
      e.title?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Title */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white">
              53 张实验任务卡知识库（T-001 ~ T-055 研发历程全记录）
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            记录量化系统从最初原型一步步演进至今日的每个实验登记、对决产物与终审判定
          </p>
        </div>

        {/* Search */}
        <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs w-full sm:w-60">
          <Search className="w-3.5 h-3.5 text-slate-400 mr-2 shrink-0" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索任务卡 如 T-052..."
            className="bg-transparent text-white placeholder-slate-500 focus:outline-none w-full"
          />
        </div>
      </div>

      {/* Two Columns: Left List & Right Markdown Content */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Task Cards List */}
        <div className="lg:col-span-4 space-y-1.5 max-h-[600px] overflow-y-auto pr-1">
          {filtered.map((exp) => {
            const isSelected = activeId === exp.id;
            return (
              <div
                key={exp.id}
                onClick={() => handleSelect(exp.id)}
                className={`p-2.5 rounded-lg border cursor-pointer transition text-xs ${
                  isSelected
                    ? 'bg-blue-950/40 border-blue-500 shadow-sm'
                    : 'bg-slate-900/50 hover:bg-slate-900 border-slate-800'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span
                    className={`font-mono font-bold px-1.5 py-0.2 rounded ${
                      isSelected
                        ? 'bg-blue-600 text-white'
                        : 'bg-slate-800 text-cyan-300'
                    }`}
                  >
                    {exp.id}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {(exp.bytes / 1024).toFixed(1)} KB
                  </span>
                </div>
                <div className="font-medium text-slate-200 line-clamp-2">
                  {exp.title}
                </div>
                {exp.verdict_lines && exp.verdict_lines.length > 0 && (
                  <div className="mt-1 pt-1 border-t border-slate-800/60 text-[10px] text-emerald-400/90 truncate flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
                    <span className="truncate">{exp.verdict_lines[0]}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Right Markdown Full Document */}
        <div className="lg:col-span-8 bg-slate-950/60 border border-slate-800 rounded-xl p-5 max-h-[600px] overflow-y-auto">
          {loading ? (
            <div className="h-64 flex items-center justify-center text-xs text-blue-400 font-mono animate-pulse">
              加载任务卡全文中...
            </div>
          ) : (
            <MarkdownViewer content={activeContent} />
          )}
        </div>
      </div>
    </div>
  );
};
