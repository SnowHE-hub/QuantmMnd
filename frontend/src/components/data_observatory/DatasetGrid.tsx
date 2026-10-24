import React, { useState } from 'react';
import { Search, Database, HardDrive, Filter, CheckCircle, ArrowRight } from 'lucide-react';
import { CatalogEntry } from '../../api/types';

interface DatasetGridProps {
  entries: CatalogEntry[];
  selectedEntryId?: string;
  onSelectEntry: (entry: CatalogEntry) => void;
}

export const DatasetGrid: React.FC<DatasetGridProps> = ({
  entries,
  selectedEntryId,
  onSelectEntry,
}) => {
  const [search, setSearch] = useState('');
  const [selectedCat, setSelectedCat] = useState<string>('ALL');

  const categories = ['ALL', ...Array.from(new Set(entries.map((e) => e.category).filter(Boolean)))];

  const filtered = entries.filter((e) => {
    const matchCat = selectedCat === 'ALL' || e.category === selectedCat;
    const matchSearch =
      search === '' ||
      e.title?.toLowerCase().includes(search.toLowerCase()) ||
      e.path?.toLowerCase().includes(search.toLowerCase()) ||
      e.purpose?.toLowerCase().includes(search.toLowerCase());
    return matchCat && matchSearch;
  });

  return (
    <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-4">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-white">44 个数据集全量资产目录与纳管清单</h3>
          <span className="text-xs font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
            共 {filtered.length} / {entries.length} 个
          </span>
        </div>

        {/* Search Input */}
        <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs w-full sm:w-64">
          <Search className="w-3.5 h-3.5 text-slate-400 mr-2 shrink-0" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索路径、表名、用途..."
            className="bg-transparent text-white placeholder-slate-500 focus:outline-none w-full"
          />
        </div>
      </div>

      {/* Category Pills */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCat(cat)}
            className={`text-xs px-2.5 py-1 rounded-lg transition font-medium ${
              selectedCat === cat
                ? 'bg-blue-600 text-white shadow-sm shadow-blue-500/20'
                : 'bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800'
            }`}
          >
            {cat === 'ALL' ? '全部类别' : cat}
          </button>
        ))}
      </div>

      {/* Dataset Grid Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto pr-1">
        {filtered.map((entry) => {
          const isSelected = selectedEntryId === entry.id;
          const sizeMb = (entry.size_bytes / (1024 * 1024)).toFixed(1);
          const sizeGb = (entry.size_bytes / (1024 * 1024 * 1024)).toFixed(2);
          const sizeDisplay =
            entry.size_bytes >= 1024 * 1024 * 1024 ? `${sizeGb} GB` : `${sizeMb} MB`;

          return (
            <div
              key={entry.id}
              onClick={() => onSelectEntry(entry)}
              className={`cursor-pointer p-3 rounded-lg border text-left transition-all ${
                isSelected
                  ? 'bg-blue-950/40 border-blue-500 shadow-md ring-1 ring-blue-500'
                  : 'bg-slate-900/50 hover:bg-slate-800/80 border-slate-800 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-400">
                  {entry.category}
                </span>
                <span className="text-xs font-mono font-bold text-cyan-300">
                  {sizeDisplay}
                </span>
              </div>

              <div className="text-xs font-semibold text-white truncate" title={entry.title}>
                {entry.title}
              </div>
              <div className="text-[10px] text-slate-400 font-mono truncate mt-0.5" title={entry.path}>
                {entry.path}
              </div>

              <div className="mt-2 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-500">
                <span>{Number(entry.n_files).toLocaleString()} 个文件</span>
                <div className="flex items-center gap-1 text-blue-400 group-hover:translate-x-0.5 transition">
                  <span className="text-[10px]">查看 Schema</span>
                  <ArrowRight className="w-3 h-3" />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
