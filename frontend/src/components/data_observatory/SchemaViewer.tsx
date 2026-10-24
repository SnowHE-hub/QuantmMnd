import React from 'react';
import {
  Database,
  FileCode,
  Calendar,
  Layers,
  ArrowRight,
  HardDrive,
  ShieldCheck,
  RotateCcw,
  Tag,
} from 'lucide-react';
import { CatalogEntry } from '../../api/types';

interface SchemaViewerProps {
  entry: CatalogEntry | null;
}

export const SchemaViewer: React.FC<SchemaViewerProps> = ({ entry }) => {
  if (!entry) {
    return (
      <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-8 text-center text-slate-500 text-xs">
        请在上方树状图或下方列表中选择任意数据集，查看其完整 Schema 与字段透视
      </div>
    );
  }

  const sizeMb = (entry.size_bytes / (1024 * 1024)).toFixed(1);
  const sizeGb = (entry.size_bytes / (1024 * 1024 * 1024)).toFixed(2);
  const sizeDisplay = entry.size_bytes >= 1024 * 1024 * 1024 ? `${sizeGb} GB` : `${sizeMb} MB`;

  // Extract sampled patterns
  const patterns = entry.schema?.patterns || [];

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-900/60 text-blue-300 border border-blue-700/50">
              {entry.category}
            </span>
            <h3 className="text-base font-bold text-white">{entry.title}</h3>
          </div>
          <p className="text-xs font-mono text-cyan-300 mt-1">{entry.path}</p>
        </div>

        {/* Badges */}
        <div className="flex items-center gap-2">
          <div className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-right">
            <div className="text-[10px] text-slate-400 font-medium">磁盘体积</div>
            <div className="text-xs font-bold font-mono text-white">{sizeDisplay}</div>
          </div>
          <div className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-right">
            <div className="text-[10px] text-slate-400 font-medium">文件数量</div>
            <div className="text-xs font-bold font-mono text-white">
              {Number(entry.n_files).toLocaleString()} 个
            </div>
          </div>
        </div>
      </div>

      {/* Meta Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 my-4 text-xs">
        <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1.5">
            <Calendar className="w-3.5 h-3.5 text-cyan-400" />
            <span>覆盖时间区间</span>
          </div>
          <div className="font-mono text-slate-200">
            {entry.coverage?.min_date && entry.coverage?.max_date
              ? `${entry.coverage.min_date} → ${entry.coverage.max_date}`
              : '时点切片 / 连续追加'}
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>置信度 & 可再生性</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-emerald-300">{entry.confidence || 'confirmed'}</span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-300">
              可重算: {entry.regenerable === 'yes' ? '是' : '否 / 成本极高'}
            </span>
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <span>生产者 (Producer)</span>
          </div>
          <div className="text-slate-300 truncate" title={entry.producer}>
            {entry.producer || 'scripts/line_a/pipeline.py'}
          </div>
        </div>

        <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
          <div className="text-[11px] text-slate-400 font-medium mb-1 flex items-center gap-1.5">
            <ArrowRight className="w-3.5 h-3.5 text-purple-400" />
            <span>消费者 (Consumers)</span>
          </div>
          <div className="text-slate-300 truncate" title={entry.consumers}>
            {entry.consumers || 'quantmind/line_a/ 评分与研究'}
          </div>
        </div>
      </div>

      {/* Purpose Note */}
      <div className="p-3 rounded-lg bg-blue-950/20 border border-blue-900/40 text-xs text-slate-300 mb-4">
        <span className="font-semibold text-blue-300 mr-2">业务用途:</span>
        {entry.purpose}
      </div>

      {/* Patterns and Columns Schema Table */}
      {patterns.length > 0 ? (
        <div className="space-y-4">
          <div className="text-xs font-semibold text-white flex items-center gap-1.5">
            <FileCode className="w-4 h-4 text-cyan-400" />
            <span>抽样文件结构与字段清单 (Schema)</span>
          </div>

          {patterns.map((pat, idx) => (
            <div key={idx} className="border border-slate-800 rounded-lg overflow-hidden bg-slate-950/40">
              <div className="bg-slate-900/80 px-3 py-2 border-b border-slate-800 flex items-center justify-between text-xs">
                <span className="font-mono font-semibold text-cyan-300">
                  {pat.pattern}
                </span>
                <div className="flex items-center gap-3 text-[11px] text-slate-400 font-mono">
                  {pat.n_rows && <span>行数: {pat.n_rows.toLocaleString()}</span>}
                  {pat.n_cols && <span>列数: {pat.n_cols}</span>}
                  {pat.date_column && (
                    <span>
                      日期键: {pat.date_column} ({pat.date_min} ~ {pat.date_max})
                    </span>
                  )}
                </div>
              </div>

              {pat.columns && pat.columns.length > 0 ? (
                <div className="max-h-60 overflow-y-auto">
                  <table className="w-full text-left text-xs divide-y divide-slate-800/60">
                    <thead className="bg-slate-900/60 text-slate-400 sticky top-0">
                      <tr>
                        <th className="px-3 py-1.5 font-medium">序号</th>
                        <th className="px-3 py-1.5 font-medium">字段名称 (Column Name)</th>
                        <th className="px-3 py-1.5 font-medium">数据类型 (Data Type)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/40 font-mono">
                      {pat.columns.map((col, cIdx) => (
                        <tr key={cIdx} className="hover:bg-slate-900/40">
                          <td className="px-3 py-1 text-slate-500 text-[11px]">{cIdx + 1}</td>
                          <td className="px-3 py-1 text-slate-200 font-medium">{col.name}</td>
                          <td className="px-3 py-1 text-cyan-400 text-[11px]">{col.type}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-3 text-xs text-slate-500 font-mono">
                  抽样文件: {pat.sample_file}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-xs text-slate-500 italic p-3 border border-slate-800/60 rounded-lg">
          该数据集主要为文本或模型包目录，无单张规整表抽样，支持在日常任务中按需读取。
        </div>
      )}
    </div>
  );
};
