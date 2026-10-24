import React from 'react';
import { FileText, Tag, Calendar, ExternalLink } from 'lucide-react';
import { AnnouncementItem } from '../../api/types';

interface AnnouncementsTimelineProps {
  announcements: AnnouncementItem[];
  ticker: string;
}

export const AnnouncementsTimeline: React.FC<AnnouncementsTimelineProps> = ({
  announcements,
  ticker,
}) => {
  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-3">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-bold text-white">时点公告与事件链 (MacBERT 智能分类)</h3>
        </div>
        <span className="text-xs font-mono text-slate-400">
          最近 90 日内共 {announcements.length} 篇公告
        </span>
      </div>

      <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
        {announcements.length > 0 ? (
          announcements.map((ann, idx) => (
            <div
              key={idx}
              className="p-2.5 rounded-lg bg-slate-900/50 hover:bg-slate-900 border border-slate-800/80 text-xs flex flex-col md:flex-row md:items-center justify-between gap-2 transition"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[11px] text-slate-400">
                    {ann.date || ann.ann_date || '—'}
                  </span>
                  {ann.type && (
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-blue-950 text-blue-300 border border-blue-800/50">
                      {ann.type}
                    </span>
                  )}
                  {ann.stage && (
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-300">
                      阶段: {ann.stage}
                    </span>
                  )}
                  {ann.importance && (
                    <span
                      className={`text-[10px] font-bold px-1.5 py-0.2 rounded uppercase ${
                        ann.importance === 'major'
                          ? 'bg-rose-950 text-rose-300 border border-rose-800/60'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {ann.importance}
                    </span>
                  )}
                </div>
                <div className="text-slate-200 font-medium text-xs line-clamp-2">
                  {ann.title}
                </div>
              </div>

              {ann.evidence_id && (
                <div className="shrink-0 text-right">
                  <span
                    className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-950 text-slate-500 border border-slate-800 max-w-[180px] inline-block truncate"
                    title={ann.evidence_id}
                  >
                    {ann.evidence_id}
                  </span>
                </div>
              )}
            </div>
          ))
        ) : (
          <div className="text-center py-6 text-slate-500 text-xs italic">
            90 日内未检索到该标的的重大公告披露
          </div>
        )}
      </div>
    </div>
  );
};
