import React, { useState } from 'react';
import { FormalReportReader } from './FormalReportReader';
import { BacktestT052 } from './BacktestT052';
import { ShadowBooks } from './ShadowBooks';
import { TaskCardsExplorer } from './TaskCardsExplorer';
import {
  BacktestT052Response,
  BooksOverview,
  ExperimentSummary,
} from '../../api/types';

interface ResearchReaderViewProps {
  selectedDate: string;
  t052Data: BacktestT052Response | null;
  booksOverview: BooksOverview | null;
  experiments: ExperimentSummary[];
  selectedExpId?: string;
  onSelectExpId?: (id: string) => void;
}

export const ResearchReaderView: React.FC<ResearchReaderViewProps> = ({
  selectedDate,
  t052Data,
  booksOverview,
  experiments,
  selectedExpId,
  onSelectExpId,
}) => {
  const [subTab, setSubTab] = useState<'formal' | 't052' | 'books' | 'tasks'>('formal');

  return (
    <div className="space-y-4">
      {/* Sub Tabs */}
      <div className="flex items-center gap-1.5 bg-[#0c1220] p-1.5 rounded-xl border border-slate-800 text-xs">
        <button
          onClick={() => setSubTab('formal')}
          className={`px-3 py-1.5 rounded-lg font-medium transition ${
            subTab === 'formal'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          正式荐股日报精读
        </button>
        <button
          onClick={() => setSubTab('t052')}
          className={`px-3 py-1.5 rounded-lg font-medium transition ${
            subTab === 't052'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          T-052 模拟盘网格成果
        </button>
        <button
          onClick={() => setSubTab('books')}
          className={`px-3 py-1.5 rounded-lg font-medium transition ${
            subTab === 'books'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          前瞻影子账本容量评测
        </button>
        <button
          onClick={() => setSubTab('tasks')}
          className={`px-3 py-1.5 rounded-lg font-medium transition ${
            subTab === 'tasks'
              ? 'bg-blue-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          53 张实验任务卡全记录
        </button>
      </div>

      {/* Active SubTab Component */}
      {subTab === 'formal' && <FormalReportReader selectedDate={selectedDate} />}
      {subTab === 't052' && <BacktestT052 t052Data={t052Data} />}
      {subTab === 'books' && (
        <ShadowBooks booksOverview={booksOverview} selectedDate={selectedDate} />
      )}
      {subTab === 'tasks' && (
        <TaskCardsExplorer
          experiments={experiments}
          selectedExpId={selectedExpId}
          onSelectExpId={onSelectExpId}
        />
      )}
    </div>
  );
};
