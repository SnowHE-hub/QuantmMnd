import React, { useState } from 'react';
import { Stethoscope, AlertCircle, ArrowRight, Play, CheckCircle2, X } from 'lucide-react';
import { CatalogData, HealthInfo } from '../../api/types';

interface HealthDoctorBannerProps {
  catalog: CatalogData | null;
  health: HealthInfo | null;
  selectedDate: string;
  onNavigateTab: (tab: any) => void;
  onExecuteCommand?: (cmd: string) => void;
}

export const HealthDoctorBanner: React.FC<HealthDoctorBannerProps> = ({
  catalog,
  health,
  selectedDate,
  onNavigateTab,
  onExecuteCommand,
}) => {
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const entries = catalog?.entries || [];
  const latestRun = health?.latest_run || selectedDate;

  // Find entries that might lag behind latestRun
  const staleEntries = entries.filter((e) => {
    if (!e.coverage || !e.coverage.max_date) return false;
    const maxD = String(e.coverage.max_date).slice(0, 10);
    return maxD < latestRun && !e.id.includes('gold') && !e.id.includes('legacy');
  });

  if (staleEntries.length === 0) {
    return null;
  }

  const sampleStale = staleEntries[0];
  const suggestedCmd = `python scripts/download_daily.py --date ${latestRun} --dataset ${sampleStale.id}`;

  return (
    <>
      {/* Subtle Doctor Diagnostic Badge in Bar */}
      <div className="bg-amber-950/40 border-y border-amber-800/40 px-4 py-1.5 flex items-center justify-between text-xs animate-in fade-in">
        <div className="flex items-center gap-2 text-amber-300">
          <Stethoscope className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span>
            <strong className="font-semibold">数据资产健康巡检:</strong> 发现 {staleEntries.length} 个数据集（如 {sampleStale.title || sampleStale.id}）可能滞后于最新交易日 ({latestRun})。
          </span>
        </div>
        <button
          onClick={() => setIsDrawerOpen(true)}
          className="px-2.5 py-0.5 rounded-md bg-amber-900/60 hover:bg-amber-800 text-amber-200 border border-amber-700/60 font-medium text-[11px] flex items-center gap-1 transition"
        >
          <span>查看诊断与一键修复</span>
          <ArrowRight className="w-3 h-3" />
        </button>
      </div>

      {/* Diagnostic & One-Click Fix Modal */}
      {isDrawerOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs animate-in fade-in duration-150">
          <div className="w-full max-w-xl bg-[#0c1220] border border-amber-800/70 rounded-2xl shadow-2xl p-5 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
                <Stethoscope className="w-4 h-4" />
                <span>数据健康医生 (Proactive Health Doctor)</span>
              </div>
              <button
                onClick={() => setIsDrawerOpen(false)}
                className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs text-slate-300">
              <p>
                系统在比对本地 44 个数据集目录时，检测到以下数据源的最新记录早于最新交易截面：
              </p>

              <div className="space-y-2 max-h-48 overflow-y-auto pr-1 font-mono text-[11px]">
                {staleEntries.slice(0, 4).map((entry) => (
                  <div
                    key={entry.id}
                    className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between"
                  >
                    <div>
                      <div className="font-bold text-white">{entry.title || entry.id}</div>
                      <div className="text-[10px] text-slate-400">路径: {entry.path}</div>
                    </div>
                    <div className="text-right">
                      <span className="text-amber-400">
                        最新: {entry.coverage?.max_date || '未记录'}
                      </span>
                      <div className="text-[10px] text-slate-500">目标: {latestRun}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* One-Click Command Generator */}
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2 font-mono">
                <div className="text-slate-400 text-[11px] font-sans font-semibold flex items-center justify-between">
                  <span>推荐更新启动命令 (WSL CLI):</span>
                  <span className="text-cyan-400 text-[10px]">一键直达运行</span>
                </div>
                <div className="p-2 rounded bg-[#070b14] border border-slate-800 text-cyan-300 text-[11px] overflow-x-auto">
                  $ {suggestedCmd}
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-slate-800">
              <span className="text-[11px] text-slate-500">
                支持在 Runbook 中直接实时流式执行
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    setIsDrawerOpen(false);
                    onNavigateTab('runbook');
                  }}
                  className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium flex items-center gap-1.5 transition"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>前往 Runbook 执行中心</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
