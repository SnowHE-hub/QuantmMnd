import React, { useEffect, useState } from 'react';
import { SystemDAG, STAGES, StageNode } from './SystemDAG';
import { SubsystemCard } from './SubsystemCard';
import { PipelineTimeline } from './PipelineTimeline';
import { analyticsApi, FunnelLayer } from '../../api/analytics';

const fmtDetail = (k: string, v: number | null) =>
  v === null ? '' : `${k} ${typeof v === 'number' && v > 0 && v < 1 ? (v * 100).toFixed(1) + '%' : v.toLocaleString()}`;

export const TopologyView: React.FC<{ selectedDate: string }> = ({ selectedDate }) => {
  const [selectedStage, setSelectedStage] = useState<StageNode>(STAGES[0]);
  const [layers, setLayers] = useState<FunnelLayer[]>([]);

  // 每一层当天的真实数量（原来写死在节点文案里，例如“约 4,800 只合规标的”“98 条声明”）
  useEffect(() => {
    if (!selectedDate) return;
    analyticsApi
      .funnel(selectedDate)
      .then((f) => setLayers(f.layers))
      .catch(() => setLayers([]));
  }, [selectedDate]);
  const counts = Object.fromEntries(layers.map((l) => [l.layer, l.n]));

  return (
    <div className="space-y-4">
      <SystemDAG counts={counts} selectedStageId={selectedStage.id} onSelectStage={(s) => setSelectedStage(s)} />
      {layers.length > 0 && (
        <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-4 text-xs" data-testid="funnel-table">
          <div className="font-semibold text-white mb-2">{selectedDate} 每一层的真实数量（/api/funnel）</div>
          <table className="w-full font-mono text-[11px]">
            <tbody>
              {layers.map((l) => (
                <tr key={l.layer} className="border-t border-slate-800">
                  <td className="py-1 text-cyan-300 w-14">{l.layer}</td>
                  <td className="text-slate-300 font-sans">{l.name}</td>
                  <td className="text-right text-white" data-testid={`funnel-${l.layer}`}>
                    {l.n.toLocaleString()}
                  </td>
                  <td className="pl-4 text-slate-500">
                    {Object.entries(l.detail)
                      .map(([k, v]) => fmtDetail(k, v))
                      .filter(Boolean)
                      .join(' · ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <SubsystemCard stage={selectedStage} />
      <PipelineTimeline />
    </div>
  );
};
