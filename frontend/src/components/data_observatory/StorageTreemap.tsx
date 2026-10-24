import React, { useRef } from 'react';
import { EChart as ReactECharts, EChartHandle } from '../common/EChart';
import { CatalogEntry } from '../../api/types';

interface StorageTreemapProps {
  entries: CatalogEntry[];
  onSelectEntry: (entry: CatalogEntry) => void;
  selectedEntryId?: string;
}

export const StorageTreemap: React.FC<StorageTreemapProps> = ({
  entries,
  onSelectEntry,
  selectedEntryId,
}) => {
  // Group entries by category
  const categoriesMap: Record<string, any[]> = {};
  entries.forEach((e) => {
    const cat = e.category || '其他';
    if (!categoriesMap[cat]) {
      categoriesMap[cat] = [];
    }
    categoriesMap[cat].push({
      name: e.title || e.id,
      value: e.size_bytes,
      id: e.id,
      n_files: e.n_files,
      path: e.path,
      rawEntry: e,
    });
  });

  const seriesData = Object.keys(categoriesMap).map((catName) => ({
    name: catName,
    children: categoriesMap[catName],
  }));

  const option = {
    tooltip: {
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: '#334155',
      textStyle: { color: '#f8fafc', fontSize: 12 },
      formatter: (info: any) => {
        const val = info.value;
        const sizeMb = (val / (1024 * 1024)).toFixed(1);
        const sizeGb = (val / (1024 * 1024 * 1024)).toFixed(2);
        const sizeStr = val >= 1024 * 1024 * 1024 ? `${sizeGb} GB` : `${sizeMb} MB`;
        const nFiles = info.data?.n_files || info.data?.rawEntry?.n_files || '-';
        const path = info.data?.path || info.data?.rawEntry?.path || '';

        return `
          <div style="font-weight: 600; margin-bottom: 4px; color: #38bdf8;">${info.name}</div>
          <div style="color: #94a3b8; font-size: 11px;">存储大小: <span style="color: #f1f5f9; font-weight: bold;">${sizeStr}</span></div>
          <div style="color: #94a3b8; font-size: 11px;">文件数量: <span style="color: #f1f5f9;">${Number(nFiles).toLocaleString()}</span></div>
          ${path ? `<div style="color: #64748b; font-size: 10px; margin-top: 4px; font-family: monospace;">${path}</div>` : ''}
        `;
      },
    },
    series: [
      {
        type: 'treemap',
        data: seriesData,
        leafDepth: 2,
        roam: false,
        nodeClick: false,
        label: {
          show: true,
          formatter: '{b}',
          fontSize: 11,
          color: '#ffffff',
          fontWeight: 500,
        },
        upperLabel: {
          show: true,
          height: 24,
          fontSize: 11,
          fontWeight: 'bold',
          color: '#93c5fd',
          backgroundColor: 'rgba(15, 23, 42, 0.7)',
        },
        itemStyle: {
          borderColor: '#0f172a',
          borderWidth: 1.5,
          gapWidth: 1.5,
        },
        levels: [
          {
            itemStyle: {
              borderColor: '#0b0f19',
              borderWidth: 2,
              gapWidth: 2,
            },
          },
          {
            color: [
              '#1e3a8a',
              '#0369a1',
              '#0f766e',
              '#4338ca',
              '#6d28d9',
              '#b45309',
            ],
            colorMappingBy: 'id',
            itemStyle: {
              borderColor: '#1e293b',
              borderWidth: 1,
              gapWidth: 1,
            },
          },
        ],
      },
    ],
  };

  const selectRef = useRef(onSelectEntry);
  selectRef.current = onSelectEntry;
  const onEvents = {
    click: (params: any) => {
      if (params.data && params.data.rawEntry) selectRef.current(params.data.rawEntry);
    },
  };

  return (
    <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-white">数据集磁盘空间树状图 (Treemap)</h3>
        <span className="text-xs font-mono text-slate-400">
          点击色块直接透视该数据集 Schema 与样本
        </span>
      </div>
      <div className="h-72 w-full">
        <ReactECharts
          option={option}
          onEvents={onEvents}
          style={{ height: '100%', width: '100%' }}
        />
      </div>
    </div>
  );
};
