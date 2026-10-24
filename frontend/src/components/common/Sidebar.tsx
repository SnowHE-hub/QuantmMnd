import React from 'react';
import {
  Compass,
  Database,
  BarChart3,
  TrendingUp,
  BookOpen,
  PlayCircle,
  FileCheck2,
  Cpu,
  Layers,
  Wallet,
} from 'lucide-react';

export type TabKey =
  | 'topology'
  | 'data_observatory'
  | 'cockpit'
  | 'brokerage'
  | 'stock_terminal'
  | 'research_reader'
  | 'runbook';

interface SidebarProps {
  activeTab: TabKey;
  onTabChange: (tab: TabKey) => void;
  catalogCount: number;
  taskCardsCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  catalogCount,
  taskCardsCount,
}) => {
  const menuItems = [
    {
      key: 'topology' as TabKey,
      label: '全景沙盘与拓扑',
      sublabel: 'System Topology',
      icon: Compass,
      badge: '10-Stage DAG',
    },
    {
      key: 'data_observatory' as TabKey,
      label: '数据资产与血缘',
      sublabel: 'Data Lineage & Storage',
      icon: Database,
      badge: `${catalogCount} 个数据集`,
    },
    {
      key: 'cockpit' as TabKey,
      label: '核心投研工作台',
      sublabel: 'Formal Tiers & Radar',
      icon: BarChart3,
      badge: '3档资金穿透',
    },
    {
      key: 'brokerage' as TabKey,
      label: '模拟账户与交割单',
      sublabel: 'Simulated Brokerage',
      icon: Wallet,
      badge: '按股数逐笔记账',
    },
    {
      key: 'stock_terminal' as TabKey,
      label: '个股深度透视',
      sublabel: 'K-Line & Claim Inspector',
      icon: TrendingUp,
      badge: '代码逐条核验',
    },
    {
      key: 'research_reader' as TabKey,
      label: '成果与研报精读',
      sublabel: 'Simulations & Reports',
      icon: BookOpen,
      badge: `${taskCardsCount} 张任务卡`,
    },
    {
      key: 'runbook' as TabKey,
      label: '零代码操作中心',
      sublabel: 'Zero-CLI Playbooks',
      icon: PlayCircle,
      badge: '免记命令行',
    },
  ];

  return (
    <aside className="w-64 border-r border-slate-800 bg-[#070b14] flex flex-col justify-between shrink-0 select-none">
      <div className="p-3 space-y-1">
        <div className="px-3 py-2 text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
          量化指挥作战版面
        </div>

        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.key;
          return (
            <button
              key={item.key}
              data-testid={`nav-${item.key}`}
              onClick={() => onTabChange(item.key)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-left transition-all ${
                isActive
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-500/40 shadow-sm shadow-blue-500/10 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon className={`w-4 h-4 ${isActive ? 'text-blue-400' : 'text-slate-500'}`} />
                <div>
                  <div className="text-xs font-medium text-slate-200">{item.label}</div>
                  <div className="text-[10px] text-slate-500 font-mono">{item.sublabel}</div>
                </div>
              </div>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                  isActive
                    ? 'bg-blue-500/30 text-blue-200'
                    : 'bg-slate-800/80 text-slate-400'
                }`}
              >
                {item.badge}
              </span>
            </button>
          );
        })}
      </div>

      {/* Bottom Subsystem Status summary */}
      <div className="p-3 border-t border-slate-900 bg-slate-950/60">
        <div className="rounded-lg p-2.5 bg-slate-900/60 border border-slate-800 text-[11px] space-y-2">
          <div className="flex items-center justify-between text-slate-400 font-medium">
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>智能体协同</span>
            </span>
            <span className="text-cyan-400 font-mono">6 Agents</span>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-blue-400" />
              <span>模型集成</span>
            </span>
            <span className="text-blue-400 font-mono">4 族 ENS (20d/63d)</span>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span className="flex items-center gap-1.5">
              <FileCheck2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>真实连通率</span>
            </span>
            <span className="text-emerald-400 font-mono">19 / 19 (100%)</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
