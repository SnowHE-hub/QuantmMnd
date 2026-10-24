import React from 'react';
import { ArrowRight, Database, Cpu, GitCommit, FileSpreadsheet, Award } from 'lucide-react';

export const DataLineageGraph: React.FC = () => {
  const lineageSteps = [
    {
      level: '1. 原始与数据湖层 (Lake)',
      color: 'border-blue-500/40 bg-blue-950/20 text-blue-300',
      icon: Database,
      items: [
        { name: 'lake/daily.parquet', desc: '日线行情' },
        { name: 'lake/balancesheet.parquet', desc: '时点资产负债表' },
        { name: 'lake/income.parquet', desc: '时点利润表' },
        { name: 'lake/events/*.parquet', desc: '16类全市场事件' },
        { name: 'text/cninfo/*.jsonl', desc: '巨潮公告元数据' },
      ],
    },
    {
      level: '2. 特征工程与GPU面板 (Panels)',
      color: 'border-cyan-500/40 bg-cyan-950/20 text-cyan-300',
      icon: FileSpreadsheet,
      items: [
        { name: 'raw/alpha_prices_panel_v6', desc: '全时点复权价格宽表' },
        { name: 'panel/alpha_panel_weekly', desc: '周频宽表因子' },
        { name: 'gpu_v1/panel/*.parquet', desc: 'Alpha158日频量价' },
        { name: 'gpu_v1/seq/seq_raw_f32.npy', desc: '时序融合张量' },
      ],
    },
    {
      level: '3. 模型与预测打分 (Scoring)',
      color: 'border-purple-500/40 bg-purple-950/20 text-purple-300',
      icon: Cpu,
      items: [
        { name: 'f3_ens_20d_*.parquet', desc: '20日四族模型等权打分' },
        { name: 'f3_ens_63d_*.parquet', desc: '63日四族模型等权打分' },
        { name: 'f3_ridge_*.parquet', desc: '冻结 Ridge 基准打分' },
        { name: 'f4_candidates_*.parquet', desc: '多周期综合候选池 (~1900)' },
      ],
    },
    {
      level: '4. 证据研判与代码核对 (Agents & F7)',
      color: 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300',
      icon: GitCommit,
      items: [
        { name: 'f5c_risk_cards_*.parquet', desc: '960+只风险与执行卡' },
        { name: 'f6_cases.parquet', desc: '每日约 40 个研究案例（A1–A4、A6）' },
        { name: 'f7_claims.parquet', desc: '逐条声明的代码核验（数量见 /api/funnel）' },
        { name: 'brain_memo.json', desc: 'A0大脑风险标记与研判' },
      ],
    },
    {
      level: '5. 决策产出与模拟账本 (Outputs)',
      color: 'border-amber-500/40 bg-amber-950/20 text-amber-300',
      icon: Award,
      items: [
        { name: 'formal/report.json', desc: '10万/100万/1000万持仓与委托' },
        { name: 'books/<d>/*_nav.parquet', desc: '前瞻影子账本容量评测' },
        { name: 't052_sim/nav/*.parquet', desc: '4038组网格回测净值' },
      ],
    },
  ];

  return (
    <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-white">端到端数据血缘与加工流向图 (Data Lineage)</h3>
        <span className="text-xs font-mono text-slate-400">
          从原始湖数据到 3 档推荐账本的完整加工链
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {lineageSteps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div
              key={idx}
              className={`p-3 rounded-lg border flex flex-col justify-between ${step.color}`}
            >
              <div>
                <div className="flex items-center gap-1.5 mb-2 pb-1.5 border-b border-slate-800/80">
                  <Icon className="w-3.5 h-3.5" />
                  <span className="text-[11px] font-bold tracking-tight">{step.level}</span>
                </div>

                <div className="space-y-1.5">
                  {step.items.map((item, itemIdx) => (
                    <div
                      key={itemIdx}
                      className="p-1.5 rounded bg-slate-900/80 border border-slate-800/60"
                    >
                      <div className="font-mono text-[10px] text-white font-medium truncate" title={item.name}>
                        {item.name}
                      </div>
                      <div className="text-[10px] text-slate-400">{item.desc}</div>
                    </div>
                  ))}
                </div>
              </div>

              {idx < lineageSteps.length - 1 && (
                <div className="hidden md:flex justify-end pt-2 text-slate-600">
                  <ArrowRight className="w-4 h-4" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
