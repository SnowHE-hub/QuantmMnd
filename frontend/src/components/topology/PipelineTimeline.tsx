import React from 'react';
import { Clock, CheckCircle2, AlertCircle, PlayCircle, ShieldCheck } from 'lucide-react';

interface StepStatus {
  name: string;
  label: string;
  duration: string;
  status: 'done' | 'running' | 'soft_done' | 'pending';
  description: string;
}

export const PipelineTimeline: React.FC = () => {
  const steps: StepStatus[] = [
    {
      name: 'lake',
      label: '数据湖核心表增量更新',
      duration: '13分45秒',
      status: 'done',
      description: '拉取 Tushare 行情、两融、北向、每日指标并严格幂等合并入湖',
    },
    {
      name: 'events & cmp',
      label: '全市场16类事件表增量与压缩',
      duration: '4分06秒',
      status: 'done',
      description: '增量刷新停牌、涨跌停、业绩预告、分红、质押等全市场事件',
    },
    {
      name: 'fin_queues',
      label: '财务三表时点队列 (PIT)',
      duration: '56分59秒',
      status: 'done',
      description: '资产负债表、利润表、现金流量表按公告日严格对齐时点',
    },
    {
      name: 'cninfo & report_rc',
      label: '巨潮公告与卖方盈利预测',
      duration: '7分52秒',
      status: 'soft_done',
      description: '巨潮增量爬取最近4天公告，卖方预测窗口更新（软步骤）',
    },
    {
      name: 'snapshot & prices',
      label: '价格面板与 PIT 快照冻结',
      duration: '3分16秒',
      status: 'done',
      description: '构建 alpha_prices_panel_v6 原始价格面板与季度快照',
    },
    {
      name: 'fwd_root & builders',
      label: '影子数据根与 GPU 构建器',
      duration: '42分59秒',
      status: 'done',
      description: '生成 qm_fwd_root 前瞻目录，链接冻结月份特征',
    },
    {
      name: 'day_pipeline',
      label: '日任务运算 (M, F0–F5c, F9)',
      duration: '1分53秒',
      status: 'done',
      description: '市场状态评估、ENS 20d/63d 多模型打分、风险与执行卡生成',
    },
    {
      name: 'books',
      label: '前瞻影子账本与容量模拟',
      duration: '5分17秒',
      status: 'done',
      description: '参考臂 vs 挑战臂净值计算，多权限线组合容量推演',
    },
    {
      name: 'research',
      label: '6-Agent 影子研究与代码核验',
      duration: '58分33秒',
      status: 'soft_done',
      description: 'DeepSeek 调用 A1-A4 专职角色产出声明，A6 审查，F7 逐条核对',
    },
    {
      name: 'formal',
      label: '正式荐股日报与 Agent 大脑',
      duration: '1分15秒',
      status: 'done',
      description: '三档账户纸面成交模拟、明日开盘委托计算、A0 大脑风险标记',
    },
  ];

  return (
    <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4 text-cyan-400" />
          <h4 className="text-sm font-semibold text-white">收盘后全流程流水线（Daily Shadow Pipeline）时序追踪</h4>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <span className="flex items-center gap-1 text-emerald-400">
            <CheckCircle2 className="w-3.5 h-3.5" /> 核心步骤 ALL DONE
          </span>
          <span className="text-slate-600">|</span>
          <span>北京时间 21:30 自动启动</span>
        </div>
      </div>

      <div className="relative pl-6 border-l-2 border-slate-800 space-y-3.5 my-2">
        {steps.map((step, idx) => (
          <div key={idx} className="relative group">
            {/* Timeline bullet */}
            <span
              className={`absolute -left-[31px] top-1 w-3.5 h-3.5 rounded-full border-2 bg-slate-900 ${
                step.status === 'done'
                  ? 'border-emerald-500 bg-emerald-950 text-emerald-400'
                  : 'border-cyan-500 bg-cyan-950 text-cyan-400'
              }`}
            />

            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 font-medium text-slate-200">
                <span className="font-mono text-cyan-400">{step.name}</span>
                <span>{step.label}</span>
                {step.status === 'soft_done' && (
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/60 font-mono">
                    软步骤
                  </span>
                )}
              </div>
              <span className="font-mono text-[11px] text-slate-400">{step.duration}</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">{step.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
