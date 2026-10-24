import React, { useState, useEffect } from 'react';
import {
  X,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Layers,
  Database,
  FileCheck2,
  TrendingUp,
  ShieldCheck,
  ArrowRight,
  ExternalLink,
} from 'lucide-react';

interface ShowcaseTourModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigateTab: (tab: any) => void;
}

export const ShowcaseTourModal: React.FC<ShowcaseTourModalProps> = ({
  isOpen,
  onClose,
  onNavigateTab,
}) => {
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    {
      id: 'topology',
      tab: 'topology',
      tag: '核心架构创新',
      title: '1. 10 阶段选股漏斗与 6-Agent 多智能体协同沙盘',
      subtitle: 'System Topology & Multi-Agent Collaboration',
      icon: Layers,
      color: 'from-blue-600 to-cyan-600',
      description:
        'QuantMind 将顶尖量化机构的复杂投研流水线抽象为全自动的 10 阶段漏斗：从全市场 5000+ 标的多模态数据接入，经过 Factor Panel v3 多维因子计算，到达 6-Agent 智能体协同系统（A0 大脑主管、A1 财务、A2 公告、A3 治理、A4 资金流、A6 审计），实现从数据到信号的工业级自闭环流转。',
      highlights: [
        '每个子系统均自带职责契约与调用说明书，彻底终结记忆命令历史',
        '真实展示收盘后流水线耗时时序（Pipeline Timings）',
        '支持直观交互式 DAG 架构全景下钻',
      ],
      actionLabel: '前往体验：系统架构全景沙盘 →',
    },
    {
      id: 'data',
      tab: 'data_observatory',
      tag: '数据资产壁垒',
      title: '2. 44 个本地量化数据集与 78.1 GB 全血缘资产全景',
      subtitle: 'Data Lineage, Storage & Schema Observatory',
      icon: Database,
      color: 'from-cyan-600 to-teal-600',
      description:
        '拒绝任何 Mock 假数据。系统全量纳管并透视本地 44 个真实数据集、424,969 个物理文件、共计 78.1 GB 量化资产。从原始行情（Raw）、特征因子（Factors）、模型打分（Scores）到回测与账本结果（Outputs），端到端数据血缘 DAG 清晰可溯。',
      highlights: [
        '基于 ECharts 的磁盘空间占用树状图（Treemap），点击任意色块直达字段 Schema',
        '字段级 Schema 探测器：支持透视数据类型、行列数及时间跨度',
        '44 个数据集结构化网格，支持按层级与关键字极速检索',
      ],
      actionLabel: '前往体验：数据资产全景图 →',
    },
    {
      id: 'terminal',
      tab: 'stock_terminal',
      tag: '防幻觉独创机制',
      title: '3. F7 独创代码级防幻觉核验机制 (Ground Truth AST)',
      subtitle: 'Zero-Hallucination Code & Data Verification',
      icon: FileCheck2,
      color: 'from-emerald-600 to-green-600',
      description:
        'LLM 在量化领域最大的痛点是数字幻觉。QuantMind 业内独创 F7 代码级核验底座：智能体研报中的每一句话被分解为最小事实声明，后台自动通过 Python 抽象语法树（AST）重算公式，逐条核对数据库真实底稿，输出 verified、minor_mismatch、mismatch 严苛徽标。',
      highlights: [
        '每只研究股票的事实与推断声明逐条经代码核验（数量随当天研究而变，见拓扑页的 F7）',
        '独创“双栏溯源对照视图”：声明自然语言 vs Python 底层运行事实零距离比对',
        'F5c 风险执行卡雷达：多维穿透 Beta、年化波动、HAR 预测与时点公告链',
      ],
      actionLabel: '前往体验：个股全景深度透视终端 →',
    },
    {
      id: 'research',
      tab: 'research_reader',
      tag: '研究成果实证',
      title: '4. T-052 模拟盘多网格净值与 53 张实验任务卡',
      subtitle: 'Rigorous Empirical Backtest & Shadow Books',
      icon: TrendingUp,
      color: 'from-purple-600 to-pink-600',
      description:
        '投研成果不仅是代码，更是可复现的实证档案。系统完整纳管了 T-052 模拟盘网格的净值序列，支持多组参数与基准动态对比；同时收录了 T-001 至 T-055 共 53 张量化实验任务卡与前瞻影子账本容量评测，支持出版级 Markdown 沉浸式精读。',
      highlights: [
        '动态 ECharts 净值曲线与最大回撤指标透视',
        '前瞻影子账本多臂策略（参考臂 vs 挑战臂）容量与收益追踪',
        '53 张任务卡元数据与全文报告检索',
      ],
      actionLabel: '前往体验：投研成果精读中心 →',
    },
    {
      id: 'cockpit',
      tab: 'cockpit',
      tag: '实盘级风控',
      title: '5. 三档实盘资金约束与盘前红队防呆沙盒',
      subtitle: 'Multi-Tier Execution & Red Team What-If Sandbox',
      icon: ShieldCheck,
      color: 'from-rose-600 to-orange-600',
      description:
        '量化系统不能停留在理论，必须面对真实实盘摩擦。系统内置 10万 / 100万 / 1000万 三档资金约束，严谨执行一手（100股）凑整与零股自动留存现金；同时提供“反事实推演沙盒”与“双股同屏 PK 竞技场”，让交易员能在开盘前直观微调仓位并实时观察组合健康度。',
      highlights: [
        '盘前红队防呆排雷：自动拦截 avoid_new_buy 标的与异常波动票',
        '反事实推演沙盒：勾选剔除委托实时重算现金比例与组合 Beta',
        '“事后诸葛亮”验真引擎：时间机器切换历史截面自动核算后续真实盈亏与排雷命中率',
      ],
      actionLabel: '前往体验：核心投研工作台 (Cockpit) →',
    },
  ];

  const current = steps[currentStep];

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;
      if (e.key === 'ArrowRight' && currentStep < steps.length - 1) {
        setCurrentStep((prev) => prev + 1);
      } else if (e.key === 'ArrowLeft' && currentStep > 0) {
        setCurrentStep((prev) => prev - 1);
      } else if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, currentStep, steps.length, onClose]);

  if (!isOpen) return null;

  const IconComp = current.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-in fade-in duration-200">
      <div className="w-full max-w-3xl bg-[#0c1220] border border-slate-700 rounded-3xl shadow-2xl overflow-hidden flex flex-col">
        {/* Top Header */}
        <div className="p-4 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-amber-500 to-purple-600 flex items-center justify-center text-white shadow-md">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-bold text-white">QuantMind 全景高光巡航路演模式</h3>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-950 text-purple-300 border border-purple-800/60 font-mono font-bold">
                  Step {currentStep + 1} / {steps.length}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                3 分钟通盘领略系统 5 大核心架构突破与学术工程壁垒
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Step Visual Banner */}
        <div className={`p-6 bg-gradient-to-br ${current.color} text-white space-y-3 relative overflow-hidden`}>
          <div className="absolute right-4 -bottom-6 opacity-15 pointer-events-none">
            <IconComp className="w-48 h-48" />
          </div>

          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-black/30 backdrop-blur-md text-[11px] font-bold border border-white/20">
            <Sparkles className="w-3 h-3 text-amber-300" />
            <span>{current.tag}</span>
          </div>

          <div className="relative z-10 space-y-1">
            <h2 className="text-lg font-bold leading-tight">{current.title}</h2>
            <p className="text-xs text-white/80 font-mono">{current.subtitle}</p>
          </div>
        </div>

        {/* Main Body */}
        <div className="p-6 space-y-5 text-xs text-slate-200">
          <p className="text-sm leading-relaxed text-slate-300 font-normal">
            {current.description}
          </p>

          <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2.5">
            <div className="font-semibold text-white text-xs flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span>本模块技术亮点与工程突破：</span>
            </div>
            <ul className="space-y-2 text-slate-300 text-xs">
              {current.highlights.map((h, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-cyan-400 font-bold shrink-0 mt-0.5">•</span>
                  <span>{h}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between">
          {/* Direct jump button */}
          <button
            onClick={() => {
              onNavigateTab(current.tab);
              onClose();
            }}
            className="px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-cyan-300 hover:text-cyan-200 border border-cyan-800/80 text-xs font-semibold flex items-center gap-1.5 transition"
          >
            <span>{current.actionLabel}</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </button>

          {/* Stepper Controls */}
          <div className="flex items-center gap-2">
            <button
              disabled={currentStep === 0}
              onClick={() => setCurrentStep((prev) => prev - 1)}
              className="p-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed border border-slate-800 transition"
              title="上一页 (←)"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>

            {/* Dots */}
            <div className="flex items-center gap-1.5 px-2">
              {steps.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setCurrentStep(i)}
                  className={`h-2 rounded-full transition-all ${
                    i === currentStep ? 'w-6 bg-cyan-400' : 'w-2 bg-slate-700 hover:bg-slate-500'
                  }`}
                />
              ))}
            </div>

            <button
              disabled={currentStep === steps.length - 1}
              onClick={() => setCurrentStep((prev) => prev + 1)}
              className="p-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed transition shadow-md"
              title="下一页 (→)"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
