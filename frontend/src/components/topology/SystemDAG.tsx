import React from 'react';
import {
  Database,
  Gauge,
  ShieldCheck,
  Filter,
  AlertTriangle,
  Cpu,
  ShieldAlert,
  FileText,
  Users,
  CheckCircle2,
  BookMarked,
  Award,
  ArrowRight,
} from 'lucide-react';

export interface StageNode {
  id: string;
  stage: string;
  name: string;
  tagline: string;
  input: string;
  output: string;
  method: string;
  models: string;
  configPath: string;
  cliCommand: string;
  icon: any;
  color: string;
}

export const STAGES: StageNode[] = [
  {
    id: 'L0',
    stage: 'L0',
    name: '数据就绪与时点闸门',
    tagline: '增量更新 + 完整性检查 + PIT 时点对齐',
    input: 'Tushare / 巨潮公告爬虫',
    output: 'data/lake/*.parquet',
    method: '严格 PIT 窗口读取，迟到与更正公告重爬',
    models: '无（确定性数据工程）',
    configPath: 'scripts/data_pipeline/update_lake_latest.py',
    cliCommand: 'python scripts/data_pipeline/update_lake_latest.py --end $END',
    icon: Database,
    color: 'from-blue-600 to-cyan-600',
  },
  {
    id: 'M',
    stage: 'M',
    name: '市场状态与风险预算',
    tagline: '指数趋势、风格轮动、波动率与两融情绪',
    input: 'HS300 指数、北向资金、两融余额',
    output: 'm_market_*.json',
    method: '20日均线斜率、HAR波动率模型、多空扩散指标',
    models: 'HAR-RV / 趋势规则',
    configPath: 'quantmind/line_a/m_market.py',
    cliCommand: 'python -m quantmind.line_a.m_market --as-of $END',
    icon: Gauge,
    color: 'from-cyan-600 to-teal-600',
  },
  {
    id: 'L1',
    stage: 'L1',
    name: '可交易与合规筛查',
    tagline: '过滤 ST、停牌、上市未满期、退市警示',
    input: '全市场 A 股（含北交所）',
    output: 'f1_instruments（逐股 ST / 停牌 / 涨跌停 / 可新买标记）',
    method: '硬规则一票否决',
    models: '无（确定性合规过滤器）',
    configPath: 'quantmind/line_a/f1_instruments.py',
    cliCommand: 'python -m quantmind.line_a.f1_instruments --as-of $END',
    icon: ShieldCheck,
    color: 'from-teal-600 to-emerald-600',
  },
  {
    id: 'L2',
    stage: 'L2',
    name: '流动性与支持域',
    tagline: '研究池构建（时点成交额前 1500 强）',
    input: 'F1 证券表',
    output: 'f2_domains（研究池 = PIT 成交额前 1500，其余为扩展域）',
    method: '按时点过去 20 日成交额 (ADV20) 动态排序',
    models: '无（确定性容量边界）',
    configPath: 'quantmind/line_a/f2f4.py',
    cliCommand: 'python -m quantmind.line_a.f2f4 domains --as-of $END',
    icon: Filter,
    color: 'from-emerald-600 to-green-600',
  },
  {
    id: 'L3',
    stage: 'L3',
    name: '旁路监测与结构化排雷',
    tagline: '巨潮公告 MacBERT 分类、违规立案、清仓减持排雷',
    input: '全市场公告与结构化事件',
    output: 'bypass_events（major / watch）',
    method: '16类结构化事件规则 + 文本风险分类器',
    models: 'MacBERT 文本微调模型',
    configPath: 'quantmind/line_a/event_monitor.py',
    cliCommand: 'python -m quantmind.line_a.event_monitor --as-of $END',
    icon: AlertTriangle,
    color: 'from-amber-600 to-orange-600',
  },
  {
    id: 'L4',
    stage: 'L4',
    name: '量化多模型集成打分 (ENS)',
    tagline: '四族模型等权集成（20日/63日周期）',
    input: 'GPU 日频特征宽表 (Alpha158 + 财务 + 动量)',
    output: 'f3_ens_20d / 63d（全市场截面 z 分）',
    method: 'XGBoost + CatBoost + TabM + 序列融合，等权 ENS',
    models: '4族已冻结模型包 + Ridge C 锚点',
    configPath: 'data/gpu_v1/forward_models/',
    cliCommand: 'python -m quantmind.line_a.pipeline day --as-of $END',
    icon: Cpu,
    color: 'from-purple-600 to-indigo-600',
  },
  {
    id: 'L5',
    stage: 'L5',
    name: '风险与执行卡 (Risk Cards)',
    tagline: 'Beta252、波动率预测、跳空异动距离、除权标记',
    input: 'F4 研究候选与持仓',
    output: 'f5c_risk_cards（风险与执行卡）',
    method: 'HAR 次日波动预测、下行偏度、涨跌停统计',
    models: 'HAR-Vol 预测器',
    configPath: 'quantmind/line_a/risk_card.py',
    cliCommand: 'python -m quantmind.line_a.risk_card --as-of $END',
    icon: ShieldAlert,
    color: 'from-indigo-600 to-blue-700',
  },
  {
    id: 'L6',
    stage: 'L6',
    name: '多模态文本与证据提取',
    tagline: '财务三表、研报盈利预测、巨潮公告抽取证据链',
    input: '精选候选标的',
    output: '带证据编号的结构化切片 (Evidence IDs)',
    method: 'PIT 财务指标抽取、卖方预期追踪、事件状态机',
    models: '巨潮爬虫 + Tushare API',
    configPath: 'quantmind/line_a/evidence/',
    cliCommand: 'python -m quantmind.line_a.harness.tools_data --as-of $END',
    icon: FileText,
    color: 'from-blue-700 to-cyan-700',
  },
  {
    id: 'L7',
    stage: 'L7',
    name: '6-Agent 智能体协同研判',
    tagline: 'A1财务 / A2公告 / A3治理 / A4资金 / A6审查',
    input: 'F6 选出的研究对象（每日约 40 只）',
    output: 'f6_cases + f7_claims（结构化声明与 A6 审查）',
    method: 'Harness 运行循环，角色隔离上下文，专职深挖',
    models: 'DeepSeek V4.1 Flash / 本地 2B/9B',
    configPath: 'quantmind/line_a/harness/roles.py',
    cliCommand: 'python -m quantmind.line_a.f6_research --as-of $END',
    icon: Users,
    color: 'from-cyan-700 to-teal-700',
  },
  {
    id: 'L8',
    stage: 'L8',
    name: '代码级事实与计算核验 (F7)',
    tagline: '逐条核对数值、单位、公式与原文逐字引文',
    input: '各智能体提交的声明草案',
    output: '代码核验徽标 (verified / mismatch)',
    method: '数值相对误差计算、公式重算、逐字比对',
    models: '无（纯 Python 确定性比对器，防幻觉核心）',
    configPath: 'quantmind/line_a/harness/verifier.py',
    cliCommand: 'python -m quantmind.line_a.harness.verifier --as-of $END',
    icon: CheckCircle2,
    color: 'from-emerald-700 to-green-700',
  },
  {
    id: 'L9',
    stage: 'L9',
    name: '前瞻影子账本与容量评测',
    tagline: '参考臂 vs 挑战臂，多权限线净值跟踪',
    input: '打分与合规候选',
    output: 'books/<d>/*__nav.parquet',
    method: '周频网格轮换、持仓上限约束、交易摩擦模拟',
    models: 'SimAccount 模拟撮合引擎',
    configPath: 'quantmind/line_a/f8_books.py',
    cliCommand: 'python -m quantmind.line_a.pipeline books --as-of $END',
    icon: BookMarked,
    color: 'from-amber-700 to-orange-700',
  },
  {
    id: 'L10',
    stage: 'L10',
    name: '正式荐股日报 (A0 大脑)',
    tagline: '三档纸面账户（10万/100万/1000万）+ 开盘委托',
    input: '全部数据、影子账本与 A0 智能体大脑',
    output: 'formal/<d>/report.json & report.md',
    method: '按板块最小申报单位取整（科创板 200 股起、北交所 1 股递增），买不起的跳过；大脑只打标记，不改持仓',
    models: 'A0 Agent 大脑（编排层）',
    configPath: 'quantmind/line_a/formal.py',
    cliCommand: 'python -m quantmind.line_a.formal --as-of $END --with-brain',
    icon: Award,
    color: 'from-rose-600 to-red-600',
  },
];

// 节点 → /api/funnel 的层（当天的真实数量）
const LAYER_OF: Record<string, string> = { L1: 'F1', L2: 'F2', L3: '旁路', L4: 'F3', L5: 'F5', L7: 'F6', L8: 'F7', L10: '正式' };

interface SystemDAGProps {
  counts?: Record<string, number>;
  selectedStageId: string;
  onSelectStage: (stage: StageNode) => void;
}

export const SystemDAG: React.FC<SystemDAGProps> = ({
  counts = {},
  selectedStageId,
  onSelectStage,
}) => {
  return (
    <div className="bg-[#0c1220]/80 border border-slate-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
            10-Stage 选股漏斗与智能体协同拓扑图
          </h3>
          <p className="text-xs text-slate-400">
            点击任意节点，查看该模块的职责、输入输出契约、依赖配置与日常启动方式
          </p>
        </div>
        <span className="text-xs font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/40">
          端到端全量自解释
        </span>
      </div>

      {/* Interactive Horizontal Funnel DAG Flow */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2.5">
        {STAGES.map((s, idx) => {
          const Icon = s.icon;
          const isSelected = selectedStageId === s.id;
          return (
            <div
              key={s.id}
              onClick={() => onSelectStage(s)}
              className={`relative cursor-pointer p-3 rounded-lg border transition-all text-left group ${
                isSelected
                  ? 'bg-blue-950/40 border-blue-500 shadow-md shadow-blue-500/20 ring-1 ring-blue-500'
                  : 'bg-slate-900/60 hover:bg-slate-800/80 border-slate-800/90 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-mono text-[11px] font-bold px-1.5 py-0.2 rounded bg-slate-800 text-slate-300">
                  {s.stage}
                </span>
                <div
                  className={`w-6 h-6 rounded flex items-center justify-center bg-gradient-to-br ${s.color} text-white shadow-sm`}
                >
                  <Icon className="w-3.5 h-3.5" />
                </div>
              </div>

              <div className="text-xs font-semibold text-white group-hover:text-cyan-300 transition line-clamp-1">
                {s.name}
              </div>
              <div className="text-[10px] text-slate-400 mt-1 line-clamp-2 leading-tight">
                {s.tagline}
              </div>

              <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-500">
                <span className="font-mono truncate">{s.output.split('（')[0]}</span>
                {LAYER_OF[s.id] && counts[LAYER_OF[s.id]] !== undefined && (
                  <span className="font-mono text-cyan-300" data-testid={`count-${s.id}`}>{counts[LAYER_OF[s.id]].toLocaleString()}</span>
                )}
                {idx < STAGES.length - 1 && (
                  <ArrowRight className="w-3 h-3 text-slate-600 group-hover:text-slate-400" />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
