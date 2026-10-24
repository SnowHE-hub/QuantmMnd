import React from 'react';
import { PlayCircle, Database, Search, Activity, BookOpen, Terminal, CheckCircle2 } from 'lucide-react';

interface PlaybookDeckProps {
  onNavigateTab: (tab: any) => void;
  onSelectTicker: (ticker: string) => void;
}

export const PlaybookDeck: React.FC<PlaybookDeckProps> = ({
  onNavigateTab,
  onSelectTicker,
}) => {
  const playbooks = [
    {
      title: '每日盘后全自动选股与日报查看',
      tagline: '查看今日推荐持仓、明日开盘委托与智能体大脑标记',
      steps: [
        '每个交易日北京时间约 21:30，系统自动执行 daily_auto.sh',
        '全流程耗时约 2 小时（数据+量化）与 1 小时（DeepSeek 智能体研究）',
        '在前端【核心投研工作台】直接查看三档纸面账户（10万/100万/1000万）',
        '在【明日开盘委托】表格中查看买卖指令与股数（买不起一手已自动标出）',
      ],
      actionLabel: '前往投研工作台',
      actionTab: 'cockpit',
      color: 'from-blue-600 to-cyan-600',
    },
    {
      title: '个股全景穿透与智能体事实核验',
      tagline: '查验某只股票为什么被推荐或被大脑打标记，核验数字真伪',
      steps: [
        '在任意界面按 Ctrl+K 或点击股票代码，跳转至【个股深度透视】',
        '查看历史 OHLCV K 线走势与成交量副图',
        '在【F7代码级逐条核验】透视镜中阅读 A1-A4 提交的事实、计算与推断声明',
        '核对彩色校验徽标（绿标 verified、红标 mismatch），并可反查底层证据编号',
      ],
      actionLabel: '穿透代表股票 (688383.SH)',
      actionTab: 'stock_terminal',
      ticker: '688383.SH',
      color: 'from-emerald-600 to-teal-600',
    },
    {
      title: '数据资产盘点与 Schema 查阅',
      tagline: '了解 44 个数据集占用多大磁盘、包含哪些字段、由谁生产',
      steps: [
        '点击左侧【数据资产与血缘】版面',
        '在 78.1 GB（72.8 GiB）存储树状图 (Treemap) 中直观感受各目录体量',
        '点击任意数据条目，展开 Schema 字段透视器（列名、数据类型、行数、日期范围）',
        '查看数据血缘流向图，了解原始湖表如何流转加工为特征与模型打分',
      ],
      actionLabel: '查看数据资产图',
      actionTab: 'data_observatory',
      color: 'from-purple-600 to-indigo-600',
    },
    {
      title: '回测成果精读与 53 张任务卡溯源',
      tagline: '查阅 4,038 组网格模拟盘表现与系统从 T-001 到 T-055 的演进判定',
      steps: [
        '点击左侧【成果与研报精读】版面',
        '在【T-052 模拟盘网格成果】中切换 W3/W6 调仓周期与规模净值曲线',
        '在【53 张实验任务卡全记录】中检索任意任务编号，阅读决策通过/采用判定',
        '在【正式荐股日报精读】中获取出版级排版的完整长篇研报',
      ],
      actionLabel: '精读实验与回测',
      actionTab: 'research_reader',
      color: 'from-amber-600 to-orange-600',
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <PlayCircle className="w-4 h-4 text-cyan-400" />
            场景化实操指南（Interactive Playbooks）
          </h3>
          <p className="text-xs text-slate-400">
            清晰说明“我想做什么、涉及哪些模块、界面在哪里看”，彻底消除记忆负担
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {playbooks.map((p, idx) => (
          <div
            key={idx}
            className="p-5 rounded-xl bg-[#0c1220]/90 border border-slate-800 shadow-lg flex flex-col justify-between space-y-4 group hover:border-slate-700 transition"
          >
            <div>
              <div className="flex items-center gap-2 mb-1.5">
                <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-slate-800 text-cyan-300">
                  场景 0{idx + 1}
                </span>
                <h4 className="text-sm font-bold text-white group-hover:text-cyan-300 transition">
                  {p.title}
                </h4>
              </div>
              <p className="text-xs text-slate-400 mb-3">{p.tagline}</p>

              <div className="space-y-2 text-xs text-slate-300 border-t border-slate-800/60 pt-3">
                {p.steps.map((step, sIdx) => (
                  <div key={sIdx} className="flex items-start gap-2">
                    <span className="text-slate-500 font-mono text-[11px] mt-0.5">
                      {sIdx + 1}.
                    </span>
                    <span className="leading-relaxed">{step}</span>
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={() => {
                if (p.ticker) onSelectTicker(p.ticker);
                onNavigateTab(p.actionTab);
              }}
              className="w-full py-2 px-3 rounded-lg bg-slate-900 hover:bg-blue-600 text-slate-300 hover:text-white border border-slate-800 hover:border-blue-500 font-medium text-xs transition text-center shadow-sm"
            >
              {p.actionLabel} →
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
