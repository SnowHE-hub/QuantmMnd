import React, { useState } from 'react';
import {
  X,
  FileText,
  Copy,
  Download,
  Printer,
  Check,
  ShieldCheck,
  TrendingUp,
  DollarSign,
  AlertTriangle,
  Sparkles,
} from 'lucide-react';
import { FormalReport, BrainResponse, MarketState, HealthInfo } from '../../api/types';

interface DailyBriefingExporterProps {
  isOpen: boolean;
  onClose: () => void;
  selectedDate: string;
  formal: FormalReport | null;
  brain: BrainResponse | null;
  market: MarketState | null;
  health: HealthInfo | null;
}

// 年化波动的百分数写法；没有数据就写 —（原来缺数据时显示写死的 21.4% / 20.8%，而且读的字段名在后端根本不存在）
const sgn = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? `${v > 0 ? '+' : ''}${(v * 100).toFixed(1)}%` : '—');
const annPct = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—');

export const DailyBriefingExporter: React.FC<DailyBriefingExporterProps> = ({
  isOpen,
  onClose,
  selectedDate,
  formal,
  brain,
  market,
  health,
}) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const tier1e5 = formal?.tiers?.aum_1e5;
  const tier1e6 = formal?.tiers?.aum_1e6;
  const tier1e7 = formal?.tiers?.aum_1e7;

  const orders = tier1e5?.orders_next_open || [];
  const validOrders = orders.filter((o) => o.est_shares > 0);
  const skippedOrders = orders.filter((o) => o.est_shares === 0);

  const flags = brain?.memo?.flags || [];
  const requests = brain?.memo?.research_requests || [];

  // Generate publication-grade Markdown text
  const generateMarkdown = () => {
    return `# QuantMind 每日量化投研与风控执行单 (Daily Quant Execution Briefing)
**业务截面日期**: ${selectedDate}  
**生成系统**: QuantMind AI-Agent Multi-Modal Quant Engine (v3.5)  
**数据基准路径**: ${health?.data_root || '/home/lenovo/projects/quantmind'}  
**审定状态**: ${formal?.approval?.research_report_approved ? '✅ 已通过投研审查闸口' : '⚠️ 影子模拟状态'}  

---

## 1. 宏观与大盘微观状态 (Market Regime Barometer)
- **沪深 300 状态**: ${sgn((market?.trend as any)?.['沪深300']?.vs_ma20)} · 20日收益率: ${market?.trend?.['沪深300']?.ret_20d ? (market.trend['沪深300'].ret_20d * 100).toFixed(2) + '%' : '—'}
- **中证 2000 微盘风格**: 中证2000 相对沪深300 近 20 日 ${sgn((market?.style as any)?.csi2000_vs_hs300_rs20)}、近 60 日 ${sgn((market?.style as any)?.csi2000_vs_hs300_rs60)}
- **市场年化波动率**: ${annPct((market?.volatility_hs300 as any)?.realized_vol_20d_ann)}（近 20 日已实现）；HAR 次日预测 ${annPct((market?.volatility_hs300 as any)?.next_day_vol_ann)}
- **两融杠杆与情绪**: 两融余额 ${typeof (market?.margin as any)?.balance_yuan === 'number' ? ((market?.margin as any).balance_yuan / 1e12).toFixed(2) + ' 万亿元' : '—'}，近 20 日 ${sgn((market?.margin as any)?.chg_20d)}（截至 ${(market?.margin as any)?.last_date ?? '—'}）

---

## 2. 三档模拟实盘账户资产透视 (Portfolio Status)
| 资金档位 | 总资产 (Equity) | 现金留存 (Cash) | 现金比例 | 持仓标的数 | 板块约束 |
|:---|:---|:---|:---|:---|:---|
| **10 万元档** | ¥${Number(tier1e5?.equity || 100000).toLocaleString()} | ¥${Number(tier1e5?.cash || 100000).toLocaleString()} | ${tier1e5?.equity ? ((tier1e5.cash / tier1e5.equity) * 100).toFixed(1) : 100}% | ${tier1e5?.holdings?.length || 0} / ${tier1e5?.top_n || 10} 只 | ${Array.isArray(tier1e5?.boards) ? tier1e5.boards.join('+') : tier1e5?.boards || '主板+创业板'} |
| **100 万元档** | ¥${Number(tier1e6?.equity || 1000000).toLocaleString()} | ¥${Number(tier1e6?.cash || 1000000).toLocaleString()} | ${tier1e6?.equity ? ((tier1e6.cash / tier1e6.equity) * 100).toFixed(1) : 100}% | ${tier1e6?.holdings?.length || 0} / ${tier1e6?.top_n || 30} 只 | 全市场配置 |
| **1000 万元档** | ¥${Number(tier1e7?.equity || 10000000).toLocaleString()} | ¥${Number(tier1e7?.cash || 10000000).toLocaleString()} | ${tier1e7?.equity ? ((tier1e7.cash / tier1e7.equity) * 100).toFixed(1) : 100}% | ${tier1e7?.holdings?.length || 0} / ${tier1e7?.top_n || 100} 只 | 机构级分散 |

---

## 3. 明日开盘委托执行单 (Approved Orders Next Open)
> **执行规则**: 开盘集合竞价市价单挂入，执行滑点预估 0.1%，执行前核对资金与一手（100股）凑整约束。

${
  validOrders.length > 0
    ? `| 方向 | 代码 | 名称 | 板块 | 委托股数 | 参考开盘价 | 预估金额 | 目标打分排名 |
|:---|:---|:---|:---|:---|:---|:---|:---|
${validOrders
  .map(
    (o) =>
      `| **${o.side === 'buy' ? '买入' : '卖出'}** | \`${o.ticker}\` | ${o.name} | ${o.board} | ${o.est_shares} 股 | ¥${Number(o.ref_close).toFixed(2)} | ¥${Number(o.est_value).toLocaleString()} | #${o.rank_in_target} |`
  )
  .join('\n')}`
    : '_明日无有效买入/卖出委托。_'
}

${
  skippedOrders.length > 0
    ? `\n### ⚠️ 零股买不起跳过明细（实盘自动留存现金）:
${skippedOrders.map((o) => `- **${o.ticker} (${o.name})**: 拟配置金额 ¥${Number(o.est_value).toFixed(0)} 不足一手 (100股)，自动跳过。`).join('\n')}`
    : ''
}

---

## 4. A0 Agent 大脑与红队风控备忘 (Red Team Directives)
${
  flags.length > 0
    ? flags.map((f) => `- **[${f.action === 'avoid_new_buy' ? '不建议新买入' : '重点关注'}] \`${f.ticker}\`**: ${f.reason}`).join('\n')
    : '- 本期暂无特殊大脑红队警示标记。'
}

### 明日优先研判分配 (A1-A6 研究员指派):
${
  requests.length > 0
    ? requests.map((r) => `- **\`${r.ticker}\`**: 建议指派 ${r.roles?.join(', ')} 重点核验: ${r.reason}`).join('\n')
    : '- 暂无指派标的。'
}

---
*报告生成时间: ${new Date().toLocaleString()} · QuantMind Command Center*
`;
  };

  const mdContent = generateMarkdown();

  const handleCopy = () => {
    navigator.clipboard.writeText(mdContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([mdContent], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `QuantMind_投研执行单_${selectedDate}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150 print:p-0 print:bg-white">
      <div className="w-full max-w-4xl bg-[#0c1220] border border-slate-700 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh] print:border-none print:shadow-none print:max-h-full print:bg-white print:text-black">
        {/* Header (Hidden on print) */}
        <div className="p-4 bg-slate-900 border-b border-slate-800 flex items-center justify-between print:hidden">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-600/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                每日量化投研与风控执行单 (One-Click Daily Briefing)
              </h3>
              <p className="text-[11px] text-slate-400 font-mono">
                截面: {selectedDate} · 一键导出专业投研归档 Markdown / 打印级 PDF
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center gap-1.5 transition"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? '已复制' : '复制 Markdown'}</span>
            </button>
            <button
              onClick={handleDownload}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center gap-1.5 transition"
            >
              <Download className="w-3.5 h-3.5 text-cyan-400" />
              <span>下载 .md</span>
            </button>
            <button
              onClick={handlePrint}
              className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium flex items-center gap-1.5 transition shadow-sm"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>打印 / 存为 PDF</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition ml-2"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Formatted Content Preview */}
        <div className="p-6 overflow-y-auto space-y-6 text-slate-200 text-xs flex-1 font-sans bg-[#080d1a] print:bg-white print:text-black print:p-8">
          <div className="border-b border-slate-800 print:border-black pb-4">
            <h1 className="text-xl font-bold text-white print:text-black">
              QuantMind 每日量化投研与风控执行单
            </h1>
            <div className="flex flex-wrap items-center gap-4 text-[11px] text-slate-400 print:text-slate-600 font-mono mt-2">
              <span>交易日: {selectedDate}</span>
              <span>数据引擎: Multi-Modal Quant Engine (v3.5)</span>
              <span>
                审查状态: {formal?.approval?.research_report_approved ? '✅ 已核验通过' : '⚠️ 模拟阶段'}
              </span>
            </div>
          </div>

          {/* Section 1: Market Regime */}
          <div className="space-y-2">
            <h2 className="text-sm font-bold text-cyan-300 print:text-black flex items-center gap-1.5">
              <TrendingUp className="w-4 h-4 text-cyan-400 print:hidden" />
              <span>一、宏观与大盘微观状态 (Market Regime)</span>
            </h2>
            <div className="p-3.5 rounded-xl bg-slate-900/60 print:bg-slate-50 border border-slate-800 print:border-slate-300 grid grid-cols-2 md:grid-cols-4 gap-3 font-mono">
              <div>
                <div className="text-[10px] text-slate-400">沪深300 相对 MA20</div>
                <div className="font-bold text-white print:text-black mt-0.5">
                  {sgn((market?.trend as any)?.['沪深300']?.vs_ma20)}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-slate-400">中证2000 相对沪深300（20 日）</div>
                <div className="font-bold text-white print:text-black mt-0.5">
                  {sgn((market?.style as any)?.csi2000_vs_hs300_rs20)}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-slate-400">沪深300 近 20 日已实现波动</div>
                <div className="font-bold text-amber-400 print:text-black mt-0.5">
                  {annPct((market?.volatility_hs300 as any)?.realized_vol_20d_ann)}
                </div>
              </div>
              <div>
                <div className="text-[10px] text-slate-400">两融余额 20 日变化</div>
                <div className="font-bold text-cyan-400 print:text-black mt-0.5">
                  {sgn((market?.margin as any)?.chg_20d)}
                </div>
              </div>
            </div>
          </div>

          {/* Section 2: Orders */}
          <div className="space-y-2">
            <h2 className="text-sm font-bold text-cyan-300 print:text-black flex items-center gap-1.5">
              <DollarSign className="w-4 h-4 text-cyan-400 print:hidden" />
              <span>二、明日开盘委托执行单 (Approved Orders Next Open)</span>
            </h2>
            <div className="border border-slate-800 print:border-slate-300 rounded-lg overflow-x-auto">
              <table className="w-full text-left text-xs divide-y divide-slate-800 print:divide-slate-300">
                <thead className="bg-slate-900 print:bg-slate-100 text-slate-400 print:text-slate-700">
                  <tr>
                    <th className="px-3 py-2">方向</th>
                    <th className="px-3 py-2">代码 / 名称</th>
                    <th className="px-3 py-2 text-right">拟委托股数</th>
                    <th className="px-3 py-2 text-right">参考开盘价</th>
                    <th className="px-3 py-2 text-right">预估金额</th>
                    <th className="px-3 py-2 text-right">目标排名</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 print:divide-slate-200 font-mono">
                  {validOrders.length > 0 ? (
                    validOrders.map((o, idx) => (
                      <tr key={idx}>
                        <td className="px-3 py-2">
                          <span className={`font-bold ${o.side === 'buy' ? 'text-rose-400' : 'text-emerald-400'}`}>
                            {o.side === 'buy' ? '买入' : '卖出'}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-white print:text-black">
                          {o.ticker} ({o.name})
                        </td>
                        <td className="px-3 py-2 text-right">{o.est_shares} 股</td>
                        <td className="px-3 py-2 text-right">¥{Number(o.ref_close).toFixed(2)}</td>
                        <td className="px-3 py-2 text-right font-bold">¥{Number(o.est_value).toLocaleString()}</td>
                        <td className="px-3 py-2 text-right text-cyan-300 print:text-black">#{o.rank_in_target}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="text-center py-4 text-slate-500 italic">
                        明日暂无调仓委托
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {skippedOrders.length > 0 && (
              <div className="p-2.5 rounded bg-amber-950/20 print:bg-amber-50 border border-amber-800/40 print:border-amber-300 text-[11px] text-amber-300 print:text-amber-900">
                <span className="font-bold">⚠️ 零股买不起自动跳过: </span>
                {skippedOrders.map((o) => `${o.ticker}(${o.name})`).join('、')} 因不足一手跳过，现金保留在账户中。
              </div>
            )}
          </div>

          {/* Section 3: Red Team Warnings */}
          <div className="space-y-2">
            <h2 className="text-sm font-bold text-cyan-300 print:text-black flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-400 print:hidden" />
              <span>三、A0 Agent 大脑与红队风控备忘 (Red Team Directives)</span>
            </h2>
            <div className="space-y-1.5">
              {flags.length > 0 ? (
                flags.map((f, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded bg-slate-900/40 print:bg-slate-50 border border-slate-800 print:border-slate-300 text-[11px]"
                  >
                    <span className="font-bold text-amber-400 print:text-amber-800 mr-1.5">
                      [{f.action === 'avoid_new_buy' ? '不建议新买入' : '重点关注'}] {f.ticker}:
                    </span>
                    <span className="text-slate-300 print:text-slate-700">{f.reason}</span>
                  </div>
                ))
              ) : (
                <div className="text-slate-500 italic">本期暂无特殊大脑红队警示标记。</div>
              )}
            </div>
          </div>
        </div>

        {/* Footer (Hidden on print) */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between print:hidden">
          <span className="text-[11px] text-slate-500 font-mono">快捷操作: 点击打印将直接调起浏览器 A4 排版视图</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  );
};
