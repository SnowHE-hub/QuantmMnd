import React, { useState } from 'react';
import { Copy, Check, Terminal, FileCode, ArrowDownRight, Layers, Cpu, ShieldCheck } from 'lucide-react';
import { StageNode } from './SystemDAG';

interface SubsystemCardProps {
  stage: StageNode;
}

export const SubsystemCard: React.FC<SubsystemCardProps> = ({ stage }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(stage.cliCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const Icon = stage.icon;

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden">
      {/* Header */}
      <div className="flex items-start justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${stage.color} flex items-center justify-center text-white shadow-md`}>
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-blue-900/60 text-blue-300 border border-blue-700/50">
                {stage.stage} 阶段模块
              </span>
              <h3 className="text-base font-bold text-white">{stage.name}</h3>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">{stage.tagline}</p>
          </div>
        </div>
        <div className="text-right">
          <span className="text-[11px] font-mono text-slate-500">自解释技术说明书</span>
        </div>
      </div>

      {/* Grid Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 my-4">
        {/* Input */}
        <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-800/80">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <span>输入数据契约</span>
          </div>
          <div className="text-xs font-mono text-slate-200">{stage.input}</div>
        </div>

        {/* Output */}
        <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-800/80">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
            <ArrowDownRight className="w-3.5 h-3.5 text-emerald-400" />
            <span>输出产物与格式</span>
          </div>
          <div className="text-xs font-mono text-emerald-300">{stage.output}</div>
        </div>

        {/* Method */}
        <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-800/80">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-purple-400" />
            <span>核心算法与逻辑</span>
          </div>
          <div className="text-xs text-slate-300">{stage.method}</div>
        </div>

        {/* Model */}
        <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-800/80">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
            <span>参与模型 / 规则</span>
          </div>
          <div className="text-xs font-medium text-cyan-300">{stage.models}</div>
        </div>
      </div>

      {/* Configuration & Code References */}
      <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <FileCode className="w-4 h-4 text-slate-500 shrink-0" />
          <span>核心源码与配置:</span>
          <code className="bg-slate-900 px-2 py-0.5 rounded text-blue-300 font-mono text-xs border border-slate-800">
            {stage.configPath}
          </code>
        </div>

        {/* Interactive CLI Command Launcher */}
        <div className="flex items-center gap-2 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
          <Terminal className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <span className="text-[11px] text-slate-400">平时如何独立运行:</span>
          <code className="text-xs font-mono text-cyan-200 select-all">{stage.cliCommand}</code>
          <button
            onClick={handleCopy}
            className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition"
            title="一键复制命令"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>
    </div>
  );
};
