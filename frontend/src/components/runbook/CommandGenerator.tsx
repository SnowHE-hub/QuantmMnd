import React, { useState } from 'react';
import { Terminal, Copy, Check, Play, Settings2, ShieldCheck, Loader2, RefreshCw, XCircle } from 'lucide-react';

interface CommandGeneratorProps {
  selectedDate: string;
  onRefreshData?: () => void;
}

export const CommandGenerator: React.FC<CommandGeneratorProps> = ({
  selectedDate,
  onRefreshData,
}) => {
  const [taskType, setTaskType] = useState<string>('pipeline_status');
  const [targetDate, setTargetDate] = useState<string>(selectedDate);
  const [ticker, setTicker] = useState<string>('688383.SH');
  const [copied, setCopied] = useState(false);

  // Live execution states
  const [isRunning, setIsRunning] = useState(false);
  const [terminalOutput, setTerminalOutput] = useState<string>('');
  const [exitCode, setExitCode] = useState<number | null>(null);

  // Generate dynamic CLI command
  let generatedCmd = '';
  let description = '';

  // 这里生成的文本只用于显示和复制；真正执行时只把任务编号和参数发给开发服务器，由服务器端白名单生成参数数组（live_runner.ts）
  const ymd = targetDate.replace(/-/g, '');
  const copyOnly = taskType === 'daily_full';
  switch (taskType) {
    case 'pipeline_status':
      generatedCmd = `tail -n 40 /home/lenovo/t030_scratch/logs/daily/${ymd}/day_${targetDate}.log`;
      description = `读取 ${targetDate} 盘后流水线日志的最后 40 行（原实现读的 status 文件并不存在，还会悄悄退回 10-23 的记录）。`;
      break;
    case 'health_audit':
      generatedCmd = `PYTHONPATH=~/projects/qm-t035 python -c 'from quantmind.line_a.api_server import app; ...'`;
      description = '检查只读接口能否导入、价格面板的最后一个交易日。';
      break;
    case 'research_ticker':
      generatedCmd = `python -c 'DataTools(lake, "${ymd}").financial_snapshot("${ticker}") ...'`;
      description = `真的按时点 ${targetDate} 取一次 ${ticker} 的财报、预告与快报，打印每张表的条数（原实现只打印一句固定文字）。`;
      break;
    case 'catalog_scan':
      generatedCmd = `python -c 'json.load(open("data_catalog.json"))["totals"]'`;
      description = '读取数据清单的文件数、总字节数和生成时间。';
      break;
    case 'daily_full':
      generatedCmd = 'cd ~/projects/qm-t035 && bash scripts/line_a/daily_auto.sh';
      description = '完整盘后流水线只能在终端里运行（自动判断交易日、找上一次完成的日期，已完成或正在运行就退出）；网页端不提供执行，避免从浏览器触发长任务或传错 LAST 参数。';
      break;
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(generatedCmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRunLive = async () => {
    setIsRunning(true);
    setTerminalOutput(`[LiveRunner] 发起实时执行任务...\n$ ${generatedCmd}\n\n`);
    setExitCode(null);

    try {
      const response = await fetch('/api/live_run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: taskType, date: targetDate, ticker }),
      });

      if (!response.ok) {
        const msg = await response.text();
        setTerminalOutput((prev) => prev + `[LiveRunner] 被拒绝 (${response.status}): ${msg}`);
        setExitCode(response.status);
        return;
      }
      if (!response.body) {
        setTerminalOutput((prev) => prev + '\n[LiveRunner] 无法获取流式响应');
        setIsRunning(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      let all = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        all += chunk;
        setTerminalOutput((prev) => prev + chunk);
      }
      const m = all.match(/\[exit:(-?\d+)\]\s*$/);
      setExitCode(m ? Number(m[1]) : -1);   // 用子进程真实的退出码（原来无论成败都写 0）
    } catch (err: any) {
      setTerminalOutput((prev) => prev + `\n[LiveRunner 出错]: ${err.message}`);
      setExitCode(1);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Settings2 className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-bold text-white">
            Web 端真实受控命令执行台与参数配置器 (Live Command Runner)
          </h3>
        </div>
        <span className="text-xs font-mono text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/40">
          白名单任务 · 仅本机
        </span>
      </div>

      {/* Form Inputs */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
        <div>
          <label className="block text-slate-400 text-[11px] mb-1 font-medium">任务操作场景</label>
          <select
            value={taskType}
            onChange={(e) => setTaskType(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-white font-medium focus:outline-none cursor-pointer"
          >
            <option value="pipeline_status">查看收盘流水线实时状态 (Pipeline Status)</option>
            <option value="health_audit">后端环境与数据根健康审计 (Health Audit)</option>
            <option value="research_ticker">单股时点数据工具自检 (Research Ticker Check)</option>
            <option value="catalog_scan">44 个数据集资产清单扫描 (Catalog Scan)</option>
            <option value="daily_full">每日收盘全自动流水线 (Daily Full Run)</option>
          </select>
        </div>

        <div>
          <label className="block text-slate-400 text-[11px] mb-1 font-medium">基准截面日期</label>
          <input
            type="text"
            value={targetDate}
            onChange={(e) => setTargetDate(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-white font-mono focus:outline-none"
          />
        </div>

        {taskType === 'research_ticker' && (
          <div>
            <label className="block text-slate-400 text-[11px] mb-1 font-medium">目标股票代码</label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 text-white font-mono focus:outline-none"
            />
          </div>
        )}
      </div>

      {/* Description */}
      <div className="p-2.5 rounded-lg bg-blue-950/20 border border-blue-900/40 text-xs text-blue-300">
        <span className="font-semibold mr-1.5">执行意图:</span>
        {description}
      </div>

      {/* Generated Command Box & Execution Actions */}
      <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 font-mono text-xs">
        <div className="flex items-center gap-2 overflow-x-auto text-cyan-200 max-w-xl">
          <Terminal className="w-4 h-4 text-cyan-400 shrink-0" />
          <span className="select-all truncate">{generatedCmd}</span>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* Live Run Button */}
          <button
            onClick={handleRunLive}
            disabled={isRunning || copyOnly}
            title={copyOnly ? '只能在终端里运行' : undefined}
            className="px-3.5 py-1.5 rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-sans text-xs font-bold flex items-center gap-1.5 transition shadow-sm disabled:opacity-50"
          >
            {isRunning ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>执行中...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-white" />
                <span>在 WSL 中执行</span>
              </>
            )}
          </button>

          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white font-sans text-xs font-medium flex items-center gap-1.5 transition border border-slate-700"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? '已复制' : '复制命令'}</span>
          </button>
        </div>
      </div>

      {/* Real-time Streaming Terminal Display */}
      {terminalOutput && (
        <div className="rounded-xl bg-[#070b14] border border-slate-800 overflow-hidden shadow-inner font-mono text-xs">
          <div className="px-3 py-2 bg-slate-900 border-b border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
            <span className="flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5 text-cyan-400" />
              <span>WSL 输出实时日志流</span>
            </span>
            <div className="flex items-center gap-2">
              {onRefreshData && exitCode === 0 && (
                <button
                  onClick={onRefreshData}
                  className="px-2 py-0.5 rounded bg-blue-600/30 hover:bg-blue-600/50 text-blue-300 border border-blue-500/40 text-[10px] flex items-center gap-1 transition"
                >
                  <RefreshCw className="w-3 h-3" />
                  <span>刷新前端数据</span>
                </button>
              )}
              <button
                onClick={() => setTerminalOutput('')}
                className="text-slate-500 hover:text-slate-300 text-[10px]"
              >
                清除输出
              </button>
            </div>
          </div>

          <pre className="p-4 max-h-64 overflow-y-auto text-cyan-300 leading-relaxed text-[11px] whitespace-pre-wrap select-text">
            {terminalOutput}
          </pre>
        </div>
      )}
    </div>
  );
};
