import React, { useState, useEffect } from 'react';
import { FileText, Copy, Check, Printer } from 'lucide-react';
import { quantmindApi } from '../../api/client';
import { MarkdownViewer } from '../common/MarkdownViewer';

interface FormalReportReaderProps {
  selectedDate: string;
}

export const FormalReportReader: React.FC<FormalReportReaderProps> = ({ selectedDate }) => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setLoading(true);
    quantmindApi
      .getFormalMarkdown(selectedDate)
      .then((md) => setContent(md || ''))
      .catch((err) => setContent(`未找到 ${selectedDate} 的正式荐股日报文本: ${err.message}`))
      .finally(() => setLoading(false));
  }, [selectedDate]);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Title & Actions */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-white">
            正式荐股日报出版级精读视图 ({selectedDate})
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 flex items-center gap-1.5 transition"
            title="一键复制 Markdown"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? '已复制' : '复制全文'}</span>
          </button>
        </div>
      </div>

      {/* Markdown Document Content */}
      <div className="p-5 rounded-xl bg-slate-950/80 border border-slate-800 max-h-[700px] overflow-y-auto">
        {loading ? (
          <div className="py-20 text-center text-xs text-blue-400 font-mono animate-pulse">
            加载正式日报全文中...
          </div>
        ) : (
          <MarkdownViewer content={content} />
        )}
      </div>
    </div>
  );
};
