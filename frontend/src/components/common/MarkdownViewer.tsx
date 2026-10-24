import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownViewerProps {
  content: string;
  className?: string;
}

export const MarkdownViewer: React.FC<MarkdownViewerProps> = ({ content, className = '' }) => {
  return (
    <div className={`prose prose-invert max-w-none text-slate-200 leading-relaxed ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ ...props }) => (
            <h1 className="text-2xl font-bold text-white border-b border-slate-800 pb-3 mt-6 mb-4 flex items-center gap-2" {...props} />
          ),
          h2: ({ ...props }) => (
            <h2 className="text-xl font-semibold text-blue-400 border-b border-slate-800/60 pb-2 mt-5 mb-3" {...props} />
          ),
          h3: ({ ...props }) => (
            <h3 className="text-lg font-medium text-cyan-300 mt-4 mb-2" {...props} />
          ),
          p: ({ ...props }) => (
            <p className="my-2.5 text-slate-300 text-sm leading-relaxed" {...props} />
          ),
          ul: ({ ...props }) => (
            <ul className="list-disc list-inside space-y-1 my-2 text-sm text-slate-300" {...props} />
          ),
          ol: ({ ...props }) => (
            <ol className="list-decimal list-inside space-y-1 my-2 text-sm text-slate-300" {...props} />
          ),
          li: ({ ...props }) => (
            <li className="text-slate-300 text-sm" {...props} />
          ),
          blockquote: ({ ...props }) => (
            <blockquote className="border-l-4 border-blue-500 pl-4 py-1 my-3 bg-slate-900/60 rounded-r text-slate-300 italic text-sm" {...props} />
          ),
          table: ({ ...props }) => (
            <div className="overflow-x-auto my-4 border border-slate-800 rounded-lg">
              <table className="min-w-full divide-y divide-slate-800 text-left text-xs tabular-nums" {...props} />
            </div>
          ),
          thead: ({ ...props }) => (
            <thead className="bg-slate-900 text-slate-400 font-semibold" {...props} />
          ),
          tbody: ({ ...props }) => (
            <tbody className="divide-y divide-slate-800/60 bg-slate-950/40" {...props} />
          ),
          th: ({ ...props }) => (
            <th className="px-3.5 py-2.5 font-medium text-slate-300" {...props} />
          ),
          td: ({ ...props }) => (
            <td className="px-3.5 py-2 text-slate-300" {...props} />
          ),
          code: ({ inline, className, children, ...props }: any) => {
            if (inline) {
              return (
                <code className="bg-slate-800 text-cyan-300 px-1.5 py-0.5 rounded text-xs font-mono" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <pre className="bg-[#070b14] p-4 rounded-lg overflow-x-auto my-3 border border-slate-800 text-xs font-mono text-cyan-200">
                <code {...props}>{children}</code>
              </pre>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
