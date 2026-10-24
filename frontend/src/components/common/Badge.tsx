import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle, HelpCircle, ShieldAlert, AlertOctagon } from 'lucide-react';
import { CheckStatus } from '../../api/types';

interface BadgeProps {
  status: CheckStatus | string;
  className?: string;
  showIcon?: boolean;
}

export const CheckBadge: React.FC<BadgeProps> = ({ status, className = '', showIcon = true }) => {
  switch (status) {
    case 'verified':
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/60 ${className}`}>
          {showIcon && <CheckCircle2 className="w-3 h-3 text-emerald-400" />}
          <span>已核对 (Verified)</span>
        </span>
      );
    case 'minor_mismatch':
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-amber-950/80 text-amber-300 border border-amber-800/60 ${className}`}>
          {showIcon && <AlertTriangle className="w-3 h-3 text-amber-300" />}
          <span>微小差异 (Minor Mismatch)</span>
        </span>
      );
    case 'mismatch':
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-rose-950/80 text-rose-400 border border-rose-800/60 ${className}`}>
          {showIcon && <XCircle className="w-3 h-3 text-rose-400" />}
          <span>不匹配 (Mismatch)</span>
        </span>
      );
    case 'missing_evidence':
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-purple-950/80 text-purple-300 border border-purple-800/60 ${className}`}>
          {showIcon && <HelpCircle className="w-3 h-3 text-purple-300" />}
          <span>证据缺失 (Missing)</span>
        </span>
      );
    case 'weak_basis':
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-orange-950/80 text-orange-300 border border-orange-800/60 ${className}`}>
          {showIcon && <AlertTriangle className="w-3 h-3 text-orange-300" />}
          <span>依据薄弱 (Weak Basis)</span>
        </span>
      );
    case 'unverifiable':
    default:
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-slate-800/80 text-slate-400 border border-slate-700/60 ${className}`}>
          {showIcon && <HelpCircle className="w-3 h-3 text-slate-400" />}
          <span>无法核验 ({status})</span>
        </span>
      );
  }
};

export const BrainActionBadge: React.FC<{ action: string; className?: string }> = ({ action, className = '' }) => {
  switch (action) {
    case 'caution':
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-amber-950/80 text-amber-300 border border-amber-800/60 ${className}`}>
          <AlertTriangle className="w-3 h-3 text-amber-400" />
          <span>留意 (Caution)</span>
        </span>
      );
    case 'avoid_new_buy':
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-rose-950/80 text-rose-300 border border-rose-800/60 ${className}`}>
          <ShieldAlert className="w-3 h-3 text-rose-400" />
          <span>不建议新买 (Avoid Buy)</span>
        </span>
      );
    case 'review_holding':
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-purple-950/80 text-purple-300 border border-purple-800/60 ${className}`}>
          <AlertOctagon className="w-3 h-3 text-purple-400" />
          <span>建议复核 (Review Holding)</span>
        </span>
      );
    default:
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700 ${className}`}>
          <span>{action}</span>
        </span>
      );
  }
};
