import React, { useEffect, useMemo, useState } from 'react';
import {
  ArrowUpCircle,
  ArrowDownCircle,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  ShoppingCart,
  Sliders,
  RotateCcw,
  Sparkles,
  Swords,
  Info,
} from 'lucide-react';
import { OrderItem, BrainFlag, RiskCardItem, TierData } from '../../api/types';
import {
  analyticsApi,
  Board,
  boardOf,
  ExecutionRules,
  ForwardResponse,
  legalShares,
  pct,
  RedTeamResponse,
  stepShares,
  tradeCost,
} from '../../api/analytics';

interface OrdersNextOpenProps {
  orders: OrderItem[];
  tier?: TierData;
  brainFlags?: BrainFlag[];
  riskCards?: RiskCardItem[];
  selectedDate: string;
  latestDate: string;
  onSelectTicker: (ticker: string) => void;
  onAddToCompare?: (ticker: string) => void;
}

const HORIZONS = [1, 5, 10, 20];
const BOARD_CN: Record<Board, string> = { MAIN: '主板', CHINEXT: '创业板', STAR: '科创板', BSE: '北交所' };

export const OrdersNextOpen: React.FC<OrdersNextOpenProps> = ({
  orders = [],
  tier,
  brainFlags = [],
  riskCards = [],
  selectedDate,
  latestDate,
  onSelectTicker,
  onAddToCompare,
}) => {
  const isHistorical = !!latestDate && selectedDate < latestDate;
  const [rules, setRules] = useState<ExecutionRules | null>(null);
  const [redteam, setRedteam] = useState<RedTeamResponse | null>(null);
  const [fwd, setFwd] = useState<ForwardResponse | null>(null);
  const [fwdError, setFwdError] = useState<string | null>(null);
  const [isSandboxMode, setIsSandboxMode] = useState(false);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [shareOverride, setShareOverride] = useState<Record<string, number>>({});

  useEffect(() => {
    analyticsApi.executionRules().then(setRules).catch(() => setRules(null));
  }, []);

  // 红队规则在后端按当天的数据统一判定（RT-v2：截面前 10%），这里只取每只股票命中的规则
  useEffect(() => {
    let alive = true;
    setRedteam(null);
    if (!orders.length) return;
    analyticsApi
      .redteam(selectedDate, 'formal', 5, 'v2')
      .then((r) => alive && setRedteam(r))
      .catch(() => alive && setRedteam(null));
    return () => {
      alive = false;
    };
  }, [selectedDate, orders.length]);

  // 历史截面：一次请求拿整组委托的事后收益与基准（原来是每只股票请求一次行情、前端用不复权价格自己算）
  useEffect(() => {
    let alive = true;
    setFwd(null);
    setFwdError(null);
    if (!isHistorical || !orders.length) return;
    analyticsApi
      .forward(selectedDate, { tickers: orders.map((o) => o.ticker), horizons: HORIZONS })
      .then((r) => alive && setFwd(r))
      .catch((e) => alive && setFwdError(String(e.message || e)));
    return () => {
      alive = false;
    };
  }, [isHistorical, selectedDate, orders]);

  useEffect(() => {
    setExcluded(new Set());
    setShareOverride({});
  }, [selectedDate, orders]);

  const rtRules = useMemo(() => {
    const m: Record<string, string[]> = {};
    redteam?.rows.forEach((r) => (m[r.ticker] = r.rules));
    return m;
  }, [redteam]);
  const fwdBy = useMemo(() => {
    const m: Record<string, ForwardResponse['rows'][number]> = {};
    fwd?.rows.forEach((r) => (m[r.ticker] = r));
    return m;
  }, [fwd]);
  const cardBy = useMemo(() => {
    const m: Record<string, RiskCardItem> = {};
    riskCards.forEach((r) => (m[r.ticker] = r));
    return m;
  }, [riskCards]);

  const severityOf = (ticker: string, shares: number) => {
    const hit = rtRules[ticker] || [];
    const flag = brainFlags.find((f) => f.ticker === ticker);
    const issues: string[] = hit.map((r) => redteam?.meta.rules[r] || r);
    if (flag?.reason) issues.push(`大脑理由：${flag.reason}`);
    if (shares === 0) issues.push('按当前资金买不起一个最小申报单位，正式规则跳过，现金保留');
    const danger = hit.includes('brain_avoid_new_buy');
    return { severity: danger ? 'danger' : hit.length || shares === 0 ? 'caution' : 'safe', issues } as const;
  };

  // ------------------------------------------------------------ 推演沙盒（按 T-052 模拟器同一套执行规则）
  const sandbox = useMemo(() => {
    if (!rules) return null;
    const cash0 = tier?.cash ?? 0;
    const lines = orders.map((o) => {
      const board = boardOf(o.ticker);
      const base = o.est_shares || 0;
      const shares = excluded.has(o.ticker) ? 0 : legalShares(shareOverride[o.ticker] ?? base, board, rules.lot);
      const gross = shares * (o.ref_close || 0);
      const cost = tradeCost(gross, o.side, rules);
      return { ...o, board, base, shares, gross, cost };
    });
    const buys = lines.filter((l) => l.side === 'buy');
    const sells = lines.filter((l) => l.side === 'sell');
    const buyGross = buys.reduce((s, l) => s + l.gross, 0);
    const sellGross = sells.reduce((s, l) => s + l.gross, 0);
    const costs = lines.reduce((s, l) => s + l.cost.total, 0);
    const cashAfter = cash0 + sellGross - buyGross - costs;

    // 推演后的持仓：现有持仓 − 卖出 + 买入（按参考价估值）
    const pos: Record<string, number> = {};
    (tier?.holdings || []).forEach((h) => (pos[h.ticker] = (pos[h.ticker] || 0) + h.value));
    lines.forEach((l) => (pos[l.ticker] = (pos[l.ticker] || 0) + (l.side === 'buy' ? l.gross : -l.gross)));
    const invested = Object.values(pos).reduce((s, v) => s + Math.max(0, v), 0);
    const equity = invested + cashAfter;
    let beta = 0;
    let vol = 0;
    let covered = 0;
    const ind: Record<string, number> = {};
    Object.entries(pos).forEach(([t, v]) => {
      if (v <= 0) return;
      const c = cardBy[t];
      const w = v / Math.max(equity, 1);
      if (c?.beta_252 !== undefined && c?.vol_60_ann !== undefined) {
        beta += w * c.beta_252;
        vol += w * c.vol_60_ann;
        covered += w;
      }
      const k = (c?.industry_tushare as string) || '未知行业';
      ind[k] = (ind[k] || 0) + w;
    });
    const topInd = Object.entries(ind).sort((a, b) => b[1] - a[1])[0];
    return {
      lines,
      cash0,
      buyGross,
      sellGross,
      costs,
      cashAfter,
      equity,
      investedShare: equity > 0 ? invested / equity : 0,
      names: Object.values(pos).filter((v) => v > 0).length,
      beta,
      volUpper: vol, // 权重加总的波动率：忽略相关性，是组合波动的上界
      covered,
      topInd,
    };
  }, [rules, orders, tier, excluded, shareOverride, cardBy]);

  const baseline = useMemo(() => {
    if (!rules) return null;
    return orders.reduce((s, o) => s + (o.est_shares || 0) * (o.ref_close || 0), 0);
  }, [rules, orders]);

  const g = redteam?.by_group?.excess_pool;
  const skipped = orders.filter((o) => (o.est_shares || 0) === 0);

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4" data-testid="orders-next-open">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <ShoppingCart className="w-4 h-4 text-cyan-400" />
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <span>{isHistorical ? '当日开盘委托与事后验真' : '明日开盘委托与红队提示'}</span>
              {isSandboxMode && (
                <span className="px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 text-[10px] font-mono">推演中</span>
              )}
            </h3>
            <p className="text-[11px] text-slate-400 font-mono">
              次日开盘成交 · 最小申报单位按板块 · 红队规则 {redteam?.meta.rule_version ?? 'v2'}（后端统一判定）
            </p>
          </div>
        </div>
        <button
          onClick={() => setIsSandboxMode(!isSandboxMode)}
          disabled={!rules}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition shadow-sm ${
            isSandboxMode ? 'bg-cyan-600 text-white' : 'bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-cyan-800/80'
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>{isSandboxMode ? '退出推演沙盒' : '开启反事实推演沙盒'}</span>
        </button>
      </div>

      {isSandboxMode && sandbox && (
        <div className="p-3.5 rounded-xl bg-cyan-950/30 border border-cyan-800/80 space-y-2 text-xs" data-testid="sandbox-panel">
          <div className="flex items-center justify-between">
            <span className="font-bold text-cyan-300 flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              推演沙盒（佣金万 {(rules!.execution.commission_rate * 1e4).toFixed(1)}、最低 {rules!.execution.min_commission} 元；过户费；卖出印花税{' '}
              {(rules!.stamp_duty_sell * 100).toFixed(2)}%）
            </span>
            <button
              onClick={() => {
                setExcluded(new Set());
                setShareOverride({});
              }}
              className="text-slate-400 hover:text-white flex items-center gap-1 text-[11px] font-mono transition"
            >
              <RotateCcw className="w-3 h-3" />
              重置
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-[11px]">
            <Metric label="原计划买入金额" value={`¥${(baseline ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`} />
            <Metric
              label="推演后 买入 / 卖出"
              value={`¥${sandbox.buyGross.toLocaleString(undefined, { maximumFractionDigits: 0 })} / ¥${sandbox.sellGross.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            />
            <Metric label="交易费用合计" value={`¥${sandbox.costs.toFixed(2)}`} />
            <Metric
              label="成交后现金"
              value={`¥${sandbox.cashAfter.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
              tone={sandbox.cashAfter < 0 ? 'bad' : 'ok'}
              testid="sandbox-cash-after"
            />
            <Metric label="持股数 / 仓位" value={`${sandbox.names} 只 / ${(sandbox.investedShare * 100).toFixed(1)}%`} />
            <Metric label="加权 beta_252" value={sandbox.covered > 0 ? sandbox.beta.toFixed(2) : '—'} testid="sandbox-beta" />
            <Metric label="加权 60 日波动（上界）" value={sandbox.covered > 0 ? pct(sandbox.volUpper, 1, false) : '—'} />
            <Metric
              label="最大行业占比"
              value={sandbox.topInd ? `${sandbox.topInd[0]} ${(sandbox.topInd[1] * 100).toFixed(1)}%` : '—'}
            />
          </div>
          {sandbox.cashAfter < 0 && (
            <div className="text-rose-300 text-[11px]">现金不足：推演后的买单超过可用资金 {Math.abs(sandbox.cashAfter).toFixed(0)} 元，实盘会被券商拒单。</div>
          )}
          <div className="text-[10px] text-slate-500">
            风险卡覆盖了 {(sandbox.covered * 100).toFixed(0)}% 的推演后权益；加权波动是各股波动的加权和，没有算相关性，真实组合波动更低。
          </div>
        </div>
      )}

      {isHistorical && (
        <div className="p-3 rounded-lg bg-slate-900/70 border border-slate-700 text-xs text-slate-300 space-y-1" data-testid="truth-test-banner">
          <div className="flex items-center gap-2 font-semibold text-emerald-300">
            <ShieldCheck className="w-4 h-4" /> 事后验真（{fwd?.meta.convention ?? 'T+1 开盘入场、复权收益'}）
          </div>
          {fwdError && <div className="text-rose-300">{fwdError}</div>}
          {fwd && (
            <div className="font-mono text-[11px]">
              入场 {fwd.meta.entry_date ?? '—'} · 行情截至 {fwd.meta.last_bar} · 基准：
              {HORIZONS.map((h) => (
                <span key={h} className="ml-2">
                  {h} 日 研究池等权 {pct(fwd.benchmarks[String(h)]?.pool_ew)} / 中证2000 {pct(fwd.benchmarks[String(h)]?.csi2000)}
                </span>
              ))}
            </div>
          )}
          {g && (
            <div className="text-[11px]">
              红队标记的 {g.flagged.n} 只里，5 日跑输研究池等权的比例 {pct(g.hit_rate, 0, false)}；未标记的 {g.unflagged.n} 只里这个比例是{' '}
              {pct(g.base_rate, 0, false)}（基准跑输率）。差值 {pct(g.lift, 0)} 才是规则的信息量
              {redteam?.warning ? `；${redteam.warning}` : ''}。
            </div>
          )}
        </div>
      )}

      {skipped.length > 0 && (
        <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-800/60 text-xs text-amber-200 flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-amber-300">{skipped.length} 只买不起一个最小申报单位，正式规则跳过（est_shares = 0）</div>
            <p className="text-[11px] text-amber-400/90 mt-0.5">主板、创业板 100 股起；科创板 200 股起、1 股递增；北交所 100 股起、1 股递增。T-052b 验证过的替代规则都没有更好，所以保留跳过。</p>
          </div>
        </div>
      )}

      <div className="border border-slate-800 rounded-lg overflow-x-auto">
        <table className="w-full text-left text-xs divide-y divide-slate-800">
          <thead className="bg-slate-900 text-slate-400 font-medium">
            <tr>
              {isSandboxMode && <th className="px-3 py-2 text-center">推演</th>}
              <th className="px-3.5 py-2">方向</th>
              <th className="px-3 py-2">代码 / 名称</th>
              <th className="px-3 py-2">板块</th>
              <th className="px-3 py-2 text-right">股数</th>
              <th className="px-3 py-2 text-right">参考价</th>
              <th className="px-3 py-2 text-right">金额</th>
              <th className="px-3 py-2 text-right">目标排名</th>
              <th className="px-3 py-2 text-center">红队</th>
              {isHistorical &&
                [1, 5, 20].map((h) => (
                  <th key={h} className="px-3 py-2 text-right">
                    {h} 日收益 / 超额
                  </th>
                ))}
              <th className="px-3 py-2 text-center">操作</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 bg-slate-950/40 font-mono">
            {orders.length === 0 && (
              <tr>
                <td colSpan={12} className="px-3 py-6 text-center text-slate-500 font-sans">
                  这个截面没有正式委托（正式日报从 {latestDate || '首个运行日'} 起才有）。
                </td>
              </tr>
            )}
            {(sandbox?.lines ?? orders.map((o) => ({ ...o, board: boardOf(o.ticker), base: o.est_shares, shares: o.est_shares, gross: o.est_value }))).map(
              (o) => {
                const audit = severityOf(o.ticker, o.base || 0);
                const f = fwdBy[o.ticker];
                const isExcluded = excluded.has(o.ticker);
                return (
                  <tr
                    key={`${o.side}-${o.ticker}`}
                    onClick={() => onSelectTicker(o.ticker)}
                    className={`hover:bg-slate-900/60 cursor-pointer transition ${isExcluded ? 'opacity-40 line-through' : audit.severity === 'danger' ? 'bg-rose-950/20' : ''}`}
                  >
                    {isSandboxMode && rules && (
                      <td className="px-3 py-2 text-center" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-center gap-1.5">
                          <input
                            type="checkbox"
                            checked={!isExcluded}
                            onChange={() =>
                              setExcluded((p) => {
                                const n = new Set(p);
                                n.has(o.ticker) ? n.delete(o.ticker) : n.add(o.ticker);
                                return n;
                              })
                            }
                            className="rounded border-slate-700 bg-slate-900 text-cyan-500 focus:ring-0 cursor-pointer"
                          />
                          {!isExcluded &&
                            ([-1, 1] as const).map((dir) => (
                              <button
                                key={dir}
                                data-testid={`step-${dir > 0 ? 'up' : 'down'}-${o.ticker}`}
                                onClick={() =>
                                  setShareOverride((p) => ({ ...p, [o.ticker]: stepShares(o.shares, dir, o.board, rules.lot) }))
                                }
                                className="w-4 h-4 rounded bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-300 flex items-center justify-center"
                                title={`${dir > 0 ? '增加' : '减少'}一档（${o.board === 'STAR' ? '科创板 200 股起' : o.board === 'BSE' ? '北交所 100 股起' : '100 股一手'}）`}
                              >
                                {dir > 0 ? '+' : '-'}
                              </button>
                            ))}
                        </div>
                      </td>
                    )}
                    <td className="px-3.5 py-2">
                      {o.side === 'buy' ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-bold bg-rose-950/80 text-rose-400 border border-rose-800/60">
                          <ArrowUpCircle className="w-3 h-3" />
                          买入
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-950/80 text-emerald-400 border border-emerald-800/60">
                          <ArrowDownCircle className="w-3 h-3" />
                          卖出
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2">
                      <div className="text-cyan-300 font-bold">{o.ticker}</div>
                      <div className="text-slate-400 font-sans text-[11px]">{o.name}</div>
                    </td>
                    <td className="px-3 py-2 text-slate-400 font-sans">{BOARD_CN[o.board as Board] ?? o.board}</td>
                    <td className={`px-3 py-2 text-right ${o.shares === 0 ? 'text-amber-400' : 'text-slate-100'}`} data-testid={`shares-${o.ticker}`}>
                      {(o.shares || 0).toLocaleString()}
                      {o.shares !== o.base && <span className="text-[10px] text-cyan-400 ml-1">({(o.base || 0).toLocaleString()})</span>}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">{o.ref_close?.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right text-slate-200">¥{(o.gross || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                    <td className="px-3 py-2 text-right text-slate-400">{o.rank_in_target}</td>
                    <td className="px-3 py-2 text-center" title={audit.issues.join('\n')}>
                      {audit.severity === 'danger' ? (
                        <ShieldAlert className="w-4 h-4 text-rose-400 inline" />
                      ) : audit.severity === 'caution' ? (
                        <Info className="w-4 h-4 text-amber-400 inline" />
                      ) : (
                        <ShieldCheck className="w-4 h-4 text-emerald-500 inline" />
                      )}
                      {audit.issues.length > 0 && <span className="text-[10px] text-slate-400 ml-1">{audit.issues.length}</span>}
                    </td>
                    {isHistorical &&
                      [1, 5, 20].map((h) => (
                        <td key={h} className="px-3 py-2 text-right text-[11px]">
                          {!f ? (
                            '…'
                          ) : f.status === 'pending' ? (
                            <span className="text-slate-500 font-sans">未到期</span>
                          ) : f.status !== 'filled' ? (
                            <span className="text-amber-400 font-sans">{f.status === 'limit_up_at_open' ? '开盘涨停买不进' : '入场日停牌'}</span>
                          ) : f[`ret_${h}`] === null ? (
                            <span className="text-slate-500 font-sans">未到期</span>
                          ) : (
                            <span>
                              <span className={(f[`ret_${h}`] as number) >= 0 ? 'text-rose-400' : 'text-emerald-400'}>{pct(f[`ret_${h}`] as number)}</span>
                              <span className="text-slate-500"> / </span>
                              <span className={(f[`excess_pool_${h}`] as number) >= 0 ? 'text-rose-300' : 'text-emerald-300'}>
                                {pct(f[`excess_pool_${h}`] as number)}
                              </span>
                            </span>
                          )}
                        </td>
                      ))}
                    <td className="px-3 py-2 text-center" onClick={(e) => e.stopPropagation()}>
                      {onAddToCompare && (
                        <button
                          onClick={() => onAddToCompare(o.ticker)}
                          className="p-1 rounded hover:bg-slate-800 text-purple-300"
                          title="加入 PK"
                        >
                          <Swords className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </td>
                  </tr>
                );
              }
            )}
          </tbody>
        </table>
      </div>
      <p className="text-[10px] text-slate-500">
        A 股配色：红涨绿跌。超额 = 个股收益 − 研究池（PIT 成交额前 1500）等权收益，同一入场和退出口径。红队规则 v2 = 大脑标记、重大事件、风险卡指标在当日截面前 10%；只作提示与事后评估，不改变委托。
      </p>
    </div>
  );
};

const Metric: React.FC<{ label: string; value: string; tone?: 'ok' | 'bad'; testid?: string }> = ({ label, value, tone, testid }) => (
  <div className="p-2.5 rounded bg-slate-950/80 border border-slate-800">
    <div className="text-slate-400 text-[10px] font-sans">{label}</div>
    <div className={`font-bold text-sm mt-0.5 ${tone === 'bad' ? 'text-rose-400' : 'text-white'}`} data-testid={testid}>
      {value}
    </div>
  </div>
);
