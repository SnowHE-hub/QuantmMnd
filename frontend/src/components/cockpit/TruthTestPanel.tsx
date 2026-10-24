import React, { useEffect, useState } from 'react';
import { EChart as ReactECharts, EChartHandle } from '../common/EChart';
import { FlaskConical, Loader2 } from 'lucide-react';
import { analyticsApi, ForwardIcResponse, pct, RedTeamResponse } from '../../api/analytics';

interface TruthTestPanelProps {
  selectedDate: string;
}

// 事后验真：模型（ENS 20 日）每个运行日的事后 IC，和红队规则 v1 / v2 在当天研究候选上的事后表现。
// 全部由后端用复权价格、T+1 开盘入场计算；前端只画图。
export const TruthTestPanel: React.FC<TruthTestPanelProps> = ({ selectedDate }) => {
  const [horizon, setHorizon] = useState(5);
  const [ic, setIc] = useState<ForwardIcResponse | null>(null);
  const [rt, setRt] = useState<Record<'v1' | 'v2', RedTeamResponse | null>>({ v1: null, v2: null });
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setIc(null);
    analyticsApi
      .forwardIc(horizon)
      .then((r) => alive && setIc(r))
      .catch((e) => alive && setErr(String(e.message || e)));
    return () => {
      alive = false;
    };
  }, [horizon]);

  useEffect(() => {
    let alive = true;
    setRt({ v1: null, v2: null });
    if (!selectedDate) return;
    Promise.all([analyticsApi.redteam(selectedDate, 'f4', horizon, 'v1'), analyticsApi.redteam(selectedDate, 'f4', horizon, 'v2')])
      .then(([a, b]) => alive && setRt({ v1: a, v2: b }))
      .catch((e) => alive && setErr(String(e.message || e)));
    return () => {
      alive = false;
    };
  }, [selectedDate, horizon]);

  const done = ic?.rows.filter((r) => r.ic !== null) ?? [];
  const option = {
    grid: { left: 48, right: 16, top: 24, bottom: 28 },
    tooltip: {
      trigger: 'axis',
      valueFormatter: (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '—'),
    },
    xAxis: { type: 'category', data: done.map((r) => r.as_of.slice(5)), axisLabel: { color: '#94a3b8', fontSize: 10 } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8', fontSize: 10 }, splitLine: { lineStyle: { color: '#1e293b' } } },
    series: [
      {
        name: `Rank IC（${horizon} 日）`,
        type: 'bar',
        // 每根柱子自己带颜色（A 股红涨绿跌）；正值上端圆角、负值下端圆角，都贴着 0 轴
        data: done.map((r) => ({
          value: r.ic,
          itemStyle: {
            color: (r.ic ?? 0) >= 0 ? '#e11d48' : '#059669',
            borderRadius: (r.ic ?? 0) >= 0 ? [4, 4, 0, 0] : [0, 0, 4, 4],
          },
        })),
        barMaxWidth: 18,
      },
    ],
  };

  return (
    <div className="bg-[#0c1220]/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4" data-testid="truth-test-panel">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-emerald-400" />
          <div>
            <h3 className="text-sm font-bold text-white">事后验真：模型 IC 与红队规则</h3>
            <p className="text-[11px] text-slate-400">T 日信号 → T+1 开盘复权入场 → 持有 h 个交易日；超额相对研究池等权</p>
          </div>
        </div>
        <div className="flex gap-1 text-xs">
          {[1, 5, 10].map((h) => (
            <button
              key={h}
              onClick={() => setHorizon(h)}
              className={`px-2 py-1 rounded ${h === horizon ? 'bg-emerald-700 text-white' : 'bg-slate-900 text-slate-300 border border-slate-700'}`}
            >
              {h} 日
            </button>
          ))}
        </div>
      </div>
      {err && <div className="text-rose-300 text-xs">{err}</div>}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 text-xs">
        <div className="space-y-2">
          <div className="font-semibold text-slate-200">ENS 20 日 · 每个运行日的事后 Rank IC（研究池）</div>
          {!ic ? (
            <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
          ) : done.length === 0 ? (
            <div className="text-slate-500">还没有到期的运行日。</div>
          ) : (
            <>
              <ReactECharts option={option} style={{ height: 180 }} />
              <div className="font-mono text-[11px] text-slate-300" data-testid="ic-summary">
                已到期 {ic.n_dates} 个运行日，平均 IC = {ic.mean_ic?.toFixed(3)}；前 10% 相对研究池平均{' '}
                {pct(done.reduce((s, r) => s + (r.top10_minus_pool ?? 0), 0) / Math.max(1, done.length))}
              </div>
              <div className="text-[10px] text-slate-500">{ic.note}。研究期日均 RankIC 0.145（T-030）；几天的前瞻值波动很大，看趋势，不看单日。</div>
            </>
          )}
        </div>

        <div className="space-y-2">
          <div className="font-semibold text-slate-200">红队规则在当日研究候选上的事后表现（{selectedDate}）</div>
          <table className="w-full font-mono text-[11px]">
            <thead className="text-slate-500">
              <tr>
                <th className="text-left py-1">规则</th>
                <th className="text-right">标记比例</th>
                <th className="text-right">标记组跑输率</th>
                <th className="text-right">未标记跑输率</th>
                <th className="text-right">差值</th>
                <th className="text-right">平均超额差</th>
              </tr>
            </thead>
            <tbody>
              {(['v1', 'v2'] as const).map((v) => {
                const r = rt[v];
                const g = r?.by_group.excess_pool;
                return (
                  <tr key={v} className="border-t border-slate-800" data-testid={`redteam-${v}`}>
                    <td className="py-1 text-slate-300">{v === 'v1' ? 'v1 固定阈值（原前端）' : 'v2 截面前 10%'}</td>
                    <td className="text-right">{r ? pct(r.meta.flagged_share, 0, false) : '…'}</td>
                    <td className="text-right">{g ? pct(g.hit_rate, 0, false) : '…'}</td>
                    <td className="text-right">{g ? pct(g.base_rate, 0, false) : '…'}</td>
                    <td className="text-right">{g ? pct(g.lift, 0) : '…'}</td>
                    <td className="text-right">{g ? pct(g.mean_gap) : '…'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="text-[10px] text-slate-500 leading-relaxed">
            “跑输率”= 事后超额 &lt; 0 的比例。相对等权基准，任何一组股票的跑输率天然在 50% 左右，所以只看标记组的跑输率没有意义，
            要和未标记组比较。v1 在 ENS 选出的高波动小盘股里会标记一大半，几乎没有区分度。
            {rt.v2?.warning ? ` ${rt.v2.warning}。` : ''}
            {rt.v2?.missing_inputs?.length ? ` 这一天缺少规则输入：${rt.v2.missing_inputs.join('、')}（早期运行日还没有这一步），标记比例偏低不代表风险低。` : ''}
            {rt.v2 && rt.v2.meta.entry_date === null ? ' 这一天还没有后续行情。' : ''}
          </div>
        </div>
      </div>
    </div>
  );
};
