import React, { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import * as echarts from 'echarts';

// 直接用 echarts 的最小封装，替代 echarts-for-react 3.0.6：
// 那个版本在 React 19 下先建一个没有 option 的临时实例、等它的 'finished' 事件再重建，事件不来就一直是空白，
// 只有 props 变化触发更新后才画出来；onChartReady / onEvents 也因此从不执行。
export interface EChartHandle {
  getEchartsInstance: () => echarts.ECharts | null;
}

interface Props {
  option: echarts.EChartsCoreOption;
  style?: React.CSSProperties;
  className?: string;
  notMerge?: boolean;
  onEvents?: Record<string, (params: any) => void>;
  onReady?: (chart: echarts.ECharts) => void | (() => void);
}

export const EChart = forwardRef<EChartHandle, Props>(({ option, style, className, notMerge, onEvents, onReady }, ref) => {
  const el = useRef<HTMLDivElement | null>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const events = useRef(onEvents);
  events.current = onEvents;
  const ready = useRef(onReady);
  ready.current = onReady;

  useImperativeHandle(ref, () => ({ getEchartsInstance: () => chart.current }), []);

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current);
    chart.current = c;
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    const names = Object.keys(events.current || {});
    names.forEach((n) => c.on(n, (p: unknown) => events.current?.[n]?.(p)));
    const undo = ready.current?.(c);
    return () => {
      ro.disconnect();
      if (typeof undo === 'function') undo();
      c.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(option, { notMerge: Boolean(notMerge) });
  }, [option, notMerge]);

  return <div ref={el} style={style} className={className} />;
});
EChart.displayName = 'EChart';
