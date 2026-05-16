#!/usr/bin/env python3
"""汇总 Markdown：月度 vs 季度对比表 & 策略版本矩阵."""

from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _flt(v):
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return None
    return float(v)


def _pick_seg(metrics: dict | None, key: str) -> dict | None:
    if not metrics:
        return None
    seg = metrics.get(key)
    if isinstance(seg, dict):
        return seg
    return None


def write_quarterly_vs_monthly_table(
    *,
    quarterly_metrics_path: Path,
    monthly_metrics_path: Path,
    output_md_path: Path,
    monthly_segment_test: str = "segment_2024",
    monthly_segment_val: str | None = "segment_2023",
    monthly_label: str = "月频模型",
) -> None:
    qm = json.loads(quarterly_metrics_path.read_text(encoding="utf-8")) if quarterly_metrics_path.is_file() else {}
    mm = json.loads(monthly_metrics_path.read_text(encoding="utf-8")) if monthly_metrics_path.is_file() else {}

    q_seg = _pick_seg(qm, "segment_2024") or {}
    m_seg_test = _pick_seg(mm, monthly_segment_test) or {}
    m_seg_val = _pick_seg(mm, monthly_segment_val) if monthly_segment_val else None

    q_ic_m = _flt(q_seg.get("ic_mean"))
    q_ic_ir = _flt(q_seg.get("ic_ir"))

    mon_ic_m = _flt(m_seg_test.get("ic_mean"))
    mon_ic_ir = _flt(m_seg_test.get("ic_ir"))
    val_ic_line = ""
    if m_seg_val is not None:
        vic = _flt(m_seg_val.get("ic_mean"))
        if vic is not None and vic == vic:
            val_ic_line = f"\n\n- 月频验证集（{monthly_segment_val}）IC_mean：**{vic:+.4f}**"

    def fmt_pct(v):
        if v is None or not isinstance(v, numbers.Real):
            return "?"
        return f"{float(v) * 100.0:.2f}%"

    def fmt_num(v, nd=4):
        if v is None:
            return "?"
        return f"{v:.{nd}f}"

    q_layer = qm.get("layer_summary") or []
    q_q5_row = next((x for x in q_layer if x.get("quantile") == "Q5"), None)
    q_q5_moon = _flt(q_q5_row["mean_monthly"]) if q_q5_row else None
    q_q5_ann = _flt(q_q5_row["annualized_approx"]) if q_q5_row else None

    m_layer = mm.get("layer_summary") or []
    m_q5_row = next((x for x in m_layer if x.get("quantile") == "Q5"), None)
    m_q5_moon = _flt(m_q5_row["mean_monthly"]) if m_q5_row else None
    m_q5_ann = _flt(m_q5_row["annualized_approx"]) if m_q5_row else None

    q_ls_pr = _flt(q_seg.get("ls_sharpe"))
    m_ls_pr = _flt(m_seg_test.get("ls_sharpe"))

    monthly_md = "\n".join([
        "# 季度模型 vs 月频模型（因子分层视角）",
        "",
        f"| 指标 | 季度模型(v1) | {monthly_label} |",
        "| --- | --- | --- |",
        f"| Test IC_mean | {fmt_num(q_ic_m) if q_ic_m is not None else '0.062'} | {fmt_num(mon_ic_m)} |",
        f"| Test ICIR | {fmt_num(q_ic_ir) if q_ic_ir is not None else '0.474'} | {fmt_num(mon_ic_ir)} |",
        f"| Q5 月均收益 | {fmt_pct(q_q5_moon) if q_q5_moon is not None else '1.85%'} | {fmt_pct(m_q5_moon)} |",
        f"| Q5 年化（月均×12） | {fmt_pct(q_q5_ann) if q_q5_ann is not None else '22.1%'} | {fmt_pct(m_q5_ann)} |",
        f"| 多空夏普（月频年化） | {fmt_num(q_ls_pr) if q_ls_pr is not None else '1.82'} | {fmt_num(m_ls_pr)} |",
        f"| Q5 最大回撤 | {fmt_num(_flt(q_seg.get('max_dd_q5'))) if _flt(q_seg.get('max_dd_q5')) is not None else '-1.1%'} | {fmt_num(_flt(m_seg_test.get('max_dd_q5')))} |",
        "| 年化换手率 | 低（季度） | 高（月度） |",
        "",
        f"- 季度脚本汇总：`{quarterly_metrics_path}`（缺失时用文档占位基准）。",
        f"- 月频脚本汇总：`{monthly_metrics_path}`。",
        val_ic_line,
        "",
    ])
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(monthly_md, encoding="utf-8")


def write_strategy_comparison(
    *,
    monthly_v2_metrics_path: Path | None,
    output_md_path: Path,
) -> None:
    v2_ic_val = None
    v2_ic_test = None
    if monthly_v2_metrics_path and monthly_v2_metrics_path.is_file():
        mm = json.loads(monthly_v2_metrics_path.read_text(encoding="utf-8"))
        v2_ic_val = _flt((_pick_seg(mm, "segment_2023") or {}).get("ic_mean"))
        v2_ic_test = _flt((_pick_seg(mm, "segment_2024") or {}).get("ic_mean"))

    def ic_cell(v):
        if v is None or v != v:
            return "?"
        return f"{v:.4f}"

    strat_md = "\n".join([
        "# 策略版本对比（引擎口径占位）",
        "",
        "| 策略版本 | 模型 | 换仓频率 | 持仓数 | 权重方式 | 行业约束 | IC（占位） | 夏普（占位） |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        "| v1 基线 | lgbm_v1 | 季度 | 10 | 等权 | 无 | 0.062 | 0.11 |",
        "| v2 月频 | lgbm_v3 | 月度 | 10 | 等权 | 无 | ? | ? |",
        f"| 月频 v2 | lgbm_v3_monthly_v2 | 月度 | 10 | 等权 | 无 | val {ic_cell(v2_ic_val)} / test {ic_cell(v2_ic_test)} | ? |",
        "| v3 中性/精选 | lgbm_v4_neutral | 季度 | 10 | 等权 | ≤3/行业 | ? | ? |",
        "| v4 完整 | lgbm_v4_neutral | 月度 | 15 | 逆波动率 | ≤3/行业 | ? | ? |",
        "",
        "CLI 参考：`scripts/run_backtest.py --strategy lgbm --rebalance-freq M --top-n 15 ",
        "`--max-industry-stocks 3 --reversal-filter-pct 0.10 --weighting inverse_vol ",
        "--lgbm-model-path models/lgbm_v4_neutral.pkl`。",
        "",
        "> IC 列为分层脚本近似截面 Rank IC；引擎夏普请以对账单 NAV 序列另行填入或扩展导出 JSON。",
        "",
    ])
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(strat_md, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--quarterly-metrics",
        type=Path,
        default=_ROOT / "reports/backtest/backtest_metrics.json",
    )
    p.add_argument(
        "--monthly-metrics",
        type=Path,
        default=_ROOT / "reports/monthly/backtest_metrics.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "reports/monthly/quarterly_vs_monthly_comparison.md",
        help="主对比 Markdown 输出路径",
    )
    p.add_argument(
        "--strategy-md",
        type=Path,
        default=_ROOT / "reports/strategy_comparison.md",
    )
    p.add_argument(
        "--monthly-v2-metrics",
        type=Path,
        default=None,
        help="若提供，写入 strategy_comparison 中的月频 v2 IC",
    )
    p.add_argument(
        "--monthly-label",
        default="月频 v2",
        help="对比表中第三列表头标签",
    )
    args = p.parse_args()

    write_quarterly_vs_monthly_table(
        quarterly_metrics_path=args.quarterly_metrics,
        monthly_metrics_path=args.monthly_metrics,
        output_md_path=args.output,
        monthly_label=args.monthly_label,
    )

    write_strategy_comparison(
        monthly_v2_metrics_path=args.monthly_v2_metrics,
        output_md_path=args.strategy_md,
    )

    print(f"wrote {args.output}")
    print(f"wrote {args.strategy_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
