#!/usr/bin/env python
"""scripts/run_wf_full_v1.py — 全量 Walk-Forward 评估（面板 v5 + PurgedWalkForwardSplit）.

参数说明（规格修正）：
  用户规格 H=4週/E=2週：E(10) < H(20) → 触发 ValueError。
  实际使用 H=12（= forward_return_12d 标签窗口），E=20（4 周 ≥ H，更保守），
  rolling 模式 lookback=756（≈3 年）。

用法：
  python scripts/run_wf_full_v1.py
  python scripts/run_wf_full_v1.py --fast  # 只用前 6 个 fold 快速验证
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quantmind.data.lake import load_trading_calendar
from quantmind.features.weekly_panel import WHITELIST_35, OUT_PATH
from quantmind.backtest.wf_split import PurgedWalkForwardSplit
from quantmind.backtest.wf_gate import (
    LGBMPredictor, CarryForwardLabelProbe,
    evaluate_oos_ic, purge_ablation, evaluate_gate, gate_meta,
)
from quantmind.backtest.wf_metrics import (
    per_rebalance_win_rate, net_excess_annualized, aggregate_ic,
    quantile_monotonicity,
)

# ============================================================================
# 参数
# ============================================================================
HORIZON = 12        # 交易日，= forward_return_12d
EMBARGO = 20        # 交易日，4 周 ≥ H（用户规格 E=2週=10 < H=12，已修正）
ROLLING_LOOKBACK = 756  # ≈3 年
N_VAL = 2
LABEL_COL = "forward_return_12d"
# 自 categorical 修复（factor_model.detect/encode_categorical_codes + LGBMRankerModel.fit
# 接受 categorical_feature 参数）起，可直接用全部 35 个特征（含 exposure_industry/area）。
# build_lgbm_arrays 内部自动检测并编码 string 列，无需外层 label-encode。
FEAT_COLS = list(WHITELIST_35)   # 35 特征（v2 起含 2 个 categorical）
OOS_START = pd.Timestamp("2022-01-01")
N_ESTIMATORS = 400
LEARNING_RATE = 0.05
NUM_LEAVES = 31

REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_OUT_HTML = REPORTS_DIR / "wf_full_run_v1.html"

# v1 旧基线（33 数值特征，无 categorical），用于 v2 报告里的对比表
# 数据来源：上次执行 scripts/run_wf_full_v1.py（categorical 修复前）的实测输出
V1_BASELINE = {
    "ic_mean": -0.0031,
    "ic_ir": -0.0285,
    "win_rate": 0.4718,
    "net_excess_annual": -0.0282,
    "max_drawdown": 0.2159,
    "monotonicity": -0.0592,
    "passed": False,
    "n_features": 33,
    "label": "v1 (33数值特征, no categorical)",
}


# ============================================================================
# 加载数据
# ============================================================================
def load_data():
    print(f"[load] 面板路径: {OUT_PATH}")
    panel = pd.read_parquet(OUT_PATH)
    print(f"[load] shape={panel.shape}  label={LABEL_COL}  features={len(FEAT_COLS)}")
    cal = load_trading_calendar()
    as_of_grid = sorted(panel.index.get_level_values("as_of").unique())
    print(f"[load] as_of: {as_of_grid[0].date()} → {as_of_grid[-1].date()}  n={len(as_of_grid)}")
    return panel, cal, as_of_grid


# ============================================================================
# 构造季度 cutoff（取交易日历中最近的交易日）
# ============================================================================
def make_quarterly_cutoffs(cal: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    cal_set = set(cal)
    cal_sorted = sorted(cal)
    quarterly = pd.date_range(start, end, freq="QS-JAN")  # 每季初
    cutoffs = []
    for qd in quarterly:
        after = [d for d in cal_sorted if d >= qd]
        if after:
            cutoffs.append(after[0])
    return cutoffs


# ============================================================================
# 主评估循环
# ============================================================================
def run_evaluation(panel, cal, as_of_grid, *, fast: bool = False):
    OOS_END = as_of_grid[-1]

    splitter = PurgedWalkForwardSplit(
        cal, horizon=HORIZON, embargo=EMBARGO,
        mode="rolling", rolling_lookback_td=ROLLING_LOOKBACK, n_val=N_VAL,
    )

    cutoffs = make_quarterly_cutoffs(cal, OOS_START, pd.Timestamp("2025-10-01"))
    if fast:
        cutoffs = cutoffs[:6]
    print(f"[wf] cutoffs={len(cutoffs)}  OOS={OOS_START.date()}→{OOS_END.date()}")

    folds = splitter.make_folds(as_of_grid, cutoffs, oos_start=OOS_START, oos_end=OOS_END)

    all_ics: dict[pd.Timestamp, float] = {}
    all_mono: dict[str, float] = {}
    fold_rows: list[dict] = []
    per_period_port: dict[pd.Timestamp, float] = {}
    per_period_bench: dict[pd.Timestamp, float] = {}

    t0 = time.perf_counter()
    for i, fold in enumerate(folds):
        if not fold.test_dates or not fold.train_dates:
            print(f"  [fold {i}] skip (no train/test)")
            continue
        print(f"  [fold {i}] cutoff={fold.cutoff.date()} "
              f"train={len(fold.train_dates)} val={len(fold.val_dates)} "
              f"test={len(fold.test_dates)} ...", end=" ", flush=True)
        try:
            pred = LGBMPredictor(
                n_quantiles=5,
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                num_leaves=NUM_LEAVES,
                verbose=-1,
            )
            result = evaluate_oos_ic(panel, fold, pred, FEAT_COLS, LABEL_COL)
            fold_ic_mean = float(result.ic_series.mean())
            fold_n = int(result.ic_series.notna().sum())
            print(f"IC={fold_ic_mean:+.4f}  dir={result.direction:+d}  n={fold_n}")

            # 合并 IC 序列
            for d, v in result.ic_series.items():
                all_ics[d] = v
            all_mono.update(result.monotonicity)

            # 每换仓期收益（用预测 top-quintile 的 label 均值作组合收益代理）
            for d in fold.test_dates:
                try:
                    xs = panel.xs(d, level="as_of")
                    realized = xs[LABEL_COL].dropna()
                    if len(realized) < 10:
                        continue
                    score = pred.score(panel, d, FEAT_COLS) * result.direction
                    score_a = score.reindex(realized.index).dropna()
                    realized_a = realized.reindex(score_a.index).dropna()
                    if len(realized_a) < 10:
                        continue
                    q80 = score_a.quantile(0.80)
                    top_idx = score_a[score_a >= q80].index
                    port_ret = float(realized_a.loc[top_idx].mean())
                    bench_ret = float(realized_a.mean())
                    per_period_port[d] = port_ret
                    per_period_bench[d] = bench_ret
                except Exception:
                    pass

            fold_rows.append({
                "fold": i,
                "cutoff": str(fold.cutoff.date()),
                "n_train": len(fold.train_dates),
                "n_val": len(fold.val_dates),
                "n_test": len(fold.test_dates),
                "direction": result.direction,
                "ic_mean": round(fold_ic_mean, 5),
                "ic_n": fold_n,
                "train_first": str(fold.train_dates[0].date()),
                "train_last": str(fold.train_dates[-1].date()),
                "test_first": str(fold.test_dates[0].date()) if fold.test_dates else None,
                "test_last": str(fold.test_dates[-1].date()) if fold.test_dates else None,
            })
        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR: {e}")
            print(tb)
            fold_rows.append({"fold": i, "cutoff": str(fold.cutoff.date()),
                               "error": str(e)[:120]})

    elapsed = time.perf_counter() - t0
    print(f"[wf] 完成 {len(fold_rows)} 个 fold，耗时 {elapsed:.1f}s")

    # 汇总 IC
    ic_series = pd.Series(all_ics).sort_index()
    ic_agg = aggregate_ic(ic_series)

    # 胜率 & 净超额
    port_s = pd.Series(per_period_port).sort_index()
    bench_s = pd.Series(per_period_bench).sort_index()
    win_rate = per_rebalance_win_rate(port_s, bench_s)
    periods_per_year = 250.0 / 12.0
    net_excess = net_excess_annualized(port_s, bench_s, periods_per_year=periods_per_year)

    # 最大回撤（基于累积超额）
    excess = port_s - bench_s
    cum_excess = (1 + excess).cumprod()
    rolling_max_val = cum_excess.cummax()
    dd = (cum_excess - rolling_max_val) / rolling_max_val
    max_dd = float(abs(dd.min())) if not dd.empty else float("nan")

    # 单调性均值
    mono_vals = [v for v in all_mono.values() if v == v]
    avg_mono = float(np.mean(mono_vals)) if mono_vals else float("nan")

    metrics = {
        "ic_mean": ic_agg.get("ic_mean", float("nan")),
        "ic_ir": ic_agg.get("ic_ir", float("nan")),
        "net_excess_annual": net_excess,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "monotonicity": avg_mono,
    }

    gate_result = evaluate_gate(metrics)
    meta = gate_meta({
        "n_folds": len([r for r in fold_rows if "error" not in r]),
        "horizon_td": HORIZON,
        "embargo_td": EMBARGO,
        "rolling_lookback_td": ROLLING_LOOKBACK,
        "oos_start": str(OOS_START.date()),
        "oos_end": str(OOS_END.date()),
        "label": LABEL_COL,
        "n_features": len(FEAT_COLS),
        "elapsed_s": round(elapsed, 1),
    })

    return {
        "ic_series": ic_series,
        "ic_agg": ic_agg,
        "fold_rows": fold_rows,
        "metrics": metrics,
        "gate_result": gate_result,
        "meta": meta,
        "port_s": port_s,
        "bench_s": bench_s,
        "folds": folds,
        "cutoffs": cutoffs,
    }


# ============================================================================
# Purge 消融
# ============================================================================
def run_ablation(panel, cal, as_of_grid):
    print("\n[ablation] CarryForwardLabelProbe purge 消融...")
    OOS_END = as_of_grid[-1]
    # 取中间 cutoff
    cutoffs_all = make_quarterly_cutoffs(cal, OOS_START, pd.Timestamp("2025-10-01"))
    if not cutoffs_all:
        return None
    mid = cutoffs_all[len(cutoffs_all) // 2]
    try:
        res = purge_ablation(
            panel, cal, as_of_grid, mid,
            FEAT_COLS, LABEL_COL,
            predictor_factory=CarryForwardLabelProbe,
            horizon=HORIZON, embargo=EMBARGO,
            oos_start=OOS_START, oos_end=OOS_END,
        )
        print(f"  purged IC={res.ic_purged:+.4f}  nopurge IC={res.ic_nopurge:+.4f}  "
              f"inflation={res.inflation:+.4f}  effective={res.purge_effective}")
        return res
    except Exception as e:
        print(f"  ablation failed: {e}")
        return None


# ============================================================================
# HTML 报告
# ============================================================================
def _fmt(v, pct=False, digits=4):
    if v != v or v is None:
        return "<span style='color:#999'>NaN</span>"
    if pct:
        return f"{v*100:.2f}%"
    return f"{v:.{digits}f}"


def _color_ic(v):
    if v != v:
        return "#999"
    if v > 0.04:
        return "#27ae60"
    if v > 0.02:
        return "#f39c12"
    return "#e74c3c"


def _verdict_style(passed: bool) -> tuple[str, str]:
    if passed:
        return "#27ae60", "✅ PASS（乐观上界）"
    return "#e74c3c", "❌ FAIL"


def _check_icon(v) -> str:
    return "✅" if v else "❌"


def _comparison_table_html(m: dict, gate_passed: bool, version_label: str) -> str:
    """v1（旧 33特征）vs 当前结果（v2 35特征+categorical）对比表."""
    def _delta(curr, base):
        if curr != curr or base != base:
            return "<span style='color:#999'>—</span>"
        d = curr - base
        if abs(d) < 1e-6:
            return "<span style='color:#666'>±0</span>"
        col = "#27ae60" if d > 0 else "#e74c3c"
        # 对最大回撤而言，越小越好——方向相反
        return f"<span style='color:{col}'>{d:+.4f}</span>"

    def _delta_pct(curr, base):
        if curr != curr or base != base:
            return "<span style='color:#999'>—</span>"
        d = curr - base
        if abs(d) < 1e-6:
            return "<span style='color:#666'>±0</span>"
        col = "#27ae60" if d > 0 else "#e74c3c"
        return f"<span style='color:{col}'>{d*100:+.2f}pp</span>"

    rows = [
        ("IC 均值",        _fmt(V1_BASELINE["ic_mean"]),         _fmt(m["ic_mean"]),       _delta(m["ic_mean"], V1_BASELINE["ic_mean"])),
        ("IC IR",          _fmt(V1_BASELINE["ic_ir"]),           _fmt(m["ic_ir"]),         _delta(m["ic_ir"], V1_BASELINE["ic_ir"])),
        ("换仓期胜率",      _fmt(V1_BASELINE["win_rate"], pct=True), _fmt(m["win_rate"], pct=True), _delta_pct(m["win_rate"], V1_BASELINE["win_rate"])),
        ("年化净超额",      _fmt(V1_BASELINE["net_excess_annual"], pct=True), _fmt(m["net_excess_annual"], pct=True), _delta_pct(m["net_excess_annual"], V1_BASELINE["net_excess_annual"])),
        ("最大回撤",        _fmt(V1_BASELINE["max_drawdown"], pct=True), _fmt(m["max_drawdown"], pct=True), _delta_pct(V1_BASELINE["max_drawdown"], m["max_drawdown"])),
        ("分位单调性",      _fmt(V1_BASELINE["monotonicity"]),    _fmt(m["monotonicity"]),  _delta(m["monotonicity"], V1_BASELINE["monotonicity"])),
    ]
    body = ""
    for label, v1v, v2v, diff in rows:
        body += f"<tr><td><b>{label}</b></td><td>{v1v}</td><td>{v2v}</td><td>{diff}</td></tr>\n"

    v1_verdict = "❌ FAIL" if not V1_BASELINE["passed"] else "✅ PASS"
    v2_verdict = "✅ PASS（乐观上界）" if gate_passed else "❌ FAIL"
    body += f"""<tr style='background:#fffde7'>
  <td><b>Gate 综合</b></td>
  <td>{v1_verdict}</td>
  <td>{v2_verdict}</td>
  <td>{'改善' if gate_passed and not V1_BASELINE['passed'] else '同' if gate_passed == V1_BASELINE['passed'] else '退化'}</td>
</tr>"""

    return f"""
<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:80%'>
  <tr style='background:#3498db;color:white'>
    <th>指标</th>
    <th>v1（33数值特征，无 categorical）</th>
    <th>{version_label}</th>
    <th>变化</th>
  </tr>
  {body}
</table>
<p style="font-size:12px;color:#777;margin-top:6px">
  注：变化列绿色=改善，红色=恶化。"最大回撤"行用 v1 − v2（回撤越小越好）。其它行用 v2 − v1。
</p>
"""


def generate_html(result: dict, ablation, version_label: str = "v2 (35特征 含 categorical)") -> str:
    metrics = result["metrics"]
    gate = result["gate_result"]
    meta = result["meta"]
    fold_rows = result["fold_rows"]
    ic_series = result["ic_series"]
    ic_agg = result["ic_agg"]

    v_color, v_text = _verdict_style(gate["passed"])

    # IC 序列表格（按 fold 分组）
    fold_table_rows = ""
    for r in fold_rows:
        if "error" in r:
            fold_table_rows += (
                f"<tr style='background:#fff0f0'>"
                f"<td>{r['fold']}</td><td>{r['cutoff']}</td>"
                f"<td colspan='8' style='color:red'>{r['error']}</td></tr>\n"
            )
        else:
            ic_v = r.get("ic_mean", float("nan"))
            ic_col = _color_ic(ic_v)
            fold_table_rows += (
                f"<tr>"
                f"<td>{r['fold']}</td>"
                f"<td>{r['cutoff']}</td>"
                f"<td>{r.get('train_first','')}</td>"
                f"<td>{r.get('train_last','')}</td>"
                f"<td>{r.get('n_train','')}</td>"
                f"<td>{r.get('test_first','')}</td>"
                f"<td>{r.get('test_last','')}</td>"
                f"<td>{r.get('n_test','')}</td>"
                f"<td style='color:{ic_col};font-weight:bold'>{_fmt(ic_v)}</td>"
                f"<td>{r.get('ic_n','')}</td>"
                f"<td>{r.get('direction','')}</td>"
                f"</tr>\n"
            )

    # Gate checks 表格
    checks_rows = ""
    check_labels = {
        "ic_mean":          ("IC 均值",          f"> {0.03:.2f}",         _fmt(metrics.get("ic_mean"))),
        "ic_ir":            ("IC IR",             f"> {0.40:.2f}",         _fmt(metrics.get("ic_ir"))),
        "net_excess_annual":("年化净超额",         f"> {0.05:.0%}",         _fmt(metrics.get("net_excess_annual"), pct=True)),
        "win_rate":         ("换仓期胜率",         f"> {0.52:.0%}",         _fmt(metrics.get("win_rate"), pct=True)),
        "max_drawdown":     ("最大回撤",           f"< {0.25:.0%}",         _fmt(metrics.get("max_drawdown"), pct=True)),
        "monotonicity":     ("分位单调性 Spearman", f"> {0.0:.1f}",          _fmt(metrics.get("monotonicity"))),
    }
    for k, (label, threshold, actual) in check_labels.items():
        passed_k = gate["checks"].get(k, False)
        icon = _check_icon(passed_k)
        row_color = "#f0fff0" if passed_k else "#fff0f0"
        checks_rows += (
            f"<tr style='background:{row_color}'>"
            f"<td>{icon}</td><td>{label}</td><td>{threshold}</td>"
            f"<td><b>{actual}</b></td></tr>\n"
        )

    # Ablation section
    if ablation is not None:
        abl_html = f"""
<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:60%'>
  <tr style='background:#f0f0f0'><th>配置</th><th>IC 均值</th><th>测试截面数</th></tr>
  <tr><td>Purge 开（H=12, E=20）</td>
      <td style='color:{_color_ic(ablation.ic_purged)}'><b>{_fmt(ablation.ic_purged)}</b></td>
      <td>{ablation.purged_n}</td></tr>
  <tr><td>Purge 关（H=0, E=0）</td>
      <td style='color:{_color_ic(ablation.ic_nopurge)}'><b>{_fmt(ablation.ic_nopurge)}</b></td>
      <td>{ablation.nopurge_n}</td></tr>
  <tr style='background:#fffde7'>
      <td><b>IC 抬升（nopurge − purged）</b></td>
      <td><b>{_fmt(ablation.inflation)}</b></td>
      <td>有效：{'✅ 是' if ablation.purge_effective else '❌ 否（purge 未生效）'}</td></tr>
</table>
"""
    else:
        abl_html = "<p style='color:#999'>消融测试未运行或失败。</p>"

    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>WF Full Run v1 — alpha_panel_weekly_v5</title>
<style>
  body {{font-family: 'Source Han Sans SC', 'Microsoft YaHei', sans-serif;
        font-size:14px; margin:30px; color:#333; background:#fafafa}}
  h1 {{color:#2c3e50; border-bottom:3px solid #3498db; padding-bottom:8px}}
  h2 {{color:#2c3e50; margin-top:30px; border-left:4px solid #3498db; padding-left:10px}}
  h3 {{color:#555; margin-top:20px}}
  table {{border-collapse:collapse; width:100%; margin:10px 0}}
  th {{background:#3498db; color:white; padding:8px; text-align:left}}
  td {{padding:6px 8px; border:1px solid #ddd}}
  tr:hover {{background:#f5f5f5}}
  .verdict {{font-size:22px; font-weight:bold; padding:12px 20px;
             border-radius:6px; display:inline-block; color:white; background:{v_color}}}
  .warn {{background:#fff8e1; border-left:4px solid #f39c12;
          padding:10px 15px; margin:10px 0; border-radius:4px}}
  .caveat {{background:#fce4ec; border-left:4px solid #e74c3c;
            padding:10px 15px; margin:10px 0; border-radius:4px; font-size:13px}}
  .metric-box {{display:inline-block; border:1px solid #ccc; border-radius:6px;
                padding:10px 16px; margin:6px; background:white; min-width:140px;
                text-align:center}}
  .metric-box .label {{font-size:12px; color:#777}}
  .metric-box .value {{font-size:20px; font-weight:bold; margin-top:4px}}
  pre {{background:#f4f4f4; padding:10px; border-radius:4px;
        font-size:12px; overflow-x:auto; white-space:pre-wrap}}
  .section {{background:white; border:1px solid #e0e0e0; border-radius:8px;
             padding:20px; margin:20px 0}}
</style>
</head>
<body>

<h1>Walk-Forward 全量评估报告 — {version_label}</h1>
<p>面板：<code>alpha_panel_weekly_v5.parquet</code>（452,439 × 38）&nbsp;|&nbsp;
   标签：<code>forward_return_12d</code>&nbsp;|&nbsp;
   特征：{len(FEAT_COLS)} 列&nbsp;|&nbsp;
   生成：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="warn">
  ⚠️ <b>参数修正说明</b>：用户规格 H=4週(20TD)/E=2週(10TD) → E&lt;H，触发 ValueError。
  实际使用 <b>H=12</b>（= 标签窗口）、<b>E=20</b>（4 周，≥ H），rolling lookback=756。
</div>
<div class="warn">
  ℹ️ <b>v2 特征集说明</b>：本次使用全部 35 个特征（含 <code>exposure_industry</code> 110类 +
  <code>exposure_area</code> 32类）。修复 <code>build_lgbm_arrays</code>：检测 string/object 列后用
  <code>pd.Categorical.cat.codes</code> 转 int32，再以 LightGBM 原生 <code>categorical_feature</code>
  参数声明（非 one-hot）；NaN→-1 单独成类；train/test 编码字典源自全 panel unique，无泄漏。
</div>

<!-- ============ GATE VERDICT ============ -->
<div class="section">
<h2>综合 Gate 结论</h2>
<div class="verdict">{v_text}</div>
<p style="margin-top:10px;color:#777;font-size:13px">过闸 = 乐观上界下的必要条件（非充分）；不过闸 = 强负面信号</p>

<h3>各项指标 vs 通过线</h3>
<table border='1' cellpadding='6' cellspacing='0' style='width:70%'>
  <tr style='background:#3498db;color:white'>
    <th style='width:40px'></th><th>指标</th><th>通过线</th><th>实测值</th>
  </tr>
  {checks_rows}
</table>
</div>

<!-- ============ KEY METRICS ============ -->
<div class="section">
<h2>核心指标汇总</h2>
<div class="metric-box">
  <div class="label">OOS IC 均值</div>
  <div class="value" style="color:{_color_ic(metrics.get('ic_mean',float('nan')))}">{_fmt(metrics.get('ic_mean'))}</div>
</div>
<div class="metric-box">
  <div class="label">IC IR</div>
  <div class="value">{_fmt(metrics.get('ic_ir'))}</div>
</div>
<div class="metric-box">
  <div class="label">换仓期胜率</div>
  <div class="value">{_fmt(metrics.get('win_rate'), pct=True)}</div>
</div>
<div class="metric-box">
  <div class="label">年化净超额</div>
  <div class="value">{_fmt(metrics.get('net_excess_annual'), pct=True)}</div>
</div>
<div class="metric-box">
  <div class="label">最大回撤</div>
  <div class="value">{_fmt(metrics.get('max_drawdown'), pct=True)}</div>
</div>
<div class="metric-box">
  <div class="label">分位单调性</div>
  <div class="value">{_fmt(metrics.get('monotonicity'))}</div>
</div>

<h3>IC 序列统计</h3>
<table border='1' cellpadding='6' cellspacing='0' style='width:50%'>
  <tr style='background:#f0f0f0'><th>统计量</th><th>值</th></tr>
  <tr><td>截面数（n）</td><td>{ic_agg.get('n','N/A')}</td></tr>
  <tr><td>IC 均值</td><td><b>{_fmt(ic_agg.get('ic_mean'))}</b></td></tr>
  <tr><td>IC 标准差</td><td>{_fmt(ic_agg.get('ic_std'))}</td></tr>
  <tr><td>IC IR</td><td><b>{_fmt(ic_agg.get('ic_ir'))}</b></td></tr>
</table>
</div>

<!-- ============ FOLD DETAIL ============ -->
<div class="section">
<h2>各 Fold 明细（{len(fold_rows)} 个 fold）</h2>
<table border='1' cellpadding='5' cellspacing='0'>
  <tr style='background:#3498db;color:white'>
    <th>#</th><th>Cutoff</th>
    <th>Train 起</th><th>Train 止</th><th>N_train</th>
    <th>Test 起</th><th>Test 止</th><th>N_test</th>
    <th>IC 均值</th><th>IC 截面数</th><th>方向</th>
  </tr>
  {fold_table_rows}
</table>
</div>

<!-- ============ PURGE ABLATION ============ -->
<div class="section">
<h2>Purge 消融对比（CarryForwardLabelProbe，中间 cutoff）</h2>
<p>探针"记住"最后训练日的标签并用作预测——purge 关时标签窗口与测试重叠 → IC 虚高；purge 开时窗口不重叠 → IC 应塌回 ≈ 0。</p>
{abl_html}
</div>

<!-- ============ SURVIVORSHIP ============ -->
<div class="section">
<h2>幸存者偏差标注</h2>
<div class="caveat">
  ⚠️ <b>幸存者上界（Optimistic Upper Bound）</b><br>
  {meta.get("survivorship_caveat","").replace(chr(10), "<br>")}
  <br><br>
  {meta.get("deferred","").replace(chr(10), "<br>")}
</div>
</div>

<!-- ============ V1 vs V2 COMPARISON ============ -->
<div class="section">
<h2>v1 旧基线 vs {version_label} 对比</h2>
<p style="color:#555">
  <b>v1</b>：上次跑（33 数值特征，<code>build_lgbm_arrays</code> 不支持 string 列，
  排除 <code>exposure_industry</code> / <code>exposure_area</code>）<br>
  <b>本次</b>：categorical 修复后，全部 35 特征，
  <code>LightGBM</code> 原生 <code>categorical_feature</code>（非 one-hot）
</p>
{_comparison_table_html(metrics, gate["passed"], version_label)}
</div>

<!-- ============ META ============ -->
<div class="section">
<h2>运行 Meta</h2>
<pre>{meta_json}</pre>
</div>

<hr style="margin-top:40px">
<p style="color:#999;font-size:12px">本报告由 scripts/run_wf_full_v1.py 自动生成 |
所有结论在幸存者上界意义下有效 | 退市票回补后须复核</p>

</body>
</html>"""
    return html


# ============================================================================
# 入口
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="只用前 6 个 fold 快速验证")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT_HTML),
                    help="HTML 报告输出路径（默认 reports/wf_full_run_v1.html）")
    ap.add_argument("--version-label", type=str, default="v2 (35特征 含 categorical)",
                    help="报告中的版本标签（用于对比表表头）")
    args = ap.parse_args()
    out_html = Path(args.out)

    print("=" * 72)
    print("Walk-Forward 全量评估")
    print(f"H={HORIZON}td  E={EMBARGO}td  rolling_lookback={ROLLING_LOOKBACK}td  n_val={N_VAL}")
    print(f"标签={LABEL_COL}  特征={len(FEAT_COLS)}")
    if args.fast:
        print("[fast mode] 仅前 6 个 fold")
    print("=" * 72)

    panel, cal, as_of_grid = load_data()
    result = run_evaluation(panel, cal, as_of_grid, fast=args.fast)
    ablation = run_ablation(panel, cal, as_of_grid)

    # ---- 终端关键指标 ----
    m = result["metrics"]
    g = result["gate_result"]
    print("\n" + "=" * 72)
    print("OOS 关键指标")
    print("=" * 72)
    print(f"  IC 均值       : {m.get('ic_mean', float('nan')):+.4f}")
    print(f"  IC IR         : {m.get('ic_ir', float('nan')):+.4f}")
    print(f"  换仓期胜率    : {m.get('win_rate', float('nan')):.2%}")
    print(f"  年化净超额    : {m.get('net_excess_annual', float('nan')):.2%}")
    print(f"  最大回撤      : {m.get('max_drawdown', float('nan')):.2%}")
    print(f"  分位单调性    : {m.get('monotonicity', float('nan')):+.4f}")
    print(f"  Gate 结论     : {'PASS（乐观上界）' if g['passed'] else 'FAIL'}")
    print(f"  各项检查      : { {k:('✅' if v else '❌') for k,v in g['checks'].items()} }")
    if ablation:
        print(f"  Purge 消融    : purged={ablation.ic_purged:+.4f}  nopurge={ablation.ic_nopurge:+.4f}"
              f"  inflation={ablation.inflation:+.4f}  effective={'Yes' if ablation.purge_effective else 'No'}")
    print("=" * 72)
    print(f"\n⚠️  幸存者偏差：面板 v5 全幸存者池，上述指标均为乐观上界。")

    # ---- HTML 报告 ----
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(result, ablation, version_label=args.version_label)
    out_html.write_text(html, encoding="utf-8")
    print(f"\n📊 HTML 报告: {out_html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
