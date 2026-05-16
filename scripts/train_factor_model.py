#!/usr/bin/env python3
"""Phase 3 — LightGBM LambdaRank 因子模型训练与 Walk-Forward 评估.

流程：
  1. 加载训练面板（data/features/csi300_2019Q1_2024Q2.parquet）
  2. 从 top_factors.json 读取「宽松」因子集（|IC_IR| >= 0.2，约 11 个）
  3. Walk-Forward 扩展窗口训练（12 折，2021Q3 → 2024Q2）
  4. 评估指标：Rank IC、Top-K 命中率、分层收益差
  5. 保存最终模型 → models/lgbm_ranker.pkl
  6. 输出 HTML 评估报告 → reports/model_evaluation.html

用法::

    cd /path/to/quantmind
    python scripts/train_factor_model.py
    python scripts/train_factor_model.py --ir-threshold 0.2 --min-train 10
    python scripts/train_factor_model.py --label forward_return_63d --no-early-stop
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 路径常量 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_PANEL = FEATURES_DIR / "csi300_2019Q1_2024Q2.parquet"
DEFAULT_TOP_FACTORS = FEATURES_DIR / "top_factors.json"
DEFAULT_MODEL_OUT = MODELS_DIR / "lgbm_ranker.pkl"
DEFAULT_REPORT_OUT = REPORTS_DIR / "model_evaluation.html"

# 全 NaN 因子（直接排除）
KNOWN_NAN_FACTORS = {"north_bound_30d_net_inflow"}


# ============================================================================
# 因子自动发现
# ============================================================================


def find_panel_file(hint: str | None) -> Path:
    if hint:
        p = Path(hint)
        for base in [Path.cwd(), PROJECT_ROOT]:
            full = base / p if not p.is_absolute() else p
            if full.exists():
                return full
        raise FileNotFoundError(f"Panel not found: {hint}")
    if DEFAULT_PANEL.exists():
        return DEFAULT_PANEL
    candidates = sorted(FEATURES_DIR.glob("csi300_*.parquet"))
    candidates = [c for c in candidates if "top20" not in c.name and "smoke" not in c.name]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No panel found in {FEATURES_DIR}")


def find_top_factors_file(hint: str | None) -> Path:
    if hint:
        p = Path(hint)
        return (PROJECT_ROOT / p) if not p.is_absolute() else p
    if DEFAULT_TOP_FACTORS.exists():
        return DEFAULT_TOP_FACTORS
    raise FileNotFoundError(f"top_factors.json not found at {DEFAULT_TOP_FACTORS}")


# ============================================================================
# Plotly 报告
# ============================================================================

_CSS = """
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
    margin: 0; padding: 24px; background: #f8f9fa; color: #333;
  }
  h1 { color: #1a252f; border-bottom: 4px solid #27ae60; padding-bottom: 12px; margin-top: 0; }
  .stats-row { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0; }
  .stat-card {
    background: white; border-radius: 10px; padding: 18px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); min-width: 140px; text-align: center;
  }
  .stat-val { font-size: 2.1em; font-weight: 700; color: #27ae60; line-height: 1.1; }
  .stat-val.blue  { color: #2980b9; }
  .stat-val.orange { color: #e67e22; }
  .stat-val.red   { color: #e74c3c; }
  .stat-lbl { font-size: 0.82em; color: #777; margin-top: 4px; }
  .feature-box {
    background: white; border-radius: 10px; padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px;
  }
  .feature-box h2 { margin-top: 0; }
  .feat-chip {
    display: inline-block; margin: 4px; padding: 4px 12px;
    background: #27ae60; color: white; border-radius: 16px; font-size: 12px;
  }
  .tab-wrap {
    background: white; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden;
  }
  .tab-nav { display: flex; overflow-x: auto; background: #1a252f; }
  .tab-btn {
    padding: 13px 20px; border: none; background: transparent;
    color: rgba(255,255,255,0.6); cursor: pointer; font-size: 13.5px;
    white-space: nowrap; border-bottom: 3px solid transparent; transition: all 0.15s;
  }
  .tab-btn:hover { color: rgba(255,255,255,0.9); }
  .tab-btn.active { color: white; border-bottom-color: #27ae60; font-weight: 600; }
  .tab-pane { display: none; padding: 16px; }
  .tab-pane.active { display: block; }
  footer { margin-top: 32px; color: #999; font-size: 0.8em; text-align: center; }
</style>
"""

_JS = """<script>
function showTab(n) {
  document.querySelectorAll('.tab-pane').forEach((el, i) => el.classList.toggle('active', i === n));
  document.querySelectorAll('.tab-btn').forEach((el, i) => el.classList.toggle('active', i === n));
}
</script>"""


def _fig_ic_series(result) -> "go.Figure":
    import plotly.graph_objects as go

    ic = result.ic_series
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in ic.values]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(d)[:10] for d in ic.index],
        y=ic.values,
        marker_color=colors,
        text=[f"{v:+.4f}" for v in ic.values],
        textposition="outside",
        name="Rank IC",
    ))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.add_hline(y=result.ic_mean, line_dash="dash", line_color="#2980b9",
                  annotation_text=f"均值={result.ic_mean:+.4f}", annotation_position="top right")
    fig.update_layout(
        title=f"Walk-Forward Rank IC 时间序列<br><sup>Spearman(预测分数, 实际收益) | 均值={result.ic_mean:+.4f}, IR={result.ic_ir:+.4f}</sup>",
        xaxis_title="测试截面日期",
        yaxis_title="Rank IC",
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _fig_quintile_returns(result) -> "go.Figure":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    folds_with_qret = [f for f in result.folds if f.quintile_returns]
    if not folds_with_qret:
        fig = go.Figure()
        fig.update_layout(title="分层收益（无数据）")
        return fig

    # 计算每个分位的平均收益（跨折叠）
    n_q = 5
    q_labels = [f"Q{i}" for i in range(1, n_q + 1)]
    avg_qret: dict[str, list[float]] = {ql: [] for ql in q_labels}
    for f in folds_with_qret:
        for ql in q_labels:
            v = f.quintile_returns.get(ql)
            if v is not None and not np.isnan(v):
                avg_qret[ql].append(v)

    means = [float(np.nanmean(avg_qret[ql])) * 100 if avg_qret[ql] else 0.0 for ql in q_labels]
    palette = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#2980b9"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["平均分层收益（跨所有测试期）", "各测试期分层收益热力图"],
        column_widths=[0.4, 0.6],
    )

    # 左：平均分层收益柱状图
    fig.add_trace(
        go.Bar(
            x=q_labels, y=means,
            marker_color=palette,
            text=[f"{v:.3f}%" for v in means],
            textposition="outside",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # 右：热力图（每折叠 × 分位）
    heat_z = []
    heat_y = []
    for f in folds_with_qret:
        row_vals = [f.quintile_returns.get(ql, float("nan")) * 100 for ql in q_labels]
        heat_z.append(row_vals)
        heat_y.append(str(f.test_date)[:10])

    fig.add_trace(
        go.Heatmap(
            z=heat_z, x=q_labels, y=heat_y,
            colorscale="RdYlGn", zmid=0,
            text=[[f"{v:.2f}%" if not np.isnan(v) else "" for v in row] for row in heat_z],
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="%"),
        ),
        row=1, col=2,
    )
    fig.update_layout(
        title="模型分层收益分析（Q1=最低预测分，Q5=最高预测分）",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _fig_feature_importance(result) -> "go.Figure":
    import plotly.graph_objects as go

    # 聚合所有折叠的特征重要性（平均）
    all_imps: dict[str, list[float]] = {}
    for f in result.folds:
        for fname, val in f.feature_importances.items():
            all_imps.setdefault(fname, []).append(val)

    if not all_imps:
        fig = go.Figure()
        fig.update_layout(title="特征重要性（无数据）")
        return fig

    avg_imp = {k: float(np.mean(v)) for k, v in all_imps.items()}
    avg_imp = dict(sorted(avg_imp.items(), key=lambda x: x[1]))

    fig = go.Figure(
        go.Bar(
            x=list(avg_imp.values()),
            y=list(avg_imp.keys()),
            orientation="h",
            marker_color="#2980b9",
            text=[f"{v:.3f}" for v in avg_imp.values()],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="特征重要性（按 Gain 归一化，跨折叠平均）",
        xaxis_title="Importance (normalized)",
        height=max(400, len(avg_imp) * 35 + 150),
        margin=dict(l=200),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _fig_top_k_hit_rate(result) -> "go.Figure":
    import plotly.graph_objects as go

    dates = [str(f.test_date)[:10] for f in result.folds]
    top10 = [f.top10_hit_rate * 100 for f in result.folds]
    top20 = [f.top20_hit_rate * 100 for f in result.folds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=top10, mode="lines+markers", name="Top-10 命中率",
        line=dict(color="#e74c3c", width=2), marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=top20, mode="lines+markers", name="Top-20 命中率",
        line=dict(color="#2980b9", width=2), marker=dict(size=7),
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="随机=50%")
    avg10 = float(np.nanmean([f.top10_hit_rate for f in result.folds])) * 100
    avg20 = float(np.nanmean([f.top20_hit_rate for f in result.folds])) * 100
    fig.add_hline(y=avg10, line_dash="dot", line_color="#e74c3c",
                  annotation_text=f"Top10均值={avg10:.1f}%", annotation_position="top left")
    fig.update_layout(
        title=f"Top-K 命中率（超越同期中位数收益）<br><sup>Top10均值={avg10:.1f}%, Top20均值={avg20:.1f}%</sup>",
        yaxis_title="命中率 (%)",
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _fig_summary_table(result) -> "go.Figure":
    import plotly.graph_objects as go

    df = result.summary_df()
    ic_vals = [f.rank_ic for f in result.folds]
    colors = [
        "#d5f5e3" if ic > 0 else "#fde8e8"
        for ic in ic_vals
    ]
    cell_fill = [colors] * len(df.columns)

    fig = go.Figure(
        go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in df.columns],
                fill_color="#27ae60",
                font=dict(color="white", size=12),
                align="center",
                height=32,
            ),
            cells=dict(
                values=[df[c].tolist() for c in df.columns],
                fill_color=cell_fill,
                align=["center"] * len(df.columns),
                height=27,
                font=dict(size=11),
            ),
        )
    )
    fig.update_layout(title="Walk-Forward 各折叠评估详情", height=700)
    return fig


def build_html_report(
    result,
    feature_cols: list[str],
    panel_path: Path,
    top_factors_path: Path,
    model_path: Path,
) -> str:
    import plotly.io as pio

    tabs = [
        ("Rank IC 时序",    _fig_ic_series(result)),
        ("Top-K 命中率",    _fig_top_k_hit_rate(result)),
        ("分层收益",         _fig_quintile_returns(result)),
        ("特征重要性",       _fig_feature_importance(result)),
        ("折叠明细表",       _fig_summary_table(result)),
    ]

    # 颜色：有效 IC IR 正负
    eff_ic_ir = result.effective_ic_ir
    ic_ir_color = "green" if eff_ic_ir > 0 else "red"
    direction = getattr(result, "direction", 1)
    regime_note = getattr(result, "regime_note", "")

    parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="UTF-8">',
        '<title>QuantMind — LightGBM Factor Model Evaluation</title>',
        _CSS,
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
        "</head><body>",
        "<h1>🤖 QuantMind — LightGBM 因子排序模型评估 (Phase 3)</h1>",
        f"<p style='color:#666;margin-top:-10px'>"
        f"面板：{panel_path.name} &nbsp;|&nbsp; "
        f"特征：{len(feature_cols)} 个 &nbsp;|&nbsp; "
        f"标签：{result.label_col} &nbsp;|&nbsp; "
        f"折叠数：{result.n_folds}</p>",
    ]

    # Regime 横幅（仅 direction=-1 时显示）
    if direction == -1 and regime_note:
        parts.append(
            f'<div style="background:#fff3cd;border-left:5px solid #e67e22;'
            f'padding:14px 20px;border-radius:6px;margin-bottom:16px;">'
            f'<strong>⚠ Regime 提示：模型信号已反转（auto_flip=True，direction=-1）</strong><br>'
            f'<span style="font-size:0.9em;color:#555">{regime_note}</span>'
            f'</div>'
        )

    # 统计卡片
    parts.append('<div class="stats-row">')
    def card(val, label, cls=""):
        return (f'<div class="stat-card">'
                f'<div class="stat-val {cls}">{val}</div>'
                f'<div class="stat-lbl">{label}</div></div>')

    dir_label = f"direction={direction:+d}"
    parts += [
        card(f"{result.ic_mean:+.4f}", "IC 均值（原始）", "red" if result.ic_mean < 0 else ""),
        card(f"{result.effective_ic_mean:+.4f}", f"IC 均值（有效，{dir_label}）"),
        card(f"{result.ic_std:.4f}", "IC 标准差", "blue"),
        card(f"{eff_ic_ir:+.4f}", "IC IR（有效）", ic_ir_color),
        card(f"{result.ic_win_rate:.1%}", "IC 胜率", "blue"),
        card(f"{result.avg_spread*100:+.3f}%", "Q5-Q1 均值", "orange"),
        card(f"{result.n_folds}", "测试折叠数", "blue"),
    ]
    parts.append("</div>")

    # 特征列表
    parts.append('<div class="feature-box"><h2>📌 训练特征</h2>')
    for f in feature_cols:
        parts.append(f'<span class="feat-chip">{f}</span>')
    parts += [
        f"<p style='color:#666;font-size:0.85em;margin-top:12px'>"
        f"模型保存路径：{model_path}  &nbsp;|&nbsp;  因子来源：{top_factors_path.name}</p>",
        "</div>",
    ]

    # Tabs
    parts.append('<div class="tab-wrap"><div class="tab-nav">')
    for i, (name, _) in enumerate(tabs):
        cls = "active" if i == 0 else ""
        parts.append(f'<button class="tab-btn {cls}" onclick="showTab({i})">{name}</button>')
    parts.append("</div>")
    for i, (_, fig) in enumerate(tabs):
        cls = "active" if i == 0 else ""
        div = pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})
        parts.append(f'<div class="tab-pane {cls}" id="p{i}">{div}</div>')
    parts.append("</div>")

    parts += [
        "<footer>QuantMind Phase 3 — LightGBM LambdaRank | scripts/train_factor_model.py</footer>",
        _JS,
        "</body></html>",
    ]
    return "\n".join(parts)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 — LightGBM LambdaRank 因子模型训练与 Walk-Forward 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--panel", default=None, help="面板 parquet 路径（默认自动发现）")
    p.add_argument("--top-factors", default=None, dest="top_factors",
                   help="top_factors.json 路径（默认自动发现）")
    p.add_argument("--label", default="forward_return_21d",
                   choices=["forward_return_21d", "forward_return_63d"],
                   help="目标标签（默认 forward_return_21d = 1个月）")
    p.add_argument("--ir-threshold", type=float, default=0.2, dest="ir_threshold",
                   help="宽松因子选择 |IC_IR| 阈值（默认 0.2）")
    p.add_argument("--filter-start", type=str, default=None, dest="filter_start",
                   help="仅保留 as_of >= 该日期的截面（YYYY-MM-DD）")
    p.add_argument("--filter-end", type=str, default=None, dest="filter_end",
                   help="仅保留 as_of <= 该日期的截面（YYYY-MM-DD）")
    p.add_argument("--flip-negative-ic", action="store_true", dest="flip_negative_ic",
                   help="对 IC_mean < 0 的因子取负值，使所有输入特征均与收益正相关")
    p.add_argument("--min-train", type=int, default=10, dest="min_train",
                   help="Walk-Forward 最少训练截面数（默认 10）")
    p.add_argument("--n-val", type=int, default=2, dest="n_val",
                   help="验证截面数（用于 early stopping，默认 2）")
    p.add_argument("--n-estimators", type=int, default=300, dest="n_estimators")
    p.add_argument("--num-leaves", type=int, default=15, dest="num_leaves")
    p.add_argument("--lr", type=float, default=0.05, dest="lr")
    p.add_argument("--no-early-stop", action="store_true", dest="no_early_stop",
                   help="禁用 early stopping（固定 n_estimators 轮）")
    p.add_argument("--no-auto-flip", action="store_true", dest="no_auto_flip",
                   help="禁用 auto_flip（不根据负 IC 自动反转信号方向）")
    p.add_argument("--model-out", default=None, dest="model_out",
                   help="模型保存路径（默认 models/lgbm_ranker.pkl）")
    p.add_argument("--report-out", default=None, dest="report_out",
                   help="报告保存路径（.html 文件，默认 reports/model_evaluation.html）")
    return p.parse_args()


def main() -> None:
    from quantmind.models.factor_model import FactorSelector, build_lgbm_arrays
    from quantmind.models.lgbm_ranker import LGBMRankerModel, walk_forward_evaluate

    args = parse_args()

    print(f"\n{'='*65}")
    print("QuantMind — Phase 3  LightGBM LambdaRank 因子排序模型")
    print(f"{'='*65}")

    # ── 路径 ────────────────────────────────────────────────────────────────
    panel_path = find_panel_file(args.panel)
    top_factors_path = find_top_factors_file(args.top_factors)
    model_out = Path(args.model_out) if args.model_out else DEFAULT_MODEL_OUT
    report_out = Path(args.report_out) if args.report_out else DEFAULT_REPORT_OUT
    model_out = model_out if model_out.is_absolute() else PROJECT_ROOT / model_out
    report_out = report_out if report_out.is_absolute() else PROJECT_ROOT / report_out

    print(f"面板  : {panel_path}")
    print(f"因子  : {top_factors_path}")
    print(f"标签  : {args.label}")
    print(f"IR阈值: |IC_IR| >= {args.ir_threshold}")

    # ── 加载面板 ─────────────────────────────────────────────────────────────
    print("\n[1/5] 加载训练面板 …")
    panel = pd.read_parquet(panel_path)
    print(f"  shape={panel.shape}")

    # 日期过滤（regime 切分）
    dates = panel.index.get_level_values("as_of")
    mask = pd.Series(True, index=panel.index)
    if args.filter_start:
        fs = pd.Timestamp(args.filter_start)
        mask &= dates >= fs
        print(f"  filter_start={args.filter_start}，过滤后 {mask.sum()} 行")
    if args.filter_end:
        fe = pd.Timestamp(args.filter_end)
        mask &= dates <= fe
        print(f"  filter_end={args.filter_end}，过滤后 {mask.sum()} 行")
    if args.filter_start or args.filter_end:
        panel = panel.loc[mask]
        print(f"  过滤后 shape={panel.shape}")

    n_null = panel[args.label].isna().sum()
    if n_null:
        print(f"  ⚠  {n_null} 行标签为 NaN，将在训练中自动跳过")

    # ── 因子选择 ─────────────────────────────────────────────────────────────
    print("\n[2/5] 选择特征因子 …")
    selector = FactorSelector(top_factors_path)
    feature_cols = selector.select_relaxed(
        ir_threshold=args.ir_threshold,
        exclude=list(KNOWN_NAN_FACTORS),
    )
    # 只保留面板中实际存在的列
    feature_cols = [f for f in feature_cols if f in panel.columns]
    print(f"  选中 {len(feature_cols)} 个特征（|IC_IR| >= {args.ir_threshold}）:")
    stats_df = selector.get_stats(feature_cols)
    for fname in feature_cols:
        if fname in stats_df.index:
            s = stats_df.loc[fname]
            print(f"    {fname:<40}  IC={s.get('ic_mean', 0):+.4f}  IR={s.get('ic_ir', 0):+.4f}")

    # 负向因子取负（--flip-negative-ic）
    flip_cols: list[str] = []
    if args.flip_negative_ic:
        for fname in feature_cols:
            ic_mean = stats_df.loc[fname, "ic_mean"] if fname in stats_df.index else 0.0
            if ic_mean < 0:
                flip_cols.append(fname)
        if flip_cols:
            panel = panel.copy()  # 复制一次，避免修改原始数据
            for fname in flip_cols:
                panel[fname] = -panel[fname]
        print(f"  --flip-negative-ic: 已对 {len(flip_cols)} 个负向因子取负")
        if flip_cols:
            print(f"    {flip_cols}")

    # ── Walk-Forward 训练 ─────────────────────────────────────────────────────
    print(f"\n[3/5] Walk-Forward 扩展窗口训练（min_train={args.min_train} 截面）…")

    early_stopping_rounds = 0 if args.no_early_stop else 50
    model_kwargs = dict(
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        learning_rate=args.lr,
        early_stopping_rounds=early_stopping_rounds,
    )

    result, final_model = walk_forward_evaluate(
        panel,
        feature_cols=feature_cols,
        label_col=args.label,
        min_train_periods=args.min_train,
        n_val_periods=args.n_val,
        model_kwargs=model_kwargs,
        auto_flip=not args.no_auto_flip,
        verbose=True,
    )

    # ── 保存模型 ─────────────────────────────────────────────────────────────
    print("\n[4/5] 保存最终模型 …")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    final_model._feature_names = feature_cols
    final_model.save(model_out)
    print(f"  ✅ 模型 → {model_out}")

    # 保存元数据
    as_of_vals = panel.index.get_level_values("as_of")
    meta = {
        "feature_cols": feature_cols,
        "flip_cols": flip_cols,
        "label_col": args.label,
        "ir_threshold": args.ir_threshold,
        "filter_start": args.filter_start,
        "filter_end": args.filter_end,
        "date_range": [str(as_of_vals.min().date()), str(as_of_vals.max().date())],
        "n_folds": result.n_folds,
        "ic_mean": round(result.ic_mean, 6),
        "ic_ir": round(result.ic_ir, 4),
        "effective_ic_mean": round(result.effective_ic_mean, 6),
        "effective_ic_ir": round(result.effective_ic_ir, 4),
        "direction": result.direction,
        "regime_note": result.regime_note,
        "ic_win_rate": round(result.ic_win_rate, 4),
        "avg_spread_pct": round(result.avg_spread * 100, 4),
        "panel_path": str(panel_path),
        "top_factors_path": str(top_factors_path),
        "model_params": model_kwargs,
    }
    meta_path = model_out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  ✅ 元数据 → {meta_path}")

    # ── 构建 HTML 报告 ────────────────────────────────────────────────────────
    print("\n[5/5] 构建 HTML 评估报告 …")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    html = build_html_report(
        result=result,
        feature_cols=feature_cols,
        panel_path=panel_path,
        top_factors_path=top_factors_path,
        model_path=model_out,
    )
    report_out.write_text(html, encoding="utf-8")
    print(f"  ✅ 报告 → {report_out}")

    # ── 最终汇总 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"🎉  Phase 3 完成")
    print(f"{'='*65}")
    print(f"  模型评估结果（{result.n_folds} 折 Walk-Forward，2021Q3–2024Q2）")
    print(f"  IC 均值（原始）: {result.ic_mean:+.4f}")
    print(f"  IC 均值（有效）: {result.effective_ic_mean:+.4f}  direction={result.direction:+d}")
    print(f"  IC IR  （有效）: {result.effective_ic_ir:+.4f}")
    print(f"  IC 胜率        : {result.ic_win_rate:.1%}")
    print(f"  Q5-Q1 均值     : {result.avg_spread*100:+.3f}%")
    eff_ic = result.effective_ic_mean
    eff_ir = result.effective_ic_ir
    ic_status = "✅ 有效" if abs(eff_ic) > 0.02 else "⚠ 较弱（|IC|<0.02）"
    ir_status = "✅ 稳定" if abs(eff_ir) > 0.3 else "⚠ 不稳定（|IR|<0.3）"
    print(f"  IC 质量        : {ic_status}")
    print(f"  IR 质量        : {ir_status}")
    if result.regime_note:
        print(f"\n  ⚠  Regime: {result.regime_note[:120]}…")
    print(f"\n  下一步：Phase 4 — LLM Listwise Rerank + DPO 对齐")
    print(f"  报告: open {report_out}")


if __name__ == "__main__":
    main()
