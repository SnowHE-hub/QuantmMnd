#!/usr/bin/env python3
"""Phase 2.2 — Factor Analysis & Effectiveness Validation.

Computes IC, IC_IR, Win Rate, Quintile Returns, and Factor Correlations
for all 41 factors in the training panel. Outputs:
  - reports/factor_analysis.html   (interactive Plotly report, tabbed)
  - data/features/top_factors.json (selected factors for Phase 2.3 modeling)

Usage:
  cd /path/to/quantmind
  python scripts/analyze_factors.py
  python scripts/analyze_factors.py --panel data/features/myfile.parquet
  python scripts/analyze_factors.py --label forward_return_63d
  python scripts/analyze_factors.py --ic-threshold 0.02 --ir-threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Paths & Thresholds
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_PANEL = FEATURES_DIR / "csi300_2019Q1_2024Q2.parquet"
LABEL_COLS = ["forward_return_21d", "forward_return_63d"]
PRIMARY_LABEL = "forward_return_21d"

DEFAULT_IC_THRESHOLD = 0.02
DEFAULT_IR_THRESHOLD = 0.3
N_QUANTILES = 5


# ============================================================================
# Panel discovery
# ============================================================================


def find_panel_file(hint: str | None = None) -> Path:
    """Locate the training panel parquet. Uses hint if given, then auto-discovers."""
    if hint:
        p = Path(hint)
        if p.is_absolute():
            if not p.exists():
                raise FileNotFoundError(f"Panel not found: {p}")
            return p
        # Relative path: try CWD first, then project root
        for base in [Path.cwd(), PROJECT_ROOT]:
            full = base / p
            if full.exists():
                return full
        raise FileNotFoundError(f"Panel not found at {hint} (tried CWD and project root)")

    # Auto-discover: prefer the canonical file, then any panel_*.parquet
    if DEFAULT_PANEL.exists():
        return DEFAULT_PANEL

    candidates = sorted(FEATURES_DIR.glob("panel_*.parquet")) + sorted(FEATURES_DIR.glob("csi300_*.parquet"))
    candidates = [c for c in candidates if "top20" not in c.name and "smoke" not in c.name]
    if candidates:
        chosen = candidates[-1]
        print(f"[auto-discover] Using panel: {chosen}")
        return chosen

    raise FileNotFoundError(
        f"No training panel found in {FEATURES_DIR}.\n"
        "Run: python scripts/build_panel.py --universe csi300 --freq Q\n"
        "Or specify: python scripts/analyze_factors.py --panel PATH"
    )


# ============================================================================
# Step 1: IC time series (cross-sectional Spearman per date)
# ============================================================================


def compute_ic_timeseries(
    panel: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
) -> pd.DataFrame:
    """Compute cross-sectional Spearman IC for each factor × date.

    At each rebalance date, computes spearmanr(factor_values, forward_return)
    across all stocks. No lookahead: factor at T predicts return from T onward.

    Returns:
        DataFrame  index=date, columns=factor_cols
    """
    dates = panel.index.get_level_values("as_of").unique().sort_values()
    records: list[dict] = []

    for d in tqdm(dates, desc="IC per date", ncols=80):
        xs = panel.xs(d, level="as_of")
        valid = xs[[*factor_cols, label_col]].dropna(subset=[label_col])
        if len(valid) < 20:
            continue
        row: dict = {"date": d}
        for f in factor_cols:
            sub = valid[[f, label_col]].dropna()
            if len(sub) < 10:
                row[f] = np.nan
                continue
            rho, _ = stats.spearmanr(sub[f], sub[label_col])
            row[f] = float(rho)
        records.append(row)

    ic_ts = pd.DataFrame(records).set_index("date")
    return ic_ts


# ============================================================================
# Step 2: IC summary stats
# ============================================================================


def summarize_ic(ic_ts: pd.DataFrame) -> pd.DataFrame:
    """Compute IC mean, IC_IR, win rate from the IC time series."""
    ic_mean = ic_ts.mean()
    ic_std = ic_ts.std(ddof=1)
    ic_ir = ic_mean / ic_std.replace(0, np.nan)
    summary = pd.DataFrame(
        {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ic_abs_mean": ic_ts.abs().mean(),
            "ic_win_rate": (ic_ts > 0).mean(),         # fraction of dates with positive IC
            "ic_abs_gt002": (ic_ts.abs() > 0.02).mean(),  # fraction of dates |IC| > 0.02
            "n_dates": ic_ts.count(),
        }
    )
    return summary.sort_values("ic_abs_mean", ascending=False)


# ============================================================================
# Step 3: Quintile backtest
# ============================================================================


def quintile_backtest(
    panel: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    n_quantiles: int = 5,
) -> dict[str, dict[str, float]]:
    """Mean forward return per quantile bucket, averaged across all dates.

    Q1 = lowest factor value, Q5 = highest. Returns spread = Q5 - Q1.
    """
    dates = panel.index.get_level_values("as_of").unique()
    # Accumulate returns per factor × quantile
    accum: dict[str, dict[int, list[float]]] = {
        f: {q: [] for q in range(1, n_quantiles + 1)} for f in factor_cols
    }

    for d in dates:
        xs = panel.xs(d, level="as_of")
        valid = xs[[*factor_cols, label_col]].dropna(subset=[label_col])
        if len(valid) < n_quantiles * 4:
            continue
        for f in factor_cols:
            sub = valid[[f, label_col]].dropna(subset=[f])
            if len(sub) < n_quantiles * 4:
                continue
            try:
                labels = pd.qcut(sub[f], n_quantiles, labels=False, duplicates="drop")
            except ValueError:
                continue
            for q_idx in range(n_quantiles):
                mask = labels == q_idx
                if mask.sum() > 0:
                    accum[f][q_idx + 1].append(sub.loc[mask, label_col].mean())

    results: dict[str, dict[str, float]] = {}
    for f in factor_cols:
        qret = {}
        for q in range(1, n_quantiles + 1):
            vals = accum[f][q]
            qret[f"Q{q}"] = float(np.nanmean(vals)) if vals else float("nan")
        q1 = qret.get("Q1", float("nan"))
        q5 = qret.get(f"Q{n_quantiles}", float("nan"))
        qret["spread"] = float(q5 - q1) if not (np.isnan(q1) or np.isnan(q5)) else float("nan")
        results[f] = qret

    return results


# ============================================================================
# Step 4: Factor correlation matrix
# ============================================================================


def compute_factor_correlation(
    panel: pd.DataFrame,
    factor_cols: list[str],
) -> pd.DataFrame:
    """Pool all observations and compute rank (Spearman) correlation between factors."""
    factor_data = panel[factor_cols].dropna(how="all")
    # Rank within each cross-section (date), then compute Pearson = Spearman
    ranked = (
        factor_data
        .groupby(level="as_of")
        .rank(pct=True)
    )
    return ranked.corr(method="pearson")


# ============================================================================
# Step 5: Factor selection
# ============================================================================


def select_factors(
    ic_summary: pd.DataFrame,
    ic_threshold: float,
    ir_threshold: float,
) -> list[str]:
    """Select factors meeting |IC mean| > ic_threshold AND |IC_IR| > ir_threshold."""
    mask = (
        (ic_summary["ic_abs_mean"] > ic_threshold)
        & (ic_summary["ic_ir"].abs() > ir_threshold)
    )
    return ic_summary[mask].index.tolist()


# ============================================================================
# Step 6: Plotly report builder
# ============================================================================


def _bar_ic_mean(ic_summary: pd.DataFrame, selected: list[str], label_col: str) -> "go.Figure":
    import plotly.graph_objects as go

    df = ic_summary.sort_values("ic_mean")
    colors = ["#2ecc71" if f in selected else "#85c1e9" for f in df.index]
    fig = go.Figure(
        go.Bar(
            x=df["ic_mean"],
            y=df.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in df["ic_mean"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>IC Mean: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=DEFAULT_IC_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text=f"+{DEFAULT_IC_THRESHOLD}", annotation_position="top right")
    fig.add_vline(x=-DEFAULT_IC_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text=f"-{DEFAULT_IC_THRESHOLD}", annotation_position="top left")
    fig.add_vline(x=0, line_color="gray", line_width=0.8)
    fig.update_layout(
        title=f"因子 IC 均值排名（目标：{label_col}）<br><sup>绿色=入选因子，蓝色=未入选</sup>",
        xaxis_title="IC Mean (Spearman)",
        yaxis_title="",
        height=900,
        margin=dict(l=200, r=100),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig


def _bar_ic_ir(ic_summary: pd.DataFrame, selected: list[str]) -> "go.Figure":
    import plotly.graph_objects as go

    df = ic_summary.sort_values("ic_ir")
    colors = ["#2ecc71" if f in selected else "#85c1e9" for f in df.index]
    fig = go.Figure(
        go.Bar(
            x=df["ic_ir"],
            y=df.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in df["ic_ir"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>IC IR: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=DEFAULT_IR_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_text=f"IR={DEFAULT_IR_THRESHOLD}", annotation_position="top right")
    fig.add_vline(x=-DEFAULT_IR_THRESHOLD, line_dash="dash", line_color="red",
                  annotation_position="top left")
    fig.add_vline(x=0, line_color="gray", line_width=0.8)
    fig.update_layout(
        title="因子 IC_IR（IC均值 / IC标准差）<br><sup>绿色=入选因子</sup>",
        xaxis_title="IC IR",
        height=900,
        margin=dict(l=200, r=100),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig


def _bar_win_rate(ic_summary: pd.DataFrame, selected: list[str]) -> "go.Figure":
    import plotly.graph_objects as go

    df = ic_summary.sort_values("ic_win_rate")
    colors = ["#2ecc71" if f in selected else "#85c1e9" for f in df.index]
    fig = go.Figure(
        go.Bar(
            x=df["ic_win_rate"] * 100,
            y=df.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v*100:.1f}%" for v in df["ic_win_rate"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Win Rate: %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50%")
    fig.update_layout(
        title="IC 胜率（IC > 0 的时期占比）",
        xaxis_title="Win Rate (%)",
        height=900,
        margin=dict(l=200, r=100),
        xaxis_range=[0, 105],
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def _heatmap_ic_ts(ic_ts: pd.DataFrame, selected: list[str], ic_summary: pd.DataFrame) -> "go.Figure":
    import plotly.graph_objects as go

    # Show selected factors (up to 20), fall back to top by |IC mean|
    show_factors = selected[:20] if len(selected) >= 3 else list(ic_summary.head(20).index)
    sub = ic_ts[show_factors].T
    date_labels = [str(d)[:10] for d in sub.columns]

    fig = go.Figure(
        go.Heatmap(
            z=sub.values,
            x=date_labels,
            y=sub.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            zmin=-0.4,
            zmax=0.4,
            colorbar=dict(title="IC"),
            text=[[f"{v:.3f}" for v in row] for row in sub.values],
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate="<b>%{y}</b><br>Date: %{x}<br>IC: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="IC 时间序列热力图（绿=正 IC，红=负 IC）",
        xaxis_title="Rebalance Date",
        height=max(400, len(show_factors) * 28 + 150),
        margin=dict(l=220),
    )
    return fig


def _quintile_chart(
    quintile_returns: dict[str, dict[str, float]],
    selected: list[str],
    ic_summary: pd.DataFrame,
    label_col: str,
    n_top: int = 10,
) -> "go.Figure":
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    show = selected[:n_top] if len(selected) >= 2 else list(ic_summary.head(n_top).index)
    ncols = min(5, len(show))
    nrows = (len(show) + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"{f}" for f in show],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    palette = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#2980b9"]
    q_labels = [f"Q{i}" for i in range(1, N_QUANTILES + 1)]

    for idx, f in enumerate(show):
        row = idx // ncols + 1
        col = idx % ncols + 1
        qret = quintile_returns.get(f, {})
        vals = [qret.get(ql, float("nan")) * 100 for ql in q_labels]
        fig.add_trace(
            go.Bar(
                x=q_labels,
                y=vals,
                marker_color=palette,
                showlegend=False,
                text=[f"{v:.2f}%" if not np.isnan(v) else "" for v in vals],
                textposition="outside",
                hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
            ),
            row=row,
            col=col,
        )
        # Add spread annotation
        spread = qret.get("spread", float("nan"))
        if not np.isnan(spread):
            fig.add_annotation(
                text=f"Spread: {spread*100:+.2f}%",
                xref=f"x{idx+1 if idx > 0 else ''}",
                yref=f"y{idx+1 if idx > 0 else ''}",
                x=0.5,
                y=1,
                xanchor="center",
                showarrow=False,
                font=dict(size=9, color="darkblue"),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=f"分层回测：5分位平均收益（目标：{label_col}，%）<br><sup>Q1=最低因子值，Q5=最高因子值</sup>",
        height=300 * nrows + 150,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _corr_matrix(corr: pd.DataFrame, selected: list[str]) -> "go.Figure":
    import plotly.graph_objects as go

    # Reorder: selected first, then the rest
    rest = [f for f in corr.index if f not in selected]
    order = selected + rest
    sub = corr.loc[order, order]

    text_vals = [[f"{v:.2f}" for v in row] for row in sub.values]
    fig = go.Figure(
        go.Heatmap(
            z=sub.values,
            x=sub.columns.tolist(),
            y=sub.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=text_vals,
            texttemplate="%{text}",
            textfont={"size": 7},
            colorbar=dict(title="Corr"),
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Corr: %{z:.3f}<extra></extra>",
        )
    )
    n = len(order)
    if selected:
        # Separator line after selected block
        sep = len(selected) - 0.5
        fig.add_shape(
            type="line", x0=-0.5, y0=sep, x1=n - 0.5, y1=sep,
            line=dict(color="lime", width=2, dash="dot"),
        )
        fig.add_shape(
            type="line", x0=sep, y0=-0.5, x1=sep, y1=n - 0.5,
            line=dict(color="lime", width=2, dash="dot"),
        )
        fig.add_annotation(
            text="← 入选因子 →", x=sep / 2, y=-1.5, showarrow=False,
            font=dict(size=10, color="lime"), xanchor="center",
        )
    fig.update_layout(
        title="因子相关性矩阵（Spearman Rank）<br><sup>蓝=正相关，红=负相关；入选因子排列在前（绿线分隔）</sup>",
        height=1000,
        width=1050,
        margin=dict(l=160, b=160),
    )
    return fig


def _summary_table(ic_summary: pd.DataFrame, selected: list[str]) -> "go.Figure":
    import plotly.graph_objects as go

    df = ic_summary.copy()
    df["selected"] = df.index.isin(selected)
    df = df.sort_values("ic_abs_mean", ascending=False)

    cell_fill = [
        ["#d5f5e3" if s else "white" for s in df["selected"]]
    ] * 8

    fig = go.Figure(
        go.Table(
            header=dict(
                values=["<b>Factor</b>", "<b>IC Mean</b>", "<b>IC Std</b>",
                        "<b>IC_IR</b>", "<b>|IC| Mean</b>", "<b>Win Rate</b>",
                        "<b>|IC|>0.02 Rate</b>", "<b>入选</b>"],
                fill_color="#2980b9",
                font=dict(color="white", size=12),
                align="center",
                height=32,
            ),
            cells=dict(
                values=[
                    df.index.tolist(),
                    [f"{v:+.4f}" for v in df["ic_mean"]],
                    [f"{v:.4f}" for v in df["ic_std"]],
                    [f"{v:+.3f}" for v in df["ic_ir"]],
                    [f"{v:.4f}" for v in df["ic_abs_mean"]],
                    [f"{v:.1%}" for v in df["ic_win_rate"]],
                    [f"{v:.1%}" for v in df["ic_abs_gt002"]],
                    ["✅" if v else "❌" for v in df["selected"]],
                ],
                fill_color=cell_fill,
                align=["left"] + ["center"] * 7,
                height=28,
                font=dict(size=11),
            ),
        )
    )
    fig.update_layout(title="因子分析汇总表（按 |IC均值| 降序）", height=900)
    return fig


# ============================================================================
# HTML assembly
# ============================================================================

_CSS = """
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
    margin: 0; padding: 24px; background: #f8f9fa; color: #333;
  }
  h1 { color: #1a252f; border-bottom: 4px solid #2980b9; padding-bottom: 12px; margin-top: 0; }
  .stats-row {
    display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0;
  }
  .stat-card {
    background: white; border-radius: 10px; padding: 18px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); min-width: 140px; text-align: center;
  }
  .stat-val { font-size: 2.2em; font-weight: 700; color: #2980b9; line-height: 1.1; }
  .stat-val.green { color: #27ae60; }
  .stat-val.orange { color: #e67e22; }
  .stat-lbl { font-size: 0.82em; color: #777; margin-top: 4px; }
  .criteria {
    background: #fef9e7; border-left: 4px solid #f39c12;
    padding: 10px 16px; border-radius: 0 8px 8px 0; margin: 8px 0 20px;
    font-size: 0.9em; color: #7d6608;
  }
  .factors-box {
    background: white; border-radius: 10px; padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px;
  }
  .factors-box h2 { margin-top: 0; color: #1a252f; }
  .factor-chip {
    display: inline-block; margin: 4px; padding: 5px 14px;
    background: #2980b9; color: white; border-radius: 20px;
    font-size: 13px; font-weight: 500;
  }
  .tab-wrap {
    background: white; border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden;
  }
  .tab-nav {
    display: flex; overflow-x: auto; background: #1a252f; gap: 0;
    scrollbar-width: thin;
  }
  .tab-btn {
    padding: 13px 20px; border: none; background: transparent;
    color: rgba(255,255,255,0.6); cursor: pointer; font-size: 13.5px;
    white-space: nowrap; border-bottom: 3px solid transparent;
    transition: all 0.15s;
  }
  .tab-btn:hover { color: rgba(255,255,255,0.9); }
  .tab-btn.active { color: white; border-bottom-color: #2980b9; font-weight: 600; }
  .tab-pane { display: none; padding: 16px; }
  .tab-pane.active { display: block; }
  footer { margin-top: 32px; color: #999; font-size: 0.8em; text-align: center; }
</style>
"""

_JS = """
<script>
function showTab(n) {
  document.querySelectorAll('.tab-pane').forEach((el, i) => el.classList.toggle('active', i === n));
  document.querySelectorAll('.tab-btn').forEach((el, i) => el.classList.toggle('active', i === n));
}
</script>
"""


def build_html_report(
    ic_ts: pd.DataFrame,
    ic_summary: pd.DataFrame,
    quintile_returns: dict,
    corr_matrix: pd.DataFrame,
    selected_factors: list[str],
    label_col: str,
    ic_threshold: float,
    ir_threshold: float,
) -> str:
    """Assemble a single-file interactive HTML report."""
    import plotly.io as pio

    tabs = [
        ("IC均值",     _bar_ic_mean(ic_summary, selected_factors, label_col)),
        ("IC_IR",     _bar_ic_ir(ic_summary, selected_factors)),
        ("IC胜率",     _bar_win_rate(ic_summary, selected_factors)),
        ("IC热力图",   _heatmap_ic_ts(ic_ts, selected_factors, ic_summary)),
        ("分层回测",   _quintile_chart(quintile_returns, selected_factors, ic_summary, label_col)),
        ("因子相关性", _corr_matrix(corr_matrix, selected_factors)),
        ("汇总表",     _summary_table(ic_summary, selected_factors)),
    ]

    # Stats
    best_ic = ic_summary["ic_abs_mean"].max()
    best_ir = ic_summary["ic_ir"].abs().max()
    n_sel = len(selected_factors)
    n_tot = len(ic_summary)

    parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="UTF-8">',
        f'<title>QuantMind — Factor Analysis ({label_col})</title>',
        _CSS,
        # embed plotly once
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
        "</head><body>",
        f"<h1>📊 QuantMind — 因子分析报告 (Phase 2.2)</h1>",
        f"<p style='color:#666;margin-top:-10px'>目标标签：<b>{label_col}</b> "
        f"&nbsp;|&nbsp; 筛选标准：<span style='color:#e74c3c'>|IC均值| > {ic_threshold}</span>"
        f" 且 <span style='color:#e74c3c'>|IC_IR| > {ir_threshold}</span></p>",
        # Stats row
        '<div class="stats-row">',
        f'<div class="stat-card"><div class="stat-val">{n_tot}</div><div class="stat-lbl">总因子数</div></div>',
        f'<div class="stat-card"><div class="stat-val green">{n_sel}</div><div class="stat-lbl">入选因子</div></div>',
        f'<div class="stat-card"><div class="stat-val">{best_ic:.4f}</div><div class="stat-lbl">最高 |IC均值|</div></div>',
        f'<div class="stat-card"><div class="stat-val">{best_ir:.3f}</div><div class="stat-lbl">最高 |IC_IR|</div></div>',
        f'<div class="stat-card"><div class="stat-val">{ic_ts.shape[0]}</div><div class="stat-lbl">时间截面数</div></div>',
        "</div>",
        # Selected factors
        '<div class="factors-box">',
        "<h2>✅ 入选因子列表</h2>",
    ]
    if selected_factors:
        for f in selected_factors:
            parts.append(f'<span class="factor-chip">{f}</span>')
    else:
        parts.append(
            '<p style="color:#e74c3c">⚠️ 没有因子通过筛选阈值，请降低 --ic-threshold 或 --ir-threshold</p>'
        )
    parts.append("</div>")

    # Tabs
    parts.append('<div class="tab-wrap"><div class="tab-nav">')
    for i, (name, _) in enumerate(tabs):
        cls = "active" if i == 0 else ""
        parts.append(f'<button class="tab-btn {cls}" onclick="showTab({i})">{name}</button>')
    parts.append("</div>")

    for i, (_, fig) in enumerate(tabs):
        cls = "active" if i == 0 else ""
        # Use full_html=False and no embedded plotly.js (we already have it via CDN)
        div = pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})
        parts.append(f'<div class="tab-pane {cls}" id="pane-{i}">{div}</div>')

    parts.append("</div>")  # tab-wrap

    parts += [
        "<footer>QuantMind Phase 2.2 Factor Analysis — generated by scripts/analyze_factors.py</footer>",
        _JS,
        "</body></html>",
    ]
    return "\n".join(parts)


# ============================================================================
# Output: top_factors.json
# ============================================================================


def save_top_factors_json(
    selected_factors: list[str],
    ic_summary: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    panel_path: Path,
    panel_shape: tuple,
    n_dates: int,
    ic_threshold: float,
    ir_threshold: float,
) -> Path:
    out = {
        "selected_factors": selected_factors,
        "selection_criteria": {
            "ic_abs_mean_threshold": ic_threshold,
            "ic_ir_abs_threshold": ir_threshold,
            "label_col": label_col,
        },
        "panel_info": {
            "path": str(panel_path),
            "shape": list(panel_shape),
            "n_dates": n_dates,
            "n_factors_total": len(factor_cols),
            "n_factors_selected": len(selected_factors),
        },
        "factor_stats": {
            f: {
                "ic_mean": round(float(ic_summary.loc[f, "ic_mean"]), 6),
                "ic_std": round(float(ic_summary.loc[f, "ic_std"]), 6),
                "ic_ir": round(float(ic_summary.loc[f, "ic_ir"]), 4),
                "ic_abs_mean": round(float(ic_summary.loc[f, "ic_abs_mean"]), 6),
                "ic_win_rate": round(float(ic_summary.loc[f, "ic_win_rate"]), 4),
                "selected": f in selected_factors,
            }
            for f in factor_cols
        },
    }
    path = FEATURES_DIR / "top_factors.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return path


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2.2 — Factor Analysis & Effectiveness Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--panel", default=None, help="Path to panel .parquet (auto-discovers if omitted)")
    p.add_argument(
        "--label",
        default=PRIMARY_LABEL,
        choices=LABEL_COLS,
        help=f"Forward return label to use for IC. Default: {PRIMARY_LABEL}",
    )
    p.add_argument(
        "--ic-threshold",
        type=float,
        default=DEFAULT_IC_THRESHOLD,
        dest="ic_threshold",
        help=f"|IC mean| threshold (default {DEFAULT_IC_THRESHOLD})",
    )
    p.add_argument(
        "--ir-threshold",
        type=float,
        default=DEFAULT_IR_THRESHOLD,
        dest="ir_threshold",
        help=f"|IC_IR| threshold (default {DEFAULT_IR_THRESHOLD})",
    )
    p.add_argument(
        "--quantiles",
        type=int,
        default=N_QUANTILES,
        help=f"Number of quantile buckets (default {N_QUANTILES})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── locate data ───────────────────────────────────────────────────────
    panel_path = find_panel_file(args.panel)
    print(f"\n{'='*60}")
    print("QuantMind — Phase 2.2 Factor Analysis")
    print(f"{'='*60}")
    print(f"Panel : {panel_path}")
    print(f"Label : {args.label}")
    print(f"Thresh: |IC mean| > {args.ic_threshold}, |IC_IR| > {args.ir_threshold}")

    # ── load ──────────────────────────────────────────────────────────────
    print("\n[1/6] Loading panel …")
    panel = pd.read_parquet(panel_path)
    factor_cols = [c for c in panel.columns if not c.startswith("forward_return_")]
    label_col = args.label
    print(f"  shape={panel.shape}, factors={len(factor_cols)}, label={label_col}")
    label_null = panel[label_col].isna().sum()
    if label_null:
        print(f"  ⚠️  {label_null} rows with NaN label (dropped in IC calc)")

    # ── IC time series ────────────────────────────────────────────────────
    print("\n[2/6] Computing cross-sectional Spearman IC …")
    ic_ts = compute_ic_timeseries(panel, factor_cols, label_col)
    print(f"  IC matrix: {ic_ts.shape}  (dates × factors)")

    # ── IC summary ────────────────────────────────────────────────────────
    print("\n[3/6] Summarising IC metrics …")
    ic_summary = summarize_ic(ic_ts)
    top10 = ic_summary.head(10)[["ic_mean", "ic_ir", "ic_win_rate", "ic_abs_mean"]]
    print(top10.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ── Quintile backtest ─────────────────────────────────────────────────
    print(f"\n[4/6] Running {args.quantiles}-quantile backtest …")
    quintile_returns = quintile_backtest(panel, factor_cols, label_col, args.quantiles)
    # Print top-5 spreads
    spreads = {f: quintile_returns[f].get("spread", float("nan")) for f in factor_cols}
    spread_df = pd.Series(spreads).dropna().sort_values(ascending=False)
    print("  Top-10 Q5−Q1 spreads:")
    for fname, sp in spread_df.head(10).items():
        print(f"    {fname:<40} {sp*100:+.3f}%")

    # ── Correlation matrix ────────────────────────────────────────────────
    print("\n[5/6] Computing factor correlation matrix …")
    corr_matrix = compute_factor_correlation(panel, factor_cols)
    # Warn about high-correlation pairs
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    high = [(r, c, upper.loc[r, c]) for r in corr_matrix.index for c in corr_matrix.columns
            if not np.isnan(upper.loc[r, c]) and abs(upper.loc[r, c]) > 0.8]
    if high:
        print(f"  ⚠️  High-correlation pairs (|r| > 0.8): {len(high)}")
        for r, c, v in sorted(high, key=lambda x: abs(x[2]), reverse=True)[:5]:
            print(f"    {r} × {c}: {v:+.3f}")

    # ── Factor selection ──────────────────────────────────────────────────
    selected_factors = select_factors(ic_summary, args.ic_threshold, args.ir_threshold)
    print(f"\n{'─'*60}")
    print(f"Selected: {len(selected_factors)} / {len(factor_cols)} factors")
    for f in selected_factors:
        row = ic_summary.loc[f]
        print(f"  {f:<40} IC={row['ic_mean']:+.4f}  IR={row['ic_ir']:+.4f}  WR={row['ic_win_rate']:.1%}")
    print(f"{'─'*60}")

    # ── Save top_factors.json ─────────────────────────────────────────────
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    json_path = save_top_factors_json(
        selected_factors=selected_factors,
        ic_summary=ic_summary,
        factor_cols=factor_cols,
        label_col=label_col,
        panel_path=panel_path,
        panel_shape=panel.shape,
        n_dates=int(ic_ts.shape[0]),
        ic_threshold=args.ic_threshold,
        ir_threshold=args.ir_threshold,
    )
    print(f"\n✅ top_factors.json → {json_path}")

    # ── Build HTML report ─────────────────────────────────────────────────
    print("\n[6/6] Building interactive HTML report …")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html = build_html_report(
        ic_ts=ic_ts,
        ic_summary=ic_summary,
        quintile_returns=quintile_returns,
        corr_matrix=corr_matrix,
        selected_factors=selected_factors,
        label_col=label_col,
        ic_threshold=args.ic_threshold,
        ir_threshold=args.ir_threshold,
    )
    report_path = REPORTS_DIR / "factor_analysis.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"✅ Report     → {report_path}")

    print(f"\n🎉  Phase 2.2 complete.")
    print(f"    {len(selected_factors)} factors selected for Phase 2.3 (model training).")
    print(f"    Open reports/factor_analysis.html in your browser to explore.")


if __name__ == "__main__":
    main()
