"""scripts/generate_attribution_report.py

统一归因报告管道：NAV 对比 + IC 热力图 + Regime 时间轴 + Barra 因子载荷。

输出：reports/figures/
  fig_01_nav_comparison.{pdf,png}
  fig_02_ic_heatmap.{pdf,png}
  fig_03_regime_timeline.{pdf,png}
  fig_04_barra_loadings.{pdf,png}

出图规范：scientific-figure-skill（全封闭框架，Okabe-Ito 色盲友好，serif 字体，
300dpi PNG，矢量 PDF）
"""

from __future__ import annotations

# ============================================================
# CONFIGURATION — edit only this block
# ============================================================
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "figures"

# 数据源
NAV_DIR        = ROOT / "reports" / "nav_v4"           # 使用 _e3 (cost_bps=13)
BARRA_DIR      = ROOT / "reports" / "barra"
IC_JSON        = ROOT / "data" / "paper_trading" / "ic_analysis_30day.json"
REGIME_PARQUET = ROOT / "data" / "features" / "regime_features.parquet"
INDEX_PARQUET  = ROOT / "data" / "raw" / "index_daily_panel.parquet"

# 字体回退链（首个可用的生效）
CN_SERIF_FALLBACK = ["SimSun", "Songti SC", "Source Han Serif SC",
                     "Noto Serif CJK SC", "WenQuanYi Micro Hei", "serif"]
EN_SERIF_FALLBACK = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "serif"]

# 输出格式
OUTPUT_FORMATS = ["pdf", "png"]
DPI = 300

# 统计
ALPHA = 0.05
IC_SIG_THRESH = 0.05          # IC 绝对值超过此阈值标星

# 数值精度
DEFAULT_DECIMALS = 3
SCIENTIFIC_LOW   = 1e-3
SCIENTIFIC_HIGH  = 1e5
THOUSANDS_SEP    = ","

# 图形尺寸（英寸）
FIG_WIDE   = (9.0, 4.5)   # 宽图：NAV、Regime
FIG_SQUARE = (8.0, 5.5)   # 方图：IC 热力图、Barra 条形图

# Okabe-Ito 色盲友好调色板
OKABE_ITO = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "gray":   "#888888",
}

# NAV 策略配色
STRATEGY_COLORS = {
    "hrp":   OKABE_ITO["blue"],
    "equal": OKABE_ITO["green"],
    "blend": OKABE_ITO["orange"],
    "kelly": OKABE_ITO["red"],
}
STRATEGY_LABELS = {
    "hrp":   "HRP",
    "equal": "Equal-Weight",
    "blend": "Blend",
    "kelly": "Kelly",
}

# ============================================================
# END CONFIGURATION
# ============================================================

import json
import logging
import sys
import warnings
# CJK/unicode glyph 缺失告警在无中文字体的环境是正常现象，静默处理
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── 安全检查 ──────────────────────────────────────────────────────────────────
assert OUTPUT_DIR.resolve() != NAV_DIR.resolve(), \
    "OUTPUT_DIR 不能与输入目录相同"

# ─── 字体解析 ──────────────────────────────────────────────────────────────────
from matplotlib import font_manager

def resolve_font(candidates: list[str]) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available or name == "serif":
            return name
    return "serif"

CN_FONT = resolve_font(CN_SERIF_FALLBACK)
EN_FONT = resolve_font(EN_SERIF_FALLBACK)
log.info("字体解析: EN=%s  CN=%s", EN_FONT, CN_FONT)

# ─── Matplotlib 全局样式 ───────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        [EN_FONT, CN_FONT, "serif"],
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "axes.unicode_minus": False,
    # 全封闭框架
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.edgecolor":     "#000000",
    "axes.labelcolor":    "#000000",
    # 网格
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.color":         "#e5e5e5",
    "grid.linewidth":     0.5,
    "grid.linestyle":     "-",
    # 背景
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "legend.frameon":     False,
    # 输出
    "figure.dpi":         100,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
})

# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def seal_spines(ax: plt.Axes, lw: float = 0.8) -> None:
    """确保四边框全部可见。"""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_color("black")


def save_figure(fig: plt.Figure, slug: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in OUTPUT_FORMATS:
        path = OUTPUT_DIR / f"{slug}.{fmt}"
        kw: dict = {"bbox_inches": "tight"}
        if fmt != "pdf":
            kw["dpi"] = DPI
        fig.savefig(path, **kw)
        log.info("  → %s", path)


def pct_fmt(x: float, _pos=None) -> str:
    return f"{x*100:.1f}%"


def nav_fmt(x: float, _pos=None) -> str:
    return f"{x:.2f}"


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_nav(weight: str) -> pd.Series:
    """加载 nav_v4/{weight}/nav_daily.csv（优先 _e3 cost-adjusted 版本）。"""
    # 优先用 E3 (cost_bps=13) 版本
    for suffix in ["_e3", ""]:
        p = NAV_DIR / f"{weight}{suffix}" / "nav_daily.csv"
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            df.index.name = "date"
            return df["nav"].rename(weight)
    raise FileNotFoundError(f"找不到 nav_v4/{weight}(_e3)/nav_daily.csv")


def load_nav_metrics(weight: str) -> dict:
    for suffix in ["_e3", ""]:
        p = NAV_DIR / f"{weight}{suffix}" / "nav_metrics.json"
        if p.exists():
            return json.loads(p.read_text())
    return {}


def load_csi300(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """从 index_daily_panel 提取 CSI300 日收盘，归一化到 start=1.0。"""
    df = pd.read_parquet(INDEX_PARQUET)
    csi = (
        df[df["ts_code"] == "000300.SH"][["trade_date", "close"]]
        .set_index("trade_date")["close"]
        .sort_index()
    )
    csi.index = pd.to_datetime(csi.index)
    csi = csi.loc[start:end]
    if csi.empty:
        return pd.Series(dtype=float, name="CSI300")
    csi = (csi / csi.iloc[0]).rename("CSI300")
    return csi


def load_ic_matrix() -> pd.DataFrame:
    """
    ic_analysis_30day.json → DataFrame (factor × horizon)。
    行：因子名，列：ic_1w / ic_2w / ic_21d / ic_3m。
    """
    data = json.loads(IC_JSON.read_text())
    ic_all = data.get("ic_all_stocks", {})
    horizons = ["ic_1w", "ic_2w", "ic_21d", "ic_3m"]
    horizon_labels = {"ic_1w": "1W", "ic_2w": "2W", "ic_21d": "21D", "ic_3m": "3M"}

    rows = {}
    factor_rename = {
        "lgbm_score":       "LGBM Score",
        "composite_score":  "Composite",
        "value_score":      "Value",
        "momentum_score":   "Momentum",
        "quality_score":    "Quality",
        "technical_score":  "Technical",
        "hist_win_rate":    "Hist Win Rate",
        "hist_sharpe":      "Hist Sharpe",
        "hist_maxdd":       "Hist MaxDD",
    }
    for factor, vals in ic_all.items():
        label = factor_rename.get(factor, factor)
        rows[label] = {horizon_labels[h]: vals.get(h, np.nan) for h in horizons}

    return pd.DataFrame(rows).T  # factor × horizon


def load_regime() -> pd.DataFrame:
    """加载季度 Regime 特征（含 regime_label 和 csi300_63d_return）。"""
    df = pd.read_parquet(REGIME_PARQUET)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    # regime_label: 0=弱势(bear)，1=强势(bull)
    return df


def load_barra() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (factor_returns, industry_returns)。
    factor_returns: 期数 × 风格因子（size,value,momentum,quality,volatility,liquidity,beta）
    industry_returns: 期数 × 行业（过滤 Unknown，取绝对均值 Top 15）
    """
    fr = pd.read_csv(BARRA_DIR / "factor_returns.csv", index_col=0)
    ir = pd.read_csv(BARRA_DIR / "industry_returns.csv", index_col=0)

    # 行业：去除 Unknown，取绝对均值最大的 15 个
    ir = ir.drop(columns=[c for c in ir.columns if "Unknown" in c], errors="ignore")
    ir.columns = [c.replace("ind_", "") for c in ir.columns]
    top15 = ir.abs().mean().nlargest(15).index
    ir = ir[top15]
    return fr, ir


# ═══════════════════════════════════════════════════════════════════════════════
# 图 1：NAV 曲线对比
# ═══════════════════════════════════════════════════════════════════════════════

def plot_nav_comparison() -> None:
    log.info("图 1 — NAV 曲线对比...")

    navs: dict[str, pd.Series] = {}
    metrics: dict[str, dict] = {}
    for wt in ["hrp", "equal", "blend", "kelly"]:
        navs[wt]    = load_nav(wt)
        metrics[wt] = load_nav_metrics(wt)

    # 对齐时间轴
    start = max(s.index.min() for s in navs.values())
    end   = min(s.index.max() for s in navs.values())
    for wt in navs:
        navs[wt] = navs[wt].loc[start:end]

    csi = load_csi300(start, end)

    fig, ax = plt.subplots(figsize=FIG_WIDE)

    # 策略线
    for wt, series in navs.items():
        m   = metrics[wt]
        ann = m.get("ann_return", float("nan"))
        fin = series.iloc[-1]
        lbl = f"{STRATEGY_LABELS[wt]}  (Final NAV {fin:.2f}, Ann {ann*100:.1f}%)"
        ax.plot(series.index, series.values,
                color=STRATEGY_COLORS[wt], linewidth=2.0, label=lbl, zorder=3)

    # CSI300 基准
    if not csi.empty:
        ax.plot(csi.index, csi.values,
                color=OKABE_ITO["gray"], linewidth=1.5, linestyle="--",
                label="CSI 300 (Benchmark)", zorder=2)

    # 参考线 y=1
    ax.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle=":", zorder=1)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative NAV")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(nav_fmt))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    cost_bps = metrics.get("kelly", {}).get("cost_bps", 13)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.99, 0.02,
            f"Net of {int(cost_bps):d} bps round-trip cost",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#555555")

    seal_spines(ax)
    fig.tight_layout(pad=0.6)
    save_figure(fig, "fig_01_nav_comparison")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 图 2：IC 热力图（因子 × 持有期）
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ic_heatmap() -> None:
    log.info("图 2 — IC 热力图...")

    ic_mat = load_ic_matrix()   # factor × horizon

    fig, ax = plt.subplots(figsize=FIG_SQUARE)

    vmax = max(0.25, np.abs(ic_mat.values).max())
    im = ax.imshow(
        ic_mat.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
    )

    # 轴刻度
    ax.set_xticks(range(ic_mat.shape[1]))
    ax.set_xticklabels(ic_mat.columns, fontsize=9)
    ax.set_yticks(range(ic_mat.shape[0]))
    ax.set_yticklabels(ic_mat.index, fontsize=9)

    # 数值标注 + 显著性星号
    for i in range(ic_mat.shape[0]):
        for j in range(ic_mat.shape[1]):
            val = ic_mat.iloc[i, j]
            if np.isnan(val):
                continue
            star = " *" if abs(val) > IC_SIG_THRESH else ""
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}{star}",
                    ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman IC", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Holding Period")
    ax.set_ylabel("Factor / Score")
    ax.text(0.99, 0.01,
            f"* = |IC| > {IC_SIG_THRESH:.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#555555")

    seal_spines(ax)
    fig.tight_layout(pad=0.6)
    save_figure(fig, "fig_02_ic_heatmap")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 图 3：Regime 状态时间轴
# ═══════════════════════════════════════════════════════════════════════════════

def plot_regime_timeline() -> None:
    log.info("图 3 — Regime 时间轴...")

    regime_df = load_regime()

    # 背景色方案：label=1 → 多头（浅绿），label=0 → 空头（浅红）
    BG_COLORS = {
        1: "#d5f0d5",   # bull  浅绿
        0: "#f5d5d5",   # bear  浅红
    }
    LABEL_TEXT = {1: "Bull", 0: "Bear"}

    fig, ax1 = plt.subplots(figsize=FIG_WIDE)
    ax2 = ax1.twinx()

    dates  = pd.DatetimeIndex(regime_df.index)
    labels = regime_df["regime_label"].values
    ret63  = regime_df["csi300_63d_return"].values

    # 每个季度区间涂背景色
    for i, (dt, lbl) in enumerate(zip(dates, labels)):
        x0 = dt - pd.DateOffset(months=1, days=15)
        x1 = dt + pd.DateOffset(months=1, days=15)
        color = BG_COLORS.get(int(lbl), "#eeeeee")
        ax1.axvspan(mdates.date2num(x0.to_pydatetime()),
                    mdates.date2num(x1.to_pydatetime()),
                    color=color, alpha=0.45, lw=0, zorder=1)

    # CSI300 日线叠加（主轴）
    csi_daily = pd.read_parquet(INDEX_PARQUET)
    csi_daily = (
        csi_daily[csi_daily["ts_code"] == "000300.SH"][["trade_date", "close"]]
        .set_index("trade_date")["close"]
        .sort_index()
    )
    csi_daily.index = pd.to_datetime(csi_daily.index)
    regime_start = dates.min() - pd.DateOffset(months=3)
    csi_sub = csi_daily.loc[regime_start:]
    if not csi_sub.empty:
        csi_norm = csi_sub / csi_sub.iloc[0]
        ax1.plot(csi_norm.index, csi_norm.values,
                 color=OKABE_ITO["blue"], linewidth=1.8, zorder=3,
                 label="CSI 300 (normalized)")

    # 63 日收益率柱状（右轴）
    bar_colors = [OKABE_ITO["green"] if r > 0 else OKABE_ITO["red"]
                  for r in ret63]
    ax2.bar(dates, ret63, width=60, color=bar_colors, alpha=0.55, zorder=2,
            label="CSI300 63d Return")
    ax2.axhline(0, color="#999999", linewidth=0.6)
    ax2.set_ylabel("CSI300 63-Day Return", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))

    # 图例 + 标签
    regime_counts = pd.Series(labels).value_counts()
    bull_n = regime_counts.get(1, 0)
    bear_n = regime_counts.get(0, 0)

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=BG_COLORS[1], alpha=0.7, label=f"Bull regime (n={bull_n})"),
        Patch(facecolor=BG_COLORS[0], alpha=0.7, label=f"Bear regime (n={bear_n})"),
    ]
    ax1.legend(handles=legend_patches + ax1.get_lines()[:1],
               loc="upper left", fontsize=8)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("CSI 300 (Normalized)", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(nav_fmt))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    seal_spines(ax1)
    fig.tight_layout(pad=0.6)
    save_figure(fig, "fig_03_regime_timeline")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 图 4：Barra 因子载荷条形图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_barra_loadings() -> None:
    log.info("图 4 — Barra 因子载荷...")

    factor_ret, industry_ret = load_barra()

    # ── 子图 1（上）：风格因子均值 ± 1 SD ──────────────────────────────────
    style_mean = factor_ret.mean()
    style_std  = factor_ret.std(ddof=1)

    # ── 子图 2（下）：行业因子均值（Top 15 by |mean|）─────────────────────
    ind_mean = industry_ret.mean().sort_values()

    fig, (ax_style, ax_ind) = plt.subplots(
        2, 1, figsize=(FIG_SQUARE[0], FIG_SQUARE[1] * 1.4),
        gridspec_kw={"height_ratios": [1, 1.8]},
    )

    # 风格因子
    x_s = np.arange(len(style_mean))
    bar_colors_s = [OKABE_ITO["blue"] if v >= 0 else OKABE_ITO["red"]
                    for v in style_mean.values]
    ax_style.bar(
        x_s, style_mean.values,
        yerr=style_std.values,
        color=bar_colors_s,
        edgecolor="black", linewidth=0.5,
        capsize=4,
        error_kw={"linewidth": 0.8, "ecolor": "black"},
        zorder=3,
    )
    ax_style.axhline(0, color="#444444", linewidth=0.8)
    ax_style.set_xticks(x_s)
    ax_style.set_xticklabels(
        [c.capitalize() for c in style_mean.index], fontsize=9
    )
    ax_style.set_ylabel("Mean Factor Return")
    ax_style.set_xlabel("Style Factor")
    ax_style.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.3f}"
    ))
    seal_spines(ax_style)

    # 行业因子
    y_i = np.arange(len(ind_mean))
    bar_colors_i = [OKABE_ITO["blue"] if v >= 0 else OKABE_ITO["red"]
                    for v in ind_mean.values]
    ax_ind.barh(
        y_i, ind_mean.values,
        color=bar_colors_i,
        edgecolor="black", linewidth=0.4,
        zorder=3,
    )
    ax_ind.axvline(0, color="#444444", linewidth=0.8)
    ax_ind.set_yticks(y_i)
    ax_ind.set_yticklabels(ind_mean.index, fontsize=8)
    ax_ind.set_xlabel("Mean Industry Factor Return")
    ax_ind.set_ylabel("Industry (Top 15 by |Mean|)")
    ax_ind.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.3f}"
    ))
    seal_spines(ax_ind)

    # 共同注释
    bm = json.loads((BARRA_DIR / "barra_metrics.json").read_text())
    avg_r2 = bm.get("avg_r2", float("nan"))
    fig.text(0.99, 0.01,
             f"Avg cross-sectional R² = {avg_r2:.3f}  |  "
             f"Error bars = ±1 SD  |  n = {bm.get('n_periods', '?')} periods",
             ha="right", va="bottom", fontsize=7, color="#555555")

    fig.tight_layout(pad=0.8, h_pad=1.5)
    save_figure(fig, "fig_04_barra_loadings")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("输出目录：%s", OUTPUT_DIR)

    plot_nav_comparison()
    plot_ic_heatmap()
    plot_regime_timeline()
    plot_barra_loadings()

    # 汇总
    generated = sorted(OUTPUT_DIR.glob("fig_0*"))
    log.info("")
    log.info("─── 生成完毕 (%d 文件) ───────────────────────────────", len(generated))
    for f in generated:
        log.info("  %s", f.name)

    expected = len(["nav", "ic", "regime", "barra"]) * len(OUTPUT_FORMATS)
    if len(generated) < expected:
        log.warning("预期 %d 个文件，实际 %d 个，请检查日志。",
                    expected, len(generated))
    else:
        log.info("✓ 所有 %d 个文件生成成功", expected)


if __name__ == "__main__":
    main()
