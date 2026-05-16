#!/usr/bin/env python3
"""LightGBM 截面五分位分层回测（使用面板内 forward_return_21d 标签）."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.models.factor_model import FactorModel
from quantmind.risk.drawdown import DrawdownController

DRAWDOWN_TOOL = DrawdownController(rules=[], verbose=False)


def _sanitize_json_obj(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _sanitize_json_obj(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize_json_obj(x) for x in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/lgbm_v2_base_features.pkl"))
    p.add_argument(
        "--panels",
        type=Path,
        nargs="+",
        default=[Path("data/panel/test.parquet"), Path("data/panel/val.parquet")],
    )
    p.add_argument("--label", default="forward_return_21d")
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--rf", type=float, default=0.03)
    p.add_argument("--output-dir", type=Path, default=Path("reports/backtest"))
    p.add_argument(
        "--final-report",
        action="store_true",
        help="额外写入 QuantMind「最终成绩单」final_report.md（通常仅对一次性 OOS 使用）",
    )
    p.add_argument(
        "--split-meta",
        type=Path,
        default=Path("data/panel/split_meta.json"),
        help="用于 final_report 中训练集/验证集行数与日期范围的说明",
    )
    return p.parse_args(argv)


def _monthly_rf(rf_annual: float) -> float:
    return float((1.0 + rf_annual) ** (1.0 / 12.0) - 1.0)


def attach_model_scores(panel: pd.DataFrame, model: FactorModel) -> pd.Series:
    feat_names = getattr(model, "_feature_names", None) or model.feature_names
    if not feat_names:
        raise ValueError("model has no feature names")
    missing = [c for c in feat_names if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing features: {missing[:5]}...")
    X = panel[list(feat_names)].to_numpy(dtype=np.float32, copy=True)
    pred = model.predict(X)
    return pd.Series(pred, index=panel.index, name="score")


def assign_quantile_labels(score: pd.Series, n_bins: int) -> pd.Series:
    """分数升序：最低档为 Q1，最高为 Q{n_bins}."""
    if len(score) < n_bins:
        return pd.Series(index=score.index, dtype=object)
    rk = score.rank(method="first", ascending=True)
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    return pd.Series(
        pd.qcut(rk, q=n_bins, labels=labels, duplicates="drop").astype(str),
        index=score.index,
        name="quantile",
    )


def per_period_quintile_returns(
    panel: pd.DataFrame,
    scores: pd.Series,
    label_col: str,
    n_bins: int,
) -> pd.DataFrame:
    """每行：(as_of, quantile_label) → 当期该层 등权均值收益。"""
    rows: list[dict[str, Any]] = []
    dates = sorted(panel.index.get_level_values("as_of").unique())
    for d in dates:
        xs = panel.xs(d, level="as_of")
        sc = scores.xs(d, level="as_of").reindex(xs.index)
        y = xs[label_col]
        valid = y.notna() & sc.notna()
        if valid.sum() < n_bins:
            continue
        sc_v = sc[valid]
        y_v = y[valid]
        q = assign_quantile_labels(sc_v, n_bins)
        if q.isna().any():
            continue
        df = pd.DataFrame({"quantile": q, "fwd": y_v})
        for ql in df["quantile"].unique():
            m = df.loc[df["quantile"] == ql, "fwd"].mean()
            rows.append({"as_of": d, "quantile": ql, "mean_return": float(m)})
    return pd.DataFrame(rows)


def market_median_by_date(panel: pd.DataFrame, label_col: str) -> pd.Series:
    """全市场每期截面标签中位数."""
    rows = []
    for d in sorted(panel.index.get_level_values("as_of").unique()):
        xs = panel.xs(d, level="as_of")
        y = xs[label_col].dropna()
        if len(y) == 0:
            continue
        rows.append((d, float(y.median())))
    return pd.Series({d: med for d, med in rows})


def layer_summary(tbl: pd.DataFrame) -> pd.DataFrame:
    """分层：平均月收益 / 胜率 / 年化。"""
    if tbl.empty:
        return pd.DataFrame()
    summaries = []
    for q in sorted(tbl["quantile"].unique()):
        sub = tbl[tbl["quantile"] == q]
        vals = sub["mean_return"].values.astype(float)
        summaries.append({
            "quantile": q,
            "periods": len(sub),
            "mean_monthly": float(np.mean(vals)) if len(vals) else float("nan"),
            "win_rate": float(np.mean(vals > 0)) if len(vals) else float("nan"),
            "annualized_approx": float(np.mean(vals)) * 12.0 if len(vals) else float("nan"),
        })
    return pd.DataFrame(summaries)


def rank_ic_series(
    panel: pd.DataFrame,
    scores: pd.Series,
    label_col: str,
) -> pd.Series:
    ics = {}
    for d in sorted(panel.index.get_level_values("as_of").unique()):
        xs = panel.xs(d, level="as_of")
        sc = scores.xs(d, level="as_of").reindex(xs.index)
        y = xs[label_col]
        valid = y.notna() & sc.notna()
        if valid.sum() < 8:
            ics[d] = float("nan")
            continue
        rho, _ = stats.spearmanr(sc[valid].values, y[valid].values)
        ics[d] = float(rho) if rho == rho else float("nan")
    return pd.Series(ics, name="rank_ic")


def sharpe_monthly(monthly_returns: np.ndarray, rf_annual: float) -> float:
    rfm = _monthly_rf(rf_annual)
    xs = monthly_returns[~np.isnan(monthly_returns)]
    if xs.size < 2:
        return float("nan")
    excess = xs - rfm
    s = float(np.std(excess, ddof=1))
    if s < 1e-16:
        return float("nan")
    return float(np.sqrt(12.0) * np.mean(excess) / s)


def long_short_returns(tbl: pd.DataFrame, n_bins: int) -> pd.Series:
    """每期 Q{n_bins}-Q1 价差（月均收益向量，index=as_of）."""
    piv = tbl.pivot_table(index="as_of", columns="quantile", values="mean_return", aggfunc="first")
    q1 = "Q1"
    qhi = f"Q{n_bins}"
    if piv.empty or q1 not in piv.columns or qhi not in piv.columns:
        return pd.Series(dtype=float)
    return piv[qhi].astype(float) - piv[q1].astype(float)


def q5_returns_series(tbl: pd.DataFrame, n_bins: int) -> pd.Series:
    piv = tbl.pivot_table(index="as_of", columns="quantile", values="mean_return", aggfunc="first")
    qh = f"Q{n_bins}"
    if qh not in piv.columns:
        return pd.Series(dtype=float)
    return piv[qh].astype(float)


def nav_from_returns_indexed(monthly_returns: pd.Series) -> pd.Series:
    r = monthly_returns.dropna().values.astype(float)
    if len(r) == 0:
        return pd.Series(dtype=float)
    nav = np.cumprod(1.0 + r)
    return pd.Series(nav, index=monthly_returns.dropna().index)


def subset_panel_by_year(panel: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    as_of = panel.index.get_level_values("as_of")
    mask = as_of.year.isin(set(years))
    return panel.loc[mask]


def _fmt_pct6(x: float) -> str:
    if x != x:
        return "nan"
    return f"{x * 100.0:.4f}"


@dataclass
class SegmentMetrics:
    label: str
    ic_mean: float
    ic_ir: float
    ls_mean_monthly: float
    ls_annual_approx: float
    ls_sharpe: float
    q5_ann: float
    max_dd_q5: float


def metrics_for_segment(
    panel: pd.DataFrame,
    scores: pd.Series,
    quint_per_period: pd.DataFrame,
    label_col: str,
    n_bins: int,
    rf_annual: float,
    name: str,
) -> SegmentMetrics:
    if panel.empty:
        return SegmentMetrics(
            name,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    sc_sub = scores.reindex(panel.index)
    qp_sub = quint_per_period[quint_per_period["as_of"].isin(
        sorted(panel.index.get_level_values("as_of").unique())
    )]

    ic_s = rank_ic_series(panel, sc_sub, label_col)
    ics_clean = ic_s[~ic_s.isna()].values.astype(float)
    ic_m = float(np.mean(ics_clean)) if len(ics_clean) else float("nan")
    ic_sd = float(np.std(ics_clean, ddof=1)) if len(ics_clean) > 1 else float("nan")
    ic_ir = ic_m / ic_sd if ic_sd == ic_sd and ic_sd > 1e-12 else float("nan")

    ls = long_short_returns(qp_sub, n_bins).sort_index()
    ls_vals = ls.values.astype(float)
    ls_mean = float(np.mean(ls_vals)) if len(ls_vals) else float("nan")

    q5 = q5_returns_series(qp_sub, n_bins).sort_index().reindex(ls.index)

    sharpe_ls = sharpe_monthly(ls_vals, rf_annual)

    nav = nav_from_returns_indexed(q5.dropna())
    if nav.empty:
        mdd = float("nan")
    else:
        mdd = DRAWDOWN_TOOL.compute_max_drawdown(nav)

    q5_ok = q5.dropna()
    q5_ann = float(q5_ok.mean() * 12.0) if len(q5_ok) else float("nan")

    return SegmentMetrics(
        name,
        ic_m,
        ic_ir,
        ls_mean,
        ls_mean * 12.0,
        sharpe_ls,
        q5_ann,
        mdd,
    )


def merge_panels(paths: list[Path], label_col: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        if label_col not in df.columns:
            raise SystemExit(f"{p}: missing {label_col}")
        frames.append(df)
    panel = pd.concat(frames).sort_index()
    return panel[~panel.index.duplicated(keep="last")]


def write_final_eval_report(
    *,
    output_path: Path,
    metrics_oos: SegmentMetrics,
    layer_all: pd.DataFrame,
    model_path: Path,
    model_obj: FactorModel,
    split_meta_path: Path,
    n_periods_eval: int,
    date_range_txt: str,
    report_suite: str = "legacy_test",
    label_col: str = "forward_return_21d",
    n_panel_dates: int | None = None,
) -> None:
    split_lines: list[str] = []
    if split_meta_path.is_file():
        sm = json.loads(split_meta_path.read_text(encoding="utf-8"))
        tr = sm.get("train") or {}
        va = sm.get("val") or {}
        tst = sm.get("test") or {}
        if report_suite == "holdout":
            split_lines = [
                "",
                "- 样本划分摘要（保留集仅在本表评估）：",
                f"  - 训练集：{tr.get('date_range', {})}，约 **{tr.get('n_rows', '?')}** 行（{len(tr.get('as_of_dates', []))} 期）",
                f"  - 验证集：{va.get('date_range', {})}，约 **{va.get('n_rows', '?')}** 行（{len(va.get('as_of_dates', []))} 期）",
                f"  - 测试集（2024）：{tst.get('date_range', {})}，**{tst.get('n_rows', '?')}** 行（{len(tst.get('as_of_dates', []))} 期）",
                f"  - **保留集（本报告）**：截面 **{date_range_txt}**，共 **{n_panel_dates if n_panel_dates is not None else n_periods_eval}** 期；"
                f"分层统计 **{n_periods_eval}** 期（标签完备）",
            ]
        else:
            split_lines = [
                "",
                "- 训练配置摘要：",
                f"  - 训练集：{tr.get('date_range', {})}，约 **{tr.get('n_rows', '?')}** 行（{len(tr.get('as_of_dates', []))} 期）",
                f"  - 验证集：{va.get('date_range', {})}，约 **{va.get('n_rows', '?')}** 行（{len(va.get('as_of_dates', []))} 期）",
                f"  - 本题 OOS 测试集：**{tst.get('date_range', {})}**，**{tst.get('n_rows', '?')}** 行；本表按期数统计为 **{n_periods_eval}** 期",
            ]
    else:
        split_lines = [
            "",
            "- （未找到 split_meta，`data/panel/split_meta.json`，训练摘要略）",
        ]

    feat_names = getattr(model_obj, "_feature_names", None) or getattr(
        model_obj, "feature_names", None
    )
    n_feat = len(feat_names) if feat_names else None
    bi = getattr(model_obj, "best_iteration_", None)
    try:
        bi_s = str(int(bi)) if bi is not None else "—"
    except Exception:
        bi_s = "—"

    q5_row = None
    if not layer_all.empty and "quantile" in layer_all.columns:
        qdf = layer_all[layer_all["quantile"] == "Q5"]
        if len(qdf):
            q5_row = qdf.iloc[0]
    q5_moon = float(q5_row["mean_monthly"]) if q5_row is not None else float("nan")
    q5_ann = float(q5_row["annualized_approx"]) if q5_row is not None else metrics_oos.q5_ann

    eval_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sharpe_s = (
        f"{metrics_oos.ls_sharpe:.6f}" if metrics_oos.ls_sharpe == metrics_oos.ls_sharpe else "nan"
    )
    dd_s = (
        f"{metrics_oos.max_dd_q5:.6f}" if metrics_oos.max_dd_q5 == metrics_oos.max_dd_q5 else "nan"
    )

    ic_mean_s = f"{metrics_oos.ic_mean:.6f}"
    if metrics_oos.ic_mean == metrics_oos.ic_mean and metrics_oos.ic_mean > 0:
        ic_mean_s = f"**{metrics_oos.ic_mean:.6f}**"

    if report_suite == "holdout":
        n_pd = n_panel_dates if n_panel_dates is not None else n_periods_eval
        ds_block = [
            f"**评估数据集：保留集 2025Q1–2026Q1**：截面日期 **{date_range_txt}**，共 **{n_pd}** 期因子截面（保留集）；"
            "**首次使用**，未经训练或调参。",
            "",
            f"- **分层与 IC**：依赖非空的 **`{label_col}`**。离线快照/本地日线若无法在截面日后闭合 21 个交易日，则该期剔除。"
            f"本报告分层汇总基于 **{n_periods_eval}** 个标签完备期（通常为 **2025Q1–2025Q4**）。",
            "",
            "- 时间轴：**模型训练**于 **2019–2022**，**验证（Early stopping）**于 **2023**，"
            "**测试（书面 OOS）**于 **2024**；本报告为 **2025–2026 保留集一次性样本外检验**。",
        ]
    else:
        ds_block = [
            f"**数据集：test——{date_range_txt}（test=2024，**首次使用**本集做最终 OOS；训练阶段未读此 parquet）**",
        ]

    md = "\n".join([
        "# QuantMind 因子模型最终评估报告",
        "",
        f"**评估日期：{eval_day}（UTC）**  ",
        *ds_block,
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| OOS IC_mean | {ic_mean_s} |",
        f"| OOS ICIR | {metrics_oos.ic_ir:.6f} |",
        f"| Q5 月均收益 | {_fmt_pct6(q5_moon)} % |",
        f"| Q5 年化收益（月均×12，近似） | {_fmt_pct6(q5_ann)} % |",
        f"| 多空年化价差（Q5−Q1×12） | {_fmt_pct6(metrics_oos.ls_annual_approx)} % |",
        f"| 多空夏普比率（月频年化） | {sharpe_s} |",
        f"| Q5 最大回撤 | {dd_s} |",
        f"| 分层评估期数（`forward_return_21d` 完备） | **{n_periods_eval}** 期 |",
        *split_lines,
        "",
        "- 模型与训练概要：",
        f"  - 模型：`{model_path}`",
        f"  - LambdaRank：`best_iteration=`**{bi_s}**（早停最优轮次）",
        f"  - 特征维度：**{n_feat if n_feat is not None else '—'}** 个数值特征",
        "",
        "## 说明",
        "",
        "- Q5/Q1、夏普、回撤等定义同 `reports/backtest/` 方法论。",
        "- 交易成本、流动性与标签实现细节未反映在表中。",
        (
            "- **保留集**：仅限模型与流程固化后的**单次**读取与评测；不得用于迭代调参。"
            if report_suite == "holdout"
            else ""
        ),
        "",
    ])
    output_path.write_text(md, encoding="utf-8")


def run_backtest(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = FactorModel.load(args.model)
    panel_full = merge_panels(args.panels, args.label)

    scores = attach_model_scores(panel_full, model)
    tbl = per_period_quintile_returns(
        panel_full, scores, args.label, args.n_bins
    )
    tbl.to_csv(args.output_dir / "quintile_returns_by_period.csv", index=False)

    layer_all = layer_summary(tbl)
    layer_all["_qi"] = layer_all["quantile"].str.slice(1).astype(int)
    layer_all = layer_all.sort_values("_qi").drop(columns="_qi")

    median_mkt = market_median_by_date(panel_full, args.label)
    q5s = tbl[tbl["quantile"] == f"Q{args.n_bins}"].set_index("as_of")["mean_return"].sort_index()
    q5s.index.name = "as_of"

    periods_table = q5s.rename("Q5_mean").reset_index()
    periods_table["market_median"] = median_mkt.reindex(periods_table["as_of"]).values
    periods_table["excess_vs_median"] = (
        periods_table["Q5_mean"] - periods_table["market_median"]
    )
    periods_table.to_csv(args.output_dir / "q5_vs_median_by_period.csv", index=False)

    panel_2023 = subset_panel_by_year(panel_full, [2023])
    panel_2024 = subset_panel_by_year(panel_full, [2024])

    metrics_2023 = metrics_for_segment(
        panel_2023,
        scores,
        tbl,
        args.label,
        args.n_bins,
        args.rf,
        "val(2023 调参集)",
    )
    metrics_2024 = metrics_for_segment(
        panel_2024,
        scores,
        tbl,
        args.label,
        args.n_bins,
        args.rf,
        "test(2024 H/O)",
    )
    metrics_os = metrics_for_segment(
        panel_full,
        scores,
        tbl,
        args.label,
        args.n_bins,
        args.rf,
        "合并OOS",
    )

    def _fmt(x: float) -> str:
        if x != x:
            return "nan"
        return f"{x:+.6f}"

    ls_all = long_short_returns(tbl, args.n_bins).sort_index()

    lines = [
        "# LightGBM 截面分层多空回测",
        "",
        f"- 模型: `{args.model}`",
        f"- 面板: {', '.join(str(x) for x in args.panels)}",
        f"- 标签: `{args.label}`，分层数: {args.n_bins}",
        f"- 无风险利率（年化）: {args.rf:.2%}",
        "",
        "## 截面分层收益表（Q1→Q5）（所提供面板的样本并集）",
        "",
        "| 分层 | 期数 | 平均月收益 | 胜率 | 年化收益（×12） |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in layer_all.iterrows():
        lines.append(
            f"| {row['quantile']} | {int(row['periods'])} | {_fmt(row['mean_monthly'])} | "
            f"{row['win_rate']:.1%} | {_fmt(row['annualized_approx'])} |"
        )

    lines.extend([
        "",
        "## Q5-Q1 多空组合",
        "",
        "| 指标 | val 2023 | test 2024 | 并入同一表 |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| IC_mean | {_fmt(metrics_2023.ic_mean)} | {_fmt(metrics_2024.ic_mean)} | "
            f"{_fmt(metrics_os.ic_mean)} |"
        ),
        (
            f"| ICIR | {_fmt(metrics_2023.ic_ir)} | {_fmt(metrics_2024.ic_ir)} | "
            f"{_fmt(metrics_os.ic_ir)} |"
        ),
        (
            f"| Q5-Q1月均价差 | {_fmt(metrics_2023.ls_mean_monthly)} | "
            f"{_fmt(metrics_2024.ls_mean_monthly)} | {_fmt(metrics_os.ls_mean_monthly)} |"
        ),
        (
            f"| Q5-Q1年化价差(×12) | {_fmt(metrics_2023.ls_annual_approx)} | "
            f"{_fmt(metrics_2024.ls_annual_approx)} | {_fmt(metrics_os.ls_annual_approx)} |"
        ),
        (
            f"| 多空夏普(月频年化) | {_fmt(metrics_2023.ls_sharpe)} | "
            f"{_fmt(metrics_2024.ls_sharpe)} | {_fmt(metrics_os.ls_sharpe)} |"
        ),
        (
            f"| Q5月均→年化 | {_fmt(metrics_2023.q5_ann)} | {_fmt(metrics_2024.q5_ann)} | "
            f"{_fmt(metrics_os.q5_ann)} |"
        ),
        (
            f"| Q5净值最大回撤 | {_fmt(metrics_2023.max_dd_q5)} | "
            f"{_fmt(metrics_2024.max_dd_q5)} | {_fmt(metrics_os.max_dd_q5)} |"
        ),
        "",
        "## 逐期 Q5 收益（相对全市场中位）",
        "",
        "| as_of | Q5均收益 | 全市场中位收益 | 超额收益 |",
        "| --- | ---: | ---: | ---: |",
    ])

    for _, brow in periods_table.iterrows():
        d0 = str(brow["as_of"])[:10]
        mm = brow["market_median"]
        mm_s = _fmt(float(mm)) if mm == mm else ""
        lines.append(
            f"| {d0} | {_fmt(float(brow['Q5_mean']))} | {mm_s} | {_fmt(float(brow['excess_vs_median']))} |"
        )

    lines.append("")
    lines.append("## 逐期 Q5−Q1 多空价差（月）")
    lines.append("")
    lines.append("| as_of | Q5-Q1 |")
    lines.append("| --- | ---: |")
    for idx, val in ls_all.items():
        lines.append(f"| {str(idx)[:10]} | {_fmt(float(val))} |")

    lines.extend([
        "",
        "## 风险提示",
        "",
        "- 回测基于 PIT 快照标签，未考虑交易成本、冲击与停牌。",
        "- 分层为等权，实盘需考虑市值、流动性及单票上限。",
        "- OOS 样本期数较短时，夏普与 ICIR 波动大。",
        "- 文件名约定：`data/panel/val.parquet`=调参，`test.parquet`=最终 OOS。",
        "",
    ])

    rep = args.output_dir / "backtest_report.md"
    rep.write_text("\n".join(lines), encoding="utf-8")

    metrics_json = {
        "layer_summary": layer_all.to_dict("records"),
        "segment_2023": metrics_2023.__dict__,
        "segment_2024": metrics_2024.__dict__,
        "segment_oos": metrics_os.__dict__,
        "spread_mean_monthly_oos": float(ls_all.mean()) if len(ls_all) else float("nan"),
        "spread_sharpe_oos": metrics_os.ls_sharpe,
        "max_dd_q5_oos": metrics_os.max_dd_q5,
        "mean_q5_monthly_oos": float(q5s.mean()) if len(q5s) else float("nan"),
    }
    (args.output_dir / "backtest_metrics.json").write_text(
        json.dumps(_sanitize_json_obj(metrics_json), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.final_report:
        if len(args.panels) != 1:
            print(
                "WARNING: --final-report 通常为一次性 OOS，建议 `--panels data/panel/test.parquet` 单文件。"
                f" 当前 panels={args.panels}",
                file=sys.stderr,
                flush=True,
            )
        n_periods_eval = int(tbl["as_of"].nunique()) if len(tbl) else 0
        ads = sorted(panel_full.index.get_level_values("as_of").unique())
        dr_txt = (
            f"{str(ads[0])[:10]} → {str(ads[-1])[:10]}"
            if len(ads) else "—"
        )
        min_year = int(pd.Timestamp(ads[0]).year) if len(ads) else 1999
        report_suite = "holdout" if min_year >= 2025 else "legacy_test"
        n_panel_dates = int(panel_full.index.get_level_values("as_of").nunique())
        write_final_eval_report(
            output_path=args.output_dir / "final_report.md",
            metrics_oos=metrics_os,
            layer_all=layer_all,
            model_path=args.model,
            model_obj=model,
            split_meta_path=args.split_meta,
            n_periods_eval=n_periods_eval,
            date_range_txt=dr_txt,
            report_suite=report_suite,
            label_col=args.label,
            n_panel_dates=n_panel_dates,
        )

    print(tbl.head(15).to_string(index=False))
    print("\n分层汇总:")
    print(layer_all.to_string(index=False))
    print("\n2023:", metrics_2023)
    print("2024:", metrics_2024)
    print("OOS:", metrics_os)
    print(f"\nreport → {rep.resolve()}")
    if args.final_report:
        print(f"final_report → {(args.output_dir / 'final_report.md').resolve()}")


def main(argv: list[str] | None = None) -> None:
    run_backtest(parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
