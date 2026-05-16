"""Alpha 1374 长仓 Top-N 综合回测报告。

生成指标：季度调仓净值曲线、Sharpe、MaxDD、换手率、行业暴露。
输出：reports/alpha_final/alpha_report.html

用法：
    python scripts/run_alpha_report.py \
        --panel data/panel/alpha_panel_v3.parquet \
        --model models/lgbm_v3_top18.pkl \
        --top 30 \
        --label forward_return_63d \
        --rf 0.03 \
        --out reports/alpha_final/alpha_report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=Path("data/panel/alpha_panel_v3.parquet"))
    p.add_argument("--model", type=Path, default=Path("models/lgbm_v3_top18.pkl"))
    p.add_argument("--top", type=int, default=30, help="Top-N 持仓（默认 30）")
    p.add_argument("--label", default="forward_return_63d")
    p.add_argument("--rf", type=float, default=0.03, help="年化无风险利率")
    p.add_argument("--out", type=Path, default=Path("reports/alpha_final/alpha_report.html"))
    p.add_argument(
        "--weight-method",
        choices=["equal", "hrp", "kelly", "blend"],
        default="equal",
        help="持仓权重计算方式：equal 等权 | hrp 分层风险平价 | kelly 分数 Kelly | blend 混合",
    )
    p.add_argument("--kelly-fraction", type=float, default=0.5, help="Kelly 分数（0.25~0.5 推荐）")
    p.add_argument(
        "--prices",
        type=Path,
        default=Path("data/raw/alpha_prices_panel.parquet"),
        help="用于 HRP/Kelly 计算历史收益率的价格文件",
    )
    return p.parse_args(argv)


def load_model_scores(panel: pd.DataFrame, model_path: Path) -> pd.Series:
    from quantmind.models.factor_model import FactorModel

    model = FactorModel.load(model_path)
    feat_names = getattr(model, "_feature_names", None) or model.feature_names
    if not feat_names:
        raise ValueError("模型无特征名")
    missing = [c for c in feat_names if c not in panel.columns]
    if missing:
        raise ValueError(f"panel 缺少特征：{missing[:5]}…")
    X = panel[list(feat_names)].to_numpy(dtype=np.float32, copy=True)
    pred = model.predict(X)
    return pd.Series(pred, index=panel.index, name="score")


def _load_price_wide(price_path: Path) -> pd.DataFrame | None:
    """加载价格面板，返回 wide 格式（index=trade_date, columns=tickers）。"""
    if not price_path.is_file():
        return None
    try:
        import pyarrow.parquet as pq
        schema_names = pq.ParquetFile(str(price_path)).schema_arrow.names
        price_col = "adj_close" if "adj_close" in schema_names else "close"
        df = pd.read_parquet(price_path)
        if "trade_date" in df.columns and "ts_code" in df.columns:
            wide = df.pivot_table(
                index="trade_date", columns="ts_code", values=price_col, aggfunc="last"
            )
            wide.index = pd.to_datetime(wide.index)
            return wide
        # 已经是 wide 格式
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None


def build_portfolio(
    panel: pd.DataFrame,
    scores: pd.Series,
    label_col: str,
    top_n: int,
    weight_method: str = "equal",
    kelly_fraction: float = 0.5,
    price_wide: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按 as_of 日期选 Top-N，记录持仓与期间收益。

    weight_method: 'equal' | 'hrp' | 'kelly' | 'blend'
    """
    from quantmind.portfolio.position_sizing import hrp_weights, kelly_weights, blend_weights

    dates = sorted(panel.index.get_level_values("as_of").unique())
    rows = []
    prev_holdings: set[str] = set()

    for d in dates:
        xs = panel.xs(d, level="as_of")
        sc = scores.xs(d, level="as_of").reindex(xs.index)
        y = xs[label_col] if label_col in xs.columns else pd.Series(dtype=float)
        valid_sc = sc.dropna()
        n = min(top_n, len(valid_sc))
        top_tickers = list(valid_sc.nlargest(n).index.astype(str))
        holdings = set(top_tickers)

        # 权重计算
        weights: dict[str, float] = {}
        if weight_method == "equal" or price_wide is None or len(top_tickers) == 0:
            eq = 1.0 / len(top_tickers) if top_tickers else 0.0
            weights = {t: eq for t in top_tickers}
        else:
            try:
                cutoff = pd.Timestamp(d)
                hist = price_wide.loc[price_wide.index < cutoff, top_tickers].dropna(how="all")
                rets = hist.pct_change().dropna(how="all").tail(252)
                if weight_method in ("hrp", "blend"):
                    w_hrp = hrp_weights(rets, min_periods=63)
                if weight_method in ("kelly", "blend"):
                    mu = rets.mean() * 252
                    cov = rets.cov() * 252
                    w_kelly = kelly_weights(mu, cov.values, fraction=kelly_fraction)
                    w_kelly.index = mu.index
                if weight_method == "hrp":
                    w_final = w_hrp
                elif weight_method == "kelly":
                    w_final = w_kelly
                else:  # blend
                    w_final = blend_weights(w_hrp, w_kelly, alpha=0.5)
                weights = w_final.to_dict()
            except Exception:
                eq = 1.0 / len(top_tickers) if top_tickers else 0.0
                weights = {t: eq for t in top_tickers}

        if y.notna().sum() > 0:
            y_sel = y.reindex(top_tickers).dropna()
            w_arr = pd.Series(weights).reindex(y_sel.index).fillna(0)
            if w_arr.sum() > 1e-9:
                w_arr /= w_arr.sum()
            port_ret = float(w_arr.dot(y_sel)) if len(y_sel) > 0 else float("nan")
        else:
            port_ret = float("nan")

        # 行业分布
        ind_col = "exposure_industry"
        ind_dist: dict[str, float] = {}
        if ind_col in xs.columns:
            ind_s = xs.loc[[t for t in holdings if t in xs.index], ind_col].dropna()
            if len(ind_s) > 0:
                vc = ind_s.value_counts(normalize=True)
                ind_dist = vc.head(5).round(3).to_dict()

        # 换手率
        common = holdings & prev_holdings
        turnover = 1.0 - len(common) / n if n > 0 else float("nan")

        rows.append({
            "as_of": d,
            "holdings": top_tickers,
            "weights": weights,
            "portfolio_return": float(port_ret) if pd.notna(port_ret) else float("nan"),
            "turnover": turnover,
            "n_stocks": n,
            "industry_dist": ind_dist,
        })
        prev_holdings = holdings

    return pd.DataFrame(rows).set_index("as_of")


def calc_metrics(returns: np.ndarray, rf_annual: float, periods_per_year: float = 4.0) -> dict:
    """季度收益序列 → 净值/年化 Sharpe/MaxDD。"""
    valid = returns[~np.isnan(returns)]
    if len(valid) == 0:
        return {}

    rf_per_period = (1 + rf_annual) ** (1.0 / periods_per_year) - 1
    excess = valid - rf_per_period

    nav = np.cumprod(1 + valid)
    cum_max = np.maximum.accumulate(nav)
    drawdowns = nav / cum_max - 1
    max_dd = float(drawdowns.min())

    ann_ret = float((1 + valid.mean()) ** periods_per_year - 1)
    ann_vol = float(valid.std() * np.sqrt(periods_per_year))
    sharpe = float(excess.mean() / valid.std() * np.sqrt(periods_per_year)) if valid.std() > 0 else float("nan")

    return {
        "n_periods": len(valid),
        "ann_return": round(ann_ret, 4),
        "ann_volatility": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(float((valid > 0).mean()), 4),
        "avg_turnover": float("nan"),
    }


def rank_ic_series(panel: pd.DataFrame, scores: pd.Series, label_col: str) -> pd.Series:
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


def build_html(
    args: argparse.Namespace,
    portfolio: pd.DataFrame,
    metrics: dict,
    ic_series: pd.Series,
    panel: pd.DataFrame,
) -> str:
    import json

    dates_str = [str(d)[:10] for d in portfolio.index]
    port_rets = portfolio["portfolio_return"].tolist()
    turnovers = portfolio["turnover"].tolist()
    ics_vals = [ic_series.get(d, float("nan")) for d in portfolio.index]

    # NAV
    nav_vals = []
    nav = 1.0
    for r in port_rets:
        if not np.isnan(r):
            nav *= (1 + r)
        nav_vals.append(round(nav, 4))

    # 行业分布（最新一期）
    latest_ind = {}
    for d in reversed(portfolio.index):
        ind = portfolio.loc[d, "industry_dist"]
        if ind:
            latest_ind = ind
            break

    # 汇总换手率
    turnovers_valid = [t for t in turnovers if not np.isnan(t)]
    avg_turnover = round(float(np.mean(turnovers_valid)), 3) if turnovers_valid else float("nan")
    metrics["avg_turnover"] = avg_turnover

    ic_valid = ic_series.dropna()
    ic_mean = round(float(ic_valid.mean()), 4) if len(ic_valid) > 0 else float("nan")
    ic_ir = round(float(ic_valid.mean() / ic_valid.std()), 4) if len(ic_valid) > 1 else float("nan")

    # Aggregate sector exposure across all periods
    all_industry: dict[str, list] = {}
    for _, row in portfolio.iterrows():
        for ind, w in row["industry_dist"].items():
            all_industry.setdefault(ind, []).append(w)
    avg_industry = {k: round(float(np.mean(v)), 3) for k, v in all_industry.items()}
    avg_industry = dict(sorted(avg_industry.items(), key=lambda x: -x[1])[:10])

    # Markdown-style summary table
    def fmt(v, pct=True):
        if np.isnan(v):
            return "—"
        return f"{v*100:.2f}%" if pct else f"{v:.4f}"

    table_rows = "\n".join([
        f"<tr><td>年化收益</td><td>{fmt(metrics.get('ann_return', float('nan')))}</td></tr>",
        f"<tr><td>年化波动率</td><td>{fmt(metrics.get('ann_volatility', float('nan')))}</td></tr>",
        f"<tr><td>Sharpe</td><td>{fmt(metrics.get('sharpe', float('nan')), pct=False)}</td></tr>",
        f"<tr><td>最大回撤</td><td>{fmt(metrics.get('max_drawdown', float('nan')))}</td></tr>",
        f"<tr><td>胜率</td><td>{fmt(metrics.get('win_rate', float('nan')))}</td></tr>",
        f"<tr><td>平均换手率</td><td>{fmt(avg_turnover)}</td></tr>",
        f"<tr><td>IC 均值</td><td>{fmt(ic_mean, pct=False)}</td></tr>",
        f"<tr><td>ICIR</td><td>{fmt(ic_ir, pct=False)}</td></tr>",
        f"<tr><td>回测期间</td><td>{dates_str[0]} → {dates_str[-1]}</td></tr>",
        f"<tr><td>期数</td><td>{metrics.get('n_periods', '—')}</td></tr>",
    ])

    ind_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v*100:.1f}%</td></tr>"
        for k, v in avg_industry.items()
    )

    per_period_rows = "\n".join(
        f"<tr><td>{d}</td><td>{fmt(r)}</td><td>{fmt(ic_series.get(pd.Timestamp(d), float('nan')), pct=False)}</td><td>{fmt(t)}</td></tr>"
        for d, r, t in zip(dates_str, port_rets, turnovers)
    )

    chart_data = json.dumps({
        "dates": dates_str,
        "nav": nav_vals,
        "ic": [round(v, 4) if not np.isnan(v) else None for v in ics_vals],
        "turnover": [round(t, 3) if not np.isnan(t) else None for t in turnovers],
    })

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Alpha 1374 Top-{args.top} 长仓回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {{font-family: "Microsoft YaHei", sans-serif; background:#f5f6fa; margin:0; padding:20px;}}
  h1 {{color:#2c3e50; border-bottom:3px solid #3498db; padding-bottom:8px;}}
  h2 {{color:#34495e; margin-top:32px; font-size:1.2em;}}
  .summary {{display:flex; gap:16px; flex-wrap:wrap; margin:16px 0;}}
  .card {{background:#fff; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.08);
          padding:16px 24px; min-width:160px;}}
  .card .val {{font-size:1.8em; font-weight:bold; color:#2980b9;}}
  .card .lbl {{font-size:.85em; color:#7f8c8d; margin-top:4px;}}
  table {{border-collapse:collapse; width:100%; background:#fff; border-radius:8px;
          box-shadow:0 2px 8px rgba(0,0,0,.06); overflow:hidden;}}
  th {{background:#3498db; color:#fff; padding:10px 14px; text-align:left; font-weight:600;}}
  td {{padding:8px 14px; border-bottom:1px solid #ecf0f1; font-size:.92em;}}
  tr:last-child td {{border-bottom:none;}}
  tr:nth-child(even) td {{background:#f9fbfc;}}
  .chart {{background:#fff; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.06);
           padding:12px; margin:16px 0; height:360px;}}
  .meta {{color:#7f8c8d; font-size:.85em; margin:8px 0;}}
  .warning {{background:#fff3cd; border-left:4px solid #f0ad4e; padding:12px 16px;
             border-radius:4px; margin:12px 0; font-size:.9em;}}
</style>
</head>
<body>
<h1>📈 Alpha 1374 Top-{args.top} 长仓回测报告</h1>
<p class="meta">
  模型：<code>{args.model}</code> &nbsp;|&nbsp;
  面板：<code>{args.panel}</code> &nbsp;|&nbsp;
  标签：<code>{args.label}</code> &nbsp;|&nbsp;
  无风险利率：{args.rf*100:.1f}%/年
</p>

<div class="warning">
  ⚠️ 注意：本回测为等权持仓、季度调仓，不含交易成本。2020-2022 因子有效，2023-2024 受 regime shift 影响 IC 转负。
</div>

<h2>📊 绩效汇总</h2>
<div class="summary">
  <div class="card"><div class="val">{metrics.get('ann_return', 0)*100:.1f}%</div><div class="lbl">年化收益</div></div>
  <div class="card"><div class="val">{metrics.get('sharpe', 0):.2f}</div><div class="lbl">Sharpe 比</div></div>
  <div class="card"><div class="val">{metrics.get('max_drawdown', 0)*100:.1f}%</div><div class="lbl">最大回撤</div></div>
  <div class="card"><div class="val">{avg_turnover*100:.1f}%</div><div class="lbl">平均换手率</div></div>
  <div class="card"><div class="val">{ic_mean:.3f}</div><div class="lbl">IC 均值</div></div>
  <div class="card"><div class="val">{ic_ir:.3f}</div><div class="lbl">ICIR</div></div>
</div>

<table>
<tr><th>指标</th><th>值</th></tr>
{table_rows}
</table>

<h2>📉 净值曲线 / IC / 换手率</h2>
<div id="main_chart" class="chart"></div>

<h2>🏭 行业暴露（平均权重 Top 10）</h2>
<table>
<tr><th>行业</th><th>平均权重</th></tr>
{ind_rows}
</table>

<h2>📋 分期明细</h2>
<table>
<tr><th>期末日期</th><th>组合收益</th><th>Rank IC</th><th>换手率</th></tr>
{per_period_rows}
</table>

<script>
const data = {chart_data};
const chart = echarts.init(document.getElementById('main_chart'));
chart.setOption({{
  tooltip: {{trigger: 'axis', axisPointer: {{type: 'cross'}}}},
  legend: {{data: ['净值', 'Rank IC', '换手率']}},
  xAxis: {{type: 'category', data: data.dates, axisLabel: {{rotate: 30}}}},
  yAxis: [
    {{type: 'value', name: '净值', position: 'left'}},
    {{type: 'value', name: 'IC / 换手', position: 'right', min: -0.3, max: 1.0}},
  ],
  series: [
    {{name: '净值', type: 'line', data: data.nav, smooth: true,
      lineStyle: {{width: 2.5, color: '#3498db'}},
      areaStyle: {{color: 'rgba(52,152,219,.1)'}}}},
    {{name: 'Rank IC', type: 'bar', yAxisIndex: 1, data: data.ic,
      itemStyle: {{color: d => (d.value || 0) >= 0 ? '#2ecc71' : '#e74c3c'}}}},
    {{name: '换手率', type: 'line', yAxisIndex: 1, data: data.turnover,
      lineStyle: {{width: 1.5, color: '#e67e22', type: 'dashed'}}}},
  ],
}});
window.addEventListener('resize', () => chart.resize());
</script>
</body>
</html>"""


def main(argv=None):
    args = parse_args(argv)
    print(f"[Alpha Report] 加载面板：{args.panel}")
    panel = pd.read_parquet(args.panel)

    # 确保 as_of 是 Timestamp
    if "as_of" not in panel.index.names:
        raise ValueError("面板 index 需含 'as_of' level")

    print(f"[Alpha Report] 加载模型：{args.model}")
    scores = load_model_scores(panel, args.model)

    # 加载价格面板（HRP/Kelly 需要历史收益率）
    price_wide = None
    if getattr(args, "weight_method", "equal") != "equal":
        price_path = args.prices if args.prices.is_absolute() else ROOT / args.prices
        print(f"[Alpha Report] 加载价格面板（{args.weight_method}）：{price_path}")
        price_wide = _load_price_wide(price_path)
        if price_wide is None:
            print("  ⚠ 价格面板不可用，退化为等权")

    print(f"[Alpha Report] 构建 Top-{args.top} 组合（权重方法={getattr(args,'weight_method','equal')}）…")
    portfolio = build_portfolio(
        panel, scores, args.label, args.top,
        weight_method=getattr(args, "weight_method", "equal"),
        kelly_fraction=getattr(args, "kelly_fraction", 0.5),
        price_wide=price_wide,
    )

    returns = portfolio["portfolio_return"].to_numpy()
    metrics = calc_metrics(returns, args.rf, periods_per_year=4.0)

    print("[Alpha Report] 计算 Rank IC…")
    ic_series = rank_ic_series(panel, scores, args.label)
    ic_valid = ic_series.dropna()
    print(f"  IC 均值={ic_valid.mean():.4f}, ICIR={ic_valid.mean()/ic_valid.std():.4f}")
    print(f"  年化收益={metrics['ann_return']*100:.2f}%, Sharpe={metrics['sharpe']:.3f}, MaxDD={metrics['max_drawdown']*100:.2f}%")

    print("[Alpha Report] 生成 HTML…")
    html = build_html(args, portfolio, metrics, ic_series, panel)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"[Alpha Report] ✅ 报告已写入：{args.out}")


if __name__ == "__main__":
    main()
