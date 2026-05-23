"""日频真实持仓 NAV 回测。

使用 alpha_prices_panel.parquet 的实际 adj_close 价格，而非面板期望收益，
构建完整的日频净值曲线。

输出：
  reports/alpha_final/nav_daily.csv         日频 NAV（策略 + CSI300 基准）
  reports/alpha_final/nav_metrics.json      汇总绩效指标
  reports/alpha_final/nav_holdings.csv      每期持仓明细
  reports/alpha_final/nav_report.html       交互式 HTML 报告（ECharts）

用法：
  python scripts/run_nav_backtest.py \\
      --panel  data/panel/alpha_panel_v3.parquet \\
      --prices data/raw/alpha_prices_panel.parquet \\
      --model  models/lgbm_v3_top18.pkl \\
      --top    30 \\
      --weight-method equal \\
      --rf     0.03 \\
      --out    reports/alpha_final/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",  type=Path, default=Path("data/panel/alpha_panel_v3.parquet"))
    p.add_argument("--prices", type=Path, default=Path("data/raw/alpha_prices_panel.parquet"))
    p.add_argument("--index-prices", type=Path, default=None,
                   help="CSI300 基准价格（可选，若不提供则跳过基准对比）")
    p.add_argument("--model",  type=Path, default=Path("models/lgbm_v3_top18.pkl"))
    p.add_argument("--top",    type=int,  default=30)
    p.add_argument("--weight-method", choices=["equal", "hrp", "kelly", "blend"], default="equal")
    p.add_argument("--kelly-fraction", type=float, default=0.5)
    p.add_argument("--rf",     type=float, default=0.03, help="年化无风险利率")
    p.add_argument("--cost-bps", type=float, default=0.0,
                   help="单边交易成本（bps）：卖方印花税+佣金 ≈ 13 bps，买方佣金 ≈ 3 bps，"
                        "默认 0（不扣成本）。E3 成本修正建议使用 13")
    p.add_argument("--out",    type=Path, default=Path("reports/alpha_final/"))
    p.add_argument("--no-mlflow", action="store_true",
                   help="跳过 MLflow 实验记录（离线调试时使用）")
    p.add_argument("--mlflow-uri", type=str, default="mlruns",
                   help="MLflow tracking URI 或本地目录，默认 'mlruns'")
    p.add_argument("--phase", type=str, default="",
                   help="实验阶段标签（如 E3），写入 MLflow tag")
    p.add_argument("--model-version", type=str, default="v6",
                   help="模型版本标签，写入 MLflow tag，默认 'v6'")
    return p.parse_args(argv)


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_price_wide(prices_path: Path) -> pd.DataFrame:
    """从长格式价格面板读取 adj_close，pivot 为宽格式。
    
    Returns: DataFrame(index=trade_date DatetimeIndex, columns=ts_code)
    """
    import pyarrow.parquet as pq
    schema_names = pq.ParquetFile(str(prices_path)).schema_arrow.names
    price_col = "adj_close" if "adj_close" in schema_names else "close"
    print(f"  价格列: {price_col}")
    df = pd.read_parquet(prices_path, columns=["ts_code", "trade_date", price_col])
    wide = df.pivot_table(index="trade_date", columns="ts_code", values=price_col, aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    print(f"  价格宽表: {wide.shape}（{wide.index[0].date()} → {wide.index[-1].date()}）")
    return wide


def load_index_price(path: Path) -> pd.Series | None:
    """加载基准指数（如 CSI300）的价格序列。"""
    if path is None or not path.is_file():
        # 尝试从 alpha_prices_panel 中找 000300.SH
        return None
    df = pd.read_parquet(path)
    if "close" in df.columns:
        s = df["close"]
    elif "adj_close" in df.columns:
        s = df["adj_close"]
    else:
        s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def load_model_scores(panel: pd.DataFrame, model_path: Path) -> pd.Series:
    """运行模型打分，返回 MultiIndex(as_of, ticker) Score 序列。"""
    from quantmind.models.factor_model import FactorModel

    model = FactorModel.load(model_path)
    feat_names = getattr(model, "_feature_names", None) or model.feature_names
    if not feat_names:
        raise ValueError("模型无特征名")
    missing = [c for c in feat_names if c not in panel.columns]
    if missing:
        raise ValueError(f"panel 缺失特征：{missing[:5]}…")
    X = panel[list(feat_names)].to_numpy(dtype=np.float32, copy=True)
    pred = model.predict(X)
    return pd.Series(pred, index=panel.index, name="score")


# ─────────────────────────────────────────────
# 持仓权重
# ─────────────────────────────────────────────

def compute_weights(
    tickers: list[str],
    as_of_date,
    price_wide: pd.DataFrame,
    method: str,
    kelly_fraction: float,
) -> dict[str, float]:
    """计算调仓日持仓权重。"""
    n = len(tickers)
    if n == 0:
        return {}
    if method == "equal":
        return {t: 1.0 / n for t in tickers}

    from quantmind.portfolio.position_sizing import hrp_weights, kelly_weights, blend_weights

    cutoff = pd.Timestamp(as_of_date)
    avail = [t for t in tickers if t in price_wide.columns]
    if len(avail) < 3:
        return {t: 1.0 / n for t in tickers}

    hist = price_wide.loc[price_wide.index < cutoff, avail].dropna(how="all").tail(252)
    rets = hist.pct_change().dropna(how="all")

    try:
        if method in ("hrp", "blend"):
            w_hrp = hrp_weights(rets, min_periods=63)
        if method in ("kelly", "blend"):
            mu = rets.mean() * 252
            cov = rets.cov() * 252
            w_kelly = kelly_weights(mu, cov.values, fraction=kelly_fraction)
            w_kelly.index = mu.index

        if method == "hrp":
            w = w_hrp
        elif method == "kelly":
            w = w_kelly
        else:
            w = blend_weights(w_hrp, w_kelly, alpha=0.5)

        # 补上 price_wide 中没有的 ticker（用等权填充）
        missing_t = [t for t in tickers if t not in w.index]
        for mt in missing_t:
            w[mt] = 1.0 / n
        w = w.reindex(tickers).fillna(0)
        w /= w.sum()
        return w.to_dict()
    except Exception:
        return {t: 1.0 / n for t in tickers}


# ─────────────────────────────────────────────
# 核心 NAV 构建
# ─────────────────────────────────────────────

def _turnover_rate(prev_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    """计算两期持仓的单边换手率（0~1）。

    换手率 = 新增买入权重之和 = 所有股票 max(0, w_new - w_old) 之和
    等价于 1 - sum(min(w_old, w_new))，对卖出方对称。
    首期建仓（prev_weights为空）换手率 = 1.0。
    """
    if not prev_weights:
        return 1.0
    all_tickers = set(prev_weights) | set(new_weights)
    buys = sum(max(0.0, new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
               for t in all_tickers)
    return min(buys, 1.0)


def build_daily_nav(
    panel: pd.DataFrame,
    scores: pd.Series,
    price_wide: pd.DataFrame,
    top_n: int,
    weight_method: str,
    kelly_fraction: float,
    cost_bps: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame, list[dict]]:
    """从日频价格和季末打分构建日频 NAV（含可选交易成本）。

    Args:
        cost_bps: 单边交易成本（bps）。买入成本 = turnover * cost_bps/10000，
                  卖出成本 = turnover * cost_bps/10000（印花税+佣金合并为单边对称处理）。
                  E3 建议值：13 bps（卖方 0.10% 印花税 + 0.03% 佣金）。

    Returns:
        nav:           pd.Series（DatetimeIndex，策略净值，从 1.0 开始）
        holdings:      pd.DataFrame（每期持仓明细）
        turnover_log:  list[dict]（每期换手率 + 成本记录）
    """
    rebalance_dates = sorted(panel.index.get_level_values("as_of").unique())
    cost_rate = cost_bps / 10_000  # 单边成本率

    # 每期调仓：选 Top-N，记录调仓日 + 持仓 + 权重
    periods: list[dict] = []
    for i, as_of in enumerate(rebalance_dates):
        xs = panel.xs(as_of, level="as_of")
        sc = scores.xs(as_of, level="as_of").reindex(xs.index)
        valid_sc = sc.dropna()
        n = min(top_n, len(valid_sc))
        top_tickers = list(valid_sc.nlargest(n).index.astype(str))

        weights = compute_weights(top_tickers, as_of, price_wide, weight_method, kelly_fraction)

        # 持仓区间：as_of 后第一个交易日 → 下一个 as_of（含）
        entry_date = pd.Timestamp(as_of)
        exit_date = pd.Timestamp(rebalance_dates[i + 1]) if i + 1 < len(rebalance_dates) else price_wide.index[-1]

        periods.append({
            "as_of": as_of,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "tickers": top_tickers,
            "weights": weights,
        })

    # ── 日频 NAV 计算 ──
    all_dates = price_wide.index
    nav_series: dict[pd.Timestamp, float] = {}
    nav_val = 1.0
    prev_weights: dict[str, float] = {}
    turnover_log: list[dict] = []

    for period in periods:
        tickers = period["tickers"]
        w_dict = period["weights"]
        entry = period["entry_date"]
        exit_ = period["exit_date"]

        # ── 换手率 & 成本扣减（在持仓期首日前计入） ──
        turnover = _turnover_rate(prev_weights, w_dict)
        round_trip_cost = turnover * cost_rate * 2   # 买入 + 卖出各一次单边成本
        nav_val *= (1.0 - round_trip_cost)           # 直接从 NAV 扣除，首日前完成
        turnover_log.append({
            "as_of": str(period["as_of"]),
            "turnover": round(turnover, 4),
            "round_trip_cost": round(round_trip_cost, 6),
            "nav_after_cost": round(nav_val, 6),
        })
        prev_weights = dict(w_dict)

        # 持仓期内的交易日（entry 后一天 → exit）
        mask = (all_dates > entry) & (all_dates <= exit_)
        period_dates = all_dates[mask]

        for t in period_dates:
            t_idx = all_dates.get_loc(t)
            prev_idx = t_idx - 1

            day_ret = 0.0
            weight_sum = 0.0
            for ticker, w in w_dict.items():
                if ticker not in price_wide.columns:
                    continue
                p_t = price_wide.at[t, ticker]
                p_prev = price_wide.iloc[prev_idx][ticker]
                if pd.isna(p_t) or pd.isna(p_prev) or p_prev <= 0:
                    continue
                day_ret += w * (p_t / p_prev - 1)
                weight_sum += w

            if weight_sum > 1e-6:
                day_ret /= weight_sum

            nav_val *= (1 + day_ret)
            nav_series[t] = nav_val

    nav = pd.Series(nav_series, name="nav").sort_index()

    if len(nav) > 0:
        first_date = pd.Timestamp(periods[0]["entry_date"])
        nav_pre = pd.Series({first_date: 1.0}, name="nav")
        nav = pd.concat([nav_pre[~nav_pre.index.isin(nav.index)], nav]).sort_index()

    holdings_rows = []
    for p in periods:
        for ticker in p["tickers"]:
            holdings_rows.append({
                "as_of": p["as_of"],
                "ticker": ticker,
                "weight": p["weights"].get(ticker, 0.0),
                "entry_date": p["entry_date"],
                "exit_date": p["exit_date"],
            })
    holdings_df = pd.DataFrame(holdings_rows)

    return nav, holdings_df, turnover_log


# ─────────────────────────────────────────────
# 指数基准（CSI300）
# ─────────────────────────────────────────────

def build_benchmark_nav(price_wide: pd.DataFrame, nav: pd.Series) -> pd.Series | None:
    """提取 CSI300 基准净值（从 index_daily_panel 或 price_wide）。"""
    # 优先从 index_daily_panel 读
    idx_panel = ROOT / "data" / "raw" / "index_daily_panel.parquet"
    if idx_panel.is_file():
        try:
            idf = pd.read_parquet(idx_panel)
            csi = idf[idf["ts_code"] == "000300.SH"].copy()
            if len(csi) > 10:
                csi["trade_date"] = pd.to_datetime(csi["trade_date"])
                csi = csi.set_index("trade_date")["close"].sort_index()
                csi_aligned = csi.reindex(nav.index).ffill().bfill()
                start_val = csi_aligned.iloc[0]
                return (csi_aligned / start_val).rename("CSI300")
        except Exception:
            pass

    # 回退：从 alpha_prices_panel 中找
    candidates = ["000300.SH", "399300.SZ", "sh000300"]
    for c in candidates:
        if c in price_wide.columns:
            idx_prices = price_wide[c].reindex(nav.index).ffill()
            idx_prices = idx_prices.dropna()
            if len(idx_prices) < 10:
                continue
            start_val = idx_prices.iloc[0]
            return (idx_prices / start_val).rename("CSI300")
    return None


# ─────────────────────────────────────────────
# 绩效指标
# ─────────────────────────────────────────────

def calc_metrics(
    nav: pd.Series,
    rf: float = 0.03,
    turnover_log: list[dict] | None = None,
    cost_bps: float = 0.0,
) -> dict:
    """日频 NAV → 综合绩效指标（含成本摘要）。"""
    rets = nav.pct_change().dropna()
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    n_years = len(rets) / 252
    ann_ret = float((1 + total) ** (1 / max(n_years, 0.1)) - 1)
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)
    running_max = nav.cummax()
    dd = (nav / running_max - 1)
    max_dd = float(dd.min())
    calmar = ann_ret / (-max_dd + 1e-9) if max_dd < 0 else float("inf")
    monthly = nav.resample("ME").last().pct_change().dropna()
    win_rate = float((monthly > 0).mean()) if len(monthly) > 0 else float("nan")

    result = {
        "start_date": str(nav.index[0].date()),
        "end_date":   str(nav.index[-1].date()),
        "total_return":    round(total, 4),
        "ann_return":      round(ann_ret, 4),
        "ann_volatility":  round(ann_vol, 4),
        "sharpe_ratio":    round(sharpe, 3),
        "calmar_ratio":    round(calmar, 3),
        "max_drawdown":    round(max_dd, 4),
        "monthly_win_rate": round(win_rate, 4),
        "n_trading_days":  len(rets),
        "cost_bps":        cost_bps,
    }

    if turnover_log:
        turnovers = [r["turnover"] for r in turnover_log]
        costs = [r["round_trip_cost"] for r in turnover_log]
        result["avg_turnover"]       = round(float(np.mean(turnovers)), 4)
        result["avg_round_trip_cost"] = round(float(np.mean(costs)), 6)
        result["total_cost_drag"]    = round(float(np.prod([1 - c for c in costs])) - 1.0, 6)
        result["n_rebalances"]       = len(turnover_log)

    return result


# ─────────────────────────────────────────────
# HTML 报告
# ─────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Alpha 1374 日频真实 NAV 回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {{ font-family: 'Microsoft YaHei', sans-serif; background:#0f1117; color:#e0e0e0; margin:0; padding:20px; }}
  h1   {{ color:#00d4aa; font-size:22px; border-bottom:1px solid #333; padding-bottom:8px; }}
  h2   {{ color:#5bc8f5; font-size:16px; margin-top:30px; }}
  .metrics-grid {{ display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }}
  .metric-card  {{ background:#1e2130; border-radius:8px; padding:14px 18px; min-width:140px; }}
  .metric-card .val {{ font-size:24px; font-weight:bold; color:#00d4aa; }}
  .metric-card .lbl {{ font-size:12px; color:#888; margin-top:4px; }}
  .neg {{ color:#ff6b6b !important; }}
  #nav-chart, #dd-chart {{ width:100%; height:380px; background:#1e2130; border-radius:8px; margin:12px 0; }}
</style>
</head>
<body>
<h1>Alpha 1374 · 日频真实 NAV 回测报告</h1>
<p style="color:#888;font-size:12px;">模型: {model_name} · Top-{top_n} 季度调仓 · 权重方法: {weight_method} · 交易成本: {cost_note} · 生成: {gen_date}</p>

<h2>绩效指标</h2>
<div class="metrics-grid">
{metric_cards}
</div>

<h2>日频净值曲线</h2>
<div id="nav-chart"></div>

<h2>水下回撤曲线</h2>
<div id="dd-chart"></div>

<script>
const navDates = {nav_dates};
const navVals  = {nav_vals};
const ddVals   = {dd_vals};
const bmVals   = {bm_vals};

const navChart = echarts.init(document.getElementById('nav-chart'));
navChart.setOption({{
  backgroundColor:'#1e2130',
  tooltip:{{ trigger:'axis', axisPointer:{{ type:'cross' }} }},
  legend:{{ data:['策略 NAV'{bm_legend}], textStyle:{{ color:'#ccc' }} }},
  xAxis:{{ type:'category', data:navDates, axisLabel:{{ color:'#888', rotate:30 }} }},
  yAxis:{{ type:'value', axisLabel:{{ color:'#888' }}, splitLine:{{ lineStyle:{{ color:'#2a2a3a' }} }} }},
  series:[
    {{ name:'策略 NAV', type:'line', data:navVals, smooth:true,
      lineStyle:{{ color:'#00d4aa', width:2 }}, showSymbol:false }}{bm_series}
  ]
}});

const ddChart = echarts.init(document.getElementById('dd-chart'));
ddChart.setOption({{
  backgroundColor:'#1e2130',
  tooltip:{{ trigger:'axis' }},
  xAxis:{{ type:'category', data:navDates, axisLabel:{{ color:'#888', rotate:30 }} }},
  yAxis:{{ type:'value', axisLabel:{{ formatter:v=>v+'%', color:'#888' }},
           splitLine:{{ lineStyle:{{ color:'#2a2a3a' }} }} }},
  series:[
    {{ name:'回撤', type:'line', data:ddVals.map(v=>+(v*100).toFixed(2)),
      lineStyle:{{ color:'#ff6b6b', width:1.5 }},
      areaStyle:{{ color:'rgba(255,107,107,0.15)' }}, showSymbol:false }}
  ]
}});
window.addEventListener('resize', ()=>{{ navChart.resize(); ddChart.resize(); }});
</script>
</body>
</html>
"""


def build_html(
    nav: pd.Series,
    metrics: dict,
    bm_nav: pd.Series | None,
    args: argparse.Namespace,
) -> str:
    dates_str = [str(d.date()) for d in nav.index]
    nav_vals  = [round(v, 6) for v in nav.values]
    running_max = nav.cummax()
    dd_vals = [(v / m - 1) for v, m in zip(nav.values, running_max.values)]

    # 基准
    if bm_nav is not None:
        bm_nav_aligned = bm_nav.reindex(nav.index).ffill()
        bm_vals_str = json.dumps([round(v, 6) if pd.notna(v) else None
                                  for v in bm_nav_aligned.values])
        bm_legend = ", 'CSI300'"
        bm_series = (",\n    { name:'CSI300', type:'line', data:bmVals, smooth:true,"
                     " lineStyle:{color:'#f0a500',width:1.5}, showSymbol:false }")
    else:
        bm_vals_str = "[]"
        bm_legend = ""
        bm_series = ""

    def _fmt(v, pct=True):
        if v != v:
            return "—"
        return f"{v*100:.2f}%" if pct else f"{v:.3f}"

    def _card(val_str, label, neg=False):
        cls = ' neg' if neg else ''
        return (f'<div class="metric-card"><div class="val{cls}">{val_str}</div>'
                f'<div class="lbl">{label}</div></div>')

    cost_bps = metrics.get("cost_bps", 0.0)
    avg_to = metrics.get("avg_turnover", float("nan"))
    cost_drag = metrics.get("total_cost_drag", float("nan"))

    cards = [
        _card(_fmt(metrics.get("ann_return", float("nan"))),  "年化收益（含成本）" if cost_bps > 0 else "年化收益"),
        _card(_fmt(metrics.get("ann_volatility", float("nan"))), "年化波动率"),
        _card(_fmt(metrics.get("sharpe_ratio", float("nan")), pct=False), "Sharpe 比率"),
        _card(_fmt(metrics.get("max_drawdown", float("nan"))), "最大回撤", neg=True),
        _card(_fmt(metrics.get("calmar_ratio", float("nan")), pct=False), "Calmar 比率"),
        _card(_fmt(metrics.get("monthly_win_rate", float("nan"))), "月度胜率"),
        _card(_fmt(metrics.get("total_return", float("nan"))), "累计收益"),
        _card(f"{metrics.get('n_trading_days',0)}", "交易日数"),
    ]
    if cost_bps > 0:
        cards += [
            _card(f"{cost_bps:.0f} bps", "单边成本"),
            _card(f"{avg_to*100:.1f}%" if avg_to == avg_to else "—", "平均换手率"),
            _card(_fmt(cost_drag) if cost_drag == cost_drag else "—", "累计成本拖累", neg=True),
        ]

    cost_note = f"{cost_bps:.0f} bps/单边" if cost_bps > 0 else "不含成本"
    from datetime import date as dt_date
    return _HTML_TEMPLATE.format(
        model_name=args.model.name,
        top_n=args.top,
        weight_method=getattr(args, "weight_method", "equal"),
        cost_note=cost_note,
        gen_date=str(dt_date.today()),
        metric_cards="\n".join(cards),
        nav_dates=json.dumps(dates_str),
        nav_vals=json.dumps(nav_vals),
        dd_vals=json.dumps([round(d, 6) for d in dd_vals]),
        bm_vals=bm_vals_str,
        bm_legend=bm_legend,
        bm_series=bm_series,
    )


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    panel_path  = args.panel  if args.panel.is_absolute()  else ROOT / args.panel
    prices_path = args.prices if args.prices.is_absolute() else ROOT / args.prices
    model_path  = args.model  if args.model.is_absolute()  else ROOT / args.model
    out_dir     = args.out    if args.out.is_absolute()    else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[NAV Backtest] 加载因子面板：{panel_path}")
    panel = pd.read_parquet(panel_path)
    if "as_of" not in panel.index.names:
        raise ValueError("面板 index 需含 'as_of' level")
    n_periods = panel.index.get_level_values("as_of").nunique()
    print(f"  ✓ 面板：{panel.shape}，{n_periods} 个季度截面")

    print(f"[NAV Backtest] 加载价格面板：{prices_path}")
    price_wide = load_price_wide(prices_path)

    print(f"[NAV Backtest] 加载模型并打分：{model_path}")
    scores = load_model_scores(panel, model_path)

    cost_bps = getattr(args, "cost_bps", 0.0)
    cost_note = f"单边 {cost_bps:.0f} bps" if cost_bps > 0 else "不含成本"
    print(f"[NAV Backtest] 构建日频 NAV（Top-{args.top}，{getattr(args,'weight_method','equal')}，{cost_note}）…")
    nav, holdings_df, turnover_log = build_daily_nav(
        panel, scores, price_wide, args.top,
        weight_method=getattr(args, "weight_method", "equal"),
        kelly_fraction=getattr(args, "kelly_fraction", 0.5),
        cost_bps=cost_bps,
    )
    print(f"  ✓ NAV 序列长度：{len(nav)}，区间：{nav.index[0].date()} → {nav.index[-1].date()}")
    if cost_bps > 0 and turnover_log:
        avg_to = np.mean([r["turnover"] for r in turnover_log])
        total_cost = 1 - np.prod([1 - r["round_trip_cost"] for r in turnover_log])
        print(f"  ✓ 成本：{len(turnover_log)} 次调仓，平均换手率 {avg_to*100:.1f}%，"
              f"累计成本拖累 {total_cost*100:.2f}%")

    # 基准
    bm_nav = build_benchmark_nav(price_wide, nav)
    if bm_nav is not None:
        print(f"  ✓ 发现 CSI300 基准（{len(bm_nav)} 日）")
    else:
        print("  ⚠ 未找到 CSI300 基准，跳过对比")

    print("[NAV Backtest] 计算绩效指标…")
    metrics = calc_metrics(nav, rf=args.rf, turnover_log=turnover_log, cost_bps=cost_bps)
    print(f"  年化收益: {metrics['ann_return']*100:.2f}%  "
          f"Sharpe: {metrics['sharpe_ratio']:.3f}  "
          f"MaxDD: {metrics['max_drawdown']*100:.2f}%")
    if bm_nav is not None:
        bm_metrics = calc_metrics(bm_nav.reindex(nav.index).ffill().dropna(), rf=args.rf)
        print(f"  [CSI300]  年化收益: {bm_metrics['ann_return']*100:.2f}%  "
              f"Sharpe: {bm_metrics['sharpe_ratio']:.3f}  "
              f"MaxDD: {bm_metrics['max_drawdown']*100:.2f}%")
        metrics["benchmark_ann_return"] = bm_metrics["ann_return"]
        metrics["benchmark_sharpe"]     = bm_metrics["sharpe_ratio"]
        metrics["benchmark_max_dd"]     = bm_metrics["max_drawdown"]
        metrics["excess_ann_return"]    = round(metrics["ann_return"] - bm_metrics["ann_return"], 4)

    # 保存
    nav_csv = out_dir / "nav_daily.csv"
    nav.rename("nav").to_csv(nav_csv, header=True)
    print(f"[NAV Backtest] ✅ 日频 NAV 已写入：{nav_csv}")

    metrics_json = out_dir / "nav_metrics.json"
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[NAV Backtest] ✅ 绩效指标已写入：{metrics_json}")

    holdings_csv = out_dir / "nav_holdings.csv"
    holdings_df.to_csv(holdings_csv, index=False)
    print(f"[NAV Backtest] ✅ 持仓明细已写入：{holdings_csv}")

    if turnover_log:
        to_json = out_dir / "nav_turnover.json"
        to_json.write_text(json.dumps(turnover_log, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[NAV Backtest] ✅ 换手率明细已写入：{to_json}")

    print("[NAV Backtest] 生成 HTML 报告…")
    html = build_html(nav, metrics, bm_nav, args)
    html_path = out_dir / "nav_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[NAV Backtest] ✅ HTML 报告已写入：{html_path}")

    # ── MLflow 实验追踪 ────────────────────────────────────────────────────────
    if not getattr(args, "no_mlflow", False):
        try:
            from quantmind.workflow.mlflow_tracker import ExperimentTracker

            tracker = ExperimentTracker(
                tracking_uri=getattr(args, "mlflow_uri", "mlruns"),
            )
            run_id = tracker.log_backtest_run(
                params={
                    "weight_method":    getattr(args, "weight_method", "equal"),
                    "top_n":            args.top,
                    "kelly_fraction":   getattr(args, "kelly_fraction", 0.5),
                    "transaction_cost": getattr(args, "cost_bps", 0.0) / 10000,
                    "rf":               args.rf,
                    "phase":            getattr(args, "phase", ""),
                    "model_version":    getattr(args, "model_version", "v6"),
                },
                metrics={
                    "ann_return":        metrics.get("ann_return", float("nan")),
                    "sharpe":            metrics.get("sharpe_ratio", float("nan")),
                    "max_dd":            metrics.get("max_drawdown", float("nan")),
                    "total_return":      metrics.get("total_return", float("nan")),
                    "ann_volatility":    metrics.get("ann_volatility", float("nan")),
                    "calmar":            metrics.get("calmar_ratio", float("nan")),
                    "win_rate":          metrics.get("monthly_win_rate", float("nan")),
                    "avg_turnover":      metrics.get("avg_turnover", float("nan")),
                    "excess_ann_return": metrics.get("excess_ann_return", float("nan")),
                },
                artifacts_dir=str(out_dir),
                run_name=(
                    f"{getattr(args, 'weight_method', 'equal')}"
                    f"_{getattr(args, 'phase', '') or 'backtest'}"
                ),
            )
            print(f"[NAV Backtest] ✅ MLflow run_id: {run_id}")
            print(f"[NAV Backtest]    查看实验: mlflow ui --port 5000")
        except Exception as mlflow_exc:
            print(f"[NAV Backtest] ⚠ MLflow 记录失败（{mlflow_exc}），回测结果已保存到文件")


if __name__ == "__main__":
    main()
