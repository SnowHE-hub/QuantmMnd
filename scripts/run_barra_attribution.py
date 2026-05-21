"""E4 Barra 因子归因报告生成器.

功能：
  1. 读取 alpha_panel_v4.parquet
  2. 可选加载 LGBM 模型打分（用于 Pure Alpha IC 计算）
  3. 运行 BarraAttributor（截面 WLS，7 个风格因子 + 行业哑变量）
  4. 生成 reports/barra/ 下的 CSV + HTML 报告

用法：
  python scripts/run_barra_attribution.py \\
      --panel  data/panel/alpha_panel_v4.parquet \\
      --model  models/lgbm_v6_alpha.pkl \\
      --out    reports/barra/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantmind.risk.barra import BarraAttributor, BarraResult, STYLE_MAP


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",  type=Path, default=Path("data/panel/alpha_panel_v4.parquet"))
    p.add_argument("--model",  type=Path, default=Path("models/lgbm_v6_alpha.pkl"),
                   help="LGBM 模型路径（用于 Pure Alpha IC，传 none 跳过）")
    p.add_argument("--label",  default="forward_return_63d")
    p.add_argument("--out",    type=Path, default=Path("reports/barra/"))
    return p.parse_args(argv)


# ─────────────────────────────────────────────
# 模型打分
# ─────────────────────────────────────────────

def _load_scores(panel: pd.DataFrame, model_path: Path) -> pd.Series | None:
    if str(model_path).lower() == "none" or not model_path.is_file():
        return None
    try:
        from quantmind.models.factor_model import FactorModel
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = FactorModel.load(model_path)
        feat = getattr(model, "_feature_names", None) or model.feature_names
        missing = [c for c in feat if c not in panel.columns]
        if missing:
            print(f"  [warn] 模型特征缺失 {len(missing)} 个，跳过打分")
            return None
        X = panel[list(feat)].to_numpy(dtype=np.float32, copy=True)
        pred = model.predict(X)
        print(f"  ✓ 模型打分完成（{len(pred)} 行）")
        return pd.Series(pred, index=panel.index, name="score")
    except Exception as e:
        print(f"  [warn] 模型加载失败: {e}")
        return None


# ─────────────────────────────────────────────
# HTML 报告
# ─────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Barra 因子归因报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {{ font-family: 'Microsoft YaHei', sans-serif; background:#0f1117; color:#e0e0e0; margin:0; padding:20px; }}
  h1   {{ color:#00d4aa; font-size:22px; border-bottom:1px solid #333; padding-bottom:8px; }}
  h2   {{ color:#5bc8f5; font-size:16px; margin-top:30px; }}
  .card-row {{ display:flex; flex-wrap:wrap; gap:12px; margin:16px 0; }}
  .card {{ background:#1e2130; border-radius:8px; padding:14px 20px; min-width:150px; }}
  .card .val {{ font-size:22px; font-weight:bold; color:#00d4aa; }}
  .card .lbl {{ font-size:12px; color:#888; margin-top:4px; }}
  .card.warn .val {{ color:#f0a500; }}
  .card.bad  .val {{ color:#ff6b6b; }}
  table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:13px; }}
  th {{ background:#1e2130; padding:8px 12px; text-align:left; color:#5bc8f5; }}
  td {{ padding:6px 12px; border-bottom:1px solid #222; }}
  tr:hover td {{ background:#1a1f2e; }}
  .pos {{ color:#00d4aa; }} .neg {{ color:#ff6b6b; }}
  #ic-chart, #cumret-chart, #r2-chart, #ind-chart {{ width:100%; height:360px;
    background:#1e2130; border-radius:8px; margin:12px 0; }}
</style>
</head>
<body>
<h1>Barra 因子归因报告（E4）</h1>
<p style="color:#888;font-size:12px;">面板：{panel_name} · 标签：{label} · 期数：{n_periods} · 生成：{gen_date}</p>

<h2>核心摘要</h2>
<div class="card-row">
  <div class="card"><div class="val">{avg_r2}</div><div class="lbl">平均截面 R²</div></div>
  <div class="card {raw_cls}"><div class="val">{raw_ic}</div><div class="lbl">原始 IC（模型 vs 前瞻收益）</div></div>
  <div class="card {resid_cls}"><div class="val">{resid_ic}</div><div class="lbl">Pure Alpha IC（残差 IC）</div></div>
  <div class="card"><div class="val">{raw_icir}</div><div class="lbl">原始 ICIR</div></div>
  <div class="card"><div class="val">{resid_icir}</div><div class="lbl">Pure Alpha ICIR</div></div>
  <div class="card"><div class="val">{n_factors}</div><div class="lbl">因子数（风格+行业）</div></div>
</div>

<h2>风格因子 IC 均值（因子收益稳定性）</h2>
<div id="ic-chart"></div>

<h2>风格因子累计收益</h2>
<div id="cumret-chart"></div>

<h2>截面 R² 时序（模型解释力）</h2>
<div id="r2-chart"></div>

{ind_section}

<h2>风格因子明细</h2>
<table>
<tr><th>因子</th><th>IC均值</th><th>ICIR</th><th>年化收益均值</th><th>收益波动率</th><th>胜率（>0）</th></tr>
{factor_rows}
</table>

<script>
const factors   = {factors_json};
const icVals    = {ic_json};
const icirVals  = {icir_json};
const dates     = {dates_json};
const cumRetMap = {cumret_json};
const r2Dates   = {r2dates_json};
const r2Vals    = {r2vals_json};

// IC Bar
const icChart = echarts.init(document.getElementById('ic-chart'));
const colors = icVals.map(v => v >= 0 ? '#00d4aa' : '#ff6b6b');
icChart.setOption({{
  backgroundColor:'#1e2130',
  tooltip:{{ trigger:'axis' }},
  xAxis:{{ type:'category', data:factors, axisLabel:{{ color:'#ccc' }} }},
  yAxis:{{ type:'value', axisLabel:{{ color:'#888' }}, splitLine:{{ lineStyle:{{color:'#2a2a3a'}} }} }},
  series:[
    {{ name:'IC均值', type:'bar', data:icVals, itemStyle:{{ color:(p)=>colors[p.dataIndex] }} }},
    {{ name:'ICIR', type:'line', data:icirVals, yAxisIndex:0,
       lineStyle:{{color:'#f0a500',width:2}}, symbol:'circle', symbolSize:6 }}
  ],
  legend:{{ data:['IC均值','ICIR'], textStyle:{{color:'#ccc'}} }}
}});

// Cumulative Return
const cumRetChart = echarts.init(document.getElementById('cumret-chart'));
const cumSeries = factors.map(f => ({{
  name: f,
  type: 'line',
  data: cumRetMap[f] || [],
  smooth: true,
  showSymbol: false,
  lineStyle: {{ width: 1.5 }},
}}));
cumRetChart.setOption({{
  backgroundColor:'#1e2130',
  tooltip:{{ trigger:'axis' }},
  legend:{{ type:'scroll', textStyle:{{color:'#ccc'}}, top:5 }},
  xAxis:{{ type:'category', data:dates, axisLabel:{{color:'#888', rotate:30}} }},
  yAxis:{{ type:'value', axisLabel:{{color:'#888'}}, splitLine:{{lineStyle:{{color:'#2a2a3a'}}}} }},
  series: cumSeries
}});

// R² timeseries
const r2Chart = echarts.init(document.getElementById('r2-chart'));
r2Chart.setOption({{
  backgroundColor:'#1e2130',
  tooltip:{{ trigger:'axis' }},
  xAxis:{{ type:'category', data:r2Dates, axisLabel:{{color:'#888', rotate:30}} }},
  yAxis:{{ type:'value', min:0, max:1, axisLabel:{{color:'#888'}}, splitLine:{{lineStyle:{{color:'#2a2a3a'}}}} }},
  series:[{{ name:'R²', type:'line', data:r2Vals, smooth:true,
    lineStyle:{{color:'#5bc8f5',width:2}}, areaStyle:{{color:'rgba(91,200,245,0.1)'}}, showSymbol:false }}]
}});

window.addEventListener('resize', ()=>{{ icChart.resize(); cumRetChart.resize(); r2Chart.resize(); }});
</script>
{ind_script}
</body>
</html>"""

_IND_SECTION = """
<h2>行业因子收益（Top 行业）</h2>
<div id="ind-chart"></div>
"""

_IND_SCRIPT = """
<script>
const indChart = echarts.init(document.getElementById('ind-chart'));
const indData  = {ind_json};
const indSeries = Object.keys(indData).slice(0,10).map(k => ({{
  name: k.replace('ind_',''),
  type: 'bar',
  stack: 'ind',
  data: indData[k],
}}));
indChart.setOption({{
  backgroundColor:'#1e2130',
  tooltip:{{trigger:'axis', axisPointer:{{type:'shadow'}}}},
  legend:{{ type:'scroll', textStyle:{{color:'#ccc'}}, top:5 }},
  xAxis:{{ type:'category', data:{ind_dates_json}, axisLabel:{{color:'#888', rotate:30}} }},
  yAxis:{{ type:'value', axisLabel:{{color:'#888'}}, splitLine:{{lineStyle:{{color:'#2a2a3a'}}}} }},
  series: indSeries
}});
window.addEventListener('resize', ()=>{{ indChart.resize(); }});
</script>
"""


def _build_html(result: BarraResult, args: argparse.Namespace) -> str:
    fr = result.factor_returns
    style_names = [c for c in fr.columns]

    # IC 数值
    ic_vals   = [round(float(result.factor_ic.get(f, 0)), 4) for f in style_names]
    icir_vals = [round(float(result.factor_icir.get(f, 0)), 4) for f in style_names]

    # 累计收益
    dates_list = [str(d)[:10] for d in fr.index]
    cumret_map: dict[str, list] = {}
    for col in style_names:
        s = fr[col].dropna()
        cum = (1 + s).cumprod()
        cumret_map[col] = [round(v, 6) for v in cum.reindex(fr.index).ffill().fillna(1.0)]

    # R²
    r2_series = result.r_squared.reindex(fr.index).ffill().fillna(0)
    r2_dates = [str(d)[:10] for d in r2_series.index]
    r2_vals  = [round(float(v), 4) for v in r2_series.values]

    # 行业图
    if not result.industry_returns.empty:
        ir = result.industry_returns
        top_inds = ir.abs().mean().nlargest(10).index.tolist()
        ind_json_data = {k: [round(float(v), 4) for v in ir[k].reindex(fr.index).fillna(0)] for k in top_inds}
        ind_section = _IND_SECTION
        ind_script  = _IND_SCRIPT.format(
            ind_json=json.dumps(ind_json_data, ensure_ascii=False),
            ind_dates_json=json.dumps(dates_list),
        )
    else:
        ind_section = ""
        ind_script  = ""

    # 因子明细行
    factor_rows = []
    for f in style_names:
        vals = fr[f].dropna()
        ic  = result.factor_ic.get(f, float("nan"))
        icir = result.factor_icir.get(f, float("nan"))
        win = float((vals > 0).mean()) if len(vals) > 0 else float("nan")
        ic_cls = "pos" if ic > 0 else "neg"
        factor_rows.append(
            f'<tr><td>{f}</td>'
            f'<td class="{ic_cls}">{ic:+.4f}</td>'
            f'<td class="{ic_cls}">{icir:+.3f}</td>'
            f'<td>{vals.mean()*100:+.3f}%</td>'
            f'<td>{vals.std()*100:.3f}%</td>'
            f'<td>{win*100:.1f}%</td></tr>'
        )

    def _fmt(v, decimals=4):
        if v != v:
            return "—"
        return f"{v:+.{decimals}f}"

    raw_ic_v   = result.raw_ic
    resid_ic_v = result.residual_ic
    raw_cls    = "warn" if abs(raw_ic_v) < 0.03 else ""
    resid_cls  = "warn" if abs(resid_ic_v) < 0.03 else ""

    return _HTML.format(
        panel_name=args.panel.name,
        label=args.label,
        n_periods=result.n_periods,
        gen_date=str(date.today()),
        avg_r2=f"{result.meta.get('avg_r2', 0):.3f}",
        raw_ic=_fmt(raw_ic_v),
        resid_ic=_fmt(resid_ic_v),
        raw_icir=_fmt(result.raw_icir, 3),
        resid_icir=_fmt(result.residual_icir, 3),
        n_factors=len(style_names) + (result.industry_returns.shape[1] if not result.industry_returns.empty else 0),
        raw_cls=raw_cls,
        resid_cls=resid_cls,
        factors_json=json.dumps(style_names),
        ic_json=json.dumps(ic_vals),
        icir_json=json.dumps(icir_vals),
        dates_json=json.dumps(dates_list),
        cumret_json=json.dumps(cumret_map, ensure_ascii=False),
        r2dates_json=json.dumps(r2_dates),
        r2vals_json=json.dumps(r2_vals),
        factor_rows="\n".join(factor_rows),
        ind_section=ind_section,
        ind_script=ind_script,
    )


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    panel_path = args.panel if args.panel.is_absolute() else ROOT / args.panel
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    out_dir    = args.out   if args.out.is_absolute()   else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Barra] 加载面板：{panel_path}")
    panel = pd.read_parquet(panel_path)
    n_periods = panel.index.get_level_values("as_of").nunique()
    print(f"  ✓ 面板：{panel.shape}，{n_periods} 期")

    print(f"[Barra] 加载模型打分…")
    scores = _load_scores(panel, model_path)

    print("[Barra] 运行 Barra 截面归因…")
    attr = BarraAttributor(label_col=args.label)
    result = attr.fit(panel, scores=scores)

    if result.n_periods == 0:
        print("[Barra] ✗ 无有效期次，退出")
        return

    # ── 保存 CSV ──────────────────────────────────────────────────────────
    fr_path = out_dir / "factor_returns.csv"
    result.factor_returns.to_csv(fr_path)
    print(f"[Barra] ✅ 因子收益：{fr_path}")

    if not result.industry_returns.empty:
        ir_path = out_dir / "industry_returns.csv"
        result.industry_returns.to_csv(ir_path)
        print(f"[Barra] ✅ 行业收益：{ir_path}")

    r2_path = out_dir / "r_squared.csv"
    result.r_squared.to_csv(r2_path, header=["r_squared"])
    print(f"[Barra] ✅ R² 时序：{r2_path}")

    # ── 保存指标 JSON ─────────────────────────────────────────────────────
    metrics = {
        "n_periods":       result.n_periods,
        "avg_r2":          round(float(result.r_squared.mean()), 4),
        "raw_ic":          round(result.raw_ic, 4) if result.raw_ic == result.raw_ic else None,
        "raw_icir":        round(result.raw_icir, 4) if result.raw_icir == result.raw_icir else None,
        "residual_ic":     round(result.residual_ic, 4) if result.residual_ic == result.residual_ic else None,
        "residual_icir":   round(result.residual_icir, 4) if result.residual_icir == result.residual_icir else None,
        "style_factors":   result.style_names,
        "factor_ic_mean":  {k: round(float(v), 4) for k, v in result.factor_ic.items()},
        "factor_icir":     {k: round(float(v), 3) for k, v in result.factor_icir.items()},
    }
    met_path = out_dir / "barra_metrics.json"
    met_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Barra] ✅ 指标 JSON：{met_path}")

    # ── 打印摘要 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  平均截面 R²     : {metrics['avg_r2']:.3f}")
    print(f"  原始 IC / ICIR  : {metrics['raw_ic']} / {metrics['raw_icir']}")
    print(f"  Pure Alpha IC/IR: {metrics['residual_ic']} / {metrics['residual_icir']}")
    print(f"\n  风格因子 IC 均值（绝对值大→因子显著）:")
    for f, ic in sorted(metrics["factor_ic_mean"].items(), key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * int(abs(ic) * 200)
        sign = "+" if ic > 0 else ""
        print(f"    {f:<12} {sign}{ic:.4f}  {bar}")
    print(f"{'─'*55}\n")

    # ── HTML 报告 ─────────────────────────────────────────────────────────
    print("[Barra] 生成 HTML 报告…")
    html = _build_html(result, args)
    html_path = out_dir / "barra_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[Barra] ✅ HTML 报告：{html_path}")


if __name__ == "__main__":
    main()
