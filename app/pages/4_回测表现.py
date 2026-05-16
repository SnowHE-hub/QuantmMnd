"""因子分层 vs 引擎回测（只读本地指标 + AI 解读）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import charts
from app.utils.data_loader import (
    load_backtest_metrics,
    load_engine_metrics,
    load_holdout_metrics,
    load_index_close,
    load_quintile_returns_csv,
)
from app.utils.llm_commentator import LLMCommentator, analyze_chart_cached, cached_llm_comment
from app.utils.metric_cards import METRIC_EXPLANATIONS, metric_with_explanation

st.set_page_config(page_title="回测表现", layout="wide")
st.title("📉 回测表现")

# 优先读 alpha v3 回测结果，再退回旧路径
_ALPHA_BT = "reports/alpha_final/backtest_metrics.json"
_LEGACY_BT = "reports/final_eval/backtest_metrics.json"
_ALPHA_QR = "reports/alpha_final/quintile_returns_by_period.csv"
_LEGACY_QR = "reports/final_eval/quintile_returns_by_period.csv"

fe_alpha = load_backtest_metrics(_ALPHA_BT)
fe_legacy = load_backtest_metrics(_LEGACY_BT)
# 优先用 alpha，若 alpha 没有 layer_summary 则退回 legacy
fe = fe_alpha if fe_alpha.get("layer_summary") else fe_legacy
_using_alpha = bool(fe_alpha.get("layer_summary"))

ho = load_holdout_metrics()
seg24 = fe.get("segment_2024") or {}
seg_oos_h = ho.get("segment_oos") or {}

if _using_alpha:
    st.info("📊 当前展示 **Alpha 1374 v3 回测**（`reports/alpha_final/`），模型 `lgbm_v3_top18`，标签 `forward_return_63d`。")
cmt = LLMCommentator()

with st.sidebar:
    st.markdown("#### 📘 IC 模板解读")
    periods = int(seg24.get("periods") or fe.get("periods") or 4)
    if seg24.get("ic_mean") is not None and seg24.get("ic_ir") is not None:
        st.markdown(cmt.explain_ic(float(seg24["ic_mean"]), float(seg24["ic_ir"]), periods))

tab1, tab2 = st.tabs(["因子分层（无成本）", "引擎回测（含成本）"])

with tab1:
    st.subheader("📊 核心指标（含解释）")
    ic_oos = seg_oos_h.get("ic_mean")
    ic_oos_txt = f"{float(ic_oos):.4f}" if ic_oos is not None else "—"
    delta_hold = None
    try:
        if ic_oos is not None and seg24.get("ic_mean") is not None:
            delta_hold = f"Δvs测试 {float(ic_oos)-float(seg24['ic_mean']):+.4f}"
    except (TypeError, ValueError):
        pass

    mcols = st.columns(5)
    metric_specs = [
        ("OOS / 保留 IC_mean", ic_oos_txt, delta_hold, "IC_mean"),
        ("Q5 年化（无成本）", f"{float(seg24['q5_ann'])*100:.2f}%" if seg24.get("q5_ann") is not None else "—", "Test 报告口径", "Q5_ann"),
        ("多空夏普", f"{float(seg24['ls_sharpe']):.3f}" if seg24.get("ls_sharpe") is not None else "—", "分层组合", "Sharpe"),
        ("Q5 最大回撤", f"{float(seg24['max_dd_q5'])*100:.2f}%" if seg24.get("max_dd_q5") is not None else "—", "季度近似", "max_drawdown"),
        ("测试集 ICIR", f"{float(seg24['ic_ir']):.4f}" if seg24.get("ic_ir") is not None else "—", "2024 H/O", "ICIR"),
    ]
    for col, (lab, val, delta, expl_key) in zip(mcols, metric_specs):
        with col:
            metric_with_explanation(lab, val, METRIC_EXPLANATIONS.get(expl_key, "指标说明待补充。"), delta)

    layers = fe.get("layer_summary") or []
    qdf = load_quintile_returns_csv()
    if not layers:
        st.info("缺少 `reports/final_eval/backtest_metrics.json`。")
    else:
        layers = sorted(layers, key=lambda x: str(x.get("quantile", "")))
        qs = [str(x.get("quantile")) for x in layers]
        ms = [float(x.get("mean_monthly") or 0) for x in layers]
        q5_m = ms[qs.index("Q5")] if "Q5" in qs else float(ms[-1])
        q1_m = ms[qs.index("Q1")] if "Q1" in qs else float(ms[0])
        spread_v = q5_m - q1_m
        commentary = analyze_chart_cached(
            "quintile_bar",
            {"q5": q5_m, "q1": q1_m, "spread": spread_v},
            cache_key="fe_layer",
        )
        st.success(f"🤖 **图表解读**：{commentary}")
        st.plotly_chart(charts.quintile_bar_with_cumulative_spread(qdf, qs, ms), use_container_width=True)

    if not qdf.empty:
        ic_proxy = qdf.pivot_table(index="as_of", columns="quantile", values="mean_return", aggfunc="mean")
        spread_series = (ic_proxy["Q5"] - ic_proxy["Q1"]).dropna() if "Q5" in ic_proxy.columns and "Q1" in ic_proxy.columns else pd.Series(dtype=float)
        ic_comment = analyze_chart_cached(
            "ic_series",
            {
                "n": int(len(spread_series)),
                "mean": float(spread_series.mean()) if len(spread_series) else 0.0,
                "std": float(spread_series.std()) if len(spread_series) else 0.0,
            },
            cache_key="ic_proxy",
        )
        st.markdown(f"🤖 **IC 时序特征（近似）**：{ic_comment}")
        st.caption("说明：仓库未导出逐期 Rank IC，下图用 **Q5−Q1 价差** 近似多空分化；横轴为 `as_of`。")
        st.plotly_chart(charts.ic_spread_timeseries(qdf), use_container_width=True)
        seg24_ic = seg24.get("ic_mean")
        if seg24_ic is not None:
            st.plotly_chart(
                charts.line_ic_proxy(sorted(qdf["as_of"].unique()), float(seg24_ic)),
                use_container_width=True,
            )

    st.subheader("保留集 vs 测试集（结构化摘要）")
    h24 = seg24
    hho = seg_oos_h
    comp_rows = [
        {
            "segment": "Test 2024",
            "IC": float(h24["ic_mean"]) if h24.get("ic_mean") is not None else None,
            "ICIR": float(h24["ic_ir"]) if h24.get("ic_ir") is not None else None,
            "Q5年化": float(h24["q5_ann"]) if h24.get("q5_ann") is not None else None,
        },
        {
            "segment": "Holdout（reports/holdout）",
            "IC": float(hho["ic_mean"]) if hho.get("ic_mean") is not None else None,
            "ICIR": float(hho["ic_ir"]) if hho.get("ic_ir") is not None else None,
            "Q5年化": float(hho["q5_ann"]) if hho.get("q5_ann") is not None else None,
        },
    ]
    cdf = pd.DataFrame(comp_rows)
    st.dataframe(cdf, use_container_width=True, hide_index=True)

    if cdf["IC"].notna().any():
        fig_ic = charts.ic_compare_bar(
            [str(x) for x in cdf["segment"]],
            [float(x) if x == x else 0.0 for x in cdf["IC"]],
        )
        st.plotly_chart(fig_ic, use_container_width=True)
        cmp_txt = cached_llm_comment(
            "请用 2~3 句中文对比测试集与保留集 IC_mean / ICIR / Q5年化（基于下表 JSON），判断泛化是否可信。\n"
            + cdf.to_json(orient="records", force_ascii=False),
            cache_key="seg_compare_llm",
            max_tokens=260,
        )
        st.info(cmp_txt)

with tab2:
    eng = load_engine_metrics()
    idx = load_index_close("000300.SH", days=900)
    bench_ret = 0.0
    if idx.empty:
        st.warning("未加载到 CSI300 指数行情（index_daily_panel）。")
    else:
        idx_win = idx[(idx["trade_date"] >= "2023-01-01") & (idx["trade_date"] <= "2024-12-31")].copy()
        tr = float(eng.get("total_return") or 0.0)
        n = len(idx_win)
        if n >= 2:
            nav = 1.0 + np.linspace(0.0, tr, n)
            strat_df = pd.DataFrame({"trade_date": idx_win["trade_date"].values, "strat_nav": nav})
            st.plotly_chart(charts.nav_compare(idx_win, strat_df), use_container_width=True)
            bench_ret = float(idx_win["close"].iloc[-1] / idx_win["close"].iloc[0] - 1.0)
        st.caption("策略净值按「区间内线性到达 total_return」近似，用于与基准形状对比。")

    nav_note = analyze_chart_cached(
        "nav_curve",
        {"strat_ret": float(eng.get("total_return") or 0), "bench_ret": bench_ret},
        "nav_comment_only",
    )
    st.markdown(f"🤖 **净值解读**：{nav_note}")

    def _sharpe_simple(ret: pd.Series, rf: float = 0.03):
        x = ret.dropna().values.astype(float)
        if x.size < 5:
            return float("nan")
        rf_d = (1 + rf) ** (1 / 252) - 1
        ex = x - rf_d
        s = float(np.std(ex, ddof=1))
        if s < 1e-12:
            return float("nan")
        return float(np.sqrt(252) * np.mean(ex) / s)

    bench_sh = float("nan")
    max_dd_b = float("nan")
    if not idx.empty:
        idf = idx.sort_values("trade_date")
        mask = (idf["trade_date"] >= "2023-01-01") & (idf["trade_date"] <= "2024-12-31")
        r = idf.loc[mask, "close"].pct_change()
        bench_sh = _sharpe_simple(r)
        cum = idf.loc[mask, "close"] / idf.loc[mask, "close"].iloc[0]
        max_dd_b = float((cum / cum.cummax() - 1).min())

    st.markdown("### 指标对比（示意区间 2023–2024）")
    st.table(
        pd.DataFrame(
            {
                "指标": ["CAGR", "Sharpe", "Max DD"],
                "策略（含成本）": [
                    f'{float(eng.get("cagr", 0))*100:.2f}%' if eng.get("cagr") is not None else "—",
                    f'{float(eng.get("sharpe", 0)):.3f}' if eng.get("sharpe") is not None else "—",
                    f'{float(eng.get("max_drawdown", 0))*100:.2f}%' if eng.get("max_drawdown") is not None else "—",
                ],
                "CSI300（持有近似）": [
                    "—",
                    f"{bench_sh:.3f}" if bench_sh == bench_sh else "—",
                    f"{max_dd_b*100:.2f}%" if max_dd_b == max_dd_b else "—",
                ],
            }
        )
    )

    ic24 = fe.get("segment_2024", {}).get("ic_mean")
    ich = ho.get("segment_oos", {}).get("ic_mean")
    if ic24 is not None or ich is not None:
        st.plotly_chart(
            charts.ic_compare_bar(
                ["Test 2024", "Holdout 25–26"],
                [float(ic24 or 0), float(ich or 0)],
            ),
            use_container_width=True,
        )
