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

tab1, tab2, tab3 = st.tabs(["因子分层（无成本）", "引擎回测（含成本）", "Barra 风险归因"])

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
    import json
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    NAV_BASE = ROOT / "reports" / "nav_v4_cost13"
    NAV_NOCOST_BASE = ROOT / "reports" / "nav_v4"
    METHODS = ["kelly", "blend", "equal", "hrp"]
    METHOD_LABELS = {"kelly": "Kelly（最优）", "blend": "Blend", "equal": "等权", "hrp": "HRP"}
    METHOD_COLORS = {"kelly": "#00d4aa", "blend": "#5bc8f5", "equal": "#f0a500", "hrp": "#ff6b6b"}

    # ── 数据加载 ─────────────────────────────────────────────────────────────
    def _load_nav_csv(method: str, with_cost: bool) -> pd.Series:
        base = NAV_BASE if with_cost else NAV_NOCOST_BASE
        p = base / method / "nav_daily.csv"
        if not p.is_file():
            return pd.Series(dtype=float)
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df.iloc[:, 0].sort_index()

    def _load_metrics(method: str, with_cost: bool) -> dict:
        base = NAV_BASE if with_cost else NAV_NOCOST_BASE
        p = base / method / "nav_metrics.json"
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    def _load_turnover(method: str) -> list[dict]:
        p = NAV_BASE / method / "nav_turnover.json"
        if not p.is_file():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    # ── 方法选择器 ────────────────────────────────────────────────────────────
    selected = st.selectbox(
        "权重方法",
        METHODS,
        format_func=lambda m: METHOD_LABELS[m],
        index=0,
    )
    m = _load_metrics(selected, with_cost=True)
    nav_series = _load_nav_csv(selected, with_cost=True)
    nav_nocost = _load_nav_csv(selected, with_cost=False)
    turnover_log = _load_turnover(selected)

    if nav_series.empty:
        st.warning(f"未找到 `reports/nav_v4_cost13/{selected}/nav_daily.csv`，请先运行 E2 回测。")
    else:
        # ── 核心指标卡 ─────────────────────────────────────────────────────
        st.subheader("📊 核心指标（含 13 bps 单边交易成本）")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        def _pct(k): return f"{m.get(k, 0)*100:.2f}%" if m.get(k) is not None else "—"
        def _f3(k):  return f"{m.get(k, 0):.3f}"       if m.get(k) is not None else "—"
        c1.metric("年化收益（含成本）", _pct("ann_return"),
                  f"超额 +{m.get('excess_ann_return',0)*100:.2f}%" if m.get('excess_ann_return') else None)
        c2.metric("Sharpe 比率", _f3("sharpe_ratio"))
        c3.metric("最大回撤", _pct("max_drawdown"))
        c4.metric("月度胜率", _pct("monthly_win_rate"))
        c5.metric("平均换手率", f"{m.get('avg_turnover',0)*100:.1f}%")
        c6.metric("累计成本拖累", f"{m.get('total_cost_drag',0)*100:.2f}%")

        st.caption(
            f"回测区间：{m.get('start_date','?')} → {m.get('end_date','?')}  |  "
            f"CSI300 年化：{m.get('benchmark_ann_return',0)*100:.2f}%  |  "
            f"调仓次数：{m.get('n_rebalances',0)} 次（季度频）"
        )

        # ── 日频 NAV 曲线 ─────────────────────────────────────────────────
        st.subheader("📈 日频净值曲线")
        fig_nav = go.Figure()
        color = METHOD_COLORS[selected]
        fig_nav.add_trace(go.Scatter(
            x=nav_series.index, y=nav_series.values,
            name=f"{METHOD_LABELS[selected]}（含成本）",
            line=dict(color=color, width=2), mode="lines",
        ))
        if not nav_nocost.empty:
            fig_nav.add_trace(go.Scatter(
                x=nav_nocost.index, y=nav_nocost.values,
                name=f"{METHOD_LABELS[selected]}（无成本）",
                line=dict(color=color, width=1.5, dash="dot"), mode="lines",
            ))

        # 加载 CSI300 基准（从 index_daily_panel）
        idx_df = load_index_close("000300.SH", days=3000)
        if not idx_df.empty:
            idx_df = idx_df.sort_values("trade_date")
            idx_df["trade_date"] = pd.to_datetime(idx_df["trade_date"])
            mask = (idx_df["trade_date"] >= nav_series.index[0]) & (idx_df["trade_date"] <= nav_series.index[-1])
            idx_win = idx_df[mask]
            if len(idx_win) >= 2:
                bm = idx_win["close"] / idx_win["close"].iloc[0]
                fig_nav.add_trace(go.Scatter(
                    x=idx_win["trade_date"], y=bm.values,
                    name="CSI300", line=dict(color="#f0a500", width=1.5, dash="dash"), mode="lines",
                ))

        fig_nav.update_layout(
            height=420, template="plotly_dark",
            legend=dict(orientation="h", y=-0.15),
            yaxis_title="净值", xaxis_title="日期",
            hovermode="x unified",
        )
        st.plotly_chart(fig_nav, use_container_width=True)

        # ── 水下回撤 ──────────────────────────────────────────────────────
        running_max = nav_series.cummax()
        dd = (nav_series / running_max - 1) * 100
        fig_dd = go.Figure(go.Scatter(
            x=dd.index, y=dd.values,
            fill="tozeroy", fillcolor="rgba(255,107,107,0.2)",
            line=dict(color="#ff6b6b", width=1), mode="lines", name="回撤",
        ))
        fig_dd.update_layout(height=200, template="plotly_dark",
                             yaxis_title="回撤 %", margin=dict(t=20))
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── 四方法对比表 ───────────────────────────────────────────────────
        st.subheader("🔀 四种权重方法对比（v6 模型，13 bps 成本）")
        rows = []
        for mth in METHODS:
            mm = _load_metrics(mth, with_cost=True)
            mn = _load_metrics(mth, with_cost=False)
            if not mm:
                continue
            rows.append({
                "权重方法": METHOD_LABELS[mth],
                "年化收益（含成本）": f"{mm.get('ann_return',0)*100:.2f}%",
                "年化收益（无成本）": f"{mn.get('ann_return',0)*100:.2f}%",
                "Sharpe": f"{mm.get('sharpe_ratio',0):.3f}",
                "最大回撤": f"{mm.get('max_drawdown',0)*100:.2f}%",
                "月度胜率": f"{mm.get('monthly_win_rate',0)*100:.1f}%",
                "平均换手率": f"{mm.get('avg_turnover',0)*100:.1f}%",
                "成本拖累": f"{mm.get('total_cost_drag',0)*100:.2f}%",
            })
        if rows:
            cmp_df = pd.DataFrame(rows)
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        # ── 换手率时序 ────────────────────────────────────────────────────
        if turnover_log:
            st.subheader("🔄 逐期换手率 & 成本")
            to_df = pd.DataFrame(turnover_log)
            to_df["as_of"] = pd.to_datetime(to_df["as_of"]).dt.strftime("%Y-%m-%d")
            fig_to = make_subplots(specs=[[{"secondary_y": True}]])
            fig_to.add_trace(go.Bar(
                x=to_df["as_of"], y=(to_df["turnover"] * 100).round(1),
                name="换手率 %", marker_color="#5bc8f5", opacity=0.7,
            ), secondary_y=False)
            fig_to.add_trace(go.Scatter(
                x=to_df["as_of"], y=(to_df["round_trip_cost"] * 100).round(4),
                name="往返成本 %", line=dict(color="#ff6b6b", width=1.5), mode="lines+markers",
            ), secondary_y=True)
            fig_to.update_layout(height=280, template="plotly_dark",
                                 legend=dict(orientation="h", y=-0.25),
                                 margin=dict(t=10))
            fig_to.update_yaxes(title_text="换手率 %", secondary_y=False)
            fig_to.update_yaxes(title_text="往返成本 %", secondary_y=True)
            st.plotly_chart(fig_to, use_container_width=True)

        # ── LLM 解读 ──────────────────────────────────────────────────────
        nav_note = analyze_chart_cached(
            "nav_curve",
            {
                "strat_ret": float(m.get("total_return", 0)),
                "ann_ret":   float(m.get("ann_return", 0)),
                "sharpe":    float(m.get("sharpe_ratio", 0)),
                "max_dd":    float(m.get("max_drawdown", 0)),
                "bench_ret": float(m.get("benchmark_ann_return", 0)),
                "cost_bps":  float(m.get("cost_bps", 0)),
                "turnover":  float(m.get("avg_turnover", 0)),
                "weight_method": selected,
            },
            f"nav_v4_cost_{selected}",
        )
        st.markdown(f"🤖 **净值解读**：{nav_note}")

with tab3:
    import json
    import plotly.graph_objects as go

    BARRA_DIR = ROOT / "reports" / "barra"
    _bm_path = BARRA_DIR / "barra_metrics.json"

    if not _bm_path.is_file():
        st.warning("未找到 Barra 报告，请先运行：`python scripts/run_barra_attribution.py`")
    else:
        bm = json.loads(_bm_path.read_text(encoding="utf-8"))

        # ── 核心指标卡 ────────────────────────────────────────────────────────
        st.subheader("📊 Barra 因子归因核心指标")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("样本期数", f"{bm['n_periods']} 期")
        c2.metric("平均截面 R²", f"{bm['avg_r2']:.3f}", help="风格+行业因子解释的截面收益方差比例")
        c3.metric("原始 IC", f"{bm['raw_ic']:.4f}", help="Spearman(模型得分, 前瞻收益)")
        c4.metric("原始 ICIR", f"{bm['raw_icir']:.3f}")
        c5.metric("Pure Alpha IC", f"{bm['residual_ic']:.4f}", help="Spearman(模型得分, Barra残差) — 剥离因子后的纯α能力")

        st.caption(
            f"Pure Alpha ICIR: **{bm['residual_icir']:.3f}**  |  "
            f"因子：7 风格 + 20 申万行业  |  "
            f"权重：√(流通市值)  |  回归：截面 WLS"
        )

        # ── 风格因子 ICIR 条形图 ────────────────────────────────────────────
        st.subheader("📈 风格因子 ICIR（时序稳定性）")
        factors = list(bm["factor_icir"].keys())
        icirs = [bm["factor_icir"][f] for f in factors]
        colors = ["#00d4aa" if v > 0 else "#ff6b6b" for v in icirs]
        fig_icir = go.Figure(go.Bar(
            x=factors, y=icirs,
            marker_color=colors,
            text=[f"{v:+.3f}" for v in icirs],
            textposition="outside",
        ))
        fig_icir.update_layout(
            height=320, template="plotly_dark",
            yaxis_title="ICIR", xaxis_title="风格因子",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_icir, use_container_width=True)
        st.caption("ICIR = 因子收益均值 / 标准差。|ICIR| > 0.2 视为显著因子。")

        # ── 因子收益累积曲线 ────────────────────────────────────────────────
        fr_path = BARRA_DIR / "factor_returns.csv"
        if fr_path.is_file():
            fr_df = pd.read_csv(fr_path, index_col=0, parse_dates=True)
            fr_df = fr_df.sort_index()
            cum_fr = (1 + fr_df).cumprod() - 1

            st.subheader("📉 风格因子累积收益")
            FACTOR_COLORS = {
                "size": "#5bc8f5", "value": "#f0a500", "momentum": "#00d4aa",
                "quality": "#a78bfa", "volatility": "#ff6b6b", "beta": "#fb923c",
                "liquidity": "#34d399",
            }
            fig_cum = go.Figure()
            for col in cum_fr.columns:
                fig_cum.add_trace(go.Scatter(
                    x=cum_fr.index, y=cum_fr[col].values,
                    name=col, mode="lines",
                    line=dict(color=FACTOR_COLORS.get(col, "#888"), width=1.5),
                ))
            fig_cum.update_layout(
                height=380, template="plotly_dark",
                yaxis_title="累积收益", xaxis_title="日期",
                legend=dict(orientation="h", y=-0.2),
                hovermode="x unified", margin=dict(t=20),
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        # ── R² 时序 ─────────────────────────────────────────────────────────
        r2_path = BARRA_DIR / "r_squared.csv"
        if r2_path.is_file():
            r2_df = pd.read_csv(r2_path, index_col=0, parse_dates=True).squeeze()
            r2_df = r2_df.sort_index()

            st.subheader("📐 截面 R² 时序（模型解释力）")
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(
                x=r2_df.index, y=r2_df.values,
                marker_color="#5bc8f5", opacity=0.75, name="R²",
            ))
            fig_r2.add_hline(y=float(r2_df.mean()), line_dash="dash",
                             line_color="#f0a500", annotation_text=f"均值 {r2_df.mean():.3f}")
            fig_r2.update_layout(
                height=260, template="plotly_dark",
                yaxis_title="R²", xaxis_title="as_of",
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        # ── 因子 IC 均值表 ───────────────────────────────────────────────────
        st.subheader("📋 风格因子详细指标")
        tbl = pd.DataFrame({
            "因子": factors,
            "IC 均值": [bm["factor_ic_mean"][f] for f in factors],
            "ICIR": [bm["factor_icir"][f] for f in factors],
        })
        tbl["|ICIR|"] = tbl["ICIR"].abs()
        tbl = tbl.sort_values("|ICIR|", ascending=False)
        tbl["IC 均值"] = tbl["IC 均值"].map(lambda x: f"{x:+.4f}")
        tbl["ICIR"] = tbl["ICIR"].map(lambda x: f"{x:+.3f}")
        tbl["|ICIR|"] = tbl["|ICIR|"].map(lambda x: f"{x:.3f}")
        st.dataframe(tbl[["因子", "IC 均值", "ICIR", "|ICIR|"]], use_container_width=True, hide_index=True)

        # ── LLM 解读 ─────────────────────────────────────────────────────────
        barra_note = analyze_chart_cached(
            "barra_attribution",
            {
                "avg_r2": bm["avg_r2"],
                "raw_ic": bm["raw_ic"],
                "raw_icir": bm["raw_icir"],
                "pure_alpha_ic": bm["residual_ic"],
                "pure_alpha_icir": bm["residual_icir"],
                "top_factors_by_icir": sorted(
                    bm["factor_icir"].items(), key=lambda kv: abs(kv[1]), reverse=True
                )[:3],
            },
            "barra_v4_main",
        )
        st.markdown(f"🤖 **Barra 解读**：{barra_note}")
