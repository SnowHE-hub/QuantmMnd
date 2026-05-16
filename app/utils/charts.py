"""Plotly chart helpers for QuantMind dashboard."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

QUANTMIND_THEME = {
    "colors": {
        "bull": "#00B894",
        "bear": "#D63031",
        "neutral": "#636E72",
        "highlight": "#0984E3",
        "background": "#F8F9FA",
    },
    "font_family": "PingFang SC, Microsoft YaHei, sans-serif",
}


def heatmap_agents(
    z: list[list[float]],
    y_labels: list[str],
    x_labels: list[str],
    hover_text: list[list[str]] | None = None,
):
    import plotly.graph_objects as go

    ht = hover_text
    return go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#D63031"], [0.5, "#FFFFFF"], [1, "#00B894"]],
            colorbar=dict(title="信号"),
            text=ht,
            hovertemplate="%{y} · %{x}<br>信号: %{z:.2f}<br>%{text}<extra></extra>"
            if ht
            else "%{y} · %{x}<br>信号: %{z:.2f}<extra></extra>",
        ),
        layout=dict(
            margin=dict(l=40, r=20, t=40, b=40),
            height=max(320, 40 * len(y_labels)),
            xaxis=dict(side="bottom"),
            font=dict(family=QUANTMIND_THEME["font_family"]),
        ),
    )


def funnel_chart(labels: list[str], values: list[float]):
    import plotly.graph_objects as go

    return go.Figure(
        go.Funnel(y=labels, x=values, textinfo="value+percent initial"),
        layout=dict(margin=dict(l=20, r=20, t=40, b=20), height=480),
    )


def quintile_monthly_bar(q_labels: list[str], mean_monthly: list[float]):
    import plotly.graph_objects as go

    colors = ["#D63031", "#E17055", "#FDCB6E", "#55EFC4", "#00B894"][-len(q_labels) :]
    return go.Figure(
        data=go.Bar(x=q_labels, y=mean_monthly, marker_color=colors[: len(mean_monthly)]),
        layout=dict(
            title="分层月均收益（无成本因子回测）",
            yaxis=dict(title="月均收益", tickformat=".2%"),
            margin=dict(l=40, r=20, t=50, b=40),
            height=400,
            font=dict(family=QUANTMIND_THEME["font_family"]),
        ),
    )


def quintile_bar_with_cumulative_spread(qdf: pd.DataFrame, q_labels: list[str], mean_monthly: list[float]):
    """左轴：分层月均收益柱；右轴：各期 Q5−Q1 价差累计（近似解读）。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ["#D63031", "#E17055", "#FDCB6E", "#55EFC4", "#00B894"][-len(q_labels) :]
    fig.add_trace(
        go.Bar(x=q_labels, y=mean_monthly, marker_color=colors[: len(mean_monthly)], name="月均收益"),
        secondary_y=False,
    )
    cum_line_x: list = []
    cum_line_y: list = []
    if not qdf.empty and "quantile" in qdf.columns and "as_of" in qdf.columns:
        piv = qdf.pivot_table(index="as_of", columns="quantile", values="mean_return", aggfunc="mean")
        if "Q5" in piv.columns and "Q1" in piv.columns:
            spread = (piv["Q5"] - piv["Q1"]).sort_index()
            cum = spread.cumsum()
            cum_line_x = list(cum.index)
            cum_line_y = list(cum.values)
    if cum_line_x:
        fig.add_trace(
            go.Scatter(x=cum_line_x, y=cum_line_y, mode="lines+markers", name="Q5−Q1 累计价差"),
            secondary_y=True,
        )
    fig.update_yaxes(title_text="月均收益", tickformat=".2%", secondary_y=False)
    fig.update_yaxes(title_text="累计价差（近似）", tickformat=".2%", secondary_y=True)
    fig.update_layout(
        title="分层月均收益 + 各期多空价差累计（CSV 近似）",
        height=440,
        font=dict(family=QUANTMIND_THEME["font_family"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def line_ic_proxy(dates: Iterable[pd.Timestamp], ic_flat: float):
    """Flat reference line when per-period IC not persisted."""
    import plotly.graph_objects as go

    xs = list(dates)
    ys = [ic_flat for _ in xs]
    return go.Figure(
        data=go.Scatter(x=xs, y=ys, mode="lines+markers", name="IC_mean（汇总）"),
        layout=dict(
            title="IC 参考（各期截面 IC 未单独导出时展示测试集均值）",
            yaxis=dict(title="Rank IC"),
            margin=dict(l=40, r=20, t=50, b=40),
            height=380,
        ),
    )


def q5_returns_by_period(df: pd.DataFrame):
    """df columns as_of, quantile, mean_return — plot Q5 series."""
    import plotly.graph_objects as go

    if df.empty or "quantile" not in df.columns:
        return go.Figure()
    q5 = df[df["quantile"].astype(str) == "Q5"].sort_values("as_of")
    return go.Figure(
        data=go.Scatter(x=q5["as_of"], y=q5["mean_return"], mode="lines+markers", name="Q5"),
        layout=dict(title="Q5 各期月均收益", yaxis=dict(tickformat=".2%"), height=380),
    )


def radar_agent(labels: list[str], values_minus1_to_1: list[float], name: str):
    import plotly.graph_objects as go

    r = [v + 1.0 for v in values_minus1_to_1]
    r_closed = r + r[:1]
    theta = labels + labels[:1]
    return go.Figure(
        data=go.Scatterpolar(r=r_closed, theta=theta, fill="toself", name=name),
        layout=dict(
            polar=dict(radialaxis=dict(range=[0, 2], tickvals=[0, 0.5, 1, 1.5, 2])),
            showlegend=True,
            height=440,
            margin=dict(l=40, r=40, t=40, b=40),
            font=dict(family=QUANTMIND_THEME["font_family"]),
        ),
    )


def plot_agent_radar_themed(agent_signals: dict, ticker: str, composite: float | None = None):
    """五维雷达（风险信号取反以便「越大越好」），0~10 分刻度 + 中性虚线。"""
    import plotly.graph_objects as go

    cats = ["估值", "动量", "质量", "情绪", "风险(反向)"]
    keys = ["ValuationAgent", "MomentumAgent", "QualityAgent", "SentimentAgent", "RiskAgent"]
    raw: list[float] = []
    for k in keys:
        block = agent_signals.get(k) or {}
        raw.append(float(block.get("signal", 0.0) or 0.0))
    adj = raw[:4] + [-raw[4]]
    scores = [(v + 1.0) * 5.0 for v in adj]
    scores_closed = scores + scores[:1]
    theta = cats + cats[:1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[5.0] * 6,
            theta=theta,
            mode="lines",
            line=dict(color="#636E72", dash="dash"),
            name="中性基准",
            fill="none",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=scores_closed,
            theta=theta,
            fill="toself",
            fillcolor="rgba(0,184,148,0.25)",
            line=dict(color=QUANTMIND_THEME["colors"]["bull"]),
            name=ticker,
        )
    )
    title_extra = ""
    if composite is not None:
        title_extra = f" · 综合信号 {composite:+.2f}"
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10], tickvals=[0, 2.5, 5, 7.5, 10])),
        title=f"{ticker} Agent 雷达{title_extra}",
        height=420,
        margin=dict(l=36, r=36, t=56, b=36),
        font=dict(family=QUANTMIND_THEME["font_family"]),
        showlegend=True,
    )
    return fig


def plot_sparkline(price_df: pd.DataFrame, ticker: str, days: int = 30):
    """迷你收盘价曲线（弱化坐标轴）。"""
    import plotly.graph_objects as go

    if price_df.empty or "close" not in price_df.columns:
        return go.Figure()
    dfc = price_df.tail(days).copy()
    dfc["trade_date"] = pd.to_datetime(dfc["trade_date"])
    fig = go.Figure(
        go.Scatter(
            x=dfc["trade_date"],
            y=dfc["close"],
            mode="lines",
            line=dict(color=QUANTMIND_THEME["colors"]["highlight"], width=2),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=120,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(title=dict(text=f"{ticker} · {days}日走势", font=dict(size=12)))
    return fig


def plot_agent_hbars(agent_signals: dict):
    """水平条形展示五维 Agent 信号。"""
    import plotly.graph_objects as go

    keys = ["ValuationAgent", "MomentumAgent", "QualityAgent", "SentimentAgent", "RiskAgent"]
    labels = ["估值", "动量", "质量", "情绪", "风险"]
    vals = [float((agent_signals.get(k) or {}).get("signal", 0) or 0) for k in keys]
    colors = [QUANTMIND_THEME["colors"]["bull"] if v >= 0 else QUANTMIND_THEME["colors"]["bear"] for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors))
    fig.update_xaxes(range=[-1.05, 1.05], title_text="信号 (−1~1)")
    fig.update_layout(height=260, margin=dict(l=80, r=20, t=30, b=40), showlegend=False)
    return fig


def ic_spread_timeseries(qdf: pd.DataFrame):
    """无逐期 IC 时：用各期 Q5−Q1 价差近似多空分化（标注日期）。"""
    import plotly.graph_objects as go

    if qdf.empty or "quantile" not in qdf.columns:
        return go.Figure()
    piv = qdf.pivot_table(index="as_of", columns="quantile", values="mean_return", aggfunc="mean")
    if "Q5" not in piv.columns or "Q1" not in piv.columns:
        return go.Figure()
    s = (piv["Q5"] - piv["Q1"]).sort_index().reset_index()
    s.columns = ["as_of", "spread"]
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="solid", line_color="#636E72")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#00B894", annotation_text="±0.05参考")
    fig.add_hline(y=-0.05, line_dash="dash", line_color="#D63031")
    fig.add_trace(
        go.Scatter(
            x=s["as_of"],
            y=s["spread"],
            mode="lines+markers",
            name="Q5−Q1",
            fill="tozeroy",
            fillcolor="rgba(9,132,227,0.12)",
            line=dict(color=QUANTMIND_THEME["colors"]["highlight"]),
        )
    )
    fig.update_layout(
        title="分层多空价差时序（IC 未导出时的近似替代）",
        yaxis=dict(title="Q5−Q1", tickformat=".2%"),
        height=420,
        font=dict(family=QUANTMIND_THEME["font_family"]),
    )
    return fig


def price_volume_ma(price_df: pd.DataFrame, recommend_date: pd.Timestamp | None = None):
    """价格 + 成交量（次轴）+ MA20/MA60。"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if price_df.empty or "close" not in price_df.columns:
        return go.Figure()
    dfc = price_df.copy()
    dfc["trade_date"] = pd.to_datetime(dfc["trade_date"])
    dfc = dfc.sort_values("trade_date")
    dfc["ma20"] = dfc["close"].rolling(20, min_periods=5).mean()
    dfc["ma60"] = dfc["close"].rolling(60, min_periods=3).mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
    fig.add_trace(
        go.Scatter(x=dfc["trade_date"], y=dfc["close"], name="收盘", line=dict(color="#0984E3")), row=1, col=1
    )
    fig.add_trace(go.Scatter(x=dfc["trade_date"], y=dfc["ma20"], name="MA20", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfc["trade_date"], y=dfc["ma60"], name="MA60", line=dict(dash="dash")), row=1, col=1)
    vol = dfc["vol"] if "vol" in dfc.columns else None
    if vol is not None:
        fig.add_trace(go.Bar(x=dfc["trade_date"], y=vol, name="成交量", marker_color="#636E72"), row=2, col=1)
    if recommend_date is not None:
        rd = pd.Timestamp(recommend_date)
        fig.add_vline(x=rd, line_width=1, line_dash="dash", line_color="purple", row=1, col=1)
    fig.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.12))
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def radar_compare(agent_signals_by_ticker: dict[str, dict]):
    """多只股票的五维雷达对比（每条线一只股票）。"""
    import plotly.graph_objects as go

    cats = ["估值", "动量", "质量", "情绪", "风险(反向)"]
    theta = cats + cats[:1]
    fig = go.Figure()
    keys = ["ValuationAgent", "MomentumAgent", "QualityAgent", "SentimentAgent", "RiskAgent"]
    palette = ["#0984E3", "#00B894", "#E17055", "#6C5CE7", "#FDCB6E"]
    fills = ["rgba(9,132,227,0.15)", "rgba(0,184,148,0.15)", "rgba(225,112,85,0.15)", "rgba(108,92,231,0.15)", "rgba(253,203,110,0.15)"]
    for i, (tk, sigs) in enumerate(agent_signals_by_ticker.items()):
        raw = [float((sigs.get(k) or {}).get("signal", 0) or 0) for k in keys]
        adj = raw[:4] + [-raw[4]]
        scores = [(v + 1.0) * 5.0 for v in adj]
        fig.add_trace(
            go.Scatterpolar(
                r=scores + scores[:1],
                theta=theta,
                fill="toself",
                name=tk,
                fillcolor=fills[i % len(fills)],
                line=dict(color=palette[i % len(palette)]),
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10])),
        height=460,
        font=dict(family=QUANTMIND_THEME["font_family"]),
        title="多标的 Agent 信号对比（风险已反向）",
    )
    return fig


def validation_scatter(valid_rows: list[dict]):
    """简单散点：胜率 vs 平均收益。"""
    import plotly.graph_objects as go

    if not valid_rows:
        return go.Figure()
    xs = [float(r.get("win_rate", 0) or 0) for r in valid_rows]
    ys = [float(r.get("avg_return", 0) or 0) for r in valid_rows]
    labels = [r.get("ticker", "") for r in valid_rows]
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            text=labels,
            hovertemplate="%{text}<br>胜率:%{x:.0%}<br>月均:%{y:.2%}<extra></extra>",
            marker=dict(size=10, color=QUANTMIND_THEME["colors"]["highlight"]),
        )
    )
    fig.update_layout(
        title="历史验证：胜率 vs 平均收益（流水线导出）",
        xaxis=dict(title="胜率", tickformat=".0%"),
        yaxis=dict(title="平均收益", tickformat=".2%"),
        height=380,
    )
    return fig


def price_with_ma(df: pd.DataFrame, recommend_date: pd.Timestamp | None = None):
    import plotly.graph_objects as go

    if df.empty or "close" not in df.columns:
        return go.Figure()
    dfc = df.copy()
    dfc["trade_date"] = pd.to_datetime(dfc["trade_date"])
    dfc = dfc.sort_values("trade_date")
    dfc["ma20"] = dfc["close"].rolling(20, min_periods=5).mean()
    dfc["ma60"] = dfc["close"].rolling(60, min_periods=5).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dfc["trade_date"], y=dfc["close"], name="收盘", mode="lines"))
    fig.add_trace(go.Scatter(x=dfc["trade_date"], y=dfc["ma20"], name="MA20", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=dfc["trade_date"], y=dfc["ma60"], name="MA60", line=dict(dash="dash")))
    if recommend_date is not None:
        rd = pd.Timestamp(recommend_date)
        fig.add_vline(x=rd, line_width=1, line_dash="dash", line_color="purple")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="价格")
    return fig


def nav_compare(index_df: pd.DataFrame, strat_df: pd.DataFrame, bench_name: str = "CSI300"):
    import plotly.graph_objects as go

    if index_df.empty or strat_df.empty or "trade_date" not in strat_df.columns:
        return go.Figure()
    idx = index_df.copy()
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    idx = idx.sort_values("trade_date")
    idx["bench_nav"] = idx["close"].astype(float) / float(idx["close"].iloc[0])
    sd = strat_df.copy()
    sd["trade_date"] = pd.to_datetime(sd["trade_date"])
    merged = pd.merge(idx[["trade_date", "bench_nav"]], sd, on="trade_date", how="inner")
    if merged.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["trade_date"], y=merged["strat_nav"], name="策略（近似）"))
    fig.add_trace(go.Scatter(x=merged["trade_date"], y=merged["bench_nav"], name=bench_name))
    fig.update_layout(title="累计净值（策略为近似曲线）", height=440, yaxis_title="净值")
    return fig


def ic_compare_bar(labels: list[str], values: list[float]):
    import plotly.graph_objects as go

    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    return go.Figure(
        data=go.Bar(x=labels, y=values, marker_color=colors),
        layout=dict(title="IC_mean 对比", yaxis=dict(title="IC"), height=380),
    )


def industry_pie(counts: dict[str, int]):
    import plotly.graph_objects as go

    if not counts:
        return go.Figure()
    labels = list(counts.keys())
    vals = list(counts.values())
    return go.Figure(
        data=go.Pie(labels=labels, values=vals, hole=0.35),
        layout=dict(title="行业分布（候选股票）", height=400),
    )


def lgbm_bar(tickers: list[str], scores: list[float]):
    import plotly.graph_objects as go

    return go.Figure(
        data=go.Bar(x=tickers, y=scores, marker_color="#636EFA"),
        layout=dict(
            title="LGBM 得分（order-preserving 粗排分）",
            xaxis=dict(tickangle=-45),
            height=max(400, len(tickers) * 18),
            margin=dict(b=120),
        ),
    )
