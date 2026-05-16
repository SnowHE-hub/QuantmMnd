"""03 策略回测 — 量化策略配置 + 回测运行 + 结果可视化."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent


def render() -> None:
    st.title("📊 量化策略回测")
    st.markdown("配置策略参数，运行 A股历史回测（T+1/涨跌停/PIT 严格隔离）。")

    # ── 参数配置 ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ 回测参数配置", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            strategy = st.selectbox(
                "策略",
                ["equal_weight", "lgbm", "llm_rerank"],
                help="equal_weight=等权 | lgbm=LGBM多因子 | llm_rerank=LLM精排",
            )
        with col2:
            start_date = st.date_input("开始日期", value=date(2022, 1, 1))
            end_date   = st.date_input("结束日期", value=date(2023, 12, 31))
        with col3:
            initial_capital = st.number_input(
                "初始资金（万元）", value=1000.0, min_value=10.0, step=100.0
            )
            universe_size = st.slider("Universe 大小（股票数）", 10, 100, 30, step=10)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_btn = st.button("🚀 运行回测", type="primary", use_container_width=True)

    st.divider()

    if run_btn:
        _run_backtest(strategy, start_date, end_date, initial_capital * 10000, universe_size)
    else:
        st.info("👆 配置参数后点击「运行回测」")
        _show_demo_backtest()


def _run_backtest(
    strategy: str,
    start_date: date,
    end_date: date,
    initial_capital: float,
    universe_size: int,
) -> None:
    """调用 BacktestEngine 运行回测并展示结果."""
    with st.spinner(f"正在运行 {strategy} 回测（{start_date} ~ {end_date}）..."):
        try:
            from quantmind.backtest.engine import BacktestEngine, BacktestConfig, EqualWeightStrategy
            import numpy as np

            # 生成模拟价格数据（Demo 模式）
            tickers = [f"Demo{i:03d}.SZ" for i in range(universe_size)]
            n_days = (end_date - start_date).days
            dates = pd.date_range(start_date, end_date, freq="B")
            n = len(dates)

            rng = np.random.default_rng(42)
            price_records = []
            for t in tickers:
                px = 10.0
                for d in dates:
                    ret = rng.normal(0.0008, 0.015)
                    px = max(px * (1 + ret), 0.01)
                    price_records.append({
                        "date": d, "ticker": t,
                        "open": px, "close": px,
                        "high": px * 1.01, "low": px * 0.99,
                        "volume": 1e6, "prev_close": px / (1 + ret),
                    })

            prices_df = pd.DataFrame(price_records).set_index(["date", "ticker"])

            cfg = BacktestConfig(initial_capital=initial_capital)
            strat = EqualWeightStrategy(config=cfg, rebalance_freq="monthly")
            engine = BacktestEngine(config=cfg, prices_df=prices_df)

            result = engine.run(
                strategy=strat,
                start_date=start_date,
                end_date=end_date,
                universe=tickers,
            )

            st.success("✅ 回测完成")
            _render_backtest_result(result)

        except Exception as e:
            st.error(f"回测失败：{e}")
            st.info("切换到 Demo 模式展示样例结果")
            _show_demo_backtest()


def _render_backtest_result(result: object) -> None:
    """渲染回测结果：净值曲线 + 回撤曲线 + 月度收益热力图 + 指标表。"""
    metrics = result.metrics
    nav = result.nav_series
    returns = result.returns

    # ── 关键指标 ──────────────────────────────────────────────────────────────
    cols = st.columns(5)
    key_metrics = [
        ("年化收益", f"{metrics.get('annualized_return', 0):.2%}"),
        ("夏普比率", f"{metrics.get('sharpe_ratio', 0):.3f}"),
        ("最大回撤", f"{metrics.get('max_drawdown', 0):.2%}"),
        ("波动率", f"{metrics.get('volatility', 0):.2%}"),
        ("Beta", f"{metrics.get('beta', 0):.3f}"),
    ]
    for col, (label, value) in zip(cols, key_metrics):
        col.metric(label, value)

    # ── 净值曲线 ──────────────────────────────────────────────────────────────
    st.subheader("净值曲线")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav / nav.iloc[0],
        mode="lines", name="策略净值",
        line=dict(color="#58a6ff", width=2),
    ))
    fig.update_layout(
        height=320,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        font=dict(color="#c9d1d9"), margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 回撤曲线 ──────────────────────────────────────────────────────────────
    st.subheader("回撤曲线")
    nav_norm = nav / nav.cummax()
    drawdown = nav_norm - 1
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        fill="tozeroy", mode="lines", name="回撤",
        line=dict(color="#f85149", width=1),
        fillcolor="rgba(248, 81, 73, 0.15)",
    ))
    fig2.update_layout(
        height=200,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d", tickformat=".1%"),
        font=dict(color="#c9d1d9"), margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 月度收益热力图 ────────────────────────────────────────────────────────
    st.subheader("月度收益热力图")
    _render_monthly_heatmap(returns)


def _render_monthly_heatmap(returns: pd.Series) -> None:
    """月度收益率热力图."""
    if returns.empty:
        st.info("收益数据不足")
        return
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly.index = monthly.index.to_period("M")
    df = monthly.reset_index()
    df.columns = ["period", "return"]
    df["year"] = df["period"].apply(lambda p: p.year)
    df["month"] = df["period"].apply(lambda p: p.month)

    pivot = df.pivot(index="year", columns="month", values="return").fillna(0)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    z = pivot.values * 100

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[months[m - 1] for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale=[[0, "#f85149"], [0.5, "#161b22"], [1, "#3fb950"]],
        zmid=0,
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="月收益%"),
    ))
    fig.update_layout(
        height=250,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"), margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_demo_backtest() -> None:
    """展示 Demo 回测结果（模拟数据）."""
    st.subheader("Demo 回测结果（模拟数据）")
    rng = np.random.default_rng(42)
    n = 500
    returns = pd.Series(rng.normal(0.0008, 0.015, n), index=pd.date_range("2022-01-01", periods=n, freq="B"))
    nav = (1 + returns).cumprod() * 10_000_000

    cols = st.columns(5)
    for col, (label, value) in zip(cols, [
        ("年化收益", "~15.2%"), ("夏普比率", "~0.87"), ("最大回撤", "~-18.3%"),
        ("波动率", "~18.9%"), ("Beta", "~0.76"),
    ]):
        col.metric(label, value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav / nav.iloc[0], mode="lines",
                             name="策略净值", line=dict(color="#58a6ff", width=2)))
    fig.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
                      font=dict(color="#c9d1d9"), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    _render_monthly_heatmap(returns)
