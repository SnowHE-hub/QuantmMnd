"""04 Agent 回测 — 加载历史 Agent 决策结果 + 可视化分析."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent


@st.cache_data(ttl=300)
def _load_agent_results() -> pd.DataFrame:
    """加载 reports/ 下所有 Agent 决策记录."""
    records_dir = ROOT / "reports" / "agent_decisions"
    rows = []
    if records_dir.exists():
        import json
        for f in sorted(records_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    rows.extend(data)
                elif isinstance(data, dict):
                    rows.append(data)
            except Exception:
                pass

    if not rows:
        # Demo 数据
        rng = np.random.default_rng(42)
        n = 200
        ratings = rng.choice(["买入", "增持", "中性", "减持", "卖出"],
                              size=n, p=[0.25, 0.30, 0.25, 0.12, 0.08])
        actual_ret = rng.normal(
            [0.08, 0.04, 0.0, -0.04, -0.08][list(["买入","增持","中性","减持","卖出"]).index(r)]
            + rng.normal(0, 0.05)
            for r in ratings
        )
        rows = [
            {
                "decision_date": f"2023-{(i%12)+1:02d}-01",
                "ticker": f"Stock{i%30:03d}",
                "rating": ratings[i],
                "confidence": float(rng.integers(50, 95)),
                "actual_return_30d": float(rng.normal(
                    {"买入": 0.06, "增持": 0.03, "中性": 0.0, "减持": -0.03, "卖出": -0.06}[ratings[i]],
                    0.08
                )),
                "sector": rng.choice(["金融", "消费", "科技", "医药", "能源"], 1)[0],
            }
            for i in range(n)
        ]

    df = pd.DataFrame(rows)
    # 类型转换
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    if "actual_return_30d" in df.columns:
        df["actual_return_30d"] = pd.to_numeric(df["actual_return_30d"], errors="coerce")
    return df


def render() -> None:
    st.title("🤖 Agent 决策回测")
    st.markdown("分析历史 Agent 评级决策的实际预测效果。")

    df = _load_agent_results()

    if df.empty:
        st.warning("暂无 Agent 决策记录，展示 Demo 数据。")
        return

    # ── 筛选器 ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        ratings_filter = st.multiselect(
            "评级筛选",
            ["买入", "增持", "中性", "减持", "卖出"],
            default=["买入", "增持", "中性", "减持", "卖出"],
        )
    with col2:
        conf_min = st.slider("最低置信度", 0, 100, 50)
    with col3:
        sectors = ["全部"] + sorted(df.get("sector", pd.Series(dtype=str)).dropna().unique().tolist())
        sector_filter = st.selectbox("行业", sectors)

    # 应用筛选
    mask = df["rating"].isin(ratings_filter)
    if "confidence" in df.columns:
        mask &= df["confidence"] >= conf_min
    if sector_filter != "全部" and "sector" in df.columns:
        mask &= df["sector"] == sector_filter
    filtered = df[mask].copy()

    st.markdown(f"**筛选后样本：{len(filtered)} 条**")
    st.divider()

    # ── 两列布局 ──────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📦 Rating × 实际收益（箱线图）")
        _render_rating_boxplot(filtered)

    with col_right:
        st.subheader("🎯 Calibration 曲线")
        _render_calibration(filtered)

    st.divider()

    # ── 评级分布 ──────────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📊 评级分布")
        _render_rating_distribution(filtered)

    with col4:
        st.subheader("📈 命中率统计")
        _render_hit_rate(filtered)


def _render_rating_boxplot(df: pd.DataFrame) -> None:
    """Rating vs 实际 30 日收益箱线图."""
    if "actual_return_30d" not in df.columns:
        st.info("无实际收益数据")
        return

    rating_order = ["买入", "增持", "中性", "减持", "卖出"]
    colors = {"买入": "#3fb950", "增持": "#58a6ff", "中性": "#8b949e",
              "减持": "#f78166", "卖出": "#f85149"}

    fig = go.Figure()
    for r in rating_order:
        sub = df[df["rating"] == r]["actual_return_30d"].dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Box(
            y=sub, name=r, marker_color=colors.get(r, "#8b949e"),
            boxmean="sd",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="#30363d")
    fig.update_layout(
        height=320,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d", tickformat=".1%"),
        font=dict(color="#c9d1d9"), margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_calibration(df: pd.DataFrame) -> None:
    """置信度 vs 实际方向命中率（Calibration 曲线）."""
    if "confidence" not in df.columns or "actual_return_30d" not in df.columns:
        st.info("数据不足")
        return

    sub = df[["confidence", "actual_return_30d", "rating"]].dropna()
    if len(sub) < 20:
        st.info("样本量不足（<20）")
        return

    # 定义"命中"：买入/增持 → 实际收益 > 0；减持/卖出 → 实际收益 < 0
    bullish = sub["rating"].isin(["买入", "增持"])
    bearish = sub["rating"].isin(["减持", "卖出"])
    hit = ((bullish) & (sub["actual_return_30d"] > 0)) | \
          ((bearish) & (sub["actual_return_30d"] < 0))
    sub = sub.copy()
    sub["hit"] = hit

    # 按置信度分桶
    bins = list(range(50, 100, 10))
    labels = [f"{b}-{b+10}%" for b in bins[:-1]] + [f"{bins[-1]}%+"]
    sub["conf_bin"] = pd.cut(sub["confidence"], bins=bins + [100], labels=labels, right=False)
    calib = sub.groupby("conf_bin", observed=True)["hit"].agg(["mean", "count"]).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=calib["conf_bin"].astype(str),
        y=calib["mean"],
        mode="lines+markers",
        name="实际命中率",
        marker=dict(size=8, color="#58a6ff"),
        line=dict(color="#58a6ff"),
        error_y=dict(
            type="data",
            array=(calib["count"].apply(lambda n: (0.25 / max(n, 1)) ** 0.5)).tolist(),
            visible=True,
        ),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#30363d", annotation_text="随机基准")
    fig.update_layout(
        height=320,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#21262d", title="置信度区间"),
        yaxis=dict(gridcolor="#21262d", tickformat=".0%", range=[0, 1]),
        font=dict(color="#c9d1d9"), margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_rating_distribution(df: pd.DataFrame) -> None:
    rating_counts = df["rating"].value_counts()
    colors = ["#3fb950", "#58a6ff", "#8b949e", "#f78166", "#f85149"]
    fig = go.Figure(go.Pie(
        labels=rating_counts.index,
        values=rating_counts.values,
        hole=0.5,
        marker=dict(colors=colors),
    ))
    fig.update_layout(
        height=280,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"), margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_hit_rate(df: pd.DataFrame) -> None:
    """按评级统计命中率表格."""
    if "actual_return_30d" not in df.columns:
        st.info("无实际收益数据")
        return

    rating_order = ["买入", "增持", "中性", "减持", "卖出"]
    rows = []
    for r in rating_order:
        sub = df[df["rating"] == r]["actual_return_30d"].dropna()
        if len(sub) == 0:
            continue
        bullish = r in ["买入", "增持"]
        hits = (sub > 0).sum() if bullish else (sub < 0).sum()
        rows.append({
            "评级": r,
            "样本数": len(sub),
            "命中数": int(hits),
            "命中率": f"{hits/len(sub):.1%}",
            "平均收益": f"{sub.mean():.2%}",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
