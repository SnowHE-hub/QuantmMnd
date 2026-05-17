"""6_智能问答.py — 基于实际数据的系统自动问答."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.sim_data import (
    RATING_COLORS,
    horizon_portfolio_ts,
    load_ic_analysis,
    load_realized_pnl,
    load_sim30d_days,
    load_sim30d_stock_returns,
    load_strategy_config,
    sim_day_to_df,
)

st.set_page_config(page_title="智能问答 · QuantMind", page_icon="💬", layout="wide")

st.markdown("""
<div style='background:linear-gradient(90deg,#2d3436,#636E72);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:20px'>
  <h2 style='margin:0'>💬 系统智能问答</h2>
  <p style='margin:6px 0 0 0;opacity:.85'>
    基于 30日全A股模拟盘数据，自动回答关于选股、收益、因子的问题
  </p>
</div>
""", unsafe_allow_html=True)

# ── 加载数据 ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_all():
    days  = load_sim30d_days()
    sr    = load_sim30d_stock_returns()
    cfg   = load_strategy_config()
    ic    = load_ic_analysis()
    rpnl  = load_realized_pnl()
    return days, sr, cfg, ic, rpnl

days, sr, cfg, ic_data, rpnl = _load_all()

# ── 推荐问题 & 快捷入口 ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 推荐问题")
    suggested = [
        "整体绩效如何？",
        "哪个持仓期最好？",
        "表现最好的股票是哪些？",
        "表现最好的行业是哪些？",
        "质量因子为什么权重提升？",
        "System2 评分权重是多少？",
        "推荐哪些股票？",
        "整个项目怎么使用？",
    ]
    for q in suggested:
        if st.button(q, use_container_width=True, key=f"q_{q}"):
            st.session_state["auto_question"] = q

    st.divider()
    st.markdown("### 🔢 系统快照")
    ts_3m = horizon_portfolio_ts(days, "3m")
    if not ts_3m.empty:
        st.metric("3月期均收益", f"{ts_3m['mean'].mean()*100:+.2f}%")
        st.metric("3月期胜率", f"{(ts_3m['mean']>0).mean()*100:.0f}%")
    st.metric("模拟天数", f"{len(days)} 天")
    st.metric("realized_pnl", f"{len(rpnl)} 条")

# ── 问答引擎 ─────────────────────────────────────────────────────────────────
def _answer(question: str) -> tuple[str, list]:
    q = question.lower().strip()
    charts = []

    # 1. 整体绩效
    if any(kw in q for kw in ["绩效", "整体", "表现", "结果", "收益如何"]):
        ts_data = {}
        for hz in ["1w", "2w", "21d", "3m"]:
            ts = horizon_portfolio_ts(days, hz)
            if not ts.empty:
                ts_data[hz] = {"mean": ts["mean"].mean(), "wr": (ts["mean"]>0).mean(), "ir": ts["mean"].mean()/ts["mean"].std() if ts["mean"].std()>0 else 0}

        text = "## 📊 30日全A股模拟盘整体绩效\n\n"
        for hz, stats in ts_data.items():
            emoji = "✅" if stats["mean"] > 0 else "❌"
            label = {"1w":"1周","2w":"2周","21d":"21天","3m":"3月"}[hz]
            text += f"**{emoji} {label}持仓期**：均值 {stats['mean']*100:+.2f}%，胜率 {stats['wr']*100:.0f}%，IR {stats['ir']:+.3f}\n\n"

        text += "\n**💡 核心结论**：3月持仓期 IR=+1.87，期胜率 96.7%，是唯一正向期限。"
        text += "\n\n1w/2w/21d均为负收益，反映 2025Q4 A股 bull market 下的短期波动特性。"

        # 生成图表
        fig = go.Figure()
        for hz, stats in ts_data.items():
            ts = horizon_portfolio_ts(days, hz)
            colors = {"1w":"#D63031","2w":"#E17055","21d":"#FDCB6E","3m":"#00B894"}
            fig.add_trace(go.Scatter(
                x=ts["date"].dt.strftime("%m-%d"), y=ts["mean"]*100,
                name={"1w":"1周","2w":"2周","21d":"21天","3m":"3月"}[hz],
                line=dict(color=colors[hz], width=2),
            ))
        fig.add_hline(y=0, line_dash="dot", line_color="#B2BEC3")
        fig.update_layout(height=280, yaxis_title="单期收益(%)",
                          legend=dict(orientation="h"),
                          plot_bgcolor="rgba(0,0,0,0)")
        charts.append(fig)
        return text, charts

    # 2. 最优持仓期
    if any(kw in q for kw in ["持仓期", "最好", "哪个期限", "建议"]):
        text = """## ⏰ 最优持仓期推荐

**结论：强烈推荐 3个月（3m）持仓期**

| 持仓期 | 期均收益 | 期胜率 | IR |
|--------|---------|--------|-----|
| 1周    | −1.88%  | 23.3%  | −0.56 |
| 2周    | −2.02%  | 30.0%  | −0.44 |
| 21天   | −1.95%  | 30.0%  | −0.28 |
| **3月**| **+20.22%** | **96.7%** | **+1.87** |

**原因分析**：
- 2025Q4-2026Q1 A股处于 bull market，基本面筛选的股票短期（<21d）受市场轮动影响大
- 3个月窗口足以让基本面逻辑兑现，系统筛选能力在此期限效果最显著
- 短线操作在这段时期需要更高的择时能力，不在本系统优势范围内

**策略建议**：建仓后持有3个月，不做频繁调仓，每季度末更新一次持仓。
"""
        return text, charts

    # 3. 最好的股票
    if any(kw in q for kw in ["股票", "标的", "持仓", "推荐"]) and any(kw in q for kw in ["最好", "最优", "表现"]):
        if not sr.empty:
            final = sr[sr["in_final"]].copy()
            if "return_3m" in final.columns:
                top_stocks = final.groupby("ticker").agg(
                    mean_3m=("return_3m", "mean"),
                    count=("ticker", "count"),
                ).sort_values("mean_3m", ascending=False).head(10)

                text = "## 🏆 表现最佳的10只股票（3月均值收益）\n\n"
                for i, (ticker, row) in enumerate(top_stocks.iterrows(), 1):
                    text += f"{i}. **{ticker}**：{row['mean_3m']*100:+.1f}%（入选 {int(row['count'])} 次）\n"

                fig = go.Figure(go.Bar(
                    x=top_stocks.index, y=top_stocks["mean_3m"]*100,
                    marker_color="#00B894",
                    text=[f"{v*100:+.1f}%" for v in top_stocks["mean_3m"]],
                    textposition="outside",
                ))
                fig.update_layout(height=280, yaxis_title="3月均值收益(%)",
                                  xaxis_tickangle=-30, plot_bgcolor="rgba(0,0,0,0)")
                charts.append(fig)
                return text, charts
        return "数据加载中，请稍后重试。", charts

    # 4. 行业表现
    if any(kw in q for kw in ["行业", "板块", "sector"]):
        if not sr.empty and "industry" in sr.columns:
            final = sr[sr["in_final"]].copy()
            if "return_3m" in final.columns:
                ind_agg = final.groupby("industry").agg(
                    ret_3m=("return_3m", "mean"),
                    count=("ticker", "count"),
                    win_rate=("return_3m", lambda x: (x>0).mean()),
                ).sort_values("ret_3m", ascending=False).head(10)

                text = "## 🏭 行业归因分析（3月收益 Top10）\n\n"
                for ind, row in ind_agg.iterrows():
                    text += f"- **{ind}**：{row['ret_3m']*100:+.1f}%，胜率{row['win_rate']*100:.0f}%，{int(row['count'])}笔\n"

                text += "\n**核心驱动行业**：装修装饰(+252%)、建筑工程(+119%)、机械基件(+58%)"

                fig = go.Figure(go.Bar(
                    x=ind_agg.index, y=ind_agg["ret_3m"]*100,
                    marker_color=["#00B894" if v>0 else "#D63031" for v in ind_agg["ret_3m"]],
                    text=[f"{v*100:+.1f}%" for v in ind_agg["ret_3m"]],
                    textposition="outside",
                ))
                fig.update_layout(height=280, yaxis_title="3月均值收益(%)",
                                  xaxis_tickangle=-30, plot_bgcolor="rgba(0,0,0,0)")
                charts.append(fig)
                return text, charts

    # 5. 因子/质量因子
    if any(kw in q for kw in ["因子", "quality", "质量", "权重", "ic"]):
        ic_final = ic_data.get("ic_final_picks", {})
        text = "## 🔬 因子 IC 分析（3月收益方向）\n\n"
        text += "| 因子 | 3月IC | 显著性 |\n|------|-------|--------|\n"
        for factor, vals in ic_final.items():
            ic3m = vals.get("ic_3m")
            p3m  = vals.get("p_3m")
            if ic3m is not None:
                sig = "**" if (p3m or 1) < 0.05 else ("*" if (p3m or 1) < 0.10 else "")
                text += f"| {factor} | {ic3m:+.4f}{sig} | p={p3m:.3f} |\n"

        text += "\n**质量因子（quality_score）IC=+0.141，p=0.015（显著）**\n"
        text += "这是所有维度中预测力最强的因子，因此在v2校准权重中从25%提升至33.3%。\n"
        text += "原因：2025Q4 A股 bull market 环境下，高质量基本面股票（ROE高、盈利稳定）更受机构青睐。"
        return text, charts

    # 6. System2 权重
    if any(kw in q for kw in ["system2", "权重", "评分", "四维"]):
        s2 = cfg.get("system2_updates", {})
        cur = s2.get("weights_previous", {})
        new = s2.get("weights_calibrated", {})
        DIM_LABELS = {"value": "价值", "momentum": "动量", "quality": "质量", "technical": "技术"}
        text = "## ⚖️ System2 四维权重\n\n"
        text += "| 维度 | v1权重 | v2校准 | 变化 |\n|------|--------|--------|------|\n"
        for k in cur:
            delta = new.get(k,0) - cur[k]
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            text += f"| {DIM_LABELS.get(k,k)} | {cur[k]*100:.1f}% | **{new.get(k,0)*100:.1f}%** | {arrow}{abs(delta)*100:.1f}pp |\n"
        text += f"\n**校准依据**：IC-abs 归一化（3月实际收益 Spearman IC）+ 50% 平滑\n"
        text += f"**最新策略参数**：{cfg.get('holding_period',{}).get('rationale','—')}"
        return text, charts

    # 7. 如何使用
    if any(kw in q for kw in ["使用", "怎么", "如何", "流程", "步骤"]):
        text = """## 📖 QuantMind 使用指南

### 快速开始

```bash
# 激活环境
conda activate quantmind
cd /home/lenovo/projects/quantmind

# 运行30日全A股模拟
python scripts/run_30day_sim.py --step fetch    # 拉取数据
python scripts/run_30day_sim.py --step simulate  # 三系统模拟
python scripts/run_30day_sim.py --step evaluate  # 绩效评估

# 优化分析
python scripts/optimize_30day_results.py
```

### 页面导航
- **今日推荐**：查看每日三系统选出的10只股票、雷达图、热力图
- **漏斗选股**：可视化6层筛选漏斗，行业归因分析
- **单股分析**：个股四维评分、历史收益、跨期追踪
- **回测表现**：NAV曲线、逐日绩效、realized PnL分布
- **模型管理**：LGBM特征重要性、IC分析、权重校准
- **智能问答**：本页面 :)

### 核心策略建议
- 持仓期：**3个月**（期胜率96.7%，IR=+1.87）
- 关注行业：装修装饰、建筑工程、机械基件
- System2 权重：质量(33.3%) > 价值(24.2%) > 动量(22.3%) > 技术(20.2%)
"""
        return text, charts

    # 默认回复
    text = """## 💬 我可以回答以下类型的问题：

- **整体绩效**：「整体表现如何？」「这次模拟效果怎样？」
- **持仓期分析**：「哪个持仓期最好？」「建议持有多久？」
- **股票排名**：「表现最好的股票是哪些？」「有哪些推荐标的？」
- **行业归因**：「哪些行业表现最好？」「行业分布如何？」
- **因子分析**：「为什么质量因子权重提升？」「IC 分析结果如何？」
- **权重配置**：「System2 权重是多少？」「四维评分怎么算？」
- **使用指引**：「整个项目怎么用？」「怎么运行模拟？」

请从侧栏选择推荐问题，或直接输入您的问题！
"""
    return text, charts


# ── 对话界面 ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# 处理快捷按钮触发的问题
auto_q = st.session_state.pop("auto_question", None)

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="📊" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        for fig in msg.get("charts", []):
            st.plotly_chart(fig, use_container_width=True)

# 处理输入
user_input = st.chat_input("请输入您的问题...") or auto_q

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("分析中..."):
            answer_text, answer_charts = _answer(user_input)
        st.markdown(answer_text)
        for fig in answer_charts:
            st.plotly_chart(fig, use_container_width=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "charts": answer_charts,
    })
    st.rerun()

# 欢迎提示（仅在对话为空时显示）
if not st.session_state.messages:
    st.info("👈 从左侧选择推荐问题，或直接在下方输入框提问。")
    col_hint1, col_hint2 = st.columns(2)
    with col_hint1:
        st.markdown("""
**📊 可查询数据**
- 30日模拟盘：5535只全A股，30个交易日
- 最终持仓：299条记录，每日10只
- realized PnL：379条实际收益记录
        """)
    with col_hint2:
        st.markdown("""
**🎯 核心结论**
- 最优持仓期：**3个月** (IR=+1.87)
- 期胜率：**96.7%**
- 均值收益：**+20.22%**
- 最强因子：**质量分** (IC=+0.141**)
        """)
