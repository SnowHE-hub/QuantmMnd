"""带解释的指标卡片组件。"""

from __future__ import annotations

import streamlit as st

METRIC_EXPLANATIONS = {
    "IC_mean": """
**信息系数（IC）** 衡量模型预测与实际收益的相关性。
- \u003e 0.05：较强预测能力
- 0.02~0.05：中等水平（业界常见）
- \u003c 0.02：信号较弱

数值越高（正向 IC）通常表示排序越有效，但仍需结合样本期与稳定性一起看。
    """.strip(),
    "ICIR": """
**IC 信息比率（ICIR）** = IC 均值 / IC 标准差，衡量信号稳定性。
- \u003e 0.5：相对稳定
- 0.3~0.5：一般
- \u003c 0.3：不稳定，需谨慎

ICIR 越高，通常表示模型在不同评估期表现更一致。
    """.strip(),
    "Sharpe": """
**夏普比率** ≈ 超额收益 / 收益波动率，衡量风险调整后收益。
- \u003e 1.0：常见意义上的“还不错”
- \u003e 2.0：非常突出（少见）
- \u003c 0：风险调整后不占优

注意：因子分层多空夏普与引擎含成本夏普口径不同，请勿混谈。
    """.strip(),
    "max_drawdown": """
**最大回撤** 衡量从历史峰值到后续谷底的最大跌幅（负数越深回撤越大）。
- 绝对值 \u003c 10%：相对较低
- 10%~30%：常见波动区间
- 绝对值 \u003e 30%：波动剧烈

请结合样本长度与杠杆/仓位假设解读。
    """.strip(),
    "综合信号": """
**综合信号** 由多个 Agent 打分融合得到（本项目展示区间约 -1 ~ +1）。
- \u003e 0.4：偏强多头线索
- 0.1~0.4：温和偏多
- -0.1~0.1：中性观察
- \u003c -0.1：偏空或警示

具体权重以流水线配置为准，此处仅为阅读指引。
    """.strip(),
    "Q5_ann": """
**Q5 年化（近似）** 来自因子分层回测中最高分位组合的年化口径估算。
常与「无交易成本」「截面加权近似」等假设绑定；请勿直接与实盘净值类比。
    """.strip(),
}


def metric_with_explanation(label: str, value: str, explanation: str, delta: str | None = None) -> None:
    """带解释的指标卡片：右侧 ❓ 展开说明。"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(label, value, delta=delta)
    with col2:
        with st.expander("❓"):
            st.markdown(explanation)
