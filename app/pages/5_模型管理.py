"""Agent 注册表与滚动验证视图。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.data_loader import (
    list_rolling_quarters,
    load_agent_registry,
    load_validation_result,
)

st.set_page_config(page_title="模型管理", layout="wide")
st.title("🧠 模型管理")

# ── Phase C 模型概览 ────────────────────────────────────────────────────────
st.subheader("📦 Phase C 三件套模型（当前生产版本）")

phase_c_cols = st.columns(3)
MODEL_CARDS = [
    {
        "name": "RiskAgent v3",
        "icon": "🛡️",
        "model_file": "models/agents/risk_hmm_v3.pkl",
        "version": "hmm_v3",
        "metrics": "HMM 3态（低波/正常/危机）+ CVaR(95%) + OLS Beta",
        "training_data": "alpha_prices_panel.parquet（1374只，2020-2024）",
        "label": "无监督（市场状态）",
        "note": "含 market_hmm（CSI300 级别）与个股 EWMA vol 预计算",
    },
    {
        "name": "ValuationAgent v3",
        "icon": "💰",
        "model_file": "models/agents/valuation_lgbm_v3.pkl",
        "version": "lgbm_v3",
        "metrics": "CV IC ≈ 0.05–0.08（截面 Spearman）",
        "training_data": "alpha_panel_v3.parquet（27480行，2020-2024）",
        "label": "forward_return_63d（截面 rank）",
        "note": "22个特征截面百分位 rank → LGBM Regressor；含 feature_quantiles",
    },
    {
        "name": "MomentumAgent v4",
        "icon": "🚀",
        "model_file": "models/agents/momentum_patchtst_v4.pkl",
        "version": "patchtst_v4",
        "metrics": "Val Acc≈52.2%，Test Acc≈51.5%",
        "training_data": "alpha_prices_panel.parquet（5日OHLCV序列）",
        "label": "5日涨跌方向（二分类）",
        "note": "PatchTST d_model=64，dropout=0.2，label_smooth=0.05；中性带±0.07",
    },
]

for col, card in zip(phase_c_cols, MODEL_CARDS):
    with col:
        file_ok = (ROOT / card["model_file"]).is_file()
        status = "✅ 已就绪" if file_ok else "❌ 文件缺失"
        st.markdown(f"#### {card['icon']} {card['name']}")
        st.markdown(f"**版本**: `{card['version']}`  \n**状态**: {status}")
        with st.expander("详细"):
            st.markdown(
                f"- **指标**: {card['metrics']}\n"
                f"- **训练数据**: {card['training_data']}\n"
                f"- **标签**: {card['label']}\n"
                f"- **备注**: {card['note']}"
            )

st.divider()

# ── 因子模型 ────────────────────────────────────────────────────────────────
st.subheader("📊 因子排序模型（daily_update 使用）")

factor_models = [
    ("lgbm_v3_top18.pkl", "18个精选因子，alpha宇宙 v3，63d标签", "✅ 当前默认"),
    ("lgbm_v3_global.pkl", "全局因子模型 v3（ICIR+0.444）", "备用"),
    ("lgbm_v3_monthly.pkl", "月频模型 v3", "备用"),
    ("lgbm_v3_expansion_only.pkl", "扩展因子单独模型", "实验"),
]
fm_rows = []
for fn, desc, tag in factor_models:
    p = ROOT / "models" / fn
    size = f"{p.stat().st_size // 1024} KB" if p.is_file() else "—"
    fm_rows.append({"文件": fn, "描述": desc, "状态": tag, "大小": size})
st.dataframe(pd.DataFrame(fm_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Agent 注册表 ─────────────────────────────────────────────────────────────
st.subheader("🗂️ Agent 注册表")

reg = load_agent_registry()
if not reg:
    st.info("未找到 `data/agent_models/registry.json`。")
else:
    rows = []
    for agent, versions in sorted(reg.items()):
        if not isinstance(versions, list):
            continue
        active = next((v for v in versions if v.get("is_active")), versions[-1])
        perf = active.get("performance") or {}
        ic_m = perf.get("ic_mean")
        acc = perf.get("accuracy")
        metric = ic_m if ic_m is not None else acc
        rows.append(
            {
                "Agent": agent,
                "激活版本": active.get("model_version"),
                "类型": active.get("model_type"),
                "IC/准确率": f"{metric:.4f}" if metric is not None else "—",
                "创建时间": (active.get("created_at") or "")[:16],
                "历史版本数": len(versions),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("📜 历史版本详情"):
        for agent, versions in sorted(reg.items()):
            if isinstance(versions, list) and len(versions) > 1:
                st.markdown(f"**{agent}**（{len(versions)} 版本）")
                st.dataframe(pd.DataFrame(versions), use_container_width=True, hide_index=True)

    # IC 演化图
    import plotly.graph_objects as go

    fig = go.Figure()
    for agent, versions in sorted(reg.items()):
        if not isinstance(versions, list):
            continue
        xs, ys = [], []
        for i, v in enumerate(versions, start=1):
            perf = v.get("performance") or {}
            y = perf.get("ic_mean") or perf.get("accuracy")
            if y is None:
                continue
            xs.append(f"v{i}")
            ys.append(float(y))
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=agent))
    fig.update_layout(title="各 Agent 版本 IC / accuracy 演化", height=380)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── 滚动验证 ────────────────────────────────────────────────────────────────
st.subheader("🔄 滚动更新验证")
quarters = list_rolling_quarters()
if not quarters:
    st.info("未找到 `reports/rolling_update/*/validation_result.json`。")
else:
    import plotly.graph_objects as go

    xs, ics, status = [], [], []
    for q in quarters:
        vr = load_validation_result(q)
        if not vr:
            continue
        xs.append(q)
        ics.append(float(vr.get("ic", 0)))
        status.append(str(vr.get("status", "")))

    colors = ["red" if s.upper() == "RETRAIN" else "#636EFA" for s in status]
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=xs, y=ics, mode="lines+markers", name="验证 IC",
            marker=dict(color=colors, size=10),
        )
    )
    fig2.update_layout(title="季度验证 IC（红点 = RETRAIN）", height=380)
    st.plotly_chart(fig2, use_container_width=True)

    for q in quarters:
        rp = ROOT / "reports" / "rolling_update" / q / "validation_report.md"
        if rp.is_file():
            with st.expander(f"报告 {q}"):
                st.code(rp.read_text(encoding="utf-8")[:4000], language="markdown")

st.divider()

# ── 运维命令 ────────────────────────────────────────────────────────────────
st.subheader("⚙️ 运维命令速查")

tab1, tab2, tab3 = st.tabs(["每日更新", "模型重训", "回测报告"])

with tab1:
    st.code(
        """# 每日更新（交易日收盘后运行）
python scripts/daily_update.py \\
  --universe alpha \\
  --lgbm-model models/lgbm_v3_top18.pkl \\
  --lgbm-top 50 \\
  --llm-top 10 \\
  --provider dashscope \\
  --model qwen-plus

# 快速调试（跳过LLM，仅到step7）
python scripts/daily_update.py \\
  --universe alpha \\
  --no-llm \\
  --stop-after step7

# cron 示例（每日 16:30）
# 30 16 * * 1-5 cd /home/lenovo/projects/quantmind && \\
#   /home/lenovo/miniforge3/envs/quantmind/bin/python \\
#   scripts/daily_update.py --universe alpha >> logs/daily_$(date +%F).log 2>&1""",
        language="bash",
    )

with tab2:
    st.code(
        """# Risk HMM v3
python scripts/train_risk_agent_v3.py \\
  --prices data/raw/alpha_prices_panel.parquet \\
  --out models/agents/risk_hmm_v3.pkl

# Valuation LGBM v3
python scripts/train_valuation_agent_v3.py \\
  --panel data/panel/alpha_panel_v3.parquet \\
  --out models/agents/valuation_lgbm_v3.pkl

# Momentum PatchTST v4
python scripts/train_momentum_patchtst.py \\
  --prices data/raw/alpha_prices_panel.parquet \\
  --out models/agents/momentum_patchtst_v4.pkl \\
  --epochs 15

# 因子模型（Top18）
python scripts/train_factor_model.py \\
  --panel data/panel/alpha_panel_v3.parquet \\
  --label forward_return_63d \\
  --top-factors data/panel/top_factors_v3.json \\
  --out models/lgbm_v3_top18.pkl""",
        language="bash",
    )

with tab3:
    st.code(
        """# 因子分层 + IC 报告（alpha 宇宙）
python scripts/run_backtest_factor.py \\
  --model models/lgbm_v3_top18.pkl \\
  --panels data/panel/alpha_panel_v3.parquet \\
  --label forward_return_63d \\
  --n-bins 5 \\
  --output-dir reports/alpha_final

# 查看输出
ls reports/alpha_final/

# Walk-forward 评估
python scripts/train_factor_model.py \\
  --panel data/panel/alpha_panel_v3.parquet \\
  --label forward_return_63d \\
  --report-out reports/wf_alpha_v3/model_evaluation_top18.html \\
  --top-factors data/panel/top_factors_v3.json""",
        language="bash",
    )

# 最新回测报告链接
st.subheader("📄 最新报告文件")
report_dirs = {
    "alpha_final": "Alpha 1374 因子回测（最新）",
    "wf_alpha_v3": "Walk-Forward 评估 v3",
    "wf_regime_ensemble": "Regime Ensemble 评估",
    "ic_v2_63d": "IC 分析（63d 标签）",
}
for rdir, desc in report_dirs.items():
    rpath = ROOT / "reports" / rdir
    if rpath.is_dir():
        htmls = sorted(rpath.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
        if htmls:
            latest = htmls[0]
            st.markdown(f"**{desc}**: `{latest.relative_to(ROOT)}` ({latest.stat().st_size // 1024} KB)")
        else:
            st.markdown(f"**{desc}**: *(目录存在，无 HTML)*")
    else:
        st.markdown(f"**{desc}**: *(目录不存在)*")
