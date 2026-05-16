"""06 模型管理 — LightGBM 特征重要性 + DPO 训练曲线 + 模型文件列表."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent.parent.parent


def render() -> None:
    st.title("🧠 模型管理")
    st.markdown("管理 LightGBM 多因子模型与 DPO 偏好对齐模型，查看训练状态与特征重要性。")

    tab_lgbm, tab_dpo, tab_files = st.tabs(["📊 LightGBM", "🎯 DPO 模型", "📁 模型文件"])

    with tab_lgbm:
        _render_lgbm()

    with tab_dpo:
        _render_dpo()

    with tab_files:
        _render_model_files()


# ─── LightGBM ──────────────────────────────────────────────────────────────────

def _render_lgbm() -> None:
    st.subheader("LightGBM 多因子模型")

    model_path = ROOT / "models" / "lgbm_ranker.pkl"
    if not model_path.exists():
        st.warning("未找到 `models/lgbm_ranker.pkl`，展示 Demo 特征重要性。")
        _render_lgbm_demo()
        return

    try:
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # 基本信息
        col1, col2, col3 = st.columns(3)
        col1.metric("特征数", model.num_features())
        col2.metric("树数量", model.num_trees())
        col3.metric("最佳迭代", getattr(model, "best_iteration_", "N/A"))

        st.divider()
        st.subheader("📊 特征重要性（gain）")
        importance = pd.Series(
            model.feature_importance(importance_type="gain"),
            index=model.feature_name(),
        ).sort_values(ascending=True).tail(30)
        _plot_feature_importance(importance)

    except Exception as e:
        st.error(f"加载模型失败：{e}")
        _render_lgbm_demo()


def _render_lgbm_demo() -> None:
    """Demo 特征重要性（基于 Phase 2 41个因子）."""
    import numpy as np
    rng = np.random.default_rng(42)
    features = [
        "north_bound_net_flow_5d", "pe_ttm", "roe_ttm", "momentum_20d", "volume_ratio_5d",
        "pb_ratio", "gross_margin", "rsi_14", "macd_signal", "analyst_coverage",
        "revenue_growth_yoy", "net_profit_growth_yoy", "asset_turnover", "debt_ratio",
        "cash_flow_ratio", "beta_60d", "idiosyncratic_vol_20d", "price_to_book",
        "earnings_yield", "book_to_market", "size_log_mktcap", "momentum_60d",
        "reversal_5d", "illiquidity_amihud", "turnover_rate_20d", "skewness_20d",
        "max_drawdown_20d", "pe_relative_sector", "eps_surprise", "analyst_revision",
    ]
    importance = pd.Series(
        sorted(rng.exponential(1.0, len(features))),
        index=features,
    )
    _plot_feature_importance(importance)


def _plot_feature_importance(importance: pd.Series) -> None:
    fig = go.Figure(go.Bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        marker=dict(
            color=importance.values,
            colorscale=[[0, "#1f6feb"], [1, "#58a6ff"]],
        ),
    ))
    fig.update_layout(
        height=max(300, len(importance) * 18),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#21262d", title="Importance (gain)"),
        yaxis=dict(gridcolor="#21262d", tickfont=dict(size=10)),
        font=dict(color="#c9d1d9"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── DPO ──────────────────────────────────────────────────────────────────────

def _render_dpo() -> None:
    st.subheader("DPO 偏好对齐模型")

    # 模型路径
    dpo_dir = ROOT / "models" / "dpo_qwen"
    if dpo_dir.exists():
        st.success(f"✅ DPO 模型已保存：`{dpo_dir}`")
        adapter_files = list(dpo_dir.glob("*.bin")) + list(dpo_dir.glob("*.safetensors"))
        if adapter_files:
            st.markdown(f"**Adapter 文件：** {', '.join(f.name for f in adapter_files)}")
    else:
        st.info("DPO 模型未找到（`models/dpo_qwen/`），请先运行训练。")

    st.divider()
    st.subheader("📈 训练曲线")

    # 尝试读取训练日志
    log_files = [
        ROOT / "logs" / "dpo_training_v2.log",
        ROOT / "logs" / "dpo_training.log",
    ]
    log_data = None
    for lf in log_files:
        if lf.exists():
            log_data = _parse_dpo_log(lf)
            if log_data is not None and not log_data.empty:
                st.caption(f"数据来源：`{lf.name}`")
                break

    if log_data is None or log_data.empty:
        st.info("未找到训练日志，展示 Demo 曲线。")
        log_data = _demo_dpo_log()

    _plot_dpo_training(log_data)

    # 训练配置摘要
    st.divider()
    st.subheader("⚙️ 训练配置")
    config_data = {
        "参数": ["基础模型", "方法", "量化", "LoRA Rank", "LoRA Alpha", "批次大小", "学习率", "训练步数"],
        "值": ["Qwen2.5-7B", "DPO (Direct Preference Optimization)", "4-bit QLoRA (bitsandbytes)",
               "16", "32", "2", "2e-5", "500"],
    }
    st.dataframe(pd.DataFrame(config_data), use_container_width=True, hide_index=True)


def _parse_dpo_log(log_path: Path) -> pd.DataFrame | None:
    """从训练日志解析 step/loss/reward_margin."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        # 匹配格式：Step N/M - loss: X.XXXX - reward_margin: X.XXXX
        pattern = r"[Ss]tep\s+(\d+)[^\n]*loss[:\s]+([0-9.]+)"
        matches = re.findall(pattern, text)
        if not matches:
            # 尝试 JSON Lines 格式 {"step": N, "loss": X}
            import json
            rows = []
            for line in text.splitlines():
                try:
                    obj = json.loads(line)
                    if "loss" in obj and "step" in obj:
                        rows.append(obj)
                except Exception:
                    pass
            if rows:
                return pd.DataFrame(rows)
            return None

        rows = [{"step": int(s), "loss": float(l)} for s, l in matches]
        return pd.DataFrame(rows)
    except Exception:
        return None


def _demo_dpo_log() -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    steps = list(range(0, 510, 10))
    # 损失从 0.69 衰减，加轻微噪声
    losses = [0.693 * (0.97 ** (i / 10)) + rng.normal(0, 0.005) for i in range(len(steps))]
    rewards = [max(0, -0.1 + 0.003 * i + rng.normal(0, 0.02)) for i in range(len(steps))]
    return pd.DataFrame({"step": steps, "loss": losses, "reward_margin": rewards})


def _plot_dpo_training(df: pd.DataFrame) -> None:
    fig = go.Figure()

    if "loss" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.get("step", df.index),
            y=df["loss"],
            mode="lines",
            name="训练 Loss",
            line=dict(color="#58a6ff", width=2),
        ))

    if "reward_margin" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.get("step", df.index),
            y=df["reward_margin"],
            mode="lines",
            name="Reward Margin",
            line=dict(color="#3fb950", width=2),
            yaxis="y2",
        ))

    fig.update_layout(
        height=320,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#21262d", title="Step"),
        yaxis=dict(gridcolor="#21262d", title="Loss", side="left"),
        yaxis2=dict(
            title="Reward Margin", side="right", overlaying="y",
            gridcolor="#21262d",
        ),
        font=dict(color="#c9d1d9"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── 模型文件 ──────────────────────────────────────────────────────────────────

def _render_model_files() -> None:
    st.subheader("models/ 目录文件")

    models_dir = ROOT / "models"
    if not models_dir.exists():
        st.info("models/ 目录不存在")
        return

    rows = []
    for f in sorted(models_dir.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            rows.append({
                "文件路径": str(f.relative_to(ROOT)),
                "大小": f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB",
                "后缀": f.suffix,
            })

    if not rows:
        st.info("models/ 目录为空，尚未保存任何模型文件。")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 按类型统计
    st.divider()
    st.markdown("**文件类型分布**")
    type_counts = df["后缀"].value_counts()
    fig = go.Figure(go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.4,
        marker=dict(colors=["#58a6ff", "#3fb950", "#a371f7", "#f78166", "#d29922"]),
    ))
    fig.update_layout(
        height=260,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
