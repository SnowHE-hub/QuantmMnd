"""7_系统控制台.py — 强化版交互式运行中心.

功能区：
  1. 每日更新    — 带参数面板 + 实时日志流
  2. 30日模拟   — 模拟 & 评估，进度可见
  3. 模型管理   — LGBM / 板块模型 / CodeGraph 重训
  4. 数据更新   — 下载 / 重建 / 文本因子缓存
  5. 系统状态   — 冠军模型、Ollama、数据新鲜度、pytest 结果
  6. 收益曲线   — 累积收益 + 热力图
  7. 因子分析   — IC 柱状图 + 权重对比
  8. 执行历史   — 本次会话所有操作日志
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── 路径 & Python 解释器 ─────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]
PYTHON = sys.executable                 # 继承当前 conda env 的 python
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.sim_data import (
    load_sim30d_days,
    load_sim30d_stock_returns,
    load_ic_analysis,
    load_realized_pnl,
    horizon_portfolio_ts,
    load_strategy_config,
)

st.set_page_config(
    page_title="系统控制台 · QuantMind",
    page_icon="🖥️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state 初始化（刷新后保留）
# ─────────────────────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict[str, Any] = {
    # 操作锁（防重复提交）
    "running_task":        None,
    # 每日更新参数
    "du_date":             date.today(),
    "du_no_agent":         False,
    "du_agent_mode":       "auto",
    "du_board_models":     True,
    "du_universe":         "alpha_1374",
    "du_position_sizing":  "equal",
    # 30日模拟参数
    "sim_universe":        "alpha_1374",
    "sim_top_n":           10,
    # 上次运行结果缓存（key → {rc, output, ts}）
    "last_results":        {},
    # 执行历史
    "exec_history":        [],
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _is_running() -> bool:
    return st.session_state.running_task is not None


def _run_streaming(
    cmd: list[str],
    log_box,
    prog_bar=None,
    timeout: int = 900,
    env: dict | None = None,
) -> tuple[int, str]:
    """在子进程中运行命令，将 stdout+stderr 实时流到 log_box。

    Parameters
    ----------
    env : 传给 subprocess.Popen 的环境变量字典；None 表示继承父进程环境。

    Returns
    -------
    (returncode, full_output)
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
            env=env,
        )
    except FileNotFoundError as e:
        log_box.error(f"命令不存在：{e}")
        return 1, str(e)

    lines: list[str] = []
    deadline = time.time() + timeout
    t_start  = time.time()
    total_est = max(timeout, 60)

    for raw in proc.stdout:                  # type: ignore[union-attr]
        if time.time() > deadline:
            proc.terminate()
            lines.append("\n⚠️  [超时，进程已终止]")
            break
        stripped = raw.rstrip()
        if stripped:
            lines.append(stripped)
        # 只显示最近 70 行，避免 DOM 过大
        log_box.code("\n".join(lines[-70:]), language="text")
        if prog_bar is not None:
            elapsed = time.time() - t_start
            prog_bar.progress(
                min(0.95, elapsed / total_est),
                text=f"已运行 {elapsed:.0f}s / 预计 {total_est}s",
            )

    proc.wait()
    full_output = "\n".join(lines)
    if prog_bar is not None:
        prog_bar.progress(1.0, text="完成")
    return proc.returncode, full_output


def _record(task_key: str, rc: int, output: str, label: str) -> None:
    """写入执行历史和 last_results。"""
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.last_results[task_key] = {
        "rc": rc, "output": output, "ts": ts, "label": label,
    }
    st.session_state.exec_history.append({
        "ts": ts, "label": label,
        "success": rc == 0,
        "rc": rc,
        "output": output,
    })


def _show_result(task_key: str) -> None:
    """展示 last_results 中对应任务的最后结果。"""
    rec = st.session_state.last_results.get(task_key)
    if not rec:
        return
    if rec["rc"] == 0:
        st.success(f"✅ 上次成功  ·  {rec['ts']}")
    else:
        st.error(f"❌ 上次失败 (rc={rec['rc']})  ·  {rec['ts']}")
    with st.expander("查看日志", expanded=False):
        out = rec["output"]
        st.code(out[-5000:] if len(out) > 5000 else out, language="text")


def _action_button(label: str, key: str, task_name: str,
                   help: str = "", type_: str = "primary") -> bool:
    """统一按钮，运行中禁用。返回是否被点击。"""
    return st.button(
        label,
        key=key,
        disabled=_is_running(),
        help=help or ("有任务正在运行，请等待完成" if _is_running() else ""),
        type=type_,
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 系统状态探测（cached）
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _probe_ollama() -> dict:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        tags = r.json().get("models", [])
        return {"ok": True, "models": [m["name"] for m in tags]}
    except Exception:
        return {"ok": False, "models": []}


@st.cache_data(ttl=60)
def _probe_model(pkl_path: str) -> dict:
    p = Path(pkl_path)
    if not p.is_file():
        return {"exists": False}
    try:
        with open(p, "rb") as f:
            m = pickle.load(f)
        return {
            "exists":    True,
            "type":      type(m).__name__,
            "direction": getattr(m, "direction", "N/A"),
            "ic_mean":   getattr(m, "ic_mean",   "N/A"),
            "n_feats":   len(getattr(m, "_feature_names", []) or []),
            "mtime":     datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


@st.cache_data(ttl=60)
def _file_freshness(rel_paths: list[str]) -> list[dict]:
    result = []
    for rp in rel_paths:
        p = ROOT / rp
        if p.exists():
            mtime = datetime.fromtimestamp(p.stat().st_mtime)
            age_h = (datetime.now() - mtime).total_seconds() / 3600
            result.append({
                "file": p.name,
                "path": rp,
                "mtime": mtime.strftime("%Y-%m-%d %H:%M"),
                "age_h": round(age_h, 1),
                "ok": age_h < 48,
            })
        else:
            result.append({"file": Path(rp).name, "path": rp,
                           "mtime": "不存在", "age_h": 9999, "ok": False})
    return result


@st.cache_data(ttl=120)
def _last_pytest() -> dict:
    cache_dir = ROOT / ".pytest_cache" / "v" / "cache"
    try:
        # lastfailed 文件记录失败的 node ids
        failed_file = cache_dir / "lastfailed"
        if not failed_file.exists():
            return {"status": "未知", "failed": [], "mtime": "N/A"}
        mtime = datetime.fromtimestamp(failed_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        with open(failed_file) as f:
            data = json.load(f)
        failed = list(data.keys())
        return {
            "status": "✅ 全部通过" if not failed else f"❌ {len(failed)} 项失败",
            "failed": failed[:5],
            "mtime": mtime,
        }
    except Exception as e:
        return {"status": f"读取失败({e})", "failed": [], "mtime": "N/A"}


# ─────────────────────────────────────────────────────────────────────────────
# 标题栏
# ─────────────────────────────────────────────────────────────────────────────
c_title, c_lock = st.columns([4, 1])
with c_title:
    st.markdown("## 🖥️ 系统控制台")
    st.caption("每日更新 · 模型训练 · 数据管理 · 系统状态 · 实时日志")
with c_lock:
    st.markdown("<br>", unsafe_allow_html=True)
    if _is_running():
        st.warning(f"⏳ 运行中：{st.session_state.running_task}")
    else:
        st.success("🟢 空闲")

st.divider()

if _is_running():
    if st.button("⛔ 强制清除锁（任务卡死时使用）", type="secondary"):
        st.session_state.running_task = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# 8 大功能 Tab
# ─────────────────────────────────────────────────────────────────────────────
(
    tab_daily, tab_sim, tab_model,
    tab_data,  tab_status,
    tab_perf,  tab_ic, tab_history,
) = st.tabs([
    "🗓 每日更新",
    "▶ 30日模拟",
    "🤖 模型管理",
    "📥 数据更新",
    "📊 系统状态",
    "📈 收益曲线",
    "🔬 因子分析",
    "📋 执行历史",
])


# ═════════════════════════════════════════════════════════════════════════════
# Tab 1：每日更新
# ═════════════════════════════════════════════════════════════════════════════
with tab_daily:
    st.markdown("### 🗓 每日更新（daily_update.py）")
    st.caption("运行完整选股流程：数据下载 → 因子计算 → 打分排序 → Agent分析 → 报告生成")

    # ── 参数面板 ─────────────────────────────────────────────────────────────
    with st.expander("⚙️ 参数配置", expanded=True):
        pc1, pc2 = st.columns(2)
        with pc1:
            selected_date = st.date_input(
                "分析日期",
                value=st.session_state.du_date,
                max_value=date.today(),
                key="du_date_input",
            )
            st.session_state.du_date = selected_date

            universe = st.selectbox(
                "股票池",
                options=["alpha_1374", "full_a_5500"],
                index=["alpha_1374", "full_a_5500"].index(st.session_state.du_universe),
                format_func=lambda x: {
                    "alpha_1374": "🏆 Alpha池（1374只，速度快）",
                    "full_a_5500": "🌐 全A股（~5500只，较慢）",
                }[x],
                key="du_universe_sel",
            )
            st.session_state.du_universe = universe

        with pc2:
            no_agent = st.checkbox(
                "跳过 Agent 分析（--no-agent，速度快约 10×）",
                value=st.session_state.du_no_agent,
                key="du_no_agent_cb",
            )
            st.session_state.du_no_agent = no_agent

            board_models = st.checkbox(
                "使用板块分层模型（MAIN/GEM/STAR 路由）",
                value=st.session_state.du_board_models,
                key="du_board_models_cb",
            )
            st.session_state.du_board_models = board_models

            if not no_agent:
                agent_mode = st.radio(
                    "Agent 模式",
                    options=["auto", "full", "fast"],
                    index=["auto", "full", "fast"].index(st.session_state.du_agent_mode),
                    format_func=lambda x: {
                        "auto": "🔀 auto（Ollama可用用LLM，否则规则）",
                        "full": "💻 full（强制 Ollama LLM）",
                        "fast": "⚡ fast（纯规则，最快）",
                    }[x],
                    horizontal=True,
                    key="du_agent_mode_radio",
                )
                st.session_state.du_agent_mode = agent_mode
            else:
                agent_mode = "fast"

            pos_sizing = st.selectbox(
                "仓位优化",
                options=["equal", "hrp", "kelly", "blend"],
                index=["equal", "hrp", "kelly", "blend"].index(
                    st.session_state.du_position_sizing
                ),
                format_func=lambda x: {
                    "equal": "等权",
                    "hrp":   "HRP层次风险平价",
                    "kelly": "分数Kelly",
                    "blend": "混合",
                }[x],
                key="du_pos_sizing_sel",
            )
            st.session_state.du_position_sizing = pos_sizing

    # ── 运行按钮 ─────────────────────────────────────────────────────────────
    _show_result("daily_update")

    if _action_button("▶ 运行每日更新", "btn_daily_run", "daily_update"):
        st.session_state.running_task = "daily_update"
        st.rerun()

    if st.session_state.running_task == "daily_update":
        # 构建命令
        cmd = [
            PYTHON, "-u",
            str(ROOT / "scripts" / "daily_update.py"),
            "--date", st.session_state.du_date.isoformat(),
            "--universe", st.session_state.du_universe,
            "--position-sizing", st.session_state.du_position_sizing,
        ]
        if st.session_state.du_no_agent:
            cmd.append("--no-agent")
        # 通过环境变量传 agent_mode（daily_update.py 从 strategy_config 读）
        # 如需覆盖，临时写配置文件 or 用 env var
        env = os.environ.copy()
        env["QUANTMIND_AGENT_MODE"] = st.session_state.du_agent_mode

        st.info(f"🚀 执行命令：`{' '.join(cmd)}`")
        prog_bar = st.progress(0, text="启动中...")
        log_box  = st.empty()

        with st.spinner("每日更新运行中，请勿关闭页面..."):
            rc, output = _run_streaming(cmd, log_box, prog_bar, timeout=900, env=env)

        _record("daily_update", rc, output, "每日更新")
        st.session_state.running_task = None

        if rc == 0:
            st.success(f"✅ 每日更新完成！日期：{st.session_state.du_date}")
            st.cache_data.clear()

            # 显示选出股票（从 JSON 报告读取）
            report_dir = ROOT / "reports" / "investment_pipeline" / st.session_state.du_date.isoformat()
            strategies_file = report_dir / "strategies.json"
            if strategies_file.exists():
                try:
                    strats = json.loads(strategies_file.read_text(encoding="utf-8"))
                    st.markdown("#### 📋 今日 Agent 分析结果")
                    rows = []
                    for s in strats:
                        rows.append({
                            "代码": s.get("ticker", ""),
                            "评级": s.get("rating", ""),
                            "综合信号": f"{s.get('composite_signal', 0):+.3f}",
                            "置信度": f"{s.get('confidence', 0):.1%}",
                            "持仓期": s.get("holding_horizon", ""),
                        })
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                except Exception as _e:
                    st.warning(f"读取策略文件失败：{_e}")

            # 读取 agent_stats
            stats_file = report_dir / "agent_stats.json"
            if stats_file.exists():
                try:
                    stats = json.loads(stats_file.read_text(encoding="utf-8"))
                    st.info(
                        f"🤖 Agent模式：**{stats.get('agent_mode_actual')}**  "
                        f"· 分析股票：{stats.get('stocks_analyzed')}只  "
                        f"· 平均置信度：{stats.get('avg_confidence', 0):.1%}  "
                        f"· Ollama：{'✅' if stats.get('ollama_used') else '❌'}"
                    )
                except Exception:
                    pass
        else:
            st.error(f"❌ 每日更新失败（returncode={rc}）")

        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2：30日模拟
# ═════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("### ▶ 30日模拟")
    st.caption("模拟过去30个交易日的选股决策，评估策略表现")

    s1, s2 = st.columns(2)
    with s1:
        sim_uni = st.selectbox(
            "股票池",
            options=["alpha_1374", "full_a_5500"],
            index=["alpha_1374", "full_a_5500"].index(st.session_state.sim_universe),
            format_func=lambda x: "Alpha池（1374只）" if x == "alpha_1374" else "全A股（~5500只）",
            key="sim_uni_sel",
        )
        st.session_state.sim_universe = sim_uni

    with s2:
        sim_topn = st.number_input(
            "每日选股数",
            min_value=5, max_value=30,
            value=st.session_state.sim_top_n,
            step=5,
            key="sim_topn_input",
        )
        st.session_state.sim_top_n = sim_topn

    _show_result("sim_simulate")
    _show_result("sim_evaluate")

    btn_s1, btn_s2 = st.columns(2)
    with btn_s1:
        if _action_button("▶ 运行30日模拟", "btn_sim_run", "sim_simulate"):
            st.session_state.running_task = "sim_simulate"
            st.rerun()

    with btn_s2:
        if _action_button("📊 评估结果", "btn_sim_eval", "sim_evaluate", type_="secondary"):
            st.session_state.running_task = "sim_evaluate"
            st.rerun()

    for task_key, step, label in [
        ("sim_simulate", "simulate", "30日模拟"),
        ("sim_evaluate", "evaluate", "模拟评估"),
    ]:
        if st.session_state.running_task == task_key:
            cmd = [
                PYTHON, "-u",
                str(ROOT / "scripts" / "run_30day_sim.py"),
                "--step", step,
                "--universe", st.session_state.sim_universe,
                "--top-n", str(st.session_state.sim_top_n),
            ]
            st.info(f"🚀 `{' '.join(cmd)}`")
            prog_bar = st.progress(0, text="启动中...")
            log_box  = st.empty()

            with st.spinner(f"{label} 运行中..."):
                rc, output = _run_streaming(cmd, log_box, prog_bar, timeout=1800)

            _record(task_key, rc, output, label)
            st.session_state.running_task = None

            if rc == 0:
                st.success(f"✅ {label} 完成")
                st.cache_data.clear()
            else:
                st.error(f"❌ {label} 失败（rc={rc}）")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3：模型管理
# ═════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("### 🤖 模型管理")

    MODEL_ACTIONS = [
        ("重训 LGBM v6 冠军模型",    "model_lgbm",   "scripts/train_lgbm_model.py",   [], 600),
        ("重训板块专用模型（3×LGBM）", "model_board",  "scripts/train_board_models.py", [], 1200),
        ("重训 Meta-Learner",         "model_meta",   "scripts/train_meta_learner.py", [], 600),
        ("更新 CodeGraph 索引",        "model_cg",     None,                           [], 120),
    ]

    # 当前冠军模型状态
    st.markdown("#### 📦 当前冠军模型状态")
    models_to_show = {
        "混训 fallback": "models/lgbm_v6_alpha.pkl",
        "MAIN 专用":     "models/lgbm_v6_main.pkl",
        "GEM 专用":      "models/lgbm_v6_gem.pkl",
        "STAR 专用":     "models/lgbm_v6_star.pkl",
    }
    mi_cols = st.columns(len(models_to_show))
    for col, (mname, mpath) in zip(mi_cols, models_to_show.items()):
        info = _probe_model(str(ROOT / mpath))
        with col:
            if not info.get("exists"):
                st.metric(mname, "❌ 不存在")
            elif "error" in info:
                st.metric(mname, f"⚠️ 读取失败")
            else:
                dir_icon = "✅" if info["direction"] == 1 else "⚠️"
                ic_str = f"{info['ic_mean']:.4f}" if isinstance(info["ic_mean"], float) else "N/A"
                st.metric(
                    mname,
                    f"{dir_icon} {info['n_feats']}特征",
                    f"dir={info['direction']} IC={ic_str}",
                )
                st.caption(f"更新：{info['mtime']}")

    st.divider()

    # ── 操作按钮 ─────────────────────────────────────────────────────────────
    for label, task_key, script, extra_args, timeout in MODEL_ACTIONS:
        _show_result(task_key)
        if _action_button(f"▶ {label}", f"btn_{task_key}", task_key):
            st.session_state.running_task = task_key
            st.rerun()

        if st.session_state.running_task == task_key:
            if script is None:
                # CodeGraph sync
                cmd = ["bash", "-ic", "qm-sync"]
            else:
                cmd = [PYTHON, "-u", str(ROOT / script)] + extra_args

            st.info(f"🚀 `{' '.join(cmd)}`")
            prog_bar = st.progress(0, text="启动中...")
            log_box  = st.empty()

            with st.spinner(f"{label} 运行中..."):
                rc, output = _run_streaming(cmd, log_box, prog_bar, timeout=timeout)

            _record(task_key, rc, output, label)
            st.session_state.running_task = None

            if rc == 0:
                st.success(f"✅ {label} 完成")
                st.cache_data.clear()
            else:
                st.error(f"❌ {label} 失败（rc={rc}）")
            st.rerun()

        st.divider()

    # 板块路由状态
    st.markdown("#### 🔀 板块路由状态")
    try:
        from quantmind.models.board_router import BoardModelRouter
        router = BoardModelRouter()
        routing_status = router.get_routing_status()
        rows = []
        for board, info in routing_status.items():
            rows.append({
                "板块": board,
                "使用模型": "⬅ fallback" if info["is_fallback"] else "✅ 专用",
                "direction": info["direction"],
                "ic_mean": info["ic_mean"] if isinstance(info["ic_mean"], float) else "N/A",
                "特征数": info["n_features"],
                "原因": info["reason"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    except Exception as _e:
        st.warning(f"读取路由状态失败：{_e}")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 4：数据更新
# ═════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 📥 数据更新")

    DATA_ACTIONS = [
        ("下载最新行情数据",        "data_download",     "scripts/download_data.py",         [], 900),
        ("重建完整因子面板（v4）",   "data_panel",        "scripts/build_full_panel.py",      [], 1800),
        ("重建日价格宽表",           "data_prices",       "scripts/build_daily_price_panel.py",[], 600),
        ("更新公告情感因子",         "data_ann_sent",     "scripts/build_sentiment_factor.py", ["--source", "ann"], 600),
        ("更新北向资金缓存",         "data_north",        "scripts/build_sentiment_factor.py", ["--source", "north"], 300),
        ("重建知识库（KB）",         "data_kb",           "scripts/build_kb.py",              [], 1200),
    ]

    # 数据新鲜度监控
    st.markdown("#### 📅 数据新鲜度")
    TRACKED_FILES = [
        "data/raw/daily_prices_panel.parquet",
        "data/panel/alpha_panel_v4.parquet",
        "data/text/ann_sentiment_factor.parquet",
        "data/text/disclosure_surprise_factor.parquet",
        "data/text/north_flow.parquet",
        "data/alpha_universe/alpha_prices_wide.parquet",
    ]
    freshness = _file_freshness(TRACKED_FILES)
    df_fresh = pd.DataFrame(freshness)
    if not df_fresh.empty:
        df_fresh["状态"] = df_fresh["ok"].map({True: "✅ 新鲜", False: "⚠️ 陈旧"})
        df_fresh["更新时间"] = df_fresh["mtime"]
        df_fresh["距今(小时)"] = df_fresh["age_h"]
        st.dataframe(
            df_fresh[["file", "状态", "更新时间", "距今(小时)"]].rename(columns={"file": "文件"}),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    st.markdown("#### ▶ 操作")

    for label, task_key, script, extra_args, timeout in DATA_ACTIONS:
        c_btn, c_status = st.columns([3, 1])
        with c_btn:
            if _action_button(f"▶ {label}", f"btn_{task_key}", task_key):
                st.session_state.running_task = task_key
                st.rerun()
        with c_status:
            rec = st.session_state.last_results.get(task_key)
            if rec:
                icon = "✅" if rec["rc"] == 0 else "❌"
                st.caption(f"{icon} {rec['ts']}")

        if st.session_state.running_task == task_key:
            cmd = [PYTHON, "-u", str(ROOT / script)] + extra_args
            st.info(f"🚀 `{' '.join(cmd)}`")
            prog_bar = st.progress(0, text="启动中...")
            log_box  = st.empty()

            with st.spinner(f"{label} 运行中..."):
                rc, output = _run_streaming(cmd, log_box, prog_bar, timeout=timeout)

            _record(task_key, rc, output, label)
            st.session_state.running_task = None

            if rc == 0:
                st.success(f"✅ {label} 完成")
                st.cache_data.clear()
            else:
                st.error(f"❌ {label} 失败（rc={rc}）")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# Tab 5：系统状态
# ═════════════════════════════════════════════════════════════════════════════
with tab_status:
    st.markdown("### 📊 系统状态")

    if st.button("🔄 刷新状态", key="btn_refresh_status"):
        st.cache_data.clear()
        st.rerun()

    # ── Ollama 状态 ──────────────────────────────────────────────────────────
    st.markdown("#### 🦙 Ollama")
    ollama = _probe_ollama()
    if ollama["ok"]:
        st.success(f"✅ Ollama 运行中，可用模型：{len(ollama['models'])} 个")
        if ollama["models"]:
            st.caption("  ·  ".join(ollama["models"][:8]))
    else:
        st.error("❌ Ollama 未运行  （启动：`ollama serve`）")

    # ── 冠军模型 ─────────────────────────────────────────────────────────────
    st.markdown("#### 🏆 模型状态")
    ALL_MODELS = {
        "lgbm_v6_alpha（混训fallback）": "models/lgbm_v6_alpha.pkl",
        "lgbm_v6_main（主板专用）":      "models/lgbm_v6_main.pkl",
        "lgbm_v6_gem（创业板专用）":     "models/lgbm_v6_gem.pkl",
        "lgbm_v6_star（科创板专用）":    "models/lgbm_v6_star.pkl",
        "factor_cnn_v2_augmented":       "models/factor_cnn_v2_augmented.pkl",
    }
    model_rows = []
    for mname, mpath in ALL_MODELS.items():
        info = _probe_model(str(ROOT / mpath))
        if not info.get("exists"):
            model_rows.append({"模型": mname, "状态": "❌ 不存在",
                               "direction": "-", "IC": "-", "特征数": "-", "更新时间": "-"})
        elif "error" in info:
            model_rows.append({"模型": mname, "状态": f"⚠️ {info['error'][:30]}",
                               "direction": "-", "IC": "-", "特征数": "-", "更新时间": "-"})
        else:
            dir_ok = "✅" if info["direction"] == 1 else "⚠️"
            ic_str = f"{info['ic_mean']:.4f}" if isinstance(info["ic_mean"], float) else "N/A"
            model_rows.append({
                "模型": mname,
                "状态": f"{dir_ok} OK",
                "direction": info["direction"],
                "IC": ic_str,
                "特征数": info["n_feats"],
                "更新时间": info["mtime"],
            })
    st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)

    # ── 策略配置 ─────────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ strategy_config_v2.json 关键配置")
    try:
        cfg = load_strategy_config()
        cfg_cols = st.columns(4)
        with cfg_cols[0]:
            st.metric("agent_mode", cfg.get("agent_mode", "未设置"))
        with cfg_cols[1]:
            st.metric("ollama_model", cfg.get("ollama_model", "未设置"))
        with cfg_cols[2]:
            st.metric("ensemble_lgbm_weight", cfg.get("ensemble_lgbm_weight", "—"))
        with cfg_cols[3]:
            st.metric("use_board_models", str(cfg.get("use_board_models", False)))
    except Exception as _e:
        st.warning(f"策略配置读取失败：{_e}")

    # ── 数据新鲜度 ───────────────────────────────────────────────────────────
    st.markdown("#### 📅 数据文件新鲜度")
    freshness_s = _file_freshness([
        "data/raw/daily_prices_panel.parquet",
        "data/panel/alpha_panel_v4.parquet",
        "data/text/ann_sentiment_factor.parquet",
        "data/text/north_flow.parquet",
        "data/alpha_universe/alpha_prices_wide.parquet",
        "data/paper_trading/strategy_config_v2.json",
    ])
    for item in freshness_s:
        icon = "✅" if item["ok"] else "⚠️"
        st.caption(
            f"{icon} `{item['file']}`  —  更新：{item['mtime']}  "
            f"（{item['age_h']:.1f}h 前）"
        )

    # ── Pytest 状态 ──────────────────────────────────────────────────────────
    st.markdown("#### 🧪 测试状态")
    pytest_info = _last_pytest()
    st.info(
        f"**最近测试结果**：{pytest_info['status']}  "
        f"·  缓存时间：{pytest_info['mtime']}"
    )
    if pytest_info["failed"]:
        st.warning("失败项（最多5条）：\n" + "\n".join(f"- `{f}`" for f in pytest_info["failed"]))

    # 快速运行 pytest
    if _action_button("🧪 运行单元测试（non-integration）", "btn_pytest", "pytest",
                      type_="secondary"):
        st.session_state.running_task = "pytest"
        st.rerun()

    if st.session_state.running_task == "pytest":
        cmd = [
            PYTHON, "-m", "pytest",
            "tests/", "-m", "not integration",
            "-q", "--no-header", "--tb=short", "--no-cov",
        ]
        st.info(f"🚀 `{' '.join(cmd)}`")
        prog_bar = st.progress(0, text="启动 pytest...")
        log_box  = st.empty()

        with st.spinner("pytest 运行中..."):
            rc, output = _run_streaming(cmd, log_box, prog_bar, timeout=900)

        _record("pytest", rc, output, "单元测试")
        st.session_state.running_task = None

        if rc == 0:
            st.success("✅ 所有测试通过")
            st.cache_data.clear()
        else:
            st.error(f"❌ 测试有失败项（rc={rc}）")
        st.rerun()

    # ── 板块路由状态（紧凑版）───────────────────────────────────────────────
    st.markdown("#### 🔀 板块路由")
    try:
        from quantmind.models.board_router import BoardModelRouter
        rtr = BoardModelRouter()
        for board, binfo in rtr.get_routing_status().items():
            fb_tag = "⬅ fallback" if binfo["is_fallback"] else "✅ 专用"
            st.caption(
                f"**{board}**: {fb_tag}  "
                f"direction={binfo['direction']}  "
                f"ic_mean={binfo['ic_mean']}  "
                f"—  {binfo['reason']}"
            )
    except Exception as _e:
        st.caption(f"路由信息读取失败：{_e}")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 6：收益曲线（原 Tab2 内容移植）
# ═════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("### 📈 收益曲线")
    try:
        days = load_sim30d_days()
    except Exception:
        days = []

    if not days:
        st.info("暂无模拟数据，请先在「30日模拟」Tab 运行模拟。")
    else:
        HZ_META = {
            "1w":  ("1周",  "#D63031"),
            "2w":  ("2周",  "#E17055"),
            "21d": ("21天", "#FDCB6E"),
            "3m":  ("3月",  "#00B894"),
        }
        selected_hz = st.multiselect(
            "选择持仓期",
            options=list(HZ_META.keys()),
            default=["21d", "3m"],
            format_func=lambda x: HZ_META[x][0],
        )

        fig = go.Figure()
        for hz in selected_hz:
            ts = horizon_portfolio_ts(days, hz)
            if ts.empty:
                continue
            label, color = HZ_META[hz]
            fig.add_trace(go.Scatter(
                x=ts["date"].dt.strftime("%Y-%m-%d"),
                y=ts["cum_return"] * 100,
                name=label,
                line=dict(color=color, width=2.5),
                mode="lines+markers", marker_size=4,
                hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
            ))

        fig.add_hline(y=0, line_dash="dot", line_color="rgba(178,190,195,0.5)")
        fig.update_layout(
            height=360, margin=dict(t=20, b=40, l=50, r=20),
            yaxis_title="累计收益 (%)",
            legend=dict(orientation="h", y=1.05),
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(178,190,195,0.15)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(178,190,195,0.15)"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # KPI 汇总
        st.markdown("#### 📊 各持仓期绩效")
        kpi_cols = st.columns(4)
        for i, (hz, (label, _)) in enumerate(HZ_META.items()):
            ts = horizon_portfolio_ts(days, hz)
            with kpi_cols[i]:
                if not ts.empty:
                    m = float(ts["mean"].mean())
                    w = float((ts["mean"] > 0).mean())
                    cum = float(ts["cum_return"].iloc[-1])
                    st.metric(label, f"{m*100:+.2f}%",
                              f"胜率{w*100:.0f}% 累计{cum*100:+.1f}%")
                else:
                    st.metric(label, "—")

        # 热力图
        st.markdown("**每日收益热力图**")
        hz_avail = [h for h in ["1w","2w","21d","3m"]
                    if not horizon_portfolio_ts(days, h).empty]
        z_data, date_labels = [], []
        for hz in hz_avail:
            ts = horizon_portfolio_ts(days, hz)
            z_data.append((ts["mean"] * 100).tolist())
            if not date_labels:
                date_labels = ts["date"].dt.strftime("%m-%d").tolist()

        if z_data:
            fig_hm = go.Figure(go.Heatmap(
                z=z_data, x=date_labels,
                y=[HZ_META[h][0] for h in hz_avail],
                colorscale="RdYlGn", zmid=0,
                text=[[f"{v:+.1f}%" for v in row] for row in z_data],
                texttemplate="%{text}",
                colorbar=dict(title="收益%", thickness=12),
            ))
            fig_hm.update_layout(
                height=200, margin=dict(t=10, b=40, l=60, r=20),
                xaxis=dict(tickfont_size=9),
            )
            st.plotly_chart(fig_hm, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 7：因子分析（原 Tab3 内容移植）
# ═════════════════════════════════════════════════════════════════════════════
with tab_ic:
    st.markdown("### 🔬 因子分析")
    try:
        ic_data = load_ic_analysis()
        cfg_ic  = load_strategy_config()
    except Exception:
        ic_data, cfg_ic = {}, {}

    ic_final = ic_data.get("ic_final_picks", {})
    if not ic_final:
        st.info("IC 数据不可用，请先运行「IC优化分析」。")
    else:
        items = [
            (k, v.get("ic_3m") or 0, v.get("p_3m") or 1)
            for k, v in ic_final.items()
            if v.get("ic_3m") is not None
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        factors   = [x[0] for x in items]
        ic_vals   = [x[1] for x in items]
        sig_marks = ["**" if x[2]<0.05 else ("*" if x[2]<0.10 else "") for x in items]
        colors    = ["#00B894" if v > 0 else "#D63031" for v in ic_vals]

        fig_ic = go.Figure(go.Bar(
            x=factors, y=ic_vals, marker_color=colors,
            text=[f"{v:+.4f}{s}" for v, s in zip(ic_vals, sig_marks)],
            textposition="outside",
        ))
        fig_ic.add_hline(y=0, line_dash="dot", line_color="rgba(178,190,195,0.5)")
        fig_ic.update_layout(
            title="因子 3月 IC（**=p<0.05，*=p<0.10）",
            height=300, margin=dict(t=40, b=90, l=10, r=10),
            yaxis_title="Spearman IC",
            xaxis=dict(tickangle=-35, tickfont_size=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ic, use_container_width=True)

        st.markdown("**⚖️ System2 权重（v1 vs v2校准）**")
        s2 = cfg_ic.get("system2_updates", {})
        cur = s2.get("weights_previous",
                     {"value":0.30,"momentum":0.25,"quality":0.25,"technical":0.20})
        new = s2.get("weights_calibrated",
                     {"value":0.242,"momentum":0.223,"quality":0.333,"technical":0.202})
        DIM   = {"value":"价值","momentum":"动量","quality":"质量","technical":"技术"}
        CDIM  = {"value":"#FDCB6E","momentum":"#E17055","quality":"#00B894","technical":"#74B9FF"}

        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(
            name="v1启发式",
            x=[DIM[k] for k in cur], y=[v*100 for v in cur.values()],
            marker_color="rgba(178,190,195,0.6)",
        ))
        fig_w.add_trace(go.Bar(
            name="v2 IC校准",
            x=[DIM[k] for k in new], y=[v*100 for v in new.values()],
            marker_color=[CDIM[k] for k in new],
        ))
        fig_w.update_layout(
            barmode="group", height=260,
            margin=dict(t=10, b=40, l=40, r=10),
            yaxis_title="权重(%)",
            legend=dict(orientation="h", y=1.05),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_w, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 8：执行历史
# ═════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("### 📋 本次会话执行历史")

    history = st.session_state.exec_history
    if not history:
        st.info("本次会话尚未执行任何命令。在其他 Tab 点击操作按钮开始。")
    else:
        total   = len(history)
        success = sum(1 for r in history if r["success"])
        st.markdown(
            f"**共执行 {total} 次** · ✅ 成功 {success} 次 · ❌ 失败 {total-success} 次"
        )
        st.divider()

        for rec in reversed(history):
            icon = "✅" if rec["success"] else "❌"
            rc_str = f"rc={rec.get('rc', '?')}"
            with st.expander(
                f"{icon} `{rec['ts']}`  **{rec['label']}**  （{rc_str}）",
                expanded=False,
            ):
                out = rec["output"]
                st.code(out[-4000:] if len(out) > 4000 else out, language="text")

        if st.button("🗑 清空历史", key="btn_clear_history", type="secondary"):
            st.session_state.exec_history = []
            st.rerun()
