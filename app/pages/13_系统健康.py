"""13_系统健康.py — 一屏看全所有子系统健康度.

全部数据来自 DataService：
  get_model_status / get_regime / get_data_freshness / get_loss_signals
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.data_service import get_data_service

st.set_page_config(page_title="系统健康 · QuantMind", page_icon="🩺", layout="wide")

_SVC = get_data_service()

st.markdown("""
<div style='background:linear-gradient(90deg,#0984E3,#00B894);
            padding:18px 24px;border-radius:12px;color:white;margin-bottom:18px'>
  <h2 style='margin:0'>🩺 系统健康总览</h2>
  <p style='margin:6px 0 0 0;opacity:.9'>
    模型 · 板块路由 · Regime · 数据新鲜度 · 损失信号 — 一屏全览（DataService）
  </p>
</div>
""", unsafe_allow_html=True)

if st.button("🔄 刷新", type="secondary"):
    _SVC.clear_cache()
    st.cache_data.clear()
    st.rerun()

# 一次性拉取
ms = _SVC.get_model_status()
regime = _SVC.get_regime()
freshness = _SVC.get_data_freshness()
loss = _SVC.get_loss_signals()

# ── 顶部健康摘要条 ────────────────────────────────────────────────────────────
overall = loss.get("latest", {}).get("overall_health", "未知")
n_fallback = sum(1 for b in ms.get("board_router", {}).values()
                 if isinstance(b, dict) and b.get("is_fallback"))
n_stale = sum(1 for f in freshness if not f.get("ok"))
health_color = {"OK": "#27ae60", "WARNING": "#f39c12",
                "CRITICAL": "#e74c3c"}.get(overall, "#95a5a6")

h1, h2, h3, h4 = st.columns(4)
h1.markdown(
    f"<div style='text-align:center'><div style='font-size:.8rem;color:#636e72'>损失信号健康</div>"
    f"<div style='font-size:1.5rem;font-weight:700;color:{health_color}'>{overall}</div></div>",
    unsafe_allow_html=True)
h2.metric("板块降级数", f"{n_fallback}/3", delta=None if n_fallback == 0 else "需关注",
          delta_color="inverse")
h3.metric("数据超期项", f"{n_stale}", delta=None if n_stale == 0 else "需更新",
          delta_color="inverse")
cur_regime = regime.get("current", {}).get("current_regime", "—")
h4.metric("当前 Regime", cur_regime.upper() if cur_regime else "—")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 1：模型状态卡片网格
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 🤖 区域1 · 模型状态")

lgbm_specs = [("lgbm_main", "LGBM 主板"), ("lgbm_gem", "LGBM 创业板"),
              ("lgbm_star", "LGBM 科创板"), ("lgbm_alpha", "LGBM Alpha池")]
cols = st.columns(4)
for col, (key, label) in zip(cols, lgbm_specs):
    m = ms.get(key, {})
    with col:
        if not m.get("exists"):
            st.markdown(f"**{label}**")
            st.error("模型缺失")
            continue
        direction = m.get("direction")
        dir_ok = direction == 1
        dir_color = "#27ae60" if dir_ok else "#e74c3c"
        dir_str = f"{direction:+d}" if isinstance(direction, int) else "—"
        ic = m.get("ic_mean")
        ic_str = f"{ic:.4f}" if isinstance(ic, (int, float)) else "—"
        st.markdown(
            f"<div style='border:1px solid #dfe6e9;border-radius:8px;padding:10px'>"
            f"<div style='font-weight:600'>{label}</div>"
            f"<div style='font-size:1.4rem;color:{dir_color};font-weight:700'>"
            f"direction {dir_str} {'✅' if dir_ok else '⚠️'}</div>"
            f"<div style='font-size:.8rem;color:#636e72'>IC {ic_str} · "
            f"{m.get('n_features','—')} 特征</div>"
            f"<div style='font-size:.75rem;color:#95a5a6'>训练 {m.get('trained_at','—')}</div>"
            f"</div>",
            unsafe_allow_html=True)

# FactorCNN + Meta-Learner
mc1, mc2, mc3 = st.columns(3)
with mc1:
    cnn = ms.get("factor_cnn", {})
    st.markdown("**🧬 FactorCNN**")
    if cnn.get("exists"):
        vic = cnn.get("val_ic")
        st.metric("验证集 IC", f"{vic:+.4f}" if isinstance(vic, (int, float)) else "—",
                  f"ICIR {cnn.get('val_icir'):.3f}" if isinstance(cnn.get("val_icir"), (int, float)) else None)
        st.caption(f"训练 {cnn.get('trained_at','—')}")
    else:
        st.info("暂无数据")
with mc2:
    ml = ms.get("meta_learner", {})
    st.markdown("**🧠 Meta-Learner**")
    if ml:
        st.metric(f"版本 {ml.get('version','—')}",
                  f"CV-AUC {ml.get('cv_auc','—')}")
        st.caption(f"样本 {ml.get('n_samples','—')} · {str(ml.get('trained_at',''))[:10]}")
    else:
        st.info("暂无数据")
with mc3:
    st.markdown("**📦 模型计数**")
    n_lgbm = sum(1 for k, _ in lgbm_specs if ms.get(k, {}).get("exists"))
    st.metric("在用 LGBM", f"{n_lgbm}/4")
    st.caption(f"FactorCNN {'✅' if ms.get('factor_cnn',{}).get('exists') else '❌'} · "
               f"Meta {'✅' if ms.get('meta_learner') else '❌'}")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 2：板块路由状态
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔀 区域2 · 板块路由")
br = ms.get("board_router", {})
if not br or "error" in br:
    st.info(f"暂无数据{'（' + br['error'][:40] + '）' if isinstance(br, dict) and br.get('error') else ''}")
else:
    rcols = st.columns(3)
    board_cn = {"MAIN": "主板", "GEM": "创业板", "STAR": "科创板"}
    for col, (board, info) in zip(rcols, br.items()):
        with col:
            is_fb = info.get("is_fallback")
            color = "#e74c3c" if is_fb else "#27ae60"
            tag = "⬅ 降级 fallback" if is_fb else "✅ 专用模型"
            st.markdown(
                f"<div style='border:2px solid {color};border-radius:8px;padding:10px;text-align:center'>"
                f"<div style='font-weight:600;font-size:1.1rem'>{board_cn.get(board, board)}</div>"
                f"<div style='color:{color};font-weight:700'>{tag}</div>"
                f"<div style='font-size:.78rem;color:#636e72;margin-top:4px'>"
                f"direction={info.get('direction','—')}</div>"
                f"<div style='font-size:.72rem;color:#95a5a6'>{info.get('reason','')}</div>"
                f"</div>",
                unsafe_allow_html=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 3：Regime 状态
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 🌐 区域3 · Regime 状态")
cur = regime.get("current", {})
if not cur:
    st.info("暂无 Regime 数据")
else:
    rg1, rg2 = st.columns([1, 2])
    with rg1:
        rmap = {"bull": ("🐂 牛市", "#27ae60"), "neutral": ("⚖️ 中性", "#f39c12"),
                "bear": ("🐻 熊市", "#e74c3c")}
        label, rc = rmap.get(cur.get("current_regime", ""), ("—", "#95a5a6"))
        st.markdown(
            f"<div style='text-align:center;border:2px solid {rc};border-radius:8px;padding:14px'>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{rc}'>{label}</div>"
            f"<div style='font-size:.8rem;color:#636e72'>as_of {str(cur.get('as_of',''))[:10]}</div>"
            f"</div>", unsafe_allow_html=True)
        st.caption(f"Bull {cur.get('bull_prob',0):.0%} · "
                   f"Neutral {cur.get('neutral_prob',0):.0%} · "
                   f"Bear {cur.get('bear_prob',0):.0%}")
    with rg2:
        weights = regime.get("weights", {})
        s2 = weights.get("system2", {}) if isinstance(weights, dict) else {}
        if s2:
            st.markdown("**当前生效动态权重（System2）**")
            wdf = pd.DataFrame([{
                "维度": {"value": "价值", "momentum": "动量",
                        "quality": "质量", "technical": "技术"}.get(k, k),
                "权重": f"{v:.3f}",
            } for k, v in s2.items()])
            st.dataframe(wdf, use_container_width=True, hide_index=True)
            ens = weights.get("ensemble", {})
            if ens:
                st.caption(f"Ensemble: LGBM {ens.get('lgbm','—')} / CNN {ens.get('cnn','—')}  ·  "
                           f"EM因子 {weights.get('em_factor','—')} · 文本 {weights.get('text_factor','—')}")
        else:
            st.info("暂无动态权重数据")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 4：数据新鲜度
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 📅 区域4 · 数据新鲜度")
if not freshness:
    st.info("暂无数据")
else:
    rows = []
    for f in freshness:
        age = f.get("age_h")
        if age is None:
            age_str = "—"
        elif age < 48:
            age_str = f"{age:.1f}h"
        else:
            age_str = f"{age/24:.1f}天"
        rows.append({
            "数据": f["label"],
            "最后更新": f.get("mtime", "—"),
            "距今": age_str,
            "预期周期": f"≤{f['max_h']/24:.0f}天" if f.get("max_h") else "—",
            "状态": "✅ 正常" if f.get("ok") else ("❌ 缺失" if not f.get("exists") else "⚠️ 超期"),
        })
    fdf = pd.DataFrame(rows)

    def _hl(row):
        if "✅" in row["状态"]:
            return [""] * len(row)
        return ["background-color: #ffe6e6"] * len(row)

    st.dataframe(fdf.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 5：损失信号（周一 cron 产出）
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### ⚠️ 区域5 · 损失信号与行动计划")
latest = loss.get("latest", {})
factor_health = loss.get("factor_health", {})
action_plan = loss.get("action_plan", {})

if not latest and not factor_health and not action_plan:
    st.info("暂无损失信号数据（周一 cron 产出）")
else:
    # 四个信号
    st.markdown("#### 📡 监控信号")
    sig_cols = st.columns(4)
    sig_labels = {
        "signal_1_ranking_loss": "排序损失",
        "signal_2_factor_decay": "因子衰减",
        "signal_3_strategy_loss": "策略损失",
        "signal_4_realized_gap": "实盘偏离",
    }
    for col, (skey, slabel) in zip(sig_cols, sig_labels.items()):
        sv = latest.get(skey, {})
        with col:
            val = sv.get("value")
            alert = sv.get("alert")
            st.metric(slabel, f"{val:.4f}" if isinstance(val, (int, float)) else "—",
                      "🔴 告警" if alert else "🟢 正常",
                      delta_color="inverse" if alert else "normal")

    # 行动计划
    actions = action_plan.get("actions", []) if isinstance(action_plan, dict) else []
    if actions:
        st.markdown("#### 🎯 建议行动")
        for a in actions:
            urgency = a.get("urgency", "")
            ucolor = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#95a5a6"}.get(urgency, "#636e72")
            st.markdown(
                f"<div style='border-left:4px solid {ucolor};padding:8px 12px;margin:6px 0;"
                f"background:#f8f9fa;border-radius:4px'>"
                f"<b>{a.get('action','')}</b> "
                f"<span style='color:{ucolor};font-size:.8rem'>[{urgency}]</span><br>"
                f"<span style='font-size:.85rem;color:#636e72'>{a.get('reason','')}</span><br>"
                f"<code style='font-size:.75rem'>{a.get('cmd','')}</code></div>",
                unsafe_allow_html=True)

    # 因子健康度汇总
    if factor_health:
        st.markdown("#### 🔬 因子健康度")
        status_counts: dict[str, int] = {}
        weak_factors = []
        for fname, finfo in factor_health.items():
            if not isinstance(finfo, dict):
                continue
            stt = finfo.get("status", "UNKNOWN")
            status_counts[stt] = status_counts.get(stt, 0) + 1
            if stt in ("WEAK", "DECAYING", "DEAD"):
                weak_factors.append((fname, stt, finfo.get("latest_ic_mean")))
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            st.caption("状态分布")
            for stt, cnt in sorted(status_counts.items(), key=lambda x: -x[1]):
                st.markdown(f"- **{stt}**: {cnt}")
        with fc2:
            if weak_factors:
                st.caption(f"⚠️ 走弱因子（{len(weak_factors)}）")
                wdf = pd.DataFrame([
                    {"因子": f, "状态": s,
                     "IC": f"{ic:.4f}" if isinstance(ic, (int, float)) and not pd.isna(ic) else "—"}
                    for f, s, ic in weak_factors[:12]
                ])
                st.dataframe(wdf, use_container_width=True, hide_index=True, height=240)
            else:
                st.success("无走弱因子")

st.caption(f"损失信号 run_ts: {latest.get('run_ts', action_plan.get('run_ts', '—'))}")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# 区域 6：数据库 + 双写状态（E1 观察期）
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 🗄️ 区域6 · 数据库 + 双写状态")

try:
    from app.ops.db_health import overall_db_status

    @st.cache_data(ttl=15)
    def _db_status_cached() -> dict:
        return overall_db_status()

    status = _db_status_cached()
except Exception as _e:  # noqa: BLE001
    st.error(f"DB 状态读取失败：{_e}")
    status = {"data_backend": "?", "write_mode": "?",
              "pg": {"ok": False, "error": str(_e)},
              "mongo": {"ok": False, "error": str(_e)},
              "failures_24h": 0, "last_failure": None}

db1, db2, db3, db4 = st.columns(4)

# 卡片1：DATA_BACKEND
backend = status.get("data_backend", "?")
back_color = "#0984E3" if backend == "parquet" else "#6c5ce7"
db1.markdown(
    f"<div style='border:1px solid #dfe6e9;border-radius:8px;padding:10px;text-align:center'>"
    f"<div style='font-size:.8rem;color:#636e72'>DATA_BACKEND</div>"
    f"<div style='font-size:1.3rem;font-weight:700;color:{back_color}'>{backend.upper()}</div>"
    f"<div style='font-size:.75rem;color:#636e72'>WRITE_MODE: {status.get('write_mode', '?')}</div>"
    f"</div>", unsafe_allow_html=True)

# 卡片2：PG
pg = status.get("pg", {})
pg_color = "#27ae60" if pg.get("ok") else "#e74c3c"
db2.markdown(
    f"<div style='border:2px solid {pg_color};border-radius:8px;padding:10px;text-align:center'>"
    f"<div style='font-size:.8rem;color:#636e72'>PostgreSQL</div>"
    f"<div style='font-size:1.3rem;font-weight:700;color:{pg_color}'>"
    f"{'✅ 在线' if pg.get('ok') else '❌ 离线'}</div>"
    f"<div style='font-size:.75rem;color:#636e72'>"
    f"{pg.get('n_tables', 0)} 表 · {pg.get('total_rows', 0):,} 行</div>"
    f"</div>", unsafe_allow_html=True)

# 卡片3：Mongo
mg = status.get("mongo", {})
mg_color = "#27ae60" if mg.get("ok") else "#e74c3c"
db3.markdown(
    f"<div style='border:2px solid {mg_color};border-radius:8px;padding:10px;text-align:center'>"
    f"<div style='font-size:.8rem;color:#636e72'>MongoDB</div>"
    f"<div style='font-size:1.3rem;font-weight:700;color:{mg_color}'>"
    f"{'✅ 在线' if mg.get('ok') else '❌ 离线'}</div>"
    f"<div style='font-size:.75rem;color:#636e72'>"
    f"{mg.get('n_collections', 0)} coll · {mg.get('total_docs', 0):,} docs</div>"
    f"</div>", unsafe_allow_html=True)

# 卡片4：失败次数（24h）
n_fail = status.get("failures_24h", 0)
fail_color = "#27ae60" if n_fail == 0 else ("#f39c12" if n_fail < 5 else "#e74c3c")
db4.markdown(
    f"<div style='border:2px solid {fail_color};border-radius:8px;padding:10px;text-align:center'>"
    f"<div style='font-size:.8rem;color:#636e72'>双写失败 (24h)</div>"
    f"<div style='font-size:1.3rem;font-weight:700;color:{fail_color}'>{n_fail} 次</div>"
    f"<div style='font-size:.75rem;color:#636e72'>"
    f"{'🟢 健康' if n_fail == 0 else '🔴 需关注'}</div>"
    f"</div>", unsafe_allow_html=True)

# 错误详情
last_fail = status.get("last_failure")
if last_fail:
    st.warning(
        f"⚠️ 最近一次失败：`{last_fail['ts'].strftime('%Y-%m-%d %H:%M:%S')}` · "
        f"writer=`{last_fail['name']}` · {last_fail.get('info', '')}"
    )

if not pg.get("ok") and pg.get("error"):
    st.caption(f"PG 错误：{pg['error']}")
if not mg.get("ok") and mg.get("error"):
    st.caption(f"Mongo 错误：{mg['error']}")

st.caption("💡 详细监控见 系统控制台 → 🔀 数据后端 / 📊 双写监控")
