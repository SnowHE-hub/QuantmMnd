"""损失信号分发器 — Week 1 学习循环基础设施.

功能
====
读取顺序验证结果（reports/sequential_val/）和实际收益追踪（data/feedback/），
计算 4 类损失信号，并决定各组件的更新动作。

4 类损失信号
============
  Signal 1: 排序损失（Ranking Loss）
    定义: 0.5 × (1 - IC_mean)，越高越差
    驱动: LGBM ranker 重训

  Signal 2: 因子衰减损失（Factor Decay Loss）
    定义: 跨 Round 因子 IC 下滑幅度（最近 2 轮 vs 最早 2 轮）
    驱动: 因子剔除 / 降权 / 替换建议

  Signal 3: 策略损失（Strategy Loss）
    定义: max(0, CSI300 收益 - 策略收益)，即跑输基准时的损失
    驱动: meta-learner 权重重估 + 仓位优化器校准

  Signal 4: 实际收益信号偏差（Realized IC Gap）
    定义: (预期 IC - realized IC) / 预期 IC
    驱动: DPO 偏好对生成 + Agent 模型重训触发

输出
====
  data/loss_signals/
    latest.json           最新一期损失信号（供 daily cron 读取）
    history.jsonl         历史信号追加日志（每次运行追加一行）
    action_plan.json      各组件的更新动作列表
    factor_health.json    因子健康状态（IC 趋势 + 告警）

用法
====
  python scripts/dispatch_loss_signals.py \\
      --seq-val  reports/sequential_val/ \\
      --feedback data/feedback/ \\
      --out      data/loss_signals/ \\
      [--ranking-loss-threshold 0.49]   # IC < 0.02 时 loss ≈ 0.49
      [--decay-threshold 0.01]          # 因子 IC 下滑超过此值触发告警
      [--realized-ic-gap-threshold 0.5] # 实际 IC / 预期 IC < 此值触发 DPO
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
# 阈值常量
# ─────────────────────────────────────────────────────────────────────────────

RANKING_LOSS_HIGH = 0.49      # IC < 0.02 时 loss > 0.49 → 触发重训
FACTOR_DECAY_THRESH = 0.01    # IC 下滑 > 1pp → 告警
FACTOR_REMOVE_THRESH = 0.015  # |IC| < 0.015 → 建议移除
REALIZED_IC_GAP = 0.50        # realized_IC / expected_IC < 0.5 → 触发 DPO
DPO_MIN_RECORDS = 100         # feedback 记录数 >= 此值才可触发 DPO

# 动作优先级
ACTION_PRIORITY = {
    "RETRAIN_LGBM":     1,
    "UPDATE_FACTORS":   2,
    "RETRAIN_AGENTS":   3,
    "TRIGGER_DPO":      4,
    "UPDATE_POSITIONER": 5,
    "MONITOR_ONLY":     6,
}


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_round_summary(seq_val_dir: Path) -> dict | None:
    """加载顺序验证汇总（round_summary.json）."""
    fp = seq_val_dir / "round_summary.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def load_all_factor_ics(seq_val_dir: Path) -> dict[int, pd.DataFrame]:
    """加载所有 round 的因子 IC 表."""
    result: dict[int, pd.DataFrame] = {}
    for rd in sorted(seq_val_dir.glob("round_*/factor_ic.csv")):
        rid = int(rd.parent.name.split("_")[1])
        try:
            df = pd.read_csv(rd, index_col="factor")
            result[rid] = df
        except Exception:
            pass
    return result


def load_feedback(feedback_dir: Path) -> dict:
    """加载 realized IC 汇总."""
    fp = feedback_dir / "realized_ic.json"
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))


def load_realized_pnl(feedback_dir: Path) -> pd.DataFrame | None:
    """加载 realized_pnl.parquet."""
    fp = feedback_dir / "realized_pnl.parquet"
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


# ─────────────────────────────────────────────────────────────────────────────
# Signal 1: 排序损失
# ─────────────────────────────────────────────────────────────────────────────

def compute_ranking_loss(summary: dict) -> dict:
    """从 round_summary 提取近期 IC，计算排序损失."""
    trajectory = summary.get("ic_trajectory", [])
    if not trajectory:
        return {"value": 1.0, "recent_ic": None, "trend": "UNKNOWN"}

    # 取最近 2 轮的 IC 均值作为"当前表现"
    recent = [r for r in trajectory if r.get("ic_mean") is not None][-3:]
    if not recent:
        return {"value": 1.0, "recent_ic": None, "trend": "UNKNOWN"}

    recent_ic = float(np.mean([r["ic_mean"] for r in recent]))
    loss_val = float(0.5 * (1.0 - recent_ic))

    # 判断趋势（与早期 2 轮对比）
    early = [r for r in trajectory if r.get("ic_mean") is not None][:3]
    if len(early) >= 2:
        early_ic = float(np.mean([r["ic_mean"] for r in early]))
        delta = recent_ic - early_ic
        trend = "IMPROVING" if delta > 0.005 else ("DECLINING" if delta < -0.005 else "STABLE")
    else:
        trend = "STABLE"

    return {
        "value": round(loss_val, 6),
        "recent_ic": round(recent_ic, 6),
        "trend": trend,
        "n_recent_rounds": len(recent),
        "alert": loss_val > RANKING_LOSS_HIGH,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2: 因子衰减损失
# ─────────────────────────────────────────────────────────────────────────────

def compute_factor_decay(factor_ics: dict[int, pd.DataFrame]) -> dict:
    """跨 Round 分析因子 IC 趋势."""
    if len(factor_ics) < 2:
        return {"factors": {}, "summary": "数据不足"}

    rounds = sorted(factor_ics.keys())
    latest_rid = rounds[-1]
    baseline_rid = rounds[0]

    latest_ic = factor_ics[latest_rid]["ic_mean"]
    baseline_ic = factor_ics[baseline_rid]["ic_mean"]

    common_factors = latest_ic.index.intersection(baseline_ic.index)

    decay_map: dict[str, float] = {}
    for f in common_factors:
        delta = float(latest_ic[f] - baseline_ic[f])
        decay_map[f] = round(delta, 6)

    # 分类因子
    strongly_decaying = [f for f, d in decay_map.items() if d < -FACTOR_DECAY_THRESH * 2]
    mildly_decaying = [f for f, d in decay_map.items() if -FACTOR_DECAY_THRESH * 2 <= d < -FACTOR_DECAY_THRESH]
    improving = [f for f, d in decay_map.items() if d > FACTOR_DECAY_THRESH]

    # 绝对 IC 健康状态（基于最新 Round）
    weak_factors = [
        f for f in latest_ic.index
        if abs(float(latest_ic[f])) < FACTOR_REMOVE_THRESH
    ]
    strong_factors = [
        f for f in latest_ic.index
        if abs(float(latest_ic[f])) >= FACTOR_REMOVE_THRESH * 2
    ]

    total_decay_loss = sum(max(0.0, -d) for d in decay_map.values()) / max(len(decay_map), 1)

    return {
        "value": round(float(total_decay_loss), 6),
        "compared_rounds": (baseline_rid, latest_rid),
        "n_factors_analyzed": len(common_factors),
        "strongly_decaying": strongly_decaying,
        "mildly_decaying": mildly_decaying,
        "improving": improving,
        "weak_factors": weak_factors,
        "strong_factors": strong_factors,
        "decay_map": {f: d for f, d in sorted(decay_map.items(), key=lambda x: x[1])[:10]},
        "alert": len(strongly_decaying) > 3 or len(weak_factors) > 5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3: 策略损失（从 NAV 报告读取）
# ─────────────────────────────────────────────────────────────────────────────

def compute_strategy_loss(root: Path) -> dict:
    """从 NAV 报告读取策略 vs 基准表现."""
    nav_dirs = list((root / "reports").glob("nav_v4*/kelly/nav_metrics.json"))
    if not nav_dirs:
        nav_dirs = list((root / "reports").glob("nav_v4/kelly/nav_metrics.json"))

    if not nav_dirs:
        return {"value": float("nan"), "note": "未找到 NAV 报告"}

    latest = sorted(nav_dirs)[-1]
    m = json.loads(latest.read_text(encoding="utf-8"))

    excess = m.get("excess_ann_return", 0.0) or 0.0
    # 损失 = max(0, -超额收益)：只有跑输时才算损失
    loss_val = float(max(0.0, -excess))

    return {
        "value": round(loss_val, 6),
        "excess_ann_return": round(float(excess), 6),
        "strategy_ann_return": round(float(m.get("ann_return", 0) or 0), 6),
        "benchmark_ann_return": round(float(m.get("benchmark_ann_return", 0) or 0), 6),
        "sharpe": round(float(m.get("sharpe_ratio", 0) or 0), 3),
        "source": str(latest),
        "alert": excess < 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal 4: Realized IC Gap
# ─────────────────────────────────────────────────────────────────────────────

def compute_realized_ic_gap(feedback: dict, summary: dict) -> dict:
    """实际 IC 与模型预期 IC 的差距."""
    realized_ic = feedback.get("overall_ic")
    n_records = feedback.get("n_records", 0)

    # 预期 IC：来自顺序验证的近期 IC 均值
    trajectory = summary.get("ic_trajectory", [])
    recent = [r["ic_mean"] for r in trajectory if r.get("ic_mean") is not None][-3:]
    expected_ic = float(np.mean(recent)) if recent else float("nan")

    if realized_ic is None or not np.isfinite(expected_ic) or expected_ic == 0:
        return {
            "value": float("nan"),
            "realized_ic": realized_ic,
            "expected_ic": expected_ic,
            "n_records": n_records,
            "note": "数据不足",
        }

    gap = float((expected_ic - realized_ic) / abs(expected_ic))
    loss_val = float(max(0.0, gap))

    return {
        "value": round(loss_val, 6),
        "realized_ic": round(float(realized_ic), 6),
        "expected_ic": round(float(expected_ic), 6),
        "gap_ratio": round(gap, 4),
        "n_records": n_records,
        "alert": gap > REALIZED_IC_GAP,
        "dpo_ready": n_records >= DPO_MIN_RECORDS and gap > REALIZED_IC_GAP,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 动作规划
# ─────────────────────────────────────────────────────────────────────────────

def plan_actions(
    ranking_loss: dict,
    factor_decay: dict,
    strategy_loss: dict,
    realized_gap: dict,
) -> list[dict]:
    """根据 4 类信号决定各组件的更新动作."""
    actions: list[dict] = []

    # 动作 1：LGBM 重训
    if ranking_loss.get("alert"):
        actions.append({
            "action": "RETRAIN_LGBM",
            "priority": ACTION_PRIORITY["RETRAIN_LGBM"],
            "reason": f"排序损失 {ranking_loss['value']:.4f} > 阈值（IC={ranking_loss['recent_ic']:.4f}）",
            "cmd": "python scripts/train_lgbm_model.py --panel data/panel/alpha_panel_v4.parquet",
            "urgency": "HIGH",
        })

    # 动作 2：因子更新
    if factor_decay.get("alert"):
        actions.append({
            "action": "UPDATE_FACTORS",
            "priority": ACTION_PRIORITY["UPDATE_FACTORS"],
            "reason": (
                f"{len(factor_decay.get('strongly_decaying', []))} 个因子严重衰减，"
                f"{len(factor_decay.get('weak_factors', []))} 个因子绝对 IC 过低"
            ),
            "weak_factors": factor_decay.get("weak_factors", []),
            "strongly_decaying": factor_decay.get("strongly_decaying", []),
            "recommendation": "运行 scripts/select_stable_factors.py 重新筛选因子集",
            "urgency": "MEDIUM",
        })

    # 动作 3：Agent 重训（策略损失高）
    if strategy_loss.get("alert"):
        actions.append({
            "action": "RETRAIN_AGENTS",
            "priority": ACTION_PRIORITY["RETRAIN_AGENTS"],
            "reason": f"策略跑输基准：超额收益={strategy_loss.get('excess_ann_return', 0)*100:.2f}%",
            "recommendation": "检查各 agent IC，对 IC < 0 的 agent 触发重训",
            "urgency": "MEDIUM",
        })

    # 动作 4：DPO 触发
    if realized_gap.get("dpo_ready"):
        actions.append({
            "action": "TRIGGER_DPO",
            "priority": ACTION_PRIORITY["TRIGGER_DPO"],
            "reason": (
                f"Realized IC 偏差 {realized_gap['gap_ratio']:.2%}（"
                f"realized={realized_gap['realized_ic']:.4f} vs "
                f"expected={realized_gap['expected_ic']:.4f}）"
                f"，已积累 {realized_gap['n_records']} 条反馈"
            ),
            "cmd": "python scripts/dpo_trainer.py --feedback data/feedback/realized_pnl.parquet",
            "urgency": "LOW",
        })

    # 动作 5：仓位优化器校准（策略损失存在但不严重）
    if not strategy_loss.get("alert") and strategy_loss.get("value", 0) > 0:
        actions.append({
            "action": "UPDATE_POSITIONER",
            "priority": ACTION_PRIORITY["UPDATE_POSITIONER"],
            "reason": "策略表现轻微下滑，建议重新校准 Kelly 参数",
            "urgency": "LOW",
        })

    # 兜底：无告警
    if not actions:
        actions.append({
            "action": "MONITOR_ONLY",
            "priority": ACTION_PRIORITY["MONITOR_ONLY"],
            "reason": "所有信号正常，继续监控",
            "urgency": "NONE",
        })

    actions.sort(key=lambda x: x["priority"])
    return actions


# ─────────────────────────────────────────────────────────────────────────────
# 因子健康报告
# ─────────────────────────────────────────────────────────────────────────────

def build_factor_health_report(
    factor_ics: dict[int, pd.DataFrame],
    decay_signal: dict,
) -> dict:
    """生成每个因子的健康状态报告."""
    if not factor_ics:
        return {}

    latest_rid = max(factor_ics.keys())
    latest = factor_ics[latest_rid]

    report: dict[str, dict] = {}
    for factor in latest.index:
        row = latest.loc[factor]
        ic_series = {
            rid: float(factor_ics[rid].loc[factor, "ic_mean"])
            if factor in factor_ics[rid].index else None
            for rid in sorted(factor_ics.keys())
        }
        valid_ics = [v for v in ic_series.values() if v is not None]
        trend_slope = float(np.polyfit(range(len(valid_ics)), valid_ics, 1)[0]) if len(valid_ics) >= 3 else 0.0

        status = "HEALTHY"
        if abs(float(row["ic_mean"])) < FACTOR_REMOVE_THRESH:
            status = "WEAK"
        elif factor in decay_signal.get("strongly_decaying", []):
            status = "DECLINING"
        elif factor in decay_signal.get("improving", []):
            status = "IMPROVING"

        report[factor] = {
            "status": status,
            "latest_ic_mean": round(float(row["ic_mean"]), 6),
            "latest_icir": round(float(row["icir"]), 4) if pd.notna(row.get("icir")) else None,
            "ic_trend_slope": round(trend_slope, 6),
            "ic_history": {str(rid): round(v, 6) if v is not None else None for rid, v in ic_series.items()},
        }

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run(
    seq_val_dir: Path,
    feedback_dir: Path,
    out_dir: Path,
    root: Path = _ROOT,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().isoformat(timespec="seconds")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    summary = load_round_summary(seq_val_dir)
    if not summary:
        logger.error(f"未找到 round_summary.json（请先运行 run_sequential_validation.py）")
        return {}

    factor_ics = load_all_factor_ics(seq_val_dir)
    feedback = load_feedback(feedback_dir)
    logger.info(
        f"[Dispatch] 加载：{len(summary['ic_trajectory'])} 轮次，"
        f"{len(factor_ics)} 因子 IC 表，"
        f"feedback n_records={feedback.get('n_records', 0)}"
    )

    # ── 计算 4 类损失信号 ──────────────────────────────────────────────────────
    s1 = compute_ranking_loss(summary)
    s2 = compute_factor_decay(factor_ics)
    s3 = compute_strategy_loss(root)
    s4 = compute_realized_ic_gap(feedback, summary)

    print("\n" + "═" * 65)
    print("  损失信号总览")
    print("─" * 65)
    print(f"  Signal 1 排序损失:     {s1['value']:.4f}  (IC={s1.get('recent_ic', 'N/A')}, 趋势={s1['trend']})  {'⚠️' if s1.get('alert') else '✅'}")
    print(f"  Signal 2 因子衰减:     {s2['value']:.4f}  (强衰减因子={len(s2.get('strongly_decaying', []))})  {'⚠️' if s2.get('alert') else '✅'}")
    print(f"  Signal 3 策略损失:     {s3['value']:.4f}  (超额={s3.get('excess_ann_return', 0)*100:.2f}%)  {'⚠️' if s3.get('alert') else '✅'}")
    ic_gap_str = f"{s4['value']:.4f}" if np.isfinite(s4.get('value', float('nan'))) else "  N/A "
    print(f"  Signal 4 Realized Gap: {ic_gap_str}  (DPO就绪={s4.get('dpo_ready', False)})  {'⚠️' if s4.get('alert') else '✅'}")
    print("─" * 65)

    # ── 规划动作 ──────────────────────────────────────────────────────────────
    actions = plan_actions(s1, s2, s3, s4)

    print("\n  动作规划（按优先级）:")
    for act in actions:
        urgency_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "⚪"}.get(act["urgency"], "?")
        print(f"  {urgency_icon} [{act['action']}] {act['reason']}")
        if act.get("cmd"):
            print(f"     → {act['cmd']}")
    print("═" * 65)

    # ── 保存输出 ──────────────────────────────────────────────────────────────
    factor_health = build_factor_health_report(factor_ics, s2)

    latest_signals = {
        "run_ts": run_ts,
        "signal_1_ranking_loss": s1,
        "signal_2_factor_decay": s2,
        "signal_3_strategy_loss": s3,
        "signal_4_realized_gap": s4,
        "overall_health": (
            "CRITICAL" if any(a["urgency"] == "HIGH" for a in actions)
            else "WARNING" if any(a["urgency"] == "MEDIUM" for a in actions)
            else "HEALTHY"
        ),
    }

    (out_dir / "latest.json").write_text(
        json.dumps(latest_signals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    action_plan_data = {"run_ts": run_ts, "actions": actions}
    (out_dir / "action_plan.json").write_text(
        json.dumps(action_plan_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (out_dir / "factor_health.json").write_text(
        json.dumps(factor_health, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # ── DB 双写（失败不中断业务）────────────────────────────────────────────
    try:
        from app.db.writers import get_writer
        get_writer().write_loss_signals(latest_signals, action_plan_data, factor_health)
    except Exception as _dw_e:
        import logging as _log
        _log.getLogger(__name__).warning("loss_signals DB 双写跳过: %s", _dw_e)

    # 追加历史日志
    history_path = out_dir / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "run_ts": run_ts,
            "ranking_loss": s1["value"],
            "factor_decay": s2["value"],
            "strategy_loss": s3["value"] if np.isfinite(s3["value"]) else None,
            "realized_gap": s4["value"] if np.isfinite(s4.get("value", float("nan"))) else None,
            "n_actions": len(actions),
            "overall_health": latest_signals["overall_health"],
        }, ensure_ascii=False) + "\n")

    logger.info(f"✅ 损失信号已保存：{out_dir}")
    return latest_signals


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-val",  type=Path, default=Path("reports/sequential_val/"))
    p.add_argument("--feedback", type=Path, default=Path("data/feedback/"))
    p.add_argument("--out",      type=Path, default=Path("data/loss_signals/"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        seq_val_dir=args.seq_val,
        feedback_dir=args.feedback,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
