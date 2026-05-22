"""顺序验证框架（Sequential Validation）— Week 1 学习循环基础设施.

设计
====
每 Round 使用扩展训练窗口 + 单（或多）期测试，模拟"每个新季度 = 一次真实考试"：

  Round 0 (baseline): Train 2019Q1-2023Q4, Test 2024Q1-Q4   ← 多期基线
  Round 1: Train 2019Q1-2024Q1, Test 2024Q2
  Round 2: Train 2019Q1-2024Q2, Test 2024Q3
  Round 3: Train 2019Q1-2024Q3, Test 2024Q4
  Round 4: Train 2019Q1-2024Q4, Test 2025Q1   ← 2025 数据开始喂入
  Round 5: Train 2019Q1-2025Q1, Test 2025Q2
  Round 6: Train 2019Q1-2025Q2, Test 2025Q3
  Round 7: Train 2019Q1-2025Q3, Test 2025Q4

每轮输出
========
  reports/sequential_val/round_{N}/
    round_result.json       汇总指标（IC、ICIR、loss 信号、champion 决策）
    ic_per_period.csv       每个测试期的截面 IC
    factor_ic.csv           每个因子的截面 IC（测试期平均）
    model.pkl               本轮训练的候选模型（challenger）

  reports/sequential_val/
    round_summary.json      所有轮次汇总（用于 IC 曲线可视化）
    champion_history.json   champion 版本变更记录

用法
====
  python scripts/run_sequential_validation.py \\
      --panel data/panel/alpha_panel_v4.parquet \\
      --features data/features/top_factors_v3_63d_flat.json \\
      --out reports/sequential_val/ \\
      [--start-round 0] [--end-round 7] [--ic-threshold 0.015]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.models.factor_model import build_lgbm_arrays
from quantmind.models.lgbm_ranker import LGBMRankerModel

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COL = "forward_return_63d"
MIN_TRAIN_PERIODS = 8   # 最少训练截面数
MIN_TEST_STOCKS = 30    # 测试期最少股票数
IC_PROMOTE_MARGIN = -0.003  # challenger.IC > champion.IC + margin 才升级
IC_WATCHLIST = 0.015         # IC 低于此值触发告警

# 默认模型超参（与 lgbm_v6 保持一致）
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "early_stopping_rounds": 50,
    "verbose": -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# Round 定义
# ─────────────────────────────────────────────────────────────────────────────

def _build_rounds(all_dates: list[pd.Timestamp]) -> list[dict]:
    """根据面板中实际存在的日期动态生成 Round 配置."""

    def _find_date(target_str: str) -> pd.Timestamp | None:
        t = pd.Timestamp(target_str)
        # 找最近的实际日期（±45天内）
        for d in all_dates:
            if abs((d - t).days) <= 45:
                return d
        return None

    def _dates_in_range(start_str: str, end_str: str) -> list[pd.Timestamp]:
        s, e = pd.Timestamp(start_str), pd.Timestamp(end_str)
        return [d for d in all_dates if s <= d <= e]

    rounds = [
        # Round 0: 多期基线（训练 2019-2023，测试全 2024）
        {
            "id": 0, "name": "baseline",
            "train_end": "2023-12-31",
            "test_range": ("2024-01-01", "2024-12-31"),
            "note": "训练 2019Q1-2023Q4，测试全 2024 年（4 期）",
        },
        # Round 1-3: 逐步纳入 2024 数据
        {
            "id": 1, "name": "2024Q1→Q2",
            "train_end": "2024-03-31", "test_range": ("2024-04-01", "2024-06-30"),
            "note": "首次纳入 2024Q1 数据",
        },
        {
            "id": 2, "name": "2024Q2→Q3",
            "train_end": "2024-06-30", "test_range": ("2024-07-01", "2024-09-30"),
            "note": "纳入 2024Q2 数据",
        },
        {
            "id": 3, "name": "2024Q3→Q4",
            "train_end": "2024-09-30", "test_range": ("2024-10-01", "2024-12-31"),
            "note": "纳入 2024Q3 数据",
        },
        # Round 4-7: 2025 数据逐步喂入
        {
            "id": 4, "name": "2024Q4→2025Q1",
            "train_end": "2024-12-31", "test_range": ("2025-01-01", "2025-03-31"),
            "note": "首次纳入 2025 数据（2024 全年）",
        },
        {
            "id": 5, "name": "2025Q1→Q2",
            "train_end": "2025-03-31", "test_range": ("2025-04-01", "2025-06-30"),
            "note": "纳入 2025Q1 数据",
        },
        {
            "id": 6, "name": "2025Q2→Q3",
            "train_end": "2025-06-30", "test_range": ("2025-07-01", "2025-09-30"),
            "note": "纳入 2025Q2 数据",
        },
        {
            "id": 7, "name": "2025Q3→Q4",
            "train_end": "2025-09-30", "test_range": ("2025-10-01", "2025-12-31"),
            "note": "纳入 2025Q3 数据（2025 全年完整验证）",
        },
        # Round 8-9: 2026 数据逐步喂入
        {
            "id": 8, "name": "2025Q4→2026Q1",
            "train_end": "2025-12-31", "test_range": ("2026-01-01", "2026-03-31"),
            "note": "首次纳入 2026 数据（2025 全年完整训练）",
        },
        {
            "id": 9, "name": "2026Q1→Q2",
            "train_end": "2026-03-31", "test_range": ("2026-04-01", "2026-06-30"),
            "note": "纳入 2026Q1 数据（2026H1 完整验证）",
        },
    ]

    result = []
    for r in rounds:
        train_end = pd.Timestamp(r["train_end"])
        train_dates = [d for d in all_dates if d <= train_end]
        test_dates = _dates_in_range(*r["test_range"])

        if len(train_dates) < MIN_TRAIN_PERIODS:
            logger.warning(f"Round {r['id']} 训练期不足 {MIN_TRAIN_PERIODS} 期，跳过")
            continue
        if not test_dates:
            logger.warning(f"Round {r['id']} 测试期无有效日期，跳过")
            continue

        result.append({
            **r,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "n_train_periods": len(train_dates),
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 核心计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_rank_ic(
    model: LGBMRankerModel,
    panel: pd.DataFrame,
    test_dates: list[pd.Timestamp],
    feature_cols: list[str],
    direction: int = 1,
) -> tuple[dict[pd.Timestamp, float], float, float]:
    """计算测试期的截面 Rank IC.

    Returns:
        (ic_per_period, ic_mean, icir)
    """
    ic_map: dict[pd.Timestamp, float] = {}
    for d in test_dates:
        if d not in panel.index.get_level_values("as_of"):
            continue
        xs = panel.xs(d, level="as_of")
        sub = xs[feature_cols + [LABEL_COL]].dropna(subset=[LABEL_COL])
        if len(sub) < MIN_TEST_STOCKS:
            continue
        X = sub[feature_cols].fillna(0.0).values.astype(np.float32)
        y = sub[LABEL_COL].values.astype(np.float64)
        raw_pred = model._model.predict(X)  # 原始 booster 分数
        pred = raw_pred * direction
        rho, _ = stats.spearmanr(pred, y)
        ic_map[d] = float(rho) if np.isfinite(rho) else float("nan")

    ics = [v for v in ic_map.values() if np.isfinite(v)]
    ic_mean = float(np.mean(ics)) if ics else float("nan")
    icir = (
        float(ic_mean / np.std(ics, ddof=1))
        if len(ics) > 1 and np.std(ics, ddof=1) > 1e-9
        else float("nan")
    )
    return ic_map, ic_mean, icir


def compute_factor_ic(
    panel: pd.DataFrame,
    test_dates: list[pd.Timestamp],
    feature_cols: list[str],
) -> pd.DataFrame:
    """计算每个因子在测试期的截面 IC（原始 Spearman，不经过模型）.

    Returns:
        DataFrame(index=factor, columns=[ic_mean, ic_std, icir, positive_rate])
    """
    rows: dict[str, list[float]] = {col: [] for col in feature_cols}
    for d in test_dates:
        if d not in panel.index.get_level_values("as_of"):
            continue
        xs = panel.xs(d, level="as_of")
        sub = xs[feature_cols + [LABEL_COL]].dropna(subset=[LABEL_COL])
        if len(sub) < MIN_TEST_STOCKS:
            continue
        y = sub[LABEL_COL].values
        for col in feature_cols:
            fv = sub[col].fillna(sub[col].median())
            rho, _ = stats.spearmanr(fv.values, y)
            if np.isfinite(rho):
                rows[col].append(float(rho))

    records = []
    for col, ics in rows.items():
        arr = np.array(ics)
        mean = float(np.mean(arr)) if len(arr) > 0 else float("nan")
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")
        icir = mean / std if std and np.isfinite(std) and std > 1e-9 else float("nan")
        pos_rate = float(np.mean(arr > 0)) if len(arr) > 0 else float("nan")
        records.append({"factor": col, "ic_mean": mean, "ic_std": std,
                        "icir": icir, "positive_rate": pos_rate, "n_periods": len(arr)})

    return pd.DataFrame(records).set_index("factor").sort_values("icir", ascending=False)


def train_challenger(
    panel: pd.DataFrame,
    train_dates: list[pd.Timestamp],
    feature_cols: list[str],
    params: dict[str, Any],
    val_periods: int = 2,
) -> tuple[LGBMRankerModel, int]:
    """在训练期数据上训练候选模型，返回 (model, direction).

    direction: auto_flip 检测（若 IC 均值 < 0 则取 -1）。
    """
    # 用最后 val_periods 期作为 early stopping 验证集
    val_dates = train_dates[-val_periods:] if len(train_dates) > val_periods else train_dates[-1:]
    actual_train = train_dates[: len(train_dates) - len(val_dates)]
    if len(actual_train) < MIN_TRAIN_PERIODS:
        actual_train = train_dates  # 数据太少时直接全量训练

    X_train, y_train, g_train = build_lgbm_arrays(panel, actual_train, feature_cols, LABEL_COL)
    X_val, y_val, g_val = build_lgbm_arrays(panel, val_dates, feature_cols, LABEL_COL)

    if X_train.shape[0] == 0:
        raise ValueError("训练集为空")

    model = LGBMRankerModel(
        n_estimators=params.get("n_estimators", 400),
        num_leaves=params.get("num_leaves", 31),
        learning_rate=params.get("learning_rate", 0.05),
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        groups=g_train,
        X_val=X_val if X_val.shape[0] > 0 else None,
        y_val=y_val if X_val.shape[0] > 0 else None,
        groups_val=g_val if X_val.shape[0] > 0 else None,
        feature_names=feature_cols,
    )

    # auto_flip：用验证集原始分数检测方向
    direction = 1
    try:
        if X_val.shape[0] > 0 and model._model is not None:
            raw_pred = model._model.predict(X_val)
            # 重建验证集标签（连续收益值而非分位数）
            y_cont = []
            for d in val_dates:
                if d not in panel.index.get_level_values("as_of"):
                    continue
                xs = panel.xs(d, level="as_of")
                sub = xs[feature_cols + [LABEL_COL]].dropna(subset=[LABEL_COL])
                y_cont.extend(sub[LABEL_COL].tolist())
            if len(y_cont) == len(raw_pred):
                rho, _ = stats.spearmanr(raw_pred, y_cont)
                if np.isfinite(rho) and rho < 0:
                    direction = -1
    except Exception:
        pass

    return model, direction


def compute_loss_signals(
    ic_mean: float,
    factor_ic: pd.DataFrame,
    prev_factor_ic: pd.DataFrame | None,
) -> dict[str, Any]:
    """计算本 Round 的损失信号."""

    # Signal 1: 排序损失（0 = 完美排序，1 = 随机）
    ranking_loss = float(0.5 * (1.0 - ic_mean)) if np.isfinite(ic_mean) else 1.0

    # Signal 2: 因子衰减（与上轮相比 IC 变化）
    factor_decay: dict[str, float] = {}
    if prev_factor_ic is not None:
        common = factor_ic.index.intersection(prev_factor_ic.index)
        for f in common:
            delta = factor_ic.loc[f, "ic_mean"] - prev_factor_ic.loc[f, "ic_mean"]
            factor_decay[f] = float(delta)

    # 找出衰减最严重的因子
    decaying_factors = [
        f for f, delta in factor_decay.items() if delta < -0.01
    ]
    healthy_factors = [
        f for f in factor_ic.index
        if abs(factor_ic.loc[f, "ic_mean"]) >= 0.01
    ]

    # 告警标志
    alerts = []
    if np.isfinite(ic_mean) and ic_mean < IC_WATCHLIST:
        alerts.append(f"整体 IC={ic_mean:.4f} 低于告警阈值 {IC_WATCHLIST}")
    if len(decaying_factors) > 3:
        alerts.append(f"因子衰减：{len(decaying_factors)} 个因子 IC 下滑 > 0.01")
    if len(healthy_factors) < 5:
        alerts.append(f"仅 {len(healthy_factors)} 个因子 IC ≥ 0.01，因子质量下降")

    return {
        "ranking_loss": round(ranking_loss, 6),
        "factor_decay": {k: round(v, 6) for k, v in factor_decay.items()},
        "decaying_factors": decaying_factors,
        "healthy_factors": healthy_factors,
        "n_healthy_factors": len(healthy_factors),
        "alerts": alerts,
        "needs_retrain": len(alerts) > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Champion-Challenger 决策
# ─────────────────────────────────────────────────────────────────────────────

def champion_challenger_decision(
    challenger_ic: float,
    champion_ic: float,
    margin: float = IC_PROMOTE_MARGIN,
) -> str:
    """决定是否升级 champion.

    Returns:
        'PROMOTE' / 'WATCHLIST' / 'DISCARD'
    """
    if not np.isfinite(challenger_ic) or not np.isfinite(champion_ic):
        return "DISCARD"
    diff = challenger_ic - champion_ic
    if diff >= margin:
        return "PROMOTE"
    if diff >= margin - 0.01:  # 接近但未达到
        return "WATCHLIST"
    return "DISCARD"


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    panel: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    start_round: int = 0,
    end_round: int = 7,
    ic_threshold: float = IC_WATCHLIST,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {**DEFAULT_PARAMS, **(model_params or {})}

    all_dates = sorted(panel.index.get_level_values("as_of").unique())
    rounds = _build_rounds(all_dates)
    rounds = [r for r in rounds if start_round <= r["id"] <= end_round]

    # 过滤掉没有足够标签的测试期
    def _has_labels(d: pd.Timestamp) -> bool:
        xs = panel.xs(d, level="as_of")
        return xs[LABEL_COL].notna().sum() >= MIN_TEST_STOCKS

    for r in rounds:
        r["test_dates"] = [d for d in r["test_dates"] if _has_labels(d)]

    rounds = [r for r in rounds if r["test_dates"]]

    if not rounds:
        logger.error("无有效 Round，退出")
        return {}

    champion_ic = float("nan")
    champion_round = -1
    prev_factor_ic: pd.DataFrame | None = None
    champion_history: list[dict] = []
    round_summaries: list[dict] = []

    logger.info(f"[SeqVal] 开始顺序验证：{len(rounds)} 轮，特征数={len(feature_cols)}")

    for r in rounds:
        rid = r["id"]
        round_out = out_dir / f"round_{rid}"
        round_out.mkdir(exist_ok=True)

        logger.info(
            f"\n{'─'*60}\n"
            f"[Round {rid}] {r['name']}  |  "
            f"训练期 {r['n_train_periods']} 期 → {r['train_dates'][-1].date()}  |  "
            f"测试期 {[d.date() for d in r['test_dates']]}"
        )

        # ── 1. 训练候选模型 ──────────────────────────────────────────────────
        try:
            model, direction = train_challenger(
                panel, r["train_dates"], feature_cols, params
            )
        except Exception as e:
            logger.error(f"[Round {rid}] 训练失败: {e}")
            continue

        # ── 2. 计算测试期 IC ──────────────────────────────────────────────────
        ic_per_period, ic_mean, icir = compute_rank_ic(
            model, panel, r["test_dates"], feature_cols, direction
        )
        logger.info(
            f"[Round {rid}] IC_mean={ic_mean:.4f}  ICIR={icir:.3f}  "
            f"direction={direction}"
        )

        # ── 3. 计算因子 IC ────────────────────────────────────────────────────
        factor_ic = compute_factor_ic(panel, r["test_dates"], feature_cols)

        # ── 4. 计算损失信号 ───────────────────────────────────────────────────
        loss = compute_loss_signals(ic_mean, factor_ic, prev_factor_ic)
        if loss["alerts"]:
            for alert in loss["alerts"]:
                logger.warning(f"[Round {rid}] ⚠️  {alert}")

        # ── 5. Champion-Challenger 决策 ───────────────────────────────────────
        if not np.isfinite(champion_ic):
            # 第一个有效 Round 直接成为 champion
            decision = "PROMOTE"
        else:
            decision = champion_challenger_decision(ic_mean, champion_ic)

        if decision == "PROMOTE":
            prev_champion_ic = champion_ic
            champion_ic = ic_mean
            champion_round = rid
            model.save(round_out / "model.pkl")
            champion_history.append({
                "round": rid,
                "ic_mean": ic_mean,
                "direction": direction,
                "prev_champion_ic": float(prev_champion_ic) if np.isfinite(prev_champion_ic) else None,
                "ic_delta": float(ic_mean - prev_champion_ic) if np.isfinite(prev_champion_ic) else None,
            })
            logger.info(f"[Round {rid}] ✅  PROMOTE → 新 champion  IC={ic_mean:.4f}")
        else:
            logger.info(f"[Round {rid}] ⏩  {decision}（champion IC={champion_ic:.4f}）")

        # ── 6. 保存本 Round 结果 ──────────────────────────────────────────────
        ic_period_records = [
            {"as_of": str(d.date()), "rank_ic": ic}
            for d, ic in sorted(ic_per_period.items())
        ]
        pd.DataFrame(ic_period_records).to_csv(round_out / "ic_per_period.csv", index=False)
        factor_ic.reset_index().to_csv(round_out / "factor_ic.csv", index=False)

        round_result = {
            "round_id": rid,
            "name": r["name"],
            "note": r["note"],
            "n_train_periods": r["n_train_periods"],
            "train_end": str(r["train_dates"][-1].date()),
            "test_dates": [str(d.date()) for d in r["test_dates"]],
            "direction": direction,
            "ic_mean": round(float(ic_mean), 6) if np.isfinite(ic_mean) else None,
            "icir": round(float(icir), 4) if np.isfinite(icir) else None,
            "ic_per_period": {str(d.date()): round(v, 6) for d, v in ic_per_period.items()},
            "champion_ic_before": round(float(champion_ic if decision != "PROMOTE" else (champion_ic if champion_round != rid else float("nan"))), 6) if np.isfinite(champion_ic) else None,
            "champion_decision": decision,
            "loss_signals": loss,
            "top5_factors_by_icir": factor_ic.head(5)["icir"].to_dict(),
            "bottom5_factors_by_icir": factor_ic.tail(5)["icir"].to_dict(),
        }
        (round_out / "round_result.json").write_text(
            json.dumps(round_result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        round_summaries.append({
            "round_id": rid,
            "name": r["name"],
            "train_end": str(r["train_dates"][-1].date()),
            "test_dates": [str(d.date()) for d in r["test_dates"]],
            "ic_mean": round(float(ic_mean), 6) if np.isfinite(ic_mean) else None,
            "icir": round(float(icir), 4) if np.isfinite(icir) else None,
            "direction": direction,
            "champion_decision": decision,
            "n_alerts": len(loss["alerts"]),
        })

        prev_factor_ic = factor_ic

        # ── 打印本 Round 顶部因子 ─────────────────────────────────────────────
        top3 = factor_ic.head(3)
        bot3 = factor_ic.tail(3)
        print(f"\n  Top 3 因子（ICIR）: "
              + ", ".join(f"{f}={row['icir']:+.3f}" for f, row in top3.iterrows()))
        print(f"  Bot 3 因子（ICIR）: "
              + ", ".join(f"{f}={row['icir']:+.3f}" for f, row in bot3.iterrows()))

    # ── 保存全局汇总 ──────────────────────────────────────────────────────────
    summary = {
        "total_rounds": len(round_summaries),
        "champion_round": champion_round,
        "champion_ic": round(float(champion_ic), 6) if np.isfinite(champion_ic) else None,
        "ic_trajectory": [
            {"round_id": r["round_id"], "name": r["name"],
             "train_end": r["train_end"], "ic_mean": r["ic_mean"],
             "icir": r["icir"], "champion_decision": r["champion_decision"]}
            for r in round_summaries
        ],
        "feature_cols": feature_cols,
        "label_col": LABEL_COL,
    }
    (out_dir / "round_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "champion_history.json").write_text(
        json.dumps(champion_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ── 打印最终汇总表 ────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  {'Round':<8} {'名称':<20} {'训练截止':<12} {'IC_mean':<10} {'ICIR':<8} {'决策'}")
    print("─" * 70)
    for r in round_summaries:
        ic_str = f"{r['ic_mean']:+.4f}" if r["ic_mean"] is not None else "  N/A  "
        ir_str = f"{r['icir']:+.3f}"   if r["icir"] is not None else " N/A "
        decision_icon = {"PROMOTE": "✅", "WATCHLIST": "⚠️ ", "DISCARD": "❌"}.get(
            r["champion_decision"], "?"
        )
        print(f"  {r['round_id']:<8} {r['name']:<20} {r['train_end']:<12} "
              f"{ic_str:<10} {ir_str:<8} {decision_icon} {r['champion_decision']}")
    print("═" * 70)
    print(f"\n  Champion → Round {champion_round}，IC={champion_ic:.4f}")
    print(f"  结果已保存至：{out_dir}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path,
                   default=Path("data/panel/alpha_panel_v4.parquet"))
    p.add_argument("--features", type=Path,
                   default=Path("data/features/top_factors_v3_63d_flat.json"),
                   help="因子列表 JSON（list 格式）或每行一个特征名的 txt")
    p.add_argument("--out", type=Path,
                   default=Path("reports/sequential_val/"))
    p.add_argument("--start-round", type=int, default=0)
    p.add_argument("--end-round",   type=int, default=9)
    p.add_argument("--label-col", type=str, default=None,
                   help="覆盖 LABEL_COL（默认 forward_return_63d）；"
                        "2026 数据建议改用 forward_return_21d")
    p.add_argument("--ic-threshold", type=float, default=IC_WATCHLIST,
                   help="IC 告警阈值（默认 0.015）")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--learning-rate", type=float, default=0.05)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── 加载面板 ──────────────────────────────────────────────────────────────
    logger.info(f"[SeqVal] 加载面板：{args.panel}")
    panel = pd.read_parquet(args.panel)
    n_periods = panel.index.get_level_values("as_of").nunique()
    logger.info(f"  面板：{panel.shape}，{n_periods} 期")

    # ── 加载特征列表 ──────────────────────────────────────────────────────────
    feat_path = args.features
    if feat_path.suffix == ".json":
        raw = json.loads(feat_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            feature_cols = raw
        elif isinstance(raw, dict):
            # 尝试多个常见 key
            for key in ("feature_cols", "selected_factors", "factors", "features"):
                if key in raw and isinstance(raw[key], list):
                    feature_cols = raw[key]
                    break
            else:
                raise SystemExit(f"无法解析特征文件（未知结构）：{feat_path}")
    else:
        feature_cols = [
            line.strip().split()[0]
            for line in feat_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    # 过滤掉面板中不存在的列
    available = [c for c in feature_cols if c in panel.columns]
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        logger.warning(f"  {len(missing)} 个特征列不在面板中（已忽略）: {missing[:5]}")
    feature_cols = available
    logger.info(f"  有效特征数：{len(feature_cols)}")

    # ── 运行验证 ──────────────────────────────────────────────────────────────
    model_params = {
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "early_stopping_rounds": 50,
    }

    # 支持 --label-col 覆盖全局 LABEL_COL
    if args.label_col:
        import quantmind  # noqa: F401 (side-effect-free import guard)
        import scripts.run_sequential_validation as _self
        _self.LABEL_COL = args.label_col
        global LABEL_COL
        LABEL_COL = args.label_col
        logger.info(f"[SeqVal] 覆盖 LABEL_COL → {LABEL_COL}")

    run_validation(
        panel=panel,
        feature_cols=feature_cols,
        out_dir=args.out,
        start_round=args.start_round,
        end_round=args.end_round,
        ic_threshold=args.ic_threshold,
        model_params=model_params,
    )


if __name__ == "__main__":
    main()
