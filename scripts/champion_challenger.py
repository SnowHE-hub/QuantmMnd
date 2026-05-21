"""Champion-Challenger 模型门控 — Week 2 学习循环.

对当前生产模型（champion）和候选模型（challenger）在共同测试期上进行
多维度评估，根据评分决定是否升级生产模型。

评估维度（全部满足才 PROMOTE）
================================
  1. IC 提升：challenger.IC > champion.IC + IC_MARGIN（默认 -0.003）
  2. 正向 IC：challenger.IC > IC_MIN（默认 0.0，至少不是负的）
  3. 稳定性：如有多测试期，challenger ICIR > -0.5

决策结果
========
  PROMOTE    → 备份旧 champion，将 challenger 复制为新 production 模型
  WATCHLIST  → IC 提升明显但未达标，继续观察下期
  DISCARD    → 丢弃 challenger，记录衰减原因

输出
====
  models/
    lgbm_production.pkl           实时指向当前最优模型（软链接/覆盖）
    archive/lgbm_{ts}_{tag}.pkl   备份旧 champion
    model_lineage.json            版本血统完整记录

  reports/champion_challenger/
    latest_eval.json              最新一次评估详情
    decision_history.jsonl        历史决策日志

用法
====
  # 用 Round 7 challenger PK 当前 lgbm_v6
  python scripts/champion_challenger.py \\
      --panel     data/panel/alpha_panel_v4.parquet \\
      --features  data/features/top_factors_v3_63d_flat.json \\
      --champion  models/lgbm_v6_alpha.pkl \\
      --challenger reports/sequential_val/round_7/model.pkl \\
      --challenger-tag round7_2025q3 \\
      [--test-end 2025-12-31] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantmind.models.lgbm_ranker import LGBMRankerModel

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

IC_MARGIN         = -0.003   # challenger IC 只需不比 champion 差 0.003 以上
IC_MIN            = -0.02    # challenger IC 至少不能明显为负（-0.02 容忍噪声）
IC_MIN_PROMOTE    = 0.0      # 晋升门禁：有效 IC 必须严格为正
ICIR_MIN          = -0.5     # 多期 ICIR 底线
DIRECTION_REQUIRED = 1       # 生产冠军模型必须 direction=+1
LABEL_COL         = "forward_return_63d"
MIN_STOCKS        = 30


class PromotionBlockedError(RuntimeError):
    """晋升被门禁拦截：challenger 不符合生产要求."""


# ─────────────────────────────────────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(path: Path) -> LGBMRankerModel:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, LGBMRankerModel):
        return obj
    # 兼容只存了 lgb.Booster 的旧格式
    m = LGBMRankerModel()
    m._model = obj
    return m


def evaluate_model(
    model: LGBMRankerModel,
    panel: pd.DataFrame,
    test_dates: list[pd.Timestamp],
    feature_cols: list[str],
    direction: int = 1,
) -> dict:
    """在指定测试期评估模型，返回 IC、ICIR、逐期明细."""
    ic_map: dict[str, float] = {}

    for d in test_dates:
        if d not in panel.index.get_level_values("as_of"):
            continue
        xs = panel.xs(d, level="as_of")
        sub = xs[feature_cols + [LABEL_COL]].dropna(subset=[LABEL_COL])
        if len(sub) < MIN_STOCKS:
            continue

        X = sub[feature_cols].fillna(0.0).values.astype(np.float32)
        y = sub[LABEL_COL].values.astype(np.float64)

        try:
            raw = model._model.predict(X)
        except Exception:
            continue
        pred = raw * direction
        rho, _ = stats.spearmanr(pred, y)
        ic_map[str(d.date())] = float(rho) if np.isfinite(rho) else float("nan")

    valid = [v for v in ic_map.values() if np.isfinite(v)]
    ic_mean = float(np.mean(valid)) if valid else float("nan")
    icir    = float(ic_mean / np.std(valid, ddof=1)) if len(valid) > 1 and np.std(valid, ddof=1) > 1e-9 else float("nan")
    win_rate = float(np.mean([v > 0 for v in valid])) if valid else float("nan")

    return {
        "ic_mean":  round(ic_mean, 6)  if np.isfinite(ic_mean) else None,
        "icir":     round(icir, 4)     if np.isfinite(icir) else None,
        "win_rate": round(win_rate, 4) if np.isfinite(win_rate) else None,
        "n_periods": len(valid),
        "ic_per_period": ic_map,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 决策逻辑
# ─────────────────────────────────────────────────────────────────────────────

def decide(chall: dict, champ: dict) -> tuple[str, list[str]]:
    """返回 (decision, reasons)."""
    reasons: list[str] = []
    criteria: list[bool] = []

    # 1. IC 提升
    c_ic = chall.get("ic_mean")
    p_ic = champ.get("ic_mean")
    if c_ic is None:
        return "DISCARD", ["challenger IC 无法计算"]
    if p_ic is None:
        # champion 无法评估 → challenger 直接升级
        criteria.append(True)
        reasons.append(f"champion 无有效 IC，challenger IC={c_ic:.4f} 直接升级")
    else:
        diff = c_ic - p_ic
        ok = diff >= IC_MARGIN
        criteria.append(ok)
        reasons.append(
            f"IC 提升 {diff:+.4f}（challenger={c_ic:.4f} vs champion={p_ic:.4f}）"
            + ("  ✅" if ok else f"  ❌（需 >= {IC_MARGIN}）")
        )

    # 2. 正向 IC
    ok = c_ic >= IC_MIN
    criteria.append(ok)
    reasons.append(
        f"challenger IC={c_ic:.4f} >= {IC_MIN}"
        + ("  ✅" if ok else "  ❌（负 IC 禁止升级）")
    )

    # 3. 稳定性
    c_icir = chall.get("icir")
    if c_icir is not None:
        ok = c_icir >= ICIR_MIN
        criteria.append(ok)
        reasons.append(
            f"challenger ICIR={c_icir:.3f} >= {ICIR_MIN}"
            + ("  ✅" if ok else "  ❌")
        )

    n_pass = sum(criteria)
    n_total = len(criteria)

    if n_pass == n_total:
        return "PROMOTE", reasons
    if n_pass >= n_total - 1:
        return "WATCHLIST", reasons
    return "DISCARD", reasons


# ─────────────────────────────────────────────────────────────────────────────
# 升级动作
# ─────────────────────────────────────────────────────────────────────────────

def promote(
    challenger_path: Path,
    champion_path: Path,
    challenger_tag: str,
    chall_eval: dict,
    champ_eval: dict,
    feature_cols: list[str],
    direction: int,
    dry_run: bool = False,
) -> dict:
    """执行模型升级：备份旧 champion，将 challenger 写入生产路径.

    晋升门禁（以下任一不满足则抛出 PromotionBlockedError 并写入 lineage）：
      1. challenger pkl 中 model.direction == DIRECTION_REQUIRED（+1）
      2. 有效 IC 均值 > IC_MIN_PROMOTE（0.0）
    """
    lineage_path = champion_path.parent / "model_lineage.json"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 门禁校验 ─────────────────────────────────────────────────────────────
    block_reasons: list[str] = []
    model_direction: int | None = None
    try:
        with open(challenger_path, "rb") as _f:
            _m = pickle.load(_f)
        model_direction = getattr(_m, "direction", None)
        if model_direction != DIRECTION_REQUIRED:
            block_reasons.append(
                f"model.direction={model_direction} (要求 {DIRECTION_REQUIRED})"
            )
    except Exception as _e:
        block_reasons.append(f"无法加载 challenger pkl: {_e}")

    challenger_ic = chall_eval.get("ic_mean")
    if challenger_ic is None or challenger_ic <= IC_MIN_PROMOTE:
        block_reasons.append(
            f"有效 ic_mean={challenger_ic} <= {IC_MIN_PROMOTE}（必须严格为正）"
        )

    if block_reasons:
        reason_str = "; ".join(block_reasons)
        logger.warning(f"  ⛔ [晋升门禁] 拒绝 {challenger_tag}: {reason_str}")
        blocked_record = {
            "ts":                  ts,
            "action":              "PROMOTE_BLOCKED",
            "challenger_tag":      challenger_tag,
            "challenger_path":     str(challenger_path),
            "challenger_ic":       challenger_ic,
            "challenger_direction": model_direction,
            "reason":              reason_str,
        }
        if not dry_run:
            _lineage: list[dict] = []
            if lineage_path.exists():
                _lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            _lineage.append(blocked_record)
            lineage_path.write_text(
                json.dumps(_lineage, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        raise PromotionBlockedError(reason_str)

    # ── 门禁通过，构造晋升记录 ────────────────────────────────────────────────
    archive_dir = champion_path.parent / "archive"
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / f"lgbm_{ts}_{champion_path.stem}.pkl"

    record = {
        "ts":                ts,
        "action":            "PROMOTE",
        "challenger_tag":    challenger_tag,
        "challenger_path":   str(challenger_path),
        "champion_path":     str(champion_path),
        "archive_path":      str(archive_path),
        "challenger_ic":     challenger_ic,
        "champion_ic":       champ_eval.get("ic_mean"),
        "ic_delta":          (
            round(challenger_ic - (champ_eval.get("ic_mean") or 0), 6)
            if challenger_ic is not None else None
        ),
        "direction":         direction,
        "feature_cols":      feature_cols,
    }

    # ── 文件写入（持文件锁，防并发竞态）─────────────────────────────────────
    lock_path = champion_path.with_suffix(".lock")
    with FileLock(str(lock_path), timeout=30):
        lineage: list[dict] = []
        if lineage_path.exists():
            lineage = json.loads(lineage_path.read_text(encoding="utf-8"))

        if not dry_run:
            shutil.copy2(champion_path, archive_path)
            shutil.copy2(challenger_path, champion_path)
            meta_path = champion_path.with_suffix(".meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            meta.update({
                "promoted_at":    ts,
                "challenger_tag": challenger_tag,
                "ic_mean":        challenger_ic,
                "ic_ir":          chall_eval.get("icir"),
                "direction":      direction,
                "feature_cols":   feature_cols,
            })
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            lineage.append(record)
            lineage_path.write_text(
                json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(f"  ✅ 已备份旧 champion → {archive_path}")
            logger.info(f"  ✅ 已写入新 champion → {champion_path}")
        else:
            logger.info(f"  [DRY-RUN] 将备份 → {archive_path}")
            logger.info(f"  [DRY-RUN] 将写入 → {champion_path}")

    return record


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run(
    panel: pd.DataFrame,
    feature_cols: list[str],
    champion_path: Path,
    challenger_path: Path,
    challenger_tag: str,
    test_end: str | None,
    out_dir: Path,
    champion_direction: int = -1,
    challenger_direction: int = -1,
    dry_run: bool = False,
    challenger_feature_cols: list[str] | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 确定测试期 ─────────────────────────────────────────────────────────────
    all_dates = sorted(panel.index.get_level_values("as_of").unique())
    if test_end:
        cutoff = pd.Timestamp(test_end)
        # 取训练截止（challenger 的 train_end = Round 7 = 2025Q3）后的所有有标签的期
        # 这里用 challenger round_result 获取 test_dates
        round_result_path = challenger_path.parent / "round_result.json"
        if round_result_path.exists():
            rr = json.loads(round_result_path.read_text(encoding="utf-8"))
            test_dates = [pd.Timestamp(d) for d in rr.get("test_dates", [])]
            if not test_dates:
                test_dates = [d for d in all_dates if d <= cutoff][-4:]
        else:
            test_dates = [d for d in all_dates if d <= cutoff][-4:]
    else:
        # 默认：最近 4 期有标签的日期
        valid_dates = []
        for d in all_dates:
            xs = panel.xs(d, level="as_of")
            if xs[LABEL_COL].notna().sum() >= MIN_STOCKS:
                valid_dates.append(d)
        test_dates = valid_dates[-4:]

    if not test_dates:
        logger.error("无有效测试期")
        return {}

    logger.info(f"测试期: {[str(d.date()) for d in test_dates]}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    logger.info(f"加载 champion: {champion_path}")
    champion = _load_model(champion_path)

    logger.info(f"加载 challenger ({challenger_tag}): {challenger_path}")
    challenger = _load_model(challenger_path)

    # ── 评估 ──────────────────────────────────────────────────────────────────
    chall_fcols = challenger_feature_cols if challenger_feature_cols is not None else feature_cols

    logger.info("评估 champion...")
    champ_eval = evaluate_model(champion, panel, test_dates, feature_cols, champion_direction)

    if challenger_feature_cols is not None:
        logger.info(f"评估 challenger（独立特征集，{len(chall_fcols)} 个特征）...")
    else:
        logger.info("评估 challenger...")
    chall_eval = evaluate_model(challenger, panel, test_dates, chall_fcols, challenger_direction)

    # ── 决策 ──────────────────────────────────────────────────────────────────
    decision, reasons = decide(chall_eval, champ_eval)

    print("\n" + "═" * 65)
    print(f"  Champion-Challenger 评估结果")
    print("─" * 65)
    print(f"  {'指标':<20} {'Champion':<15} {'Challenger':<15}")
    print(f"  {'IC_mean':<20} {str(champ_eval.get('ic_mean') or 'N/A'):<15} {str(chall_eval.get('ic_mean') or 'N/A'):<15}")
    print(f"  {'ICIR':<20} {str(champ_eval.get('icir') or 'N/A'):<15} {str(chall_eval.get('icir') or 'N/A'):<15}")
    print(f"  {'IC Win Rate':<20} {str(champ_eval.get('win_rate') or 'N/A'):<15} {str(chall_eval.get('win_rate') or 'N/A'):<15}")
    print(f"  {'测试期数':<20} {champ_eval['n_periods']:<15} {chall_eval['n_periods']:<15}")
    print("─" * 65)
    print(f"  评估标准:")
    for r in reasons:
        print(f"    {r}")
    print("─" * 65)
    decision_icon = {"PROMOTE": "✅ PROMOTE", "WATCHLIST": "⚠️  WATCHLIST", "DISCARD": "❌ DISCARD"}[decision]
    print(f"  决策: {decision_icon}")
    print("═" * 65)

    # ── 执行动作 ──────────────────────────────────────────────────────────────
    promote_record = None
    if decision == "PROMOTE":
        try:
            promote_record = promote(
                challenger_path=challenger_path,
                champion_path=champion_path,
                challenger_tag=challenger_tag,
                chall_eval=chall_eval,
                champ_eval=champ_eval,
                feature_cols=feature_cols,
                direction=challenger_direction,
                dry_run=dry_run,
            )
        except PromotionBlockedError as _e:
            logger.warning(f"  ⛔ 晋升被门禁拦截，实际决策降为 DISCARD: {_e}")
            decision = "DISCARD"

    # ── 保存评估结果 ──────────────────────────────────────────────────────────
    ts = datetime.now().isoformat(timespec="seconds")
    eval_record = {
        "ts":              ts,
        "challenger_tag":  challenger_tag,
        "test_dates":      [str(d.date()) for d in test_dates],
        "champion_eval":   champ_eval,
        "challenger_eval": chall_eval,
        "decision":        decision,
        "reasons":         reasons,
        "dry_run":         dry_run,
        "promoted":        promote_record is not None,
    }
    (out_dir / "latest_eval.json").write_text(
        json.dumps(eval_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history_path = out_dir / "decision_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": ts, "challenger_tag": challenger_tag,
            "champion_ic": champ_eval.get("ic_mean"),
            "challenger_ic": chall_eval.get("ic_mean"),
            "decision": decision,
        }, ensure_ascii=False) + "\n")

    logger.info(f"评估结果已保存：{out_dir}")
    return eval_record


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel",       type=Path, default=Path("data/panel/alpha_panel_v4.parquet"))
    p.add_argument("--features",    type=Path, default=Path("data/features/top_factors_v3_63d_flat.json"))
    p.add_argument("--challenger-features", type=Path, default=None,
                   help="challenger 独立特征文件（不同于 champion 特征集时使用）")
    p.add_argument("--champion",    type=Path, default=Path("models/lgbm_v6_alpha.pkl"))
    p.add_argument("--challenger",  type=Path, default=Path("reports/sequential_val/round_7/model.pkl"))
    p.add_argument("--challenger-tag", default="round7_2025q3")
    p.add_argument("--champion-direction",   type=int, default=-1)
    p.add_argument("--challenger-direction", type=int, default=-1)
    p.add_argument("--test-end",    default=None, help="测试截止日期，默认取最近 4 期")
    p.add_argument("--out",         type=Path, default=Path("reports/champion_challenger/"))
    p.add_argument("--dry-run",     action="store_true", help="不实际覆盖文件")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 加载面板
    logger.info(f"加载面板：{args.panel}")
    panel = pd.read_parquet(args.panel)

    # 加载特征列
    raw = json.loads(args.features.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        feature_cols = raw
    elif isinstance(raw, dict):
        for key in ("feature_cols", "selected_factors", "factors", "features"):
            if key in raw and isinstance(raw[key], list):
                feature_cols = raw[key]
                break
        else:
            raise SystemExit(f"无法解析特征文件：{args.features}")
    feature_cols = [c for c in feature_cols if c in panel.columns]

    # 独立 challenger 特征集（可选）
    challenger_feature_cols: list[str] | None = None
    if args.challenger_features and args.challenger_features.exists():
        raw_c = json.loads(args.challenger_features.read_text(encoding="utf-8"))
        if isinstance(raw_c, list):
            challenger_feature_cols = raw_c
        elif isinstance(raw_c, dict):
            for key in ("feature_cols", "selected_factors", "factors", "features"):
                if key in raw_c and isinstance(raw_c[key], list):
                    challenger_feature_cols = raw_c[key]
                    break
        if challenger_feature_cols:
            challenger_feature_cols = [c for c in challenger_feature_cols if c in panel.columns]
            logger.info(f"Challenger 独立特征集: {len(challenger_feature_cols)} 个特征")

    run(
        panel=panel,
        feature_cols=feature_cols,
        champion_path=args.champion,
        challenger_path=args.challenger,
        challenger_tag=args.challenger_tag,
        test_end=args.test_end,
        out_dir=args.out,
        champion_direction=args.champion_direction,
        challenger_direction=args.challenger_direction,
        dry_run=args.dry_run,
        challenger_feature_cols=challenger_feature_cols,
    )


if __name__ == "__main__":
    main()
