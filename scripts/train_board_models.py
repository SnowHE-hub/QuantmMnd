"""scripts/train_board_models.py — 按板块分别训练 LGBM 专用模型.

背景
----
混训模型（lgbm_v6_alpha）把主板/创业板/科创板混在一起训练：
- 科创板 PE 中位数 ~60，主板 ~20；模型会错误惩罚科创板高估值
- 小盘股 reversal_1w IC 比大盘股强 2-3 倍，方向和幅度都有差异
本脚本为三个子宇宙各训练一个专用 LGBM 模型。

Usage
-----
python scripts/train_board_models.py \\
    --panel data/panel/alpha_panel_v4.parquet \\
    --out-dir models/ \\
    --n-folds 8

python scripts/train_board_models.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantmind.models.lgbm_ranker import LGBMRankerModel, walk_forward_evaluate
from quantmind.models.factor_model import FactorModel

# ── 板块识别 ──────────────────────────────────────────────────────────────────

def get_board(ticker: str) -> str:
    """按证券代码判断 A 股板块。

    规则：
    - 688xxx.SH → STAR  （科创板）
    - 300xxx.SZ → GEM   （创业板）
    - 其余 .SH/.SZ → MAIN（主板，含北交所合并）
    """
    code = ticker.split(".")[0]
    if ticker.endswith(".SH") and code.startswith("688"):
        return "STAR"
    if ticker.endswith(".SZ") and code.startswith("300"):
        return "GEM"
    return "MAIN"


def filter_panel_by_board(
    panel: pd.DataFrame,
    board: str,
    ticker_level: str = "ticker",
) -> pd.DataFrame:
    """过滤 panel 只保留指定板块的股票."""
    all_tickers = panel.index.get_level_values(ticker_level)
    mask = pd.Series(all_tickers, index=panel.index).apply(get_board) == board
    return panel.loc[mask]


# ── 板块配置 ──────────────────────────────────────────────────────────────────

BOARD_CONFIGS: dict[str, dict[str, Any]] = {
    "main": {
        "board_code":   "MAIN",
        "display_name": "主板（沪深主板+北交所）",
        "model_path":   "models/lgbm_v6_main.pkl",
        "lgbm_params": {
            # 主板：样本最多，可用更复杂模型
            "num_leaves":      63,
            "min_child_samples": 50,
            "lambda_l1":       0.05,
            "lambda_l2":       0.05,
        },
        "min_ticker_threshold": 50,
    },
    "gem": {
        "board_code":   "GEM",
        "display_name": "创业板",
        "model_path":   "models/lgbm_v6_gem.pkl",
        "lgbm_params": {
            # 创业板：成长股，降低叶子数防过拟合
            "num_leaves":      31,
            "min_child_samples": 30,
            "lambda_l1":       0.1,
            "lambda_l2":       0.1,
        },
        "min_ticker_threshold": 50,
    },
    "star": {
        "board_code":   "STAR",
        "display_name": "科创板",
        "model_path":   "models/lgbm_v6_star.pkl",
        "lgbm_params": {
            # 科创板：样本最少，强正则化防过拟合
            "num_leaves":      15,
            "min_child_samples": 20,
            "lambda_l1":       0.2,
            "lambda_l2":       0.5,
        },
        "min_ticker_threshold": 30,
    },
}


# ── IC 评估工具 ───────────────────────────────────────────────────────────────

def evaluate_model_ic(
    model: LGBMRankerModel,
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    min_train_periods: int = 8,
    n_val_periods: int = 2,
) -> float:
    """计算已训练模型在 panel 上的 Walk-Forward 平均 Rank IC。

    使用相同的折叠切分，对每个测试截面计算 Spearman IC，返回均值。
    注意：这里不重新训练，只在测试截面上评估传入的模型。
    """
    from quantmind.models.factor_model import WalkForwardSplit

    dates = sorted(panel.index.get_level_values("as_of").unique())
    if len(dates) < min_train_periods + 1:
        return float("nan")

    splitter = WalkForwardSplit(
        min_train_periods=min_train_periods,
        n_val_periods=n_val_periods,
    )
    try:
        folds = splitter.split(dates)
    except ValueError:
        return float("nan")

    ics: list[float] = []
    for fold in folds:
        td = fold.test_date
        if td not in panel.index.get_level_values("as_of"):
            continue
        xs = panel.xs(td, level="as_of")
        valid = xs[[*feature_cols, label_col]].dropna(subset=[label_col])
        if len(valid) < 5:
            continue
        X = valid[feature_cols].fillna(0.0).values.astype(np.float32)
        y = valid[label_col].values
        try:
            pred = model.predict(X)
            rho, _ = spearmanr(pred, y)
            if not np.isnan(rho):
                ics.append(float(rho))
        except Exception:
            continue

    return float(np.mean(ics)) if ics else float("nan")


# ── 主训练逻辑 ────────────────────────────────────────────────────────────────

def train_one_board(
    board_name: str,
    config: dict[str, Any],
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    out_dir: Path,
    min_train_periods: int = 8,
    mixed_model: LGBMRankerModel | None = None,
) -> dict[str, Any] | None:
    """训练单个板块模型，返回评估结果 dict。"""
    board_code   = config["board_code"]
    display_name = config["display_name"]
    model_path   = out_dir / Path(config["model_path"]).name
    min_tickers  = config["min_ticker_threshold"]

    print(f"\n{'='*60}")
    print(f"  训练 {display_name} ({board_code}) 专用模型")
    print(f"{'='*60}")

    # 过滤板块数据
    board_panel = filter_panel_by_board(panel, board_code)
    n_tickers   = board_panel.index.get_level_values("ticker").nunique()
    n_rows      = len(board_panel)
    print(f"  股票数: {n_tickers}  样本行: {n_rows}")

    if n_tickers < min_tickers:
        print(f"  ⚠️  样本不足（< {min_tickers} 只），跳过训练")
        return None

    # Walk-Forward 训练
    lgbm_params = {
        "n_estimators":         300,
        "learning_rate":        0.05,
        "feature_fraction":     0.8,
        "bagging_fraction":     0.8,
        "bagging_freq":         5,
        "early_stopping_rounds": 80,
        "auto_flip":            True,
        "verbose":              -1,
        **config["lgbm_params"],
    }

    t0 = time.time()
    try:
        result, last_model = walk_forward_evaluate(
            panel              = board_panel,
            feature_cols       = feature_cols,
            label_col          = label_col,
            min_train_periods  = min_train_periods,
            n_val_periods      = 2,
            model_kwargs       = lgbm_params,
            auto_flip          = True,
            verbose            = True,
        )
    except Exception as exc:
        print(f"  ❌ 训练失败: {exc}")
        return None

    elapsed = time.time() - t0
    board_ic = result.effective_ic_mean

    # 评估混训模型在该板块上的 IC（作为基准）
    mixed_ic = float("nan")
    if mixed_model is not None:
        mixed_ic = evaluate_model_ic(
            model             = mixed_model,
            panel             = board_panel,
            feature_cols      = feature_cols,
            label_col         = label_col,
            min_train_periods = min_train_periods,
        )

    # ── direction + ic_mean 双重质量校验 ─────────────────────────────────────
    direction      = last_model.direction
    ic_mean_val    = getattr(last_model, "ic_mean", None)
    direction_warn = direction != 1
    ic_warn        = (ic_mean_val is not None) and (ic_mean_val <= 0)
    quality_fail   = direction_warn or ic_warn

    if direction_warn:
        print(
            f"\n  ⚠️  [门禁①] {display_name} direction={direction}（raw IC 为负，"
            f"auto_flip 触发）。\n"
            f"     保存到 debug 路径，不覆盖生产模型。\n"
            f"     BoardModelRouter 会在运行时自动降级到 lgbm_v6_alpha。"
        )
    elif ic_warn:
        print(
            f"\n  ⚠️  [门禁②] {display_name} ic_mean={ic_mean_val:.4f} ≤ 0，"
            f"模型无有效预测力。\n"
            f"     保存到 debug 路径，不覆盖生产模型。"
        )
    else:
        print(f"\n  ✅ direction=+1, ic_mean={ic_mean_val:.4f if ic_mean_val else 'N/A'}，"
              f"通过双重质量门禁，部署到生产路径。")

    # 保存逻辑：只有通过质量门禁的模型才写入生产路径
    if quality_fail:
        # 保存到 debug 路径供离线分析，不覆盖生产模型
        debug_path = model_path.parent / f"{model_path.stem}_debug{model_path.suffix}"
        last_model.save(debug_path)
        print(f"  📦 debug 模型保存 → {debug_path}（耗时 {elapsed:.0f}s）")
        print(f"  ⏭  跳过生产路径 {model_path.name}（质量门禁未通过）")
    else:
        last_model.save(model_path)
        print(f"  ✅ 已保存 → {model_path}（耗时 {elapsed:.0f}s）")

    improvement = board_ic - mixed_ic if not np.isnan(mixed_ic) else float("nan")
    print(f"\n  IC 对比：混训={mixed_ic:+.4f}  专用={board_ic:+.4f}  "
          f"改善={improvement:+.4f}" if not np.isnan(mixed_ic)
          else f"\n  专用模型 IC={board_ic:+.4f}（混训 IC 不可用）")

    return {
        "board":             board_name,
        "display":           display_name,
        "n_tickers":         n_tickers,
        "n_rows":            n_rows,
        "mixed_ic":          round(mixed_ic, 5) if not np.isnan(mixed_ic) else None,
        "board_ic":          round(board_ic, 5),
        "improvement":       round(improvement, 5) if not np.isnan(improvement) else None,
        "ic_std":            round(result.ic_std, 5),
        "ic_ir":             round(result.effective_ic_ir, 5) if not np.isnan(result.effective_ic_ir) else None,
        "ic_win_rate":       round(result.ic_win_rate, 3),
        "n_folds":           result.n_folds,
        "direction":         direction,
        "ic_mean":           round(ic_mean_val, 5) if ic_mean_val is not None else None,
        "direction_warning": direction_warn,
        "ic_warning":        ic_warn,
        "production_ready":  not quality_fail,
        "model_path":        str(model_path),
        "elapsed_s":         round(elapsed, 1),
    }


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[train_board_models] 加载 alpha_panel_v4…")
    panel = pd.read_parquet(args.panel)
    print(f"  Panel shape: {panel.shape}, dates: {panel.index.get_level_values('as_of').nunique()}")

    # 加载 lgbm_v6_alpha 作为混训基准
    mixed_model: LGBMRankerModel | None = None
    fallback_path = ROOT / "models" / "lgbm_v6_alpha.pkl"
    if fallback_path.exists():
        try:
            mixed_model = FactorModel.load(fallback_path)
            print(f"  混训基准模型: {fallback_path.name}  direction={mixed_model.direction}")
        except Exception as exc:
            print(f"  ⚠️  无法加载混训模型: {exc}")

    # 确定特征列（从 v6 模型或 meta 中取）
    feature_cols = args.feature_cols
    if not feature_cols:
        if mixed_model is not None and mixed_model._feature_names:
            feature_cols = list(mixed_model._feature_names)
        else:
            meta_path = ROOT / "models" / "lgbm_v6_alpha.meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                feature_cols = meta.get("feature_cols", [])
    if not feature_cols:
        raise SystemExit("❌ 无法确定特征列，请通过 --feature-cols 指定或确保 lgbm_v6_alpha 存在")

    # 过滤 panel 中实际存在的特征
    feature_cols = [c for c in feature_cols if c in panel.columns]
    label_col = args.label
    if label_col not in panel.columns:
        raise SystemExit(f"❌ 标签列 {label_col!r} 不在 panel 中。可用: {[c for c in panel.columns if 'return' in c or 'label' in c]}")

    print(f"  Features: {len(feature_cols)}, Label: {label_col}")
    print(f"  Min train periods: {args.min_train_periods}")

    # 打印板块分布
    all_tickers = panel.index.get_level_values("ticker").unique()
    print("\n板块分布:")
    for board in ["MAIN", "GEM", "STAR"]:
        cnt = sum(1 for t in all_tickers if get_board(t) == board)
        print(f"  {board}: {cnt} 只")

    # 逐板块训练
    results: list[dict] = []
    boards_to_train = [b for b in BOARD_CONFIGS if b not in (args.skip or [])]

    for board_name in boards_to_train:
        config = BOARD_CONFIGS[board_name]
        row = train_one_board(
            board_name        = board_name,
            config            = config,
            panel             = panel,
            feature_cols      = feature_cols,
            label_col         = label_col,
            out_dir           = out_dir,
            min_train_periods = args.min_train_periods,
            mixed_model       = mixed_model,
        )
        if row is not None:
            results.append(row)

    # 打印对比表
    if results:
        print("\n" + "="*70)
        print("  板块专用模型 vs 混训模型 IC 对比表")
        print("="*70)
        header = f"{'板块':<10}{'股票数':>7}{'样本数':>8}{'混训IC':>10}{'专用IC':>10}{'改善':>10}{'IC_IR':>8}{'生产':>6}"
        print(header)
        print("-"*78)
        for r in results:
            mixed_str  = f"{r['mixed_ic']:+.4f}" if r["mixed_ic"] is not None else "   N/A"
            board_str  = f"{r['board_ic']:+.4f}"
            impr_str   = f"{r['improvement']:+.4f}" if r["improvement"] is not None else "   N/A"
            ir_str     = f"{r['ic_ir']:+.4f}" if r["ic_ir"] is not None else "   N/A"
            flag       = "🔺" if (r["improvement"] or 0) > 0 else "🔻"
            prod_flag  = "✅" if r.get("production_ready") else "⚠️ fallback"
            print(f"  {r['display']:<10}{r['n_tickers']:>6}{r['n_rows']:>8}  "
                  f"{mixed_str:>9}  {board_str:>9}  {flag}{impr_str:>8}  {ir_str:>7}  {prod_flag}")
        print("="*78)
        # 如有 direction 警告，汇总提示
        warned = [r["display"] for r in results if r.get("direction_warning")]
        if warned:
            print(f"\n  ⚠️  以下板块 direction=-1，BoardModelRouter 推理时将自动降级 fallback：")
            for d in warned:
                print(f"     · {d}")

        # 保存结果 JSON
        result_path = out_dir / "board_model_results.json"
        result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  结果已保存 → {result_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按主板/创业板/科创板三子宇宙分别训练 LGBM 排序模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--panel",    default="data/panel/alpha_panel_v4.parquet",
                   help="alpha panel parquet 路径")
    p.add_argument("--out-dir",  default="models/",
                   help="模型输出目录")
    p.add_argument("--label",    default="forward_return_63d",
                   help="目标标签列名")
    p.add_argument("--feature-cols", nargs="*", default=[],
                   help="特征列列表（默认从 lgbm_v6_alpha 读取）")
    p.add_argument("--min-train-periods", type=int, default=8,
                   help="Walk-Forward 最少训练截面数（越小折叠越多）")
    p.add_argument("--skip",     nargs="*", default=[],
                   help="跳过的板块，如 --skip star")
    return p.parse_args()


if __name__ == "__main__":
    main()
