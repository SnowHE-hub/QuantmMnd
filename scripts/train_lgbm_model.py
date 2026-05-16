#!/usr/bin/env python3
"""在 train/val/test panel 上训练 LGBMRanker（LambdaRank）并输出报告。

使用 ``LGBMRankerModel.fit()``（与 ``train_factor_model`` 的 Walk-Forward 流程不同）。

特征集：
  - all：split_meta 中数值列（排除 exposure_industry / exposure_area）
  - base：list_age_years 之前的数值列（同 meta 顺序）
  - expansion：list_age_years 及之后的数值列
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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

EXCLUDE_STRING = frozenset({"exposure_industry", "exposure_area"})
DEFAULT_META = Path("data/panel/split_meta.json")
MIN_CS = 8


def load_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_list(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0])
    return out


def meta_numeric_features(panel: pd.DataFrame, meta: dict[str, Any]) -> list[str]:
    cols = meta.get("feature_columns") or []
    return [
        c
        for c in cols
        if c not in EXCLUDE_STRING
        and c in panel.columns
        and pd.api.types.is_numeric_dtype(panel[c])
    ]


def resolve_feature_columns(
    panel: pd.DataFrame,
    meta: dict[str, Any],
    feature_set: str,
) -> list[str]:
    meta_order = [c for c in meta["feature_columns"] if c not in EXCLUDE_STRING]
    numeric_in_panel = [c for c in meta_order if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]
    if feature_set == "all":
        return numeric_in_panel
    try:
        i = meta["feature_columns"].index("list_age_years")
    except ValueError:
        raise SystemExit("split_meta missing list_age_years anchor for base/expansion split")
    base_names = [c for c in meta["feature_columns"][:i] if c not in EXCLUDE_STRING]
    exp_names = [c for c in meta["feature_columns"][i:] if c not in EXCLUDE_STRING]
    if feature_set == "base":
        return [c for c in base_names if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]
    if feature_set == "expansion":
        return [c for c in exp_names if c in panel.columns and pd.api.types.is_numeric_dtype(panel[c])]
    raise SystemExit(f"unknown --feature-set {feature_set}")


def filter_panel_from_date(panel: pd.DataFrame, start: str | None) -> pd.DataFrame:
    if not start:
        return panel
    d = pd.Timestamp(start)
    idx = panel.index.get_level_values("as_of")
    return panel.loc[idx >= d]


def rank_ic_one_date_raw_boost(
    model: LGBMRankerModel,
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_cols: list[str],
    label_col: str,
) -> float:
    """与 rank_ic_one_date 相同，但使用 booster 原始输出（乘 direction 之前）。"""
    if as_of not in panel.index.get_level_values("as_of"):
        return float("nan")
    xs = panel.xs(as_of, level="as_of")
    sub = xs[[*feature_cols, label_col]].dropna(subset=[label_col])
    if len(sub) < MIN_CS:
        return float("nan")
    X = sub[feature_cols].fillna(0.0).values.astype(np.float32)
    y = sub[label_col].values.astype(np.float64)
    if model._model is None:
        return float("nan")
    pred = model._model.predict(X)
    rho, _ = stats.spearmanr(pred, y)
    return float(rho) if rho == rho else float("nan")


def rank_ic_one_date(
    model: LGBMRankerModel,
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    feature_cols: list[str],
    label_col: str,
) -> float:
    if as_of not in panel.index.get_level_values("as_of"):
        return float("nan")
    xs = panel.xs(as_of, level="as_of")
    sub = xs[[*feature_cols, label_col]].dropna(subset=[label_col])
    if len(sub) < MIN_CS:
        return float("nan")
    X = sub[feature_cols].fillna(0.0).values.astype(np.float32)
    y = sub[label_col].values.astype(np.float64)
    pred = model.predict(X)
    rho, _ = stats.spearmanr(pred, y)
    return float(rho) if rho == rho else float("nan")


def summarize_ics(ics: list[float]) -> tuple[float, float, float]:
    arr = np.array([x for x in ics if x == x], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")
    ir = m / s if s == s and s > 1e-12 else float("nan")
    win = float(np.mean([x > 0 for x in arr]))
    return m, ir, win


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=Path("data/panel/train.parquet"))
    p.add_argument("--val", type=Path, default=Path("data/panel/val.parquet"))
    p.add_argument("--test", type=Path, default=Path("data/panel/test.parquet"))
    p.add_argument("--label", default="forward_return_21d")
    p.add_argument("--feature-set", choices=["all", "base", "expansion"], default="all")
    p.add_argument("--filter-start", default=None, help="仅保留 train 中 as_of>=该日期")
    p.add_argument("--split-meta", type=Path, default=DEFAULT_META)
    p.add_argument(
        "--feature-list",
        type=Path,
        default=None,
        help="可选：每行一个因子名；与 train 列求交集后作为特征（优先级高于 --feature-set）",
    )
    p.add_argument("--model-output", type=Path, default=Path("models/lgbm_ranker_v1.pkl"))
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--output-dir", type=Path, default=Path("reports/model_training"))
    p.add_argument("--n-quantiles", type=int, default=5)
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="写入 IC / best_iter 等，供 model_comparison 汇总",
    )
    p.add_argument("--run-tag", default="", help="输出文件后缀（默认用 model-output 的 stem）")
    return p.parse_args(argv)


def run_training(args: argparse.Namespace) -> None:
    run_tag = (args.run_tag or args.model_output.stem).replace(" ", "_")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {}
    if args.split_meta.is_file():
        meta = load_meta(args.split_meta)

    train_p = pd.read_parquet(args.train)
    val_p = pd.read_parquet(args.val)
    test_p = pd.read_parquet(args.test)

    train_p = filter_panel_from_date(train_p, args.filter_start)

    if args.feature_list is not None and args.feature_list.is_file():
        raw = load_feature_list(args.feature_list)
        feature_cols = [
            c
            for c in raw
            if c in train_p.columns and pd.api.types.is_numeric_dtype(train_p[c])
        ]
        if not feature_cols:
            raise SystemExit("feature-list ∩ train numeric columns is empty")
    elif meta.get("feature_columns"):
        feature_cols = meta_numeric_features(train_p, meta)
        if not feature_cols:
            raise SystemExit("split_meta.feature_columns has no usable numeric columns in train")
    elif meta:
        feature_cols = resolve_feature_columns(train_p, meta, args.feature_set)
    else:
        raise SystemExit(f"split_meta not found: {args.split_meta} (and no --feature-list)")
    if not feature_cols:
        raise SystemExit("no feature columns resolved")

    train_dates = sorted(train_p.index.get_level_values("as_of").unique())
    val_dates = sorted(val_p.index.get_level_values("as_of").unique())
    test_dates = sorted(test_p.index.get_level_values("as_of").unique())

    print(f"train rows={len(train_p)} periods={len(train_dates)}  val periods={len(val_dates)}  test periods={len(test_dates)}")
    print(f"feature_set={args.feature_set}  n_features={len(feature_cols)}")

    X_tr, y_tr, g_tr = build_lgbm_arrays(
        train_p, train_dates, feature_cols, args.label, args.n_quantiles
    )
    X_va, y_va, g_va = build_lgbm_arrays(
        val_p, val_dates, feature_cols, args.label, args.n_quantiles
    )
    if len(X_tr) == 0 or len(X_va) == 0:
        raise SystemExit("empty train or val arrays — check dates / label / features")

    model = LGBMRankerModel(
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        feature_fraction=args.colsample_bytree,
        bagging_fraction=args.bagging_fraction,
        min_child_samples=args.min_child_samples,
        early_stopping_rounds=args.early_stopping_rounds,
        auto_flip=True,
        verbose=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        groups=g_tr,
        X_val=X_va,
        y_val=y_va,
        groups_val=g_va,
        feature_names=feature_cols,
    )
    best_iter = int(model.best_iteration_)

    val_ics_raw: list[float] = []
    print("\n=== Validation Rank IC（每期，LightGBM 原始打分） ===")
    for d in val_dates:
        ic = rank_ic_one_date_raw_boost(model, val_p, d, feature_cols, args.label)
        val_ics_raw.append(ic)
        print(f"  val as_of={str(d)[:10]}  Rank_IC_raw={ic:+.4f}" if ic == ic else f"  val as_of={str(d)[:10]}  Rank_IC_raw=nan")
    raw_ic_mean, _, _ = summarize_ics(val_ics_raw)
    model.set_direction(raw_ic_mean)

    val_ics: list[float] = []
    print("\n=== Validation Rank IC（每期，direction 修正后 = 实盘 predict） ===")
    for d in val_dates:
        ic = rank_ic_one_date(model, val_p, d, feature_cols, args.label)
        val_ics.append(ic)
        print(f"  val as_of={str(d)[:10]}  Rank_IC={ic:+.4f}" if ic == ic else f"  val as_of={str(d)[:10]}  Rank_IC=nan")
    val_ic_mean, val_ic_ir, val_win = summarize_ics(val_ics)
    print(f"val 汇总（有效）: IC_mean={val_ic_mean:+.4f}  ICIR={val_ic_ir:+.4f}  IC>0率={val_win:.1%}  direction={model.direction:+d}")

    test_ics: list[float] = []
    test_ic_by_date: dict[str, float] = {}
    print("\n=== Test Rank IC（每期） ===")
    for d in test_dates:
        ic = rank_ic_one_date(model, test_p, d, feature_cols, args.label)
        test_ics.append(ic)
        test_ic_by_date[str(d)[:10]] = ic
        print(f"  test as_of={str(d)[:10]}  Rank_IC={ic:+.4f}" if ic == ic else f"  test as_of={str(d)[:10]}  Rank_IC=nan")
    test_ic_mean, test_ic_ir, test_win = summarize_ics(test_ics)
    print(f"test 汇总: IC_mean={test_ic_mean:+.4f}  ICIR={test_ic_ir:+.4f}  IC>0率={test_win:.1%}")

    model._feature_names = list(feature_cols)
    model.save(args.model_output)
    print(f"\nmodel saved → {args.model_output.resolve()}")

    imp: dict[str, float] = {}
    if model._model is not None:
        gains = model._model.booster_.feature_importance(importance_type="gain")
        imp = {n: float(g) for n, g in zip(feature_cols, gains)}
    imp_sorted = sorted(imp.items(), key=lambda x: -x[1])
    fi_path = args.output_dir / f"feature_importance_{run_tag}.csv"
    pd.DataFrame(
        [{"feature": n, "gain": g} for n, g in imp_sorted],
    ).to_csv(fi_path, index=False)
    # 便于单版本查阅的固定文件名
    pd.DataFrame(
        [{"feature": n, "gain": g} for n, g in imp_sorted],
    ).to_csv(args.output_dir / "feature_importance.csv", index=False)

    hp = {
        "run_tag": run_tag,
        "num_leaves": args.num_leaves,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "min_child_samples": args.min_child_samples,
        "feature_fraction(colsample_bytree)": args.colsample_bytree,
        "bagging_fraction": args.bagging_fraction,
        "early_stopping_rounds": args.early_stopping_rounds,
        "label": args.label,
        "feature_set": args.feature_set,
        "n_features": len(feature_cols),
        "filter_start": args.filter_start,
        "best_iteration_val": best_iter,
        "train_periods": len(train_dates),
        "val_periods": len(val_dates),
        "test_periods": len(test_dates),
    }
    top20_md = "\n".join([f"| {i+1} | {n} | {g:.6f} |" for i, (n, g) in enumerate(imp_sorted[:20])])

    test_ic_lines = "\n".join([f"| {ds} | {v:+.6f} |" if v == v else f"| {ds} | nan |" for ds, v in sorted(test_ic_by_date.items())])
    md = "\n".join([
        "# LGBMRanker 训练报告",
        "",
        f"生成时间: {datetime.utcnow().isoformat()}Z",
        f"run_tag: `{run_tag}`",
        "",
        "## 超参数",
        "",
        "```json",
        json.dumps(hp, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Early stopping",
        "",
        f"- 验证集最优迭代 ``best_iteration_``: **{best_iter}**",
        "",
        "## 验证集 / 测试集 IC",
        "",
        "| 数据集 | IC_mean | ICIR | IC>0 率 | direction |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| val | {val_ic_mean:+.6f} | {val_ic_ir:+.6f} | {val_win:.2%} | {model.direction:+d} |",
        f"| test | {test_ic_mean:+.6f} | {test_ic_ir:+.6f} | {test_win:.2%} | {model.direction:+d} |",
        "",
        "## Test 各截面 Rank IC",
        "",
        "| as_of | Rank_IC |",
        "| --- | ---: |",
        test_ic_lines,
        "",
        "## Top 20 特征重要性（gain）",
        "",
        "| rank | feature | gain |",
        "| --- | --- | ---: |",
        top20_md,
        "",
        f"完整 gain: `feature_importance_{run_tag}.csv`（及 `feature_importance.csv` 末次覆盖）",
        f"模型: `{args.model_output}`",
        "",
    ])
    report_path = args.output_dir / f"train_report_{run_tag}.md"
    report_path.write_text(md, encoding="utf-8")
    (args.output_dir / "train_report.md").write_text(md, encoding="utf-8")
    print(f"report → {report_path.resolve()}")

    metrics = {
        "run_tag": run_tag,
        "feature_set": args.feature_set,
        "n_features": len(feature_cols),
        "train_periods": len(train_dates),
        "filter_start": args.filter_start,
        "best_iteration": best_iter,
        "val_IC_mean": val_ic_mean,
        "val_ICIR": val_ic_ir,
        "val_IC_positive_rate": val_win,
        "test_IC_mean": test_ic_mean,
        "test_ICIR": test_ic_ir,
        "test_IC_positive_rate": test_win,
        "direction": model.direction,
        "model_path": str(args.model_output),
        "feature_importance_csv": str(fi_path),
    }
    mj = args.metrics_json or (args.output_dir / f"metrics_{run_tag}.json")
    mj.parent.mkdir(parents=True, exist_ok=True)
    mj.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"metrics json → {mj.resolve()}")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
