"""scripts/build_train_test_split.py — 时序划分 train / val / test / predict / holdout.

默认按日历年切片（与 csi300_full_panel 季度末 as_of 一致）：
  train  : year(as_of) <= --train-max-year（默认 2022，即 2019-Q1 → 2022-Q4）
  val    : year(as_of) == --val-year（默认 2023，长度须等于 --val-periods）
  test   : year(as_of) == --test-year（默认 2024，长度须等于 --test-periods）
  holdout: year(as_of) >= --holdout-min-year（默认 2025），仅当面板中存在对应行时写出；
           若仅有快照而无特征行则跳过并记录在 meta。
predict.parquet：取 test 期内最近 --predict-periods 个截面（默认 2），用于 forward 打分场景。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantmind.core.config import PROJECT_ROOT

HOLDOUT_WARNING_ZH = (
    "【禁止在模型固化前使用 holdout 数据】\n"
    "此文件对应 2025 及以后预留截面，不得用于训练、超参搜索、因子筛选或任何形式的反复调参。"
    "仅在模型与流程完全冻结后，可用于一次性样本外检验。\n"
)

DEFAULT_HOLDOUT_SNAPSHOT_MARKERS = (
    "2025-03-31",
    "2025-06-30",
    "2025-09-30",
    "2025-12-31",
    "2026-03-31",
)


def _resolve(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def _parse_opt_dates(raw: list[str] | None) -> list[pd.Timestamp] | None:
    if not raw:
        return None
    out = [_ts(s) for s in raw]
    return out


def _snapshot_holdout_ready(snap_root: Path) -> bool:
    if not snap_root.is_dir():
        return False
    return all((snap_root / d).is_dir() for d in DEFAULT_HOLDOUT_SNAPSHOT_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Time-ordered train/val/test (+ optional holdout) split by calendar year",
    )
    parser.add_argument(
        "--panel-file",
        type=Path,
        default=Path("data/features/csi300_full_panel.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/panel"))
    parser.add_argument("--val-periods", type=int, default=4)
    parser.add_argument("--test-periods", type=int, default=4)
    parser.add_argument("--predict-periods", type=int, default=2)
    parser.add_argument("--train-max-year", type=int, default=2022)
    parser.add_argument("--val-year", type=int, default=2023)
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--holdout-min-year", type=int, default=2025)
    parser.add_argument(
        "--val-dates",
        nargs="*",
        default=None,
        help="可选：显式 val 截面日期（ISO），须与 panel 中 as_of 一致；省略则按 --val-year 自动筛选",
    )
    parser.add_argument(
        "--test-dates",
        nargs="*",
        default=None,
        help="可选：显式 test 截面日期；省略则按 --test-year 自动筛选",
    )
    parser.add_argument(
        "--holdout-dates",
        nargs="*",
        default=None,
        help="可选：显式 holdout 截面日期；省略则按 --holdout-min-year 自动筛选（须存在于 panel）",
    )
    parser.add_argument(
        "--snapshots-root",
        type=Path,
        default=Path("data/snapshots"),
        help="用于检测 Track A 是否已落盘季度快照（缺少则无 holdout 期望来源提示）",
    )
    parser.add_argument("--label", type=str, default="forward_return_21d")
    parser.add_argument("--drop-nan-label", action="store_true")
    args = parser.parse_args()

    panel_path = _resolve(args.panel_file)
    out_dir = _resolve(args.output_dir)
    snap_root = _resolve(args.snapshots_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(panel_path)
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must have MultiIndex")
    if panel.index.names != ["as_of", "ticker"]:
        panel.index = panel.index.set_names(["as_of", "ticker"])

    as_ofs = sorted({_ts(x) for x in panel.index.get_level_values(0).unique()})

    label_col = args.label
    if label_col not in panel.columns:
        raise KeyError(label_col)

    explicit_val = _parse_opt_dates(args.val_dates)
    explicit_test = _parse_opt_dates(args.test_dates)
    explicit_holdout = _parse_opt_dates(args.holdout_dates)

    as_of_set = set(as_ofs)

    def _must_be_in_panel(name: str, dates: list[pd.Timestamp]) -> None:
        missing = [d for d in dates if d not in as_of_set]
        if missing:
            raise ValueError(
                f"{name} dates not found in panel index: {[str(d.date()) for d in missing]}. "
                f"Panel has {len(as_ofs)} as_of values."
            )

    if explicit_val is not None:
        _must_be_in_panel("val", explicit_val)
        val_as_ofs = sorted(explicit_val)
    else:
        val_as_ofs = sorted({d for d in as_ofs if d.year == args.val_year})

    if explicit_test is not None:
        _must_be_in_panel("test", explicit_test)
        test_as_ofs = sorted(explicit_test)
    else:
        test_as_ofs = sorted({d for d in as_ofs if d.year == args.test_year})

    if explicit_holdout is not None:
        _must_be_in_panel("holdout", explicit_holdout)
        holdout_as_ofs = sorted(explicit_holdout)
    else:
        holdout_as_ofs = sorted({d for d in as_ofs if d.year >= args.holdout_min_year})

    used = set(val_as_ofs) | set(test_as_ofs) | set(holdout_as_ofs)
    train_as_ofs = sorted([d for d in as_ofs if d not in used])
    stray = [d for d in train_as_ofs if d.year > args.train_max_year]
    if stray:
        raise ValueError(
            "train split contains as_of with year > train_max_year "
            f"({args.train_max_year}): {[str(x.date()) for x in stray]}. "
            "Check panel or explicit --val-dates/--test-dates/--holdout-dates."
        )

    if len(val_as_ofs) != args.val_periods:
        raise ValueError(
            f"val split has {len(val_as_ofs)} periods, expected --val-periods={args.val_periods}. "
            f"val dates: {[str(x.date()) for x in val_as_ofs]}"
        )
    if len(test_as_ofs) != args.test_periods:
        raise ValueError(
            f"test split has {len(test_as_ofs)} periods, expected --test-periods={args.test_periods}. "
            f"test dates: {[str(x.date()) for x in test_as_ofs]}"
        )

    snap_ready = _snapshot_holdout_ready(snap_root)
    holdout_status = "written"
    holdout_note = ""
    if not holdout_as_ofs:
        holdout_status = "skipped_empty"
        if snap_ready:
            holdout_note = (
                "快照目录已包含 2025–2026 季度，但当前 panel-file 无对应 as_of 行；"
                "请在不改动特征公式的前提下生成扩展面板后再划分 holdout。"
            )
        else:
            holdout_note = "未检测到完整 holdout 快照目录；跳过 holdout.parquet。"

    predict_as_ofs = sorted(test_as_ofs)[-args.predict_periods :] if test_as_ofs else []
    if len(predict_as_ofs) != args.predict_periods:
        raise ValueError(
            f"cannot take predict_periods={args.predict_periods} from test (has {len(test_as_ofs)})."
        )

    def _take(dates: list[pd.Timestamp]) -> pd.DataFrame:
        return panel.loc[panel.index.get_level_values(0).isin(dates)]

    train_df = _take(train_as_ofs)
    val_df = _take(val_as_ofs)
    test_df = _take(test_as_ofs)
    predict_df = _take(predict_as_ofs)
    holdout_df = _take(holdout_as_ofs) if holdout_as_ofs else pd.DataFrame()

    if args.drop_nan_label:
        train_df = train_df.loc[train_df[label_col].notna()]
        val_df = val_df.loc[val_df[label_col].notna()]
        test_df = test_df.loc[test_df[label_col].notna()]
        predict_df = predict_df.loc[predict_df[label_col].notna()]
        if not holdout_df.empty:
            holdout_df = holdout_df.loc[holdout_df[label_col].notna()]

    feat_cols = [c for c in panel.columns if not str(c).startswith("forward_return_")]

    train_df.to_parquet(out_dir / "train.parquet")
    val_df.to_parquet(out_dir / "val.parquet")
    test_df.to_parquet(out_dir / "test.parquet")
    predict_df.to_parquet(out_dir / "predict.parquet")

    warn_path = out_dir / "holdout_DO_NOT_USE_BEFORE_MODEL_FREEZE.txt"
    holdout_path = out_dir / "holdout.parquet"
    if not holdout_df.empty:
        holdout_df.to_parquet(holdout_path)
        warn_path.write_text(HOLDOUT_WARNING_ZH, encoding="utf-8")
    else:
        if warn_path.is_file():
            warn_path.unlink()
        if holdout_path.is_file():
            holdout_path.unlink()

    def _bounds(idxs: pd.Index) -> dict[str, str | None]:
        if len(idxs) == 0:
            return {"min": None, "max": None}
        lev = idxs.get_level_values(0)
        tmin, tmax = lev.min(), lev.max()
        return {"min": pd.Timestamp(tmin).date().isoformat(), "max": pd.Timestamp(tmax).date().isoformat()}

    meta = {
        "panel_file": str(panel_path),
        "label_column": label_col,
        "drop_nan_label": bool(args.drop_nan_label),
        "train_max_year": args.train_max_year,
        "val_year": args.val_year,
        "test_year": args.test_year,
        "holdout_min_year": args.holdout_min_year,
        "val_periods": args.val_periods,
        "test_periods": args.test_periods,
        "predict_periods": args.predict_periods,
        "snapshots_root": str(snap_root),
        "track_a_snapshots_expected_ready": snap_ready,
        "semantics": {
            "train": "拟合用历史窗口（默认截至 2022 年末截面）",
            "val": "验证 / early stopping / 因子筛选（默认 2023 全年 4 期），严禁用 test 调参",
            "test": "最终一次性 OOS（默认 2024 全年 4 期），勿反复刷指标",
            "predict": "默认取 test 期内最近若干截面，用于 forward 打分流水线",
            "holdout": "预留样本（≥2025），仅在模型固化后做一次评测；见 holdout_DO_NOT_USE_BEFORE_MODEL_FREEZE.txt",
        },
        "feature_columns": feat_cols,
        "n_periods_total": len(as_ofs),
        "train": {
            "as_of_dates": [pd.Timestamp(x).date().isoformat() for x in train_as_ofs],
            "date_range": _bounds(train_df.index),
            "n_rows": int(len(train_df)),
            "n_periods": len(train_as_ofs),
        },
        "val": {
            "as_of_dates": [pd.Timestamp(x).date().isoformat() for x in val_as_ofs],
            "date_range": _bounds(val_df.index),
            "n_rows": int(len(val_df)),
            "n_periods": len(val_as_ofs),
        },
        "test": {
            "as_of_dates": [pd.Timestamp(x).date().isoformat() for x in test_as_ofs],
            "date_range": _bounds(test_df.index),
            "n_rows": int(len(test_df)),
            "n_periods": len(test_as_ofs),
        },
        "predict": {
            "as_of_dates": [pd.Timestamp(x).date().isoformat() for x in predict_as_ofs],
            "date_range": _bounds(predict_df.index),
            "n_rows": int(len(predict_df)),
            "n_periods": len(predict_as_ofs),
        },
        "holdout": {
            "_warning": HOLDOUT_WARNING_ZH.strip().replace("\n", " "),
            "status": holdout_status,
            "note": holdout_note,
            "as_of_dates": [pd.Timestamp(x).date().isoformat() for x in holdout_as_ofs],
            "date_range": _bounds(holdout_df.index) if not holdout_df.empty else {"min": None, "max": None},
            "n_rows": int(len(holdout_df)),
            "n_periods": len(holdout_as_ofs),
            "readme_file": "holdout_DO_NOT_USE_BEFORE_MODEL_FREEZE.txt" if not holdout_df.empty else None,
        },
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        "语义: train=拟合 | val=2023 调参 | test=2024 一次性 OOS | predict=test 末截面 | "
        "holdout=≥2025 预留（仅当面板含对应行）",
        flush=True,
    )
    print(
        f"train periods={len(train_as_ofs)} rows={len(train_df)} range {meta['train']['date_range']}",
        flush=True,
    )
    print(
        f"val   periods={len(val_as_ofs)} rows={len(val_df)} range {meta['val']['date_range']}",
        flush=True,
    )
    print(
        f"test  periods={len(test_as_ofs)} rows={len(test_df)} range {meta['test']['date_range']}",
        flush=True,
    )
    print(
        f"pred  periods={len(predict_as_ofs)} rows={len(predict_df)} range {meta['predict']['date_range']}",
        flush=True,
    )
    print(
        f"holdout periods={len(holdout_as_ofs)} rows={len(holdout_df)} status={holdout_status}",
        flush=True,
    )
    if holdout_note:
        print(f"holdout note: {holdout_note}", flush=True)

    sets = {
        "train": set(train_as_ofs),
        "val": set(val_as_ofs),
        "test": set(test_as_ofs),
        "predict": set(predict_as_ofs),
        "holdout": set(holdout_as_ofs),
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    if holdout_as_ofs:
        pairs += [("train", "holdout"), ("val", "holdout"), ("test", "holdout")]
    for a, b in pairs:
        inter = sets[a] & sets[b]
        if inter:
            raise ValueError(f"overlap {a}/{b}: {inter}")
    # predict 故意与 test 重叠 — 不纳入互斥检查

    if train_as_ofs and val_as_ofs:
        if val_as_ofs[0] <= train_as_ofs[-1]:
            raise ValueError("val 起始应严格晚于 train 末尾截面")
    if val_as_ofs and test_as_ofs:
        if test_as_ofs[0] <= val_as_ofs[-1]:
            raise ValueError("test 起始应严格晚于 val 末尾截面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
