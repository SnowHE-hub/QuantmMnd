"""P7：20 日 XGBoost 前瞻打分。只追加写 data/gpu_v1/forward/，不接定时任务。

用最后一折已经保存的 ubj（不重新训练）。CatBoost 没有保存模型，所以不产 ENS。
候选日期是 model/v1 里严格晚于最后一折 pred_days 的 as_of。冻结面板里没有这种日期时，
不写文件，退出码 0。已有输出文件则拒绝覆盖。

用法（仓库根目录）：PYTHONPATH=. python docs/work/evidence/T-030/forward_20d.py /home/lenovo/projects/quantmind
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

import numpy as np

from quantmind.research.gpu.data import Block, ModelData
from quantmind.research.gpu.ensemble import rank_average
from quantmind.research.gpu.folds import calendar_and_grid, make_folds
from quantmind.research.gpu.trees import XGB, TreeInput


def main() -> None:
    root = Path(sys.argv[1])
    os.environ.setdefault("WRITE_MODE", "parquet_only")
    cal, grid = calendar_and_grid(root)
    folds = make_folds(cal, grid, "20d")
    last = folds[-1]
    covered_through = max(pd.Timestamp(d) for d in last.pred_days)
    run_dir = root / "data/gpu_v1/runs/xgb_20d"
    selected = json.loads((run_dir / "ledger" / f"k{last.k:02d}_selected.json").read_text())
    rounds = int(selected["rounds"])
    models = sorted((run_dir).glob(f"k{last.k:02d}_s*.ubj"))
    if len(models) != 5:
        raise SystemExit(f"expected 5 xgb models for fold {last.k}, found {len(models)}")
    meta_dir = root / "data/gpu_v1/model/v1"
    frames = []
    for path in sorted(meta_dir.glob("*.meta.parquet")):
        meta = pd.read_parquet(path, columns=["as_of"])
        meta["as_of"] = pd.to_datetime(meta["as_of"])
        meta = meta[meta["as_of"] > covered_through]
        if len(meta):
            frames.append(meta)
    days = sorted(set(pd.concat(frames)["as_of"])) if frames else []
    print(f"last_fold={last.k} covered_through={covered_through.date()} rounds={rounds} new_days={len(days)}", flush=True)
    if not days:
        print("没有晚于冻结样本的 as_of，不写文件。", flush=True)
        return
    out_dir = root / "data/gpu_v1/forward/xgb_20d"
    out_dir.mkdir(parents=True, exist_ok=True)
    for day in days:
        dest = out_dir / f"{pd.Timestamp(day).date()}.parquet"
        if dest.exists():
            raise SystemExit(f"refuse to overwrite {dest}")
    import xgboost as xgb

    md = ModelData(root)
    block = md.load(days)
    order = np.argsort(block.meta["day"].to_numpy(), kind="mergesort")
    block = Block(block.X[order], block.meta.iloc[order].reset_index(drop=True), block.days)
    src = TreeInput(md, block)
    preds = []
    for path in models:
        model = xgb.Booster()
        model.load_model(str(path))
        preds.append(XGB().predict(model, src, rounds))
    score = rank_average(preds, block.meta["day"].to_numpy())
    frame = pd.DataFrame({
        "as_of": block.meta["as_of"],
        "ticker": block.meta["ticker"],
        "score": score,
        "top1500": block.meta["top1500"],
    })
    for day, part in frame.groupby("as_of", sort=True):
        dest = out_dir / f"{pd.Timestamp(day).date()}.parquet"
        tmp = dest.with_suffix(".tmp")
        part.to_parquet(tmp, index=False)
        os.replace(tmp, dest)
        print(f"wrote {dest} rows={len(part)}", flush=True)


if __name__ == "__main__":
    main()
