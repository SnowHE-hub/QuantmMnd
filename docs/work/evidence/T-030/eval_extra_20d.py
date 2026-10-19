"""20 日探索性比较。不覆盖 primary_20d.json，不算预登记主比较。

比较：
- xgb_20d_weekly 对 xgb_20d（周频训练减日频训练的日 RankIC）
- master、lgbm、ENS+master 对 Ridge
配对差沿用 eval_primary.paired（块长 13 和 26，10000 次，种子 0）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eval_primary import ic_by_date, load_family, paired
from quantmind.research.gpu.ensemble import rank_average_frames
from quantmind.research.gpu.folds import calendar_and_grid, make_folds

ROOT = Path("/home/lenovo/projects/quantmind")


def load_tag(root: Path, tag: str) -> pd.Series:
    d = root / "data/gpu_v1/preds" / tag
    files = sorted(d.glob("fold_*.parquet"))
    if len(files) != 19:
        raise SystemExit(f"{tag} has {len(files)} fold files, expected 19")
    df = pd.concat([pd.read_parquet(f, columns=["as_of", "ticker", "score"]) for f in files], ignore_index=True)
    df["as_of"] = pd.to_datetime(df["as_of"])
    return df.set_index(["as_of", "ticker"])["score"].sort_index()


def main() -> None:
    root = ROOT
    h = "20d"
    cal, grid = calendar_and_grid(root)
    test = sorted({d for f in make_folds(cal, grid, h) for d in f.test_asof})
    meta = pd.concat(
        [pd.read_parquet(p, columns=["as_of", "ticker", "y20", "top1500"])
         for p in sorted((root / "data/gpu_v1/model/v1").glob("*.meta.parquet"))],
        ignore_index=True)
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    meta = meta[meta["as_of"].isin(set(test))].set_index(["as_of", "ticker"]).rename(columns={"y20": "y"})
    pool = meta[meta["top1500"]]
    ridge = pd.read_parquet(root / "data/bakeoff/preds_v6_1_t028/ridge_20d_v6_1_BASE.parquet")
    ridge["as_of"] = pd.to_datetime(ridge["as_of"])
    ridge = ridge.set_index(["as_of", "ticker"])["score"].reindex(pool.index)

    scores = {
        "xgb": load_tag(root, "xgb_20d").reindex(pool.index),
        "xgb_weekly": load_tag(root, "xgb_20d_weekly").reindex(pool.index),
        "lgbm": load_family(root, "lgbm", h).reindex(pool.index),
        "master": load_family(root, "master", h).reindex(pool.index),
        "cat": load_family(root, "cat", h).reindex(pool.index),
        "tabm": load_family(root, "tabm", h).reindex(pool.index),
        "seqfusion": load_family(root, "seqfusion", h).reindex(pool.index),
    }
    members = ["xgb", "cat", "tabm", "seqfusion", "master"]
    scores["ENS+master"] = rank_average_frames({m: scores[m] for m in members}).reindex(pool.index)

    def vs_ridge(name: str) -> dict:
        both = pool.assign(e=scores[name], r=ridge).dropna()
        out = paired(ic_by_date(both, "e"), ic_by_date(both, "r"))
        out["n_rows"] = int(len(both))
        out["role"] = "探索性。不是预登记主比较，不改 20 日主结论。"
        return out

    both_w = pool.assign(w=scores["xgb_weekly"], d=scores["xgb"]).dropna()
    weekly = paired(ic_by_date(both_w, "w"), ic_by_date(both_w, "d"))
    weekly["n_rows"] = int(len(both_w))
    weekly["contrast"] = "xgb_20d_weekly 减 xgb_20d"
    weekly["role"] = "探索性。decision 只表示周频训练的日 RankIC 是否高于日频训练，不是对 Ridge 的主比较。"

    res = {
        "horizon": "20d",
        "role": "探索性比较。20 日主比较仍以 primary_20d.json 为准。",
        "weekly_minus_daily": weekly,
        "vs_ridge": {name: vs_ridge(name) for name in ("master", "lgbm", "ENS+master")},
    }
    out = Path("docs/work/evidence/T-030/p6/exploratory_20d.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({
        "weekly_mean_diff": weekly["mean_diff"],
        "weekly_decision": weekly["decision"],
        "vs_ridge": {k: {"mean_diff": v["mean_diff"], "decision": v["decision"]} for k, v in res["vs_ridge"].items()},
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
