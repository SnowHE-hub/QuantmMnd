"""T-030 负对照诊断（2026-10-17）。不改 configs.py，不覆盖已有 N1 预测和 controls_*.json。

子命令：
  n1-foldci <data_root> <out.json>
      用已有的 xgb_63d_N1shuffle（打乱种子 0）算：块长 13 的区间，以及以折为抽样单位的 t 区间（只含有有限标签的折）。
  n2-assoc <data_root> <out.json>
      折 0 内层训练日上，金丝雀（t+1 的 r_cc）与 63d 标签的截面相关，以及 1 日对数收益在 63 日对数收益方差中的占比。不训练模型。
  n2-gain <data_root> <out.json>
      重跑一次 N2 的那一个模型，把 gain 全表写出来。排队用，不覆盖 controls_63d.json。
  n1-seed <data_root> <shuffle_seed>
      再跑一遍 63d N1（XGBoost 网格 c0、500 轮、模型种子 0），只改标签打乱种子。
      预测写到 data/gpu_v1/preds/xgb_63d_N1sXX，不碰 xgb_63d_N1shuffle。
  n1-summary <data_root> <out.json>
      汇总种子 0（已有预测）和 N1s01–N1s15。各种子的 IC、块长 13 区间、9 折 t 区间，以及种子之间的分布。

用法都在仓库根目录、PYTHONPATH=. 下。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from quantmind.research.gpu.folds import calendar_and_grid, make_folds
from quantmind.research.gpu.metrics import daily_rank_ic, moving_block_bootstrap, two_sided_ci
from quantmind.research.gpu.transforms import day_bounds, gauss_rank_by_day

ROOT_DATA_PRED = "data/gpu_v1/preds"
SEEDS = [0] + list(range(1, 16))  # 0 用已有预测；1–15 是追加的打乱种子


def _meta(root: Path, days, columns: list[str]) -> pd.DataFrame:
    days = pd.DatetimeIndex(pd.to_datetime(list(days)))
    want = set(days)
    parts = []
    d = root / "data/gpu_v1/model/v1"
    for month in sorted({x.strftime("%Y-%m") for x in days}):
        meta = pd.read_parquet(d / f"{month}.meta.parquet", columns=columns)
        meta["as_of"] = pd.to_datetime(meta["as_of"])
        meta = meta.loc[meta["as_of"].isin(want)]
        if len(meta):
            parts.append(meta)
    if not parts:
        raise RuntimeError("no meta rows")
    meta = pd.concat(parts, ignore_index=True)
    got = pd.DatetimeIndex(sorted(meta["as_of"].unique()))
    meta["day"] = got.get_indexer(meta["as_of"])
    return meta.sort_values(["day", "ticker"], kind="mergesort").reset_index(drop=True)


def _pred_dir(root: Path, seed: int) -> Path:
    name = "xgb_63d_N1shuffle" if seed == 0 else f"xgb_63d_N1s{seed:02d}"
    return root / ROOT_DATA_PRED / name


def _fold_daily_ics(root: Path, seed: int) -> list[np.ndarray]:
    """与 p6_controls.pool_ic 相同的 1500 池日 RankIC，按折分开。只读 meta 列，不读特征矩阵。"""
    cal, grid = calendar_and_grid(root)
    folds = make_folds(cal, grid, "63d")
    out = []
    pred_dir = _pred_dir(root, seed)
    for k, f in enumerate(folds):
        meta = _meta(root, f.test_asof, ["as_of", "ticker", "y63", "top1500"])
        p = pd.read_parquet(pred_dir / f"fold_{k}.parquet")
        p["as_of"] = pd.to_datetime(p["as_of"])
        score = p.set_index(["as_of", "ticker"])["score"].reindex(
            pd.MultiIndex.from_frame(meta[["as_of", "ticker"]])).to_numpy()
        ic = daily_rank_ic(score, meta["y63"].to_numpy(dtype=np.float64),
                           meta["day"].to_numpy(), meta["top1500"].to_numpy())
        out.append(ic)
    return out


def _t_interval(x: np.ndarray, level: float = 0.95) -> dict:
    x = np.asarray(x, dtype=float)
    n = int(len(x))
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.5 + level / 2, n - 1))
    return {"n": n, "mean": mean, "se": se, "tcrit": tcrit,
            "ci95": [mean - tcrit * se, mean + tcrit * se]}


def _seed_stats(ics: list[np.ndarray]) -> dict:
    usable = [a for a in ics if len(a)]
    flat = np.concatenate(usable)
    block = two_sided_ci(moving_block_bootstrap(flat, 13), 0.95)
    fold_means = np.array([float(a.mean()) if len(a) else np.nan for a in ics])
    fold = _t_interval(fold_means[np.isfinite(fold_means)])
    return {
        "ic_mean": float(flat.mean()),
        "n_dates": int(len(flat)),
        "n_dates_per_fold": [int(len(a)) for a in ics],
        "n_folds_with_labels": int(len(usable)),
        "ci95_block13": block,
        "block13_contains_0": bool(block[0] <= 0 <= block[1]),
        "fold_means": [None if not np.isfinite(v) else float(v) for v in fold_means],
        "fold_t": fold,
        "fold_contains_0": bool(fold["ci95"][0] <= 0 <= fold["ci95"][1]),
    }


def cmd_n1_foldci(root: Path, out: Path) -> None:
    st = _seed_stats(_fold_daily_ics(root, 0))
    st["shuffle_seed"] = 0
    st["pred_dir"] = str(_pred_dir(root, 0))
    st["note"] = ("打乱种子 0，即已写出的 xgb_63d_N1shuffle。折级 t 区间只用有有限 63d 标签的折。"
                 "第 9 折的测试 as_of 是 2026-08-04 至 2026-09-15，数据止于 2026-10-09，y63 全缺失，所以是 8 折不是 9 折。")
    _write(out, st)


def cmd_n1_summary(root: Path, out: Path) -> None:
    per = []
    missing = []
    for seed in SEEDS:
        d = _pred_dir(root, seed)
        if len(list(d.glob("fold_*.parquet"))) != 9:
            missing.append(seed)
            continue
        st = _seed_stats(_fold_daily_ics(root, seed))
        st["shuffle_seed"] = seed
        per.append(st)
    means = np.array([s["ic_mean"] for s in per])
    summary = {
        "seeds_expected": SEEDS,
        "seeds_present": [s["shuffle_seed"] for s in per],
        "seeds_missing": missing,
        "per_seed": per,
    }
    if len(means):
        summary["across_seeds"] = {
            "n": int(len(means)),
            "mean": float(means.mean()),
            "std": float(means.std(ddof=1)) if len(means) > 1 else None,
            "min": float(means.min()),
            "p25": float(np.percentile(means, 25)),
            "median": float(np.median(means)),
            "p75": float(np.percentile(means, 75)),
            "max": float(means.max()),
            "t_interval_of_seed_means": _t_interval(means) if len(means) > 1 else None,
            "n_block13_excludes_0": int(sum(not s["block13_contains_0"] for s in per)),
            "n_fold_excludes_0": int(sum(not s["fold_contains_0"] for s in per)),
        }
    _write(out, summary)


def cmd_n2_assoc(root: Path, out: Path) -> None:
    """不训练。折 0 内层训练日，与 p6_controls 的 N2 用同一批日期。"""
    from quantmind.research.gpu.dl import SeqStore

    cal, grid = calendar_and_grid(root)
    fold0 = make_folds(cal, grid, "63d")[0]
    meta = _meta(root, fold0.inner_train_days, ["as_of", "ticker", "y63", "seq_idx"])
    seq = SeqStore(root)
    rcc = seq.channels.index("r_cc")
    gday = seq.day_index(meta["as_of"].to_numpy())
    seq_idx = meta["seq_idx"].to_numpy()
    y = meta["y63"].to_numpy(dtype=np.float64)
    n_days = seq.arr.shape[1]
    r1 = np.full(len(meta), np.nan)
    s63 = np.full(len(meta), np.nan)
    b = day_bounds(meta["day"].to_numpy())
    n_clamped = 0
    for s, e in zip(b[:-1], b[1:], strict=True):
        g = int(gday[s])
        nxt = g + 1
        if nxt >= n_days:
            nxt = n_days - 1
            n_clamped += e - s
        rows = seq_idx[s:e]
        r1[s:e] = np.asarray(seq.arr[rows, nxt, rcc], dtype=np.float64)
        if g + 63 < n_days:
            window = np.asarray(seq.arr[rows, g + 1:g + 64, rcc], dtype=np.float64)
            s63[s:e] = window.sum(axis=1)
    ok_y = np.isfinite(y) & np.isfinite(r1)
    spearman = []
    for s, e in zip(b[:-1], b[1:], strict=True):
        m = ok_y[s:e]
        if m.sum() < 30:
            continue
        spearman.append(float(stats.spearmanr(r1[s:e][m], y[s:e][m]).statistic))
    both = np.isfinite(s63) & np.isfinite(r1) & np.isfinite(y)
    r1b, s63b, yb = r1[both], s63[both], y[both]
    log_y = np.log1p(yb)
    var_r1 = float(np.var(r1b, ddof=1))
    var_s = float(np.var(s63b, ddof=1))
    corr_sum = float(np.corrcoef(r1b, s63b)[0, 1])
    corr_label = float(np.corrcoef(s63b, log_y)[0, 1])
    _write(out, {
        "sample": "63d fold 0 inner_train_days",
        "n_rows_meta": int(len(meta)),
        "n_rows_finite_y_and_r1": int(ok_y.sum()),
        "n_rows_with_63_future_log_returns": int(both.sum()),
        "n_days_spearman": int(len(spearman)),
        "n_rows_clamped_to_last_seq_day": int(n_clamped),
        "mean_daily_spearman_r1_vs_y63": float(np.mean(spearman)),
        "var_r1": var_r1,
        "var_sum63_log": var_s,
        "var_share_r1_in_sum63": var_r1 / var_s,
        "iid_benchmark_1_over_63": 1.0 / 63.0,
        "r2_sum63_on_r1": corr_sum ** 2,
        "corr_sum63_vs_log1p_y63": corr_label,
        "note": "y63 是 63 个交易日的简单收益。sum63 是随后 63 个交易日 r_cc 的和。两者若是同一复权收盘价，sum63 应等于 log(1+y63)。",
    })


def cmd_n2_gain(root: Path, out: Path) -> None:
    from quantmind.research.gpu.configs import TREE_GRIDS
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.dl import SeqStore
    from quantmind.research.gpu.runtime import start_ram_guard, wait_for_vram
    from quantmind.research.gpu.transforms import gauss_rank_by_day
    from quantmind.research.gpu.trees import XGB, TreeInput, train_target

    start_ram_guard()
    wait_for_vram(4000)
    cal, grid = calendar_and_grid(root)
    fold0 = make_folds(cal, grid, "63d")[0]
    md = ModelData(root)
    names = md.stock_features + md.market_features
    assert not any("canary" in c or "fwd" in c for c in names)
    seq = SeqStore(root)
    tr = md.load(fold0.inner_train_days)
    g = seq.day_index(tr.meta["as_of"].to_numpy())
    r_next = seq.arr[tr.meta["seq_idx"].to_numpy(), np.minimum(g + 1, seq.arr.shape[1] - 1), seq.channels.index("r_cc")]
    canary = gauss_rank_by_day(r_next.astype(np.float64), tr.meta["day"].to_numpy())
    rows, y = train_target(tr, "63d", "regression")
    model = XGB().fit(TREE_GRIDS["xgb"][0], TreeInput(md, tr, rows, extra=canary[:, None].astype(np.float32)),
                      y, tr.meta["day"].to_numpy()[rows], 500, seed=0)
    gain = model.get_score(importance_type="gain")
    names = names + ["CANARY_next_day_return"]
    order = sorted(gain, key=gain.get, reverse=True)
    table = [{"rank": i + 1, "feature": names[int(k[1:])], "gain": float(gain[k])} for i, k in enumerate(order)]
    canary_row = next(r for r in table if r["feature"] == "CANARY_next_day_return")
    total = float(sum(r["gain"] for r in table))
    _write(out, {
        "horizon": "63d",
        "fold": 0,
        "n_features": len(names),
        "canary_rank": canary_row["rank"],
        "canary_gain": canary_row["gain"],
        "canary_gain_share": canary_row["gain"] / total,
        "top1_feature": table[0]["feature"],
        "top1_gain": table[0]["gain"],
        "top10": table[:10],
    })


def _shuffle_within_day(y: np.ndarray, day: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = y.copy()
    b = day_bounds(day)
    for s, e in zip(b[:-1], b[1:], strict=True):
        out[s:e] = rng.permutation(out[s:e])
    return out


def cmd_n1_seed(root: Path, seed: int) -> None:
    """与 run_trees.py --shuffle-labels 相同，只是打乱种子可改，输出目录分开。"""
    import gc

    from quantmind.research.gpu.configs import TREE_GRIDS
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.ensemble import rank_average
    from quantmind.research.gpu.folds import check_folds
    from quantmind.research.gpu.runtime import start_ram_guard, wait_for_vram
    from quantmind.research.gpu.trees import XGB, TreeInput, train_target

    if seed == 0:
        raise SystemExit("seed 0 is the existing xgb_63d_N1shuffle run; refuse to rewrite it")
    os.environ.setdefault("WRITE_MODE", "parquet_only")
    start_ram_guard()
    tag = f"xgb_63d_N1s{seed:02d}"
    run_dir = root / "data/gpu_v1/runs" / tag
    pred_dir = root / ROOT_DATA_PRED / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    cal, wgrid = calendar_and_grid(root)
    folds = make_folds(cal, wgrid, "63d")
    check_folds(folds, cal)
    md = ModelData(root)
    params = TREE_GRIDS["xgb"][0]
    t0 = time.time()
    log = lambda m: print(f"[{tag}] {m} ({time.time() - t0:.0f}s)", flush=True)  # noqa: E731
    for k, f in enumerate(folds):
        if (pred_dir / f"fold_{k}.parquet").exists():
            log(f"fold {k} exists, skip")
            continue
        wait_for_vram(4000, log=log)
        full = md.load(f.train_days)
        rows, y = train_target(full, "63d", params["objective"])
        day = full.meta["day"].to_numpy()[rows]
        y = _shuffle_within_day(y, day, seed)
        X = TreeInput(md, full, rows)
        del full
        gc.collect()
        t1 = time.time()
        model = XGB().fit(params, X, y, day, 500, seed=0)
        pb = md.load(f.pred_days)
        pred = XGB().predict(model, TreeInput(md, pb), 500)
        score = rank_average([pred], pb.meta["day"].to_numpy())
        frame = pd.DataFrame({"as_of": pb.meta["as_of"], "ticker": pb.meta["ticker"],
                              "score": score, "top1500": pb.meta["top1500"]})
        tmp = pred_dir / f"fold_{k}.tmp"
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, pred_dir / f"fold_{k}.parquet")
        log(f"fold {k} written {frame.shape} {time.time() - t1:.0f}s")
        del model, X, pb, frame
        gc.collect()
    log("done")


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(obj, ensure_ascii=False, indent=1))


def main() -> None:
    cmd, root = sys.argv[1], Path(sys.argv[2])
    if cmd == "n1-foldci":
        cmd_n1_foldci(root, Path(sys.argv[3]))
    elif cmd == "n1-summary":
        cmd_n1_summary(root, Path(sys.argv[3]))
    elif cmd == "n2-assoc":
        cmd_n2_assoc(root, Path(sys.argv[3]))
    elif cmd == "n2-gain":
        cmd_n2_gain(root, Path(sys.argv[3]))
    elif cmd == "n1-seed":
        cmd_n1_seed(root, int(sys.argv[3]))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
