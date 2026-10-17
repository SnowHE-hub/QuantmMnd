"""T-030 负对照诊断（2026-10-17）。不改 configs.py，不覆盖已有 N1 预测和 controls_*.json。

子命令：
  n1-foldci <data_root> <out.json>
      用已有的 xgb_63d_N1shuffle（打乱种子 0）算：块长 13 的区间，以及以折为抽样单位的 t 区间（只含有有限标签的折）。
  n2-assoc <data_root> <out.json>
      折 0 内层训练日上，金丝雀（t+1 的 r_cc）与 63d 标签的截面相关，以及 1 日对数收益在 63 日对数收益方差中的占比。不训练模型。
  n2-gain <data_root> <out.json>
      重跑一次原 N2（次日收益金丝雀），把 gain 全表写出来。排队用，不覆盖 controls_63d.json。只作报告。
  n2-target-canary <data_root> <out.json>
      N2-R：63d 折 0 的诊断副本里加入训练目标本身。不覆盖正式模型。终端不打印 gain 或通过状态。
  n1-seed <data_root> <shuffle_seed>
      再跑一遍 63d N1（XGBoost 网格 c0、500 轮、模型种子 0），只改标签打乱种子。
      预测写到 data/gpu_v1/preds/xgb_63d_N1sXX，不碰 xgb_63d_N1shuffle。
  n1-summary <data_root> <out.json>
      N1-R：种子 1–15 的日期等权均值，双侧 90% t 区间（df=14）是否落在 (-0.005, +0.005)。
      终端不打印 IC、区间或通过状态。
  freeze-dates <data_root> <out.json>
      按标签成熟条件冻结 63d 可评估日期。不读预测。

用法都在仓库根目录、PYTHONPATH=. 下。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
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


FROZEN_DATES = Path(__file__).resolve().parent / "p6" / "frozen_dates_63d.json"
DELTA_N1 = 0.005
MIN_NAMES = 30


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finish(path: Path, rows: int) -> None:
    """终端只留完成标记、行数和文件哈希。IC、区间和通过状态只写进文件。"""
    print(f"完成 rows={rows} sha256={_file_sha256(path)}", flush=True)


def _evaluable_dates(root: Path, horizon: str) -> tuple[list, pd.DataFrame]:
    """有成熟标签、且 1500 池里至少 30 只股票标签有限的测试日。不读预测，也不删计划折。"""
    ycol = "y63" if horizon == "63d" else "y20"
    cal, grid = calendar_and_grid(root)
    folds = make_folds(cal, grid, horizon)
    rows = []
    for f in folds:
        meta = _meta(root, f.test_asof, ["as_of", "ticker", ycol, "top1500"])
        ok = np.isfinite(meta[ycol].to_numpy(dtype=np.float64)) & meta["top1500"].to_numpy()
        counts = meta.loc[ok].groupby("as_of", sort=True).size()
        for day, n in counts.items():
            if int(n) >= MIN_NAMES:
                rows.append({"as_of": pd.Timestamp(day).strftime("%Y-%m-%d"), "fold": int(f.k),
                             "n_names": int(n)})
    return folds, pd.DataFrame(rows)


def cmd_freeze_dates(root: Path, out: Path) -> None:
    folds, table = _evaluable_dates(root, "63d")
    by_fold = [{"fold": int(f.k), "n_dates": int((table["fold"] == f.k).sum())} for f in folds]
    obj = {
        "horizon": "63d",
        "rule": "标签成熟且 top1500 内至少 30 个有限标签的测试日。没有成熟标签的折日期数为 0，不填零。",
        "n_planned_folds": len(folds),
        "n_evaluable_folds": int(sum(1 for r in by_fold if r["n_dates"] > 0)),
        "n_dates": int(len(table)),
        "folds": by_fold,
        "dates": table.to_dict(orient="records"),
    }
    _write(out, obj, echo=False)
    _finish(out, obj["n_dates"])


def _seed_daily(root: Path, seed: int, frozen: pd.DataFrame) -> pd.Series:
    """每个冻结日期单独算 1500 池截面 RankIC，再按这个顺序排好。不把不同日期的股票合在一起。"""
    from scipy.stats import rankdata

    cal, grid = calendar_and_grid(root)
    folds = make_folds(cal, grid, "63d")
    want = set(frozen["as_of"].tolist())
    out: dict[str, float] = {}
    pred_dir = _pred_dir(root, seed)
    for k, f in enumerate(folds):
        use = [d for d in f.test_asof if pd.Timestamp(d).strftime("%Y-%m-%d") in want]
        if not use:
            continue
        path = pred_dir / f"fold_{k}.parquet"
        if not path.exists():
            continue
        meta = _meta(root, use, ["as_of", "ticker", "y63", "top1500"])
        pred = pd.read_parquet(path, columns=["as_of", "ticker", "score"])
        pred["as_of"] = pd.to_datetime(pred["as_of"])
        meta = meta.merge(pred, on=["as_of", "ticker"], how="left")
        score = meta["score"].to_numpy(dtype=np.float64)
        y = meta["y63"].to_numpy(dtype=np.float64)
        mask = meta["top1500"].to_numpy()
        bounds = day_bounds(meta["day"].to_numpy())
        for s, e in zip(bounds[:-1], bounds[1:], strict=True):
            day = meta["as_of"].iloc[s].strftime("%Y-%m-%d")
            ok = np.isfinite(score[s:e]) & np.isfinite(y[s:e]) & mask[s:e]
            if int(ok.sum()) < MIN_NAMES:
                out[day] = float("nan")
            else:
                out[day] = float(np.corrcoef(rankdata(score[s:e][ok]), rankdata(y[s:e][ok]))[0, 1])
    return pd.Series(out, dtype=np.float64).reindex(frozen["as_of"].tolist())


def _t90(values: np.ndarray) -> dict:
    n = int(values.size)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    tcrit = float(stats.t.ppf(0.95, n - 1))
    half = tcrit * sd / np.sqrt(n)
    return {"n": n, "df": n - 1, "mean": mean, "sd": sd, "t_crit_95": tcrit,
            "ci90": [mean - half, mean + half]}


def _classify(ci: list[float]) -> str:
    lo, hi = ci
    if lo > DELTA_N1 or hi < -DELTA_N1:
        return "检测到超容忍偏差"
    if lo > -DELTA_N1 and hi < DELTA_N1:
        return "通过"
    return "未能确认等价"


def _block_ci(values: np.ndarray, block: int) -> list[float] | None:
    if len(values) < block or not np.isfinite(values).all():
        return None
    lo, hi = two_sided_ci(moving_block_bootstrap(values, block, 10000, 0), 0.95)
    return [float(lo), float(hi)]


def cmd_n1_summary(root: Path, out: Path) -> None:
    """N1-R。主统计量是种子 1–15 的日期等权均值。缺种子、日期变了或 IC 无定义时记工程异常，不删、不补零、不替换。"""
    if not FROZEN_DATES.exists():
        raise SystemExit(f"missing frozen date list: {FROZEN_DATES}")
    frozen_raw = json.loads(FROZEN_DATES.read_text(encoding="utf-8"))
    frozen = pd.DataFrame(frozen_raw["dates"])
    folds, recomputed = _evaluable_dates(root, "63d")
    problems: list[str] = []
    if recomputed["as_of"].tolist() != frozen["as_of"].tolist():
        problems.append("冻结日期与按标签成熟条件重算的日期不一致")
    if [int((recomputed["fold"] == f.k).sum()) for f in folds] != [r["n_dates"] for r in frozen_raw["folds"]]:
        problems.append("冻结清单的分折日期数与重算结果不一致")
    if int(frozen_raw["n_planned_folds"]) != len(folds) or int(frozen_raw["n_dates"]) != len(frozen):
        problems.append("冻结清单的折数或日期数与清单内容不一致")
    per_seed = []
    daily: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        pred = _pred_dir(root, seed)
        missing_folds = [k for k in range(len(folds)) if not (pred / f"fold_{k}.parquet").exists()]
        if missing_folds:
            problems.append(f"seed {seed} missing folds {missing_folds}")
            per_seed.append({"shuffle_seed": seed, "defined": False, "missing_folds": missing_folds})
            continue
        series = _seed_daily(root, seed, frozen)
        values = series.to_numpy(dtype=np.float64)
        defined = bool(np.isfinite(values).all()) and len(values) == len(frozen)
        if not defined:
            problems.append(f"seed {seed} has undefined IC or a changed date set")
        fold_desc = []
        fold_ids = frozen["fold"].to_numpy()
        for k in range(len(folds)):
            part = values[fold_ids == k]
            finite = part[np.isfinite(part)]
            fold_desc.append({
                "fold": k,
                "n_dates": int(len(part)),
                "mean_ic": None if len(finite) == 0 else float(finite.mean()),
                "note": "折内日期等权，只作描述。T_s 是全部冻结日期等权，不是折等权。",
            })
        per_seed.append({
            "shuffle_seed": seed,
            "role": "已揭示诊断" if seed == 0 else "未揭示复核",
            "defined": defined,
            "T": None if not defined else float(values.mean()),
            "n_dates": int(np.isfinite(values).sum()),
            "ci95_block13": None if not defined else _block_ci(values, 13),
            "fold_desc": fold_desc,
        })
        if defined:
            daily[seed] = values
    primary = None
    combined16 = None
    time_structure = None
    verdict = "工程异常"
    if not problems and len(daily) == 16:
        primary_values = np.array([daily[s].mean() for s in range(1, 16)], dtype=np.float64)
        primary = _t90(primary_values)
        primary["seeds"] = "1-15"
        primary["delta"] = DELTA_N1
        primary["delta_note"] = "G1 裁定的审计容忍度，不是由这批种子估计出来的常数"
        verdict = _classify(primary["ci90"])
        combined16 = _t90(np.array([daily[s].mean() for s in SEEDS], dtype=np.float64))
        combined16["seeds"] = "0-15"
        combined16["role"] = "完整报告，不承担放行"
        mean_unseen = np.mean(np.vstack([daily[s] for s in range(1, 16)]), axis=0)
        mean_all = np.mean(np.vstack([daily[s] for s in SEEDS]), axis=0)
        time_structure = {
            "role": "只展示时间结构，不作为放行标准",
            "seeds_1_15": {"ci95_block13": _block_ci(mean_unseen, 13), "ci95_block26": _block_ci(mean_unseen, 26)},
            "seeds_0_15": {"ci95_block13": _block_ci(mean_all, 13), "ci95_block26": _block_ci(mean_all, 26)},
        }
    _write(out, {
        "rule": "N1-R。主判据是种子 1–15 的日期等权 T_s 的双侧 90% t 区间（df=14）是否落在 (-0.005, +0.005)。",
        "frozen_dates_sha256": _file_sha256(FROZEN_DATES),
        "n_planned_folds": frozen_raw["n_planned_folds"],
        "n_evaluable_folds": frozen_raw["n_evaluable_folds"],
        "n_dates": frozen_raw["n_dates"],
        "problems": problems,
        "verdict": verdict,
        "passed": verdict == "通过" and not problems,
        "primary": primary,
        "per_seed": per_seed,
        "combined_16_report_only": combined16,
        "time_structure_report_only": time_structure,
    }, echo=False)
    _finish(out, len(per_seed))


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
        "note": "原 1 日金丝雀，只作报告。63d 的放行金丝雀是 n2-target-canary，不使用本次排名。",
    }, echo=False)
    _finish(out, len(table))


CANARY_COLUMN = "CANARY_y63_train"


def cmd_n2_target_canary(root: Path, out: Path) -> None:
    """63d 折 0 的独立诊断副本。金丝雀列等于真正送进 XGBoost 的 gauss-rank 训练标签。"""
    from quantmind.research.gpu.configs import TREE_GRIDS
    from quantmind.research.gpu.data import ModelData, assert_official_feature_names
    from quantmind.research.gpu.runtime import start_ram_guard, wait_for_vram
    from quantmind.research.gpu.trees import XGB, TreeInput, train_target

    start_ram_guard()
    wait_for_vram(4000)
    cal, grid = calendar_and_grid(root)
    fold0 = make_folds(cal, grid, "63d")[0]
    md = ModelData(root)
    official = md.stock_features + md.market_features
    assert_official_feature_names(official)
    feature_file = root / "data/gpu_v1/model/v1/features.json"
    on_disk = json.loads(feature_file.read_text(encoding="utf-8"))
    disk_names = on_disk["stock_features"] + on_disk["market_features"]
    tr = md.load(fold0.inner_train_days)
    rows, y = train_target(tr, "63d", "regression")
    extra = np.full((len(tr.meta), 1), np.nan, dtype=np.float64)
    extra[rows, 0] = y
    keys = tr.meta.iloc[rows][["as_of", "ticker"]]
    sent = extra[rows, 0].astype(np.float32)
    construct_ok = bool(
        np.array_equal(extra[rows, 0], y)
        and np.array_equal(sent, np.asarray(y, dtype=np.float32))
        and len(keys.drop_duplicates()) == len(rows)
    )
    params = TREE_GRIDS["xgb"][0]
    model = XGB().fit(params, TreeInput(md, tr, rows, extra=extra), y,
                      tr.meta["day"].to_numpy()[rows], 500, seed=0)
    diag_dir = root / "data/gpu_v1/diag/n2_target_canary_63d"
    diag_dir.mkdir(parents=True, exist_ok=True)
    model_path = diag_dir / "fold0_c0_seed0.ubj"
    model.save_model(str(model_path))
    names = official + [CANARY_COLUMN]
    gain = model.get_score(importance_type="gain")
    total_gain = model.get_score(importance_type="total_gain")
    table = []
    for i, name in enumerate(names):
        key = f"f{i}"
        g = gain.get(key)
        tg = total_gain.get(key)
        table.append({
            "feature": name,
            "gain": None if g is None else float(g),
            "total_gain": None if tg is None else float(tg),
            "used": g is not None,
        })
    canary = next(r for r in table if r["feature"] == CANARY_COLUMN)
    other = [r["gain"] for r in table if r["feature"] != CANARY_COLUMN and r["gain"] is not None]
    canary_gain = canary["gain"]
    finite = canary_gain is not None and bool(np.isfinite(canary_gain))
    unique_first = bool(finite and other and canary_gain > max(other))
    model_rel = model_path.relative_to(root).as_posix()
    isolated = (
        CANARY_COLUMN not in disk_names
        and model_rel.startswith("data/gpu_v1/diag/")
        and not model_rel.startswith("data/gpu_v1/runs/")
        and not model_rel.startswith("data/gpu_v1/preds/")
    )
    _write(out, {
        "rule": "N2-R。gain 必须严格高于其他每一列。total_gain 只作辅助。",
        "horizon": "63d",
        "fold": 0,
        "config": "xgb c0",
        "rounds": 500,
        "model_seed": 0,
        "canary_column": CANARY_COLUMN,
        "n_train_rows": int(len(rows)),
        "construct_ok": construct_ok,
        "unique_first": unique_first,
        "isolated": isolated,
        "diagnostic_model": model_rel,
        "diagnostic_model_sha256": _file_sha256(model_path),
        "passed": bool(construct_ok and unique_first and isolated),
        "params": params,
        "importance": table,
    }, echo=False)
    _finish(out, len(table))


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
    log = lambda m: print(m, flush=True)  # noqa: E731
    for k, f in enumerate(folds):
        path = pred_dir / f"fold_{k}.parquet"
        if path.exists():
            _finish(path, len(pd.read_parquet(path, columns=["ticker"])))
            continue
        wait_for_vram(4000, log=log)
        full = md.load(f.train_days)
        rows, y = train_target(full, "63d", params["objective"])
        day = full.meta["day"].to_numpy()[rows]
        y = _shuffle_within_day(y, day, seed)
        X = TreeInput(md, full, rows)
        del full
        gc.collect()
        model = XGB().fit(params, X, y, day, 500, seed=0)
        pb = md.load(f.pred_days)
        pred = XGB().predict(model, TreeInput(md, pb), 500)
        score = rank_average([pred], pb.meta["day"].to_numpy())
        frame = pd.DataFrame({"as_of": pb.meta["as_of"], "ticker": pb.meta["ticker"],
                              "score": score, "top1500": pb.meta["top1500"]})
        tmp = pred_dir / f"fold_{k}.tmp"
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        n_rows = len(frame)
        del model, X, pb, frame
        gc.collect()
        _finish(path, n_rows)


def _write(path: Path, obj: dict, echo: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    if echo:
        print(json.dumps(obj, ensure_ascii=False, indent=1))


def main() -> None:
    cmd, root = sys.argv[1], Path(sys.argv[2])
    if cmd == "n1-foldci":
        cmd_n1_foldci(root, Path(sys.argv[3]))
    elif cmd == "n1-summary":
        cmd_n1_summary(root, Path(sys.argv[3]))
    elif cmd == "freeze-dates":
        cmd_freeze_dates(root, Path(sys.argv[3]))
    elif cmd == "n2-assoc":
        cmd_n2_assoc(root, Path(sys.argv[3]))
    elif cmd == "n2-gain":
        cmd_n2_gain(root, Path(sys.argv[3]))
    elif cmd == "n2-target-canary":
        cmd_n2_target_canary(root, Path(sys.argv[3]))
    elif cmd == "n1-seed":
        cmd_n1_seed(root, int(sys.argv[3]))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
