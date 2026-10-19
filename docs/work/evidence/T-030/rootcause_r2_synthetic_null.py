"""T-030 63d 根因清单 R2：独立合成零信号数据检查"训练 → 预测 → IC"整条链（G1 裁定 §7 的根因排查项）.

两个变体，各跑一次，固定种子，事先写定（本文件提交早于运行）：
  R2a 合成标签：特征为真实特征；训练标签与评估标签都换成与一切无关的 iid N(0, 1)（缺失位置与 y63 相同），
      评估用同一份合成标签算 1500 池日 RankIC。理论期望恰为 0。
  R2b 合成特征：标签为真实 y63（不打乱）；股票级与市场级特征全部换成 iid N(0, 1)。评估用真实 y63。理论期望恰为 0。
其余与 N1 相同：XGBoost 网格 c0、500 轮、模型种子 0、全部 9 个外层折、冻结的 189 个可评估日期、日期等权 T。
判读（事先写定，工程检查，不是放行判据、不进入 N1-R）：每个变体报 T、块长 13 的 95% 区间、折均值。
  |T| < 0.005 且区间含 0 → "链路未见异常"；否则 → "链路异常，需排查"，先定位再说别的。
输出：data/gpu_v1/preds/xgb_63d_R2a_synlabel/、xgb_63d_R2b_synfeat/；docs/work/evidence/T-030/p6/r2_synthetic_null.json
用法：PYTHONPATH=. python docs/work/evidence/T-030/rootcause_r2_synthetic_null.py <data_root>
"""
from __future__ import annotations

import gc
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20261019


def main() -> None:
    root = Path(sys.argv[1])
    os.environ.setdefault("WRITE_MODE", "parquet_only")
    from quantmind.research.gpu.configs import TREE_GRIDS
    from quantmind.research.gpu.data import ModelData
    from quantmind.research.gpu.ensemble import rank_average
    from quantmind.research.gpu.folds import calendar_and_grid, check_folds, make_folds
    from quantmind.research.gpu.metrics import daily_rank_ic, moving_block_bootstrap, two_sided_ci
    from quantmind.research.gpu.runtime import start_ram_guard, wait_for_vram
    from quantmind.research.gpu.trees import XGB, TreeInput, train_target

    start_ram_guard()
    frozen = json.loads((HERE / "p6/frozen_dates_63d.json").read_text(encoding="utf-8"))
    cal, wgrid = calendar_and_grid(root)
    folds = make_folds(cal, wgrid, "63d")
    check_folds(folds, cal)
    md = ModelData(root)
    params = TREE_GRIDS["xgb"][0]
    report = {"what": __doc__.split("\n")[0], "seed": SEED, "utc_start": datetime.now(UTC).isoformat(),
              "frozen_dates_sha256_file": "p6/frozen_dates_63d.json", "variants": {}}
    resumed: list[str] = []

    def noise(shape, *key) -> np.ndarray:
        return np.random.default_rng([SEED, *key]).standard_normal(shape)

    def noise_matrix(shape, dtype, *key) -> np.ndarray:
        """大矩阵分块按 float32 生成（2026-10-19 第一次运行时整块 float64 噪声触发内存守卫，R2b 未开始即退出）。"""
        rng = np.random.default_rng([SEED, *key])
        out = np.empty(shape, dtype=dtype)
        for s in range(0, shape[0], 200_000):
            e = min(shape[0], s + 200_000)
            out[s:e] = rng.standard_normal((e - s, *shape[1:]), dtype=np.float32)
        return out

    for variant, tag in (("R2a_synlabel", "xgb_63d_R2a_synlabel"), ("R2b_synfeat", "xgb_63d_R2b_synfeat")):
        pred_dir = root / "data/gpu_v1/preds" / tag
        pred_dir.mkdir(parents=True, exist_ok=True)
        daily = []
        for k, f in enumerate(folds):
            fold_file = pred_dir / f"fold_{k}.parquet"
            if fold_file.exists():                              # 进程被杀后续跑：已写出的折只重算 IC，不重训
                pb = md.load(f.pred_days)
                saved = pd.read_parquet(fold_file)
                if len(saved) != len(pb.meta) or not (saved["ticker"].to_numpy() == pb.meta["ticker"].to_numpy()).all():
                    raise RuntimeError(f"{tag} fold {k}: saved rows do not match the prediction block")
                score, dpb = saved["score"].to_numpy(dtype=np.float64), pb.meta["day"].to_numpy()
                y_eval = pb.meta["y63"].to_numpy(dtype=np.float64)
                if variant == "R2a_synlabel":
                    y_eval = np.where(np.isfinite(y_eval), noise(len(y_eval), 5, k), np.nan)
                test = pd.DatetimeIndex(pd.to_datetime(list(f.test_asof)))
                mask = (pb.meta["top1500"].to_numpy(dtype=bool)
                        & pd.to_datetime(pb.meta["as_of"]).isin(test).to_numpy())
                daily.append(daily_rank_ic(score, y_eval, dpb, mask))
                resumed.append(f"{variant}:{k}")
                del pb, saved
                gc.collect()
                continue
            wait_for_vram(4000, log=lambda m: print(m, flush=True))
            full = md.load(f.train_days)
            y_real = full.meta["y63"].to_numpy(dtype=np.float64)
            if variant == "R2a_synlabel":                       # 合成标签：缺失位置与 y63 相同
                syn = noise(len(y_real), 0, k)
                full.meta["y63"] = np.where(np.isfinite(y_real), syn, np.nan)
            rows, y = train_target(full, "63d", params["objective"])
            day = full.meta["day"].to_numpy()[rows]
            X = TreeInput(md, full, rows)
            if variant == "R2b_synfeat":                        # 合成特征：股票级与市场级全部换成噪声
                X.X = None
                gc.collect()
                X.X = noise_matrix(full.X.shape, np.float16, 1, k)
                X.mk = noise_matrix(X.mk.shape, np.float32, 2, k)
            del full
            gc.collect()
            model = XGB().fit(params, X, y, day, 500, seed=0)
            del X
            gc.collect()
            pb = md.load(f.pred_days)
            Xp = TreeInput(md, pb)
            if variant == "R2b_synfeat":
                Xp.X = noise_matrix(pb.X.shape, np.float16, 3, k)
                Xp.mk = noise_matrix(Xp.mk.shape, np.float32, 4, k)
            pred = XGB().predict(model, Xp, 500)
            dpb = pb.meta["day"].to_numpy()
            score = rank_average([pred], dpb)
            pd.DataFrame({"as_of": pb.meta["as_of"], "ticker": pb.meta["ticker"], "score": score,
                          "top1500": pb.meta["top1500"]}).to_parquet(pred_dir / f"fold_{k}.parquet", index=False)
            y_eval = pb.meta["y63"].to_numpy(dtype=np.float64)
            if variant == "R2a_synlabel":
                y_eval = np.where(np.isfinite(y_eval), noise(len(y_eval), 5, k), np.nan)
            test = pd.DatetimeIndex(pd.to_datetime(list(f.test_asof)))
            mask = (pb.meta["top1500"].to_numpy(dtype=bool)
                    & pd.to_datetime(pb.meta["as_of"]).isin(test).to_numpy())
            ic = daily_rank_ic(score, y_eval, dpb, mask)
            daily.append(ic)
            print(json.dumps({"variant": variant, "fold": k, "n_dates": int(len(ic)),
                              "fold_mean": float(ic.mean()) if len(ic) else None}), flush=True)
            del model, pb, Xp
            gc.collect()
        flat = np.concatenate([a for a in daily if len(a)])
        ci = two_sided_ci(moving_block_bootstrap(flat, 13), 0.95)
        T = float(flat.mean())
        report["variants"][variant] = {
            "T": T, "n_dates": int(len(flat)), "ci95_block13": ci,
            "fold_means": [float(a.mean()) if len(a) else None for a in daily],
            "reading": "链路未见异常" if (abs(T) < 0.005 and ci[0] <= 0 <= ci[1]) else "链路异常，需排查",
        }
    report["resumed_folds_ic_recomputed_from_saved_preds"] = resumed
    report["n_frozen_dates_expected"] = int(frozen.get("n_dates", 189)) if isinstance(frozen, dict) else 189
    report["utc_end"] = datetime.now(UTC).isoformat()
    (HERE / "p6/r2_synthetic_null.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("T", "n_dates", "ci95_block13", "reading")}
                      for k, v in report["variants"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
