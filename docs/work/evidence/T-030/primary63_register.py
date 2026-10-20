"""T-030 · 63d 主比较的放行前登记（预登记附件 B.3：N1-R2 已通过；正式模型与主比较保持冻结才执行）.

不计算任何 IC，不读任何主比较结果。只做：
1. `p6/primary_63d.json` 必须不存在；
2. 评估代码冻结：`eval_primary.py` 与 G1 2026-10-17 冻结清单一致；`configs.py`、`metrics.py` 与 N1-R2 登记一致；
   主比较用到的代码自标签 `t030-p7-2026-10-19` 以来没有任何提交改动；
3. 正式模型与数据冻结：P7 清单 `p7/data_sha256.json` 的每个文件重新计算 sha256，逐一比对；
   另记录主比较读入、但不在 P7 清单里的输入（Ridge C 基线、价格、周频面板）的 sha256；
4. 63d 可评估行冻结（与 20d 的 `frozen_eval_rows_20d.parquet` 同一定义）：测试 as_of 上、1500 池内、
   ENS 四族（xgb、cat、tabm、seqfusion）都有分数、Ridge C 有分数、y63 有限的 (as_of, ticker)。
输出 `p6/frozen_eval_rows_63d.parquet` 与 `p6/primary63_registration.json`；终端只打印完成标记、行数、日期数和 sha256。

用法：PYTHONPATH=. python docs/work/evidence/T-030/primary63_register.py <data_root>
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
import eval_primary as ep  # noqa: E402

from quantmind.research.gpu.folds import calendar_and_grid, make_folds  # noqa: E402

P6 = HERE / "p6"
FROZEN_CODE = ["docs/work/evidence/T-030/eval_primary.py", "quantmind/research/gpu/ensemble.py",
               "quantmind/research/gpu/folds.py", "quantmind/research/gpu/metrics.py",
               "quantmind/research/gpu/configs.py", "quantmind/research/gpu/transforms.py",
               "quantmind/execution/nav_engine.py", "quantmind/backtest/wf_costs.py"]
EXTRA_INPUTS = ["data/bakeoff/preds_v6_1/ridge_63d_v6_1_C_infnan.parquet",
                "data/raw/alpha_prices_panel_v6_20261009.parquet", "data/panel/alpha_panel_weekly_v6_1.parquet"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    root = Path(sys.argv[1])
    out_json, out_rows = P6 / "primary63_registration.json", P6 / "frozen_eval_rows_63d.parquet"
    if (P6 / "primary_63d.json").exists():
        raise SystemExit("primary_63d.json already exists; refuse to register after the fact")
    if out_json.exists() or out_rows.exists():
        raise SystemExit("registration already written; refuse to overwrite")
    problems: list[str] = []

    g1 = json.loads((HERE / "g1_freeze_manifest_2026-10-17.json").read_text(encoding="utf-8"))["files"]
    reg = json.loads((P6 / "n1r2_registration.json").read_text(encoding="utf-8"))["sha256"]
    code = {p: sha(REPO / p) for p in FROZEN_CODE}
    if code["docs/work/evidence/T-030/eval_primary.py"] != g1["docs/work/evidence/T-030/eval_primary.py"]:
        problems.append("eval_primary.py differs from the G1 2026-10-17 freeze manifest")
    for p in ("quantmind/research/gpu/configs.py", "quantmind/research/gpu/metrics.py"):
        if code[p] != reg[p]:
            problems.append(f"{p} differs from the N1-R2 registration")
    diff = subprocess.run(["git", "diff", "--name-only", "t030-p7-2026-10-19", "HEAD", "--", *FROZEN_CODE],
                          cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *FROZEN_CODE],
                           cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    if diff or dirty:
        problems.append(f"frozen code changed since t030-p7-2026-10-19: {sorted(set(diff + dirty))}")

    p7 = json.loads((HERE / "p7/data_sha256.json").read_text(encoding="utf-8"))["files"]
    mism = []
    for e in p7:
        base = root if e["path"].startswith("data/") else REPO
        f = base / e["path"]
        if not f.exists() or sha(f) != e["sha256"]:
            mism.append(e["path"])
    if mism:
        problems.append(f"{len(mism)} P7 manifest files differ or are missing")
    extra = {p: sha(root / p) for p in EXTRA_INPUTS}

    cal, grid = calendar_and_grid(root)
    test = sorted({d for f in make_folds(cal, grid, "63d") for d in f.test_asof})
    meta = pd.concat([pd.read_parquet(p, columns=["as_of", "ticker", "y63", "top1500"])
                      for p in sorted((root / "data/gpu_v1/model/v1").glob("*.meta.parquet"))], ignore_index=True)
    meta["as_of"] = pd.to_datetime(meta["as_of"])
    meta = meta[meta["as_of"].isin(set(test))].set_index(["as_of", "ticker"])
    pool = meta[meta["top1500"]]
    ridge = pd.read_parquet(root / ep.BASE["63d"])
    ridge["as_of"] = pd.to_datetime(ridge["as_of"])
    cols = {"ridge": ridge.set_index(["as_of", "ticker"])["score"].reindex(pool.index)}
    for m in ep.ENS_MEMBERS:
        s = ep.load_family(root, m, "63d")
        if s is None:
            problems.append(f"ENS member {m} has no 63d predictions")
            continue
        cols[m] = s.reindex(pool.index)
    frame = pd.DataFrame(cols, index=pool.index)
    ok = frame.notna().all(axis=1) & np.isfinite(pool["y63"].to_numpy(dtype=float))
    rows = pool.index[ok.to_numpy()].to_frame(index=False)

    frozen_dates = json.loads((P6 / "frozen_dates_63d.json").read_text(encoding="utf-8"))
    rows.to_parquet(out_rows, index=False)
    res = {
        "what": "T-030 63d 主比较放行前登记（附件 B.3）：代码与数据冻结核对、可评估行冻结；不含任何 IC",
        "utc": datetime.now(UTC).isoformat(),
        "n1r2_result": {"file": "docs/work/evidence/T-030/p6/n1r2_seeds_63d.json",
                        "sha256": sha(P6 / "n1r2_seeds_63d.json"), "verdict": "N1-R2 补充复核通过"},
        "n2r_result": {"file": "docs/work/evidence/T-030/p6/n2_target_canary_63d.json",
                       "sha256": sha(P6 / "n2_target_canary_63d.json")},
        "frozen_code_sha256": code,
        "frozen_code_changed_since_p7_tag": sorted(set(diff + dirty)),
        "p7_manifest_files": len(p7), "p7_manifest_mismatches": mism,
        "extra_inputs_sha256": extra,
        "frozen_dates_63d_sha256": sha(P6 / "frozen_dates_63d.json"),
        "frozen_dates_63d_keys": list(frozen_dates)[:10],
        "n_test_asof": len(test),
        "eval_rows": {"file": "docs/work/evidence/T-030/p6/frozen_eval_rows_63d.parquet", "n_rows": int(len(rows)),
                      "n_dates": int(rows["as_of"].nunique()), "sha256": sha(out_rows)},
        "primary_63d_json_present": False,
        "problems": problems,
    }
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"完成 rows={len(rows)} dates={rows['as_of'].nunique()} problems={len(problems)} "
          f"rows_sha256={res['eval_rows']['sha256']} reg_sha256={sha(out_json)}")


if __name__ == "__main__":
    main()
