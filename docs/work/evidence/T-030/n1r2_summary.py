"""T-030 N1-R2：固定历史面板上的一次性补充置换复核（G1 2026-10-19 收口裁定 §2.2；预登记附件 B）.

主判据只用置换种子 16–55（40 个，登记时都未揭示）：每个种子按 N1-R 原口径得到冻结日期等权的 T_s，
  T̄_new ± t(0.95, 39) · S_new / √40 的双侧 90% 区间 ⊂ (−0.005, +0.005) 且完整性检查通过 → "N1-R2 补充复核通过"。
S_new 用这 40 个种子的实际结果计算，判定用未四舍五入的数值。全部完成后只判读一次；不提前停止，不删种子、不改权重、
不截尾、不换区间；工程失败恢复同一任务，不用新种子替换。
种子 0–15 的结果并列报告；旧批、新批、全部种子的合并结果只作描述，不在"新批通过"与"合并通过"之间择优。
G1 写明：两轮等价判读合起来的错误放行概率并集上界是 0.10，不再宣称整体 5%。
盲态：结果只写文件，终端只打印完成、行数和 sha256（沿用 controls_diag._write(echo=False) / _finish）。
用法：PYTHONPATH=. python docs/work/evidence/T-030/n1r2_summary.py <data_root> <out.json>
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEW = list(range(16, 56))
OLD = list(range(0, 16))


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    spec = importlib.util.spec_from_file_location("controls_diag", HERE / "controls_diag.py")
    cd = importlib.util.module_from_spec(spec)
    sys.modules["controls_diag"] = cd
    spec.loader.exec_module(cd)
    frozen_raw = json.loads(cd.FROZEN_DATES.read_text(encoding="utf-8"))
    frozen = pd.DataFrame(frozen_raw["dates"])
    folds, recomputed = cd._evaluable_dates(root, "63d")
    problems: list[str] = []
    if recomputed["as_of"].tolist() != frozen["as_of"].tolist():
        problems.append("冻结日期与按标签成熟条件重算的日期不一致")
    daily: dict[int, np.ndarray] = {}
    per_seed = []
    for seed in OLD + NEW:
        pred = cd._pred_dir(root, seed)
        missing = [k for k in range(len(folds)) if not (pred / f"fold_{k}.parquet").exists()]
        if missing:
            problems.append(f"seed {seed} missing folds {missing}")
            per_seed.append({"shuffle_seed": seed, "defined": False, "missing_folds": missing})
            continue
        values = cd._seed_daily(root, seed, frozen).to_numpy(dtype=np.float64)
        defined = bool(np.isfinite(values).all()) and len(values) == len(frozen)
        if not defined:
            problems.append(f"seed {seed} has undefined IC or a changed date set")
        role = "已揭示诊断" if seed == 0 else ("N1-R 已揭示复核" if seed <= 15 else "N1-R2 未揭示复核")
        per_seed.append({"shuffle_seed": seed, "role": role, "defined": defined,
                         "T": None if not defined else float(values.mean()),
                         "ci95_block13": None if not defined else cd._block_ci(values, 13)})
        if defined:
            daily[seed] = values
    primary, verdict = None, "工程异常"
    if not problems and all(s in daily for s in NEW):
        primary = cd._t90(np.array([daily[s].mean() for s in NEW], dtype=np.float64))
        primary.update(seeds="16-55", delta=cd.DELTA_N1)
        verdict = cd._classify(primary["ci90"])
        if verdict == "通过":
            verdict = "N1-R2 补充复核通过"
    describe = {}
    for name, ss in (("seeds_1_15", list(range(1, 16))), ("seeds_0_15", OLD), ("seeds_0_55", OLD + NEW),
                     ("seeds_1_55", list(range(1, 56)))):
        if all(s in daily for s in ss):
            describe[name] = cd._t90(np.array([daily[s].mean() for s in ss], dtype=np.float64))
    cd._write(out, {
        "rule": "N1-R2：主判据只用种子 16–55 的日期等权 T_s，双侧 90% t 区间（df=39）是否落在 (-0.005, +0.005)。",
        "g1_ruling": "docs/work/reviews/T-030-G1-ruling-2026-10-19c.md",
        "family_error_note": "两轮等价判读（N1-R 与 N1-R2）的错误放行概率并集上界为 0.10；不宣称整体 5%。",
        "frozen_dates_sha256": cd._file_sha256(cd.FROZEN_DATES),
        "problems": problems, "verdict": verdict, "passed": verdict == "N1-R2 补充复核通过" and not problems,
        "primary": primary, "per_seed": per_seed,
        "descriptive_only_not_for_release": describe,
    }, echo=False)
    cd._finish(out, len(per_seed))


if __name__ == "__main__":
    main()
