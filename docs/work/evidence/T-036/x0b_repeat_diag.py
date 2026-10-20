"""T-036 X0-b 诊断（不是判定）：同一冻结输入、同一模型包，连续打分两次，比较各族各种子的原始预测，
看序列融合的小差异是不是 GPU 推理本身的非确定性。用法：python x0b_repeat_diag.py <data_root> <out.json>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def main() -> None:
    from quantmind.line_a.scoring import FAMS, score_ens20
    real, out = Path(sys.argv[1]), Path(sys.argv[2])
    days = [pd.Timestamp(d) for d in ("2026-09-01", "2026-09-15", "2026-10-09")]
    b = real / "data/gpu_v1/forward_models/20d_fold18"
    _, r1 = score_ens20(real, days, b)
    _, r2 = score_ens20(real, days, b)
    res = {fm: [float(np.nanmax(np.abs(a.astype(np.float64) - c.astype(np.float64)))) for a, c in zip(r1[fm], r2[fm], strict=True)]
           for fm in FAMS}
    out.write_text(json.dumps({"days": [str(d.date()) for d in days], "run1_vs_run2_max_abs_diff": res}, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
