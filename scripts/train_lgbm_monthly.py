#!/usr/bin/env python3
"""月度 LambdaRank 训练入口 —— 封装默认路径调用 ``train_lgbm_model.py``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cmd = [
        sys.executable,
        str(_ROOT / "scripts/train_lgbm_model.py"),
        "--train",
        str(_ROOT / "data/panel/monthly_train.parquet"),
        "--val",
        str(_ROOT / "data/panel/monthly_val.parquet"),
        "--test",
        str(_ROOT / "data/panel/monthly_test.parquet"),
        "--label",
        "forward_return_21d",
        "--feature-set",
        "all",
        "--n-estimators",
        "800",
        "--early-stopping-rounds",
        "50",
        "--split-meta",
        str(_ROOT / "data/panel/monthly_split_meta.json"),
        "--model-output",
        str(_ROOT / "models/lgbm_v3_monthly.pkl"),
        "--output-dir",
        str(_ROOT / "reports/monthly"),
        "--metrics-json",
        str(_ROOT / "reports/monthly/metrics_v3_monthly.json"),
        "--run-tag",
        "v3_monthly",
        *[arg for arg in sys.argv[1:] if arg],
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
