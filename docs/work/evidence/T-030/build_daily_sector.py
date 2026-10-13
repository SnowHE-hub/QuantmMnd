"""T-030 P0：PIT 申万行业与 9 个行业特征在日频网格上的取值.

直接调用 docs/work/evidence/T-022/build_sector_features_pit.py 的 pit_industry / sector_features（逐 as_of 截面，无跨日依赖）；
输入换成日频面板：动量来自 data/gpu_v1/panel/base，基本面来自 data/gpu_v1/panel/fnd（reindex 到面板索引，与原脚本相同）。
输出 <data_root>/data/gpu_v1/panel/sector/<YYYY-MM>.parquet；已存在的月份跳过，从不覆盖。
用法：PYTHONPATH=. python build_daily_sector.py <data_root>
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pandas as pd

_ev = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("t022pit", _ev / "T-022/build_sector_features_pit.py")
t22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t22)


def main() -> None:
    root = Path(sys.argv[1])
    t0 = time.time()
    P = root / "data/gpu_v1/panel"
    out_dir = P / "sector"
    out_dir.mkdir(parents=True, exist_ok=True)
    sw = pd.read_parquet(root / "data/lake/events/index_member_all_20261009.parquet")
    months = sorted(p.stem for p in (P / "base").glob("*.parquet"))
    for m in months:
        out = out_dir / f"{m}.parquet"
        if out.exists():
            continue
        assert (P / "fnd" / f"{m}.parquet").exists(), f"fnd month {m} missing"
        mom = pd.read_parquet(P / "base" / f"{m}.parquet", columns=["momentum_1m", "momentum_3m", "momentum_6m"])
        fnd = pd.read_parquet(P / "fnd" / f"{m}.parquet",
                              columns=["roe_ttm", "pb", "pe_ttm", "operating_profit_yoy"]).reindex(mom.index)
        ind = t22.pit_industry(mom.index, sw)
        feat = pd.concat([ind, t22.sector_features(pd.concat([mom, fnd, ind], axis=1))], axis=1)
        tmp = out.with_suffix(".tmp")
        feat.to_parquet(tmp)
        os.replace(tmp, out)
        print(f"  {m} {feat.shape} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[sector] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
