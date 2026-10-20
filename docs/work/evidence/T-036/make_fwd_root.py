"""T-036：搭前瞻影子数据根（不修改冻结的 T-030 构建器）.

主根 <fwd_root>/main/data/ 的布局与真实 data/ 相同：
- lake/：每个文件都是指向真实数据湖的符号链接；其中
    stock_basic_full_merged_20261009.parquet → stock_basic_full_merged_<END>.parquet
    events/<api>_20261009.parquet            → events/<api>_<END>.parquet（存在时；否则仍指向 20261009 版本）
- raw/alpha_prices_panel_v6_20261009.parquet → raw/alpha_prices_panel_v6_<END>.parquet
- loss_signals_v4/short_horizon_survivors.json → 真实文件
- gpu_v1/panel/{base,fnd,sector,event,a158}/<月>.parquet：早于 CUT 的月份链接到冻结文件（构建器会跳过），其余重算
- gpu_v1/panel/sh、gpu_v1/seq、gpu_v1/panel/market：空（整组重算）
- gpu_v1/model/v1/：features.json 与早于 CUT 的 <月>.X.npy / .meta.parquet 链接到冻结文件；market.parquet 不链接（重算）
Alpha158 用单独的根 <fwd_root>/a158/data/：
- raw/alpha_prices_panel_v6_20261009.parquet 是真实文件：新价格面板截取 A158_START 之后的部分（只用于构建器裁剪日历）
- qlib_cn_daily_v6_1 → qlib_cn_daily_fwd_<END>；bakeoff/ticker_qsym_map_v6_1.json → ticker_qsym_map_fwd_<END>.json
- gpu_v1/panel/a158/<月>：[A158_START, CUT) 的月份链接到冻结文件
所有链接写进 <fwd_root>/fwd_root_manifest.json（被重定向的输入附目标文件 sha256）。
用法：python make_fwd_root.py <data_root> <END_YYYYMMDD> <fwd_root> [--cut 2026-09] [--a158-start 2025-07-01]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

FROZEN_TAG = "20261009"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def link(dst: Path, target: Path, log: list, redirected: bool = False) -> None:
    if not target.exists():
        raise FileNotFoundError(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    os.symlink(target, dst)
    if redirected:
        log.append({"link": str(dst), "target": str(target), "sha256": sha(target)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=Path)
    ap.add_argument("end")
    ap.add_argument("fwd_root", type=Path)
    ap.add_argument("--cut", default="2026-09")
    ap.add_argument("--a158-start", default="2025-07-01")
    a = ap.parse_args()
    real = a.data_root / "data"
    if a.fwd_root.exists():
        raise SystemExit(f"refusing: {a.fwd_root} exists")
    main_d, a158_d = a.fwd_root / "main" / "data", a.fwd_root / "a158" / "data"
    redirected: list = []
    counts: dict = {}

    # ---- lake
    for f in sorted((real / "lake").iterdir()):
        if f.is_dir():
            continue
        if f.name == f"stock_basic_full_merged_{FROZEN_TAG}.parquet":
            link(main_d / "lake" / f.name, real / "lake" / f"stock_basic_full_merged_{a.end}.parquet", redirected, True)
        else:
            link(main_d / "lake" / f.name, f, [])
    for f in sorted((real / "lake/events").glob(f"*_{FROZEN_TAG}.parquet")):
        api = f.name[: -len(f"_{FROZEN_TAG}.parquet")]
        newer = real / "lake/events" / f"{api}_{a.end}.parquet"
        link(main_d / "lake/events" / f.name, newer if newer.exists() else f, redirected, True)
    # ---- raw prices, survivors list
    link(main_d / f"raw/alpha_prices_panel_v6_{FROZEN_TAG}.parquet",
         real / f"raw/alpha_prices_panel_v6_{a.end}.parquet", redirected, True)
    link(main_d / "loss_signals_v4/short_horizon_survivors.json",
         real / "loss_signals_v4/short_horizon_survivors.json", redirected, True)
    # ---- frozen panel months before CUT
    for g in ("base", "fnd", "sector", "event", "a158"):
        n = 0
        for f in sorted((real / "gpu_v1/panel" / g).glob("*.parquet")):
            if f.stem < a.cut:
                link(main_d / "gpu_v1/panel" / g / f.name, f, [])
                n += 1
        counts[f"panel/{g}"] = n
    for g in ("sh", "market"):
        (main_d / "gpu_v1/panel" / g).mkdir(parents=True, exist_ok=True)
    (main_d / "gpu_v1/seq").mkdir(parents=True, exist_ok=True)
    mv1 = real / "gpu_v1/model/v1"
    link(main_d / "gpu_v1/model/v1/features.json", mv1 / "features.json", redirected, True)
    n = 0
    for f in sorted(mv1.glob("*.meta.parquet")):
        m = f.name.split(".")[0]
        if m < a.cut:
            link(main_d / "gpu_v1/model/v1" / f.name, f, [])
            link(main_d / "gpu_v1/model/v1" / f"{m}.X.npy", mv1 / f"{m}.X.npy", [])
            n += 1
    counts["model/v1 months"] = n

    # ---- Alpha158 root
    px = pd.read_parquet(real / f"raw/alpha_prices_panel_v6_{a.end}.parquet")
    px = px[pd.to_datetime(px["trade_date"]) >= pd.Timestamp(a.a158_start)]
    (a158_d / "raw").mkdir(parents=True, exist_ok=True)
    trunc = a158_d / f"raw/alpha_prices_panel_v6_{FROZEN_TAG}.parquet"
    px.to_parquet(trunc, index=False)
    redirected.append({"file": str(trunc), "what": f"new price panel truncated to >= {a.a158_start}",
                       "rows": int(len(px)), "sha256": sha(trunc)})
    link(a158_d / "lake/trade_cal_sse.parquet", real / "lake/trade_cal_sse.parquet", [])
    link(a158_d / "qlib_cn_daily_v6_1", real / f"qlib_cn_daily_fwd_{a.end}", redirected, False)
    redirected.append({"link": str(a158_d / "qlib_cn_daily_v6_1"), "target": str(real / f"qlib_cn_daily_fwd_{a.end}")})
    link(a158_d / "bakeoff/ticker_qsym_map_v6_1.json", real / f"bakeoff/ticker_qsym_map_fwd_{a.end}.json", redirected, True)
    a_start_m = a.a158_start[:7]
    n = 0
    for f in sorted((real / "gpu_v1/panel/a158").glob("*.parquet")):
        if a_start_m <= f.stem < a.cut:
            link(a158_d / "gpu_v1/panel/a158" / f.name, f, [])
            n += 1
    counts["a158 root months"] = n
    (a158_d / "gpu_v1/panel/_a158_parts").mkdir(parents=True, exist_ok=True)

    man = {"end": a.end, "cut": a.cut, "a158_start": a.a158_start, "frozen_tag": FROZEN_TAG,
           "data_root": str(a.data_root), "redirected_inputs": redirected, "frozen_links": counts}
    (a.fwd_root / "fwd_root_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"fwd_root": str(a.fwd_root), "redirected": len(redirected), "frozen_links": counts},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
