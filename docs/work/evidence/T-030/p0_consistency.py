"""T-030 P0：日频构建器在 v6_1 的 374 个周频截面上与现有 v6_1 文件逐值一致（设计稿 §3.2）.

判据（每组、每列）：两边行集合完全相同（按参照文件的 (as_of, ticker) 索引对齐后，日频侧不得缺行）；
NaN 位置相同；±inf 位置与符号相同；有限值 |a − b| ≤ 1e-9 · max(1, |b|)；字符串列逐值相等。另报逐位相等的个数。
组：base（35 因子 + 12/21/63d 标签，对 alpha_panel_weekly_v6_1_base）、labels（20/40d，对 labels_extra_v6_1）、
sh（对 short_horizon_factors_v6_1）、fnd（对 fundamental_factors_v6_1）、a158（对 alpha158_asof_v6_1，按其自身索引）、
sector（对 sector_features_v6_1_pit）、event（对 event_factors_v6_1）。只读，结果写 <out.json>。
用法：PYTHONPATH=. python p0_consistency.py <data_root> <out.json> [group ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

root, out = Path(sys.argv[1]), Path(sys.argv[2])
groups = sys.argv[3:] or ["base", "labels", "sh", "fnd", "a158", "sector", "event"]
D = root / "data"
G = D / "gpu_v1/panel"
REF = {"base": (D / "panel/alpha_panel_weekly_v6_1_base.parquet", "base"),
       "labels": (D / "panel/labels_extra_v6_1.parquet", "base"),
       "sh": (D / "panel/short_horizon_factors_v6_1.parquet", "sh"),
       "fnd": (D / "panel/fundamental_factors_v6_1.parquet", "fnd"),
       "a158": (D / "bakeoff/alpha158_asof_v6_1.parquet", "a158"),
       "sector": (D / "panel/sector_features_v6_1_pit.parquet", "sector"),
       "event": (D / "panel/event_factors_v6_1.parquet", "event")}
TOL = 1e-9
PARTIAL = os.environ.get("T030_PARTIAL") == "1"


def compare_col(a: pd.Series, b: pd.Series) -> dict:
    if b.dtype == object or str(b.dtype) in ("string", "category"):
        av, bv = a.astype(object), b.astype(object)
        na_a, na_b = av.isna(), bv.isna()
        bad = int(((av != bv) & ~(na_a & na_b)).sum())
        return {"kind": "str", "n": int(len(b)), "nan_mismatch": int((na_a != na_b).sum()), "value_mismatch": bad,
                "ok": bad == 0 and bool((na_a == na_b).all())}
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    na_a, na_b = np.isnan(av), np.isnan(bv)
    inf_a, inf_b = np.isinf(av), np.isinf(bv)
    fin = np.isfinite(av) & np.isfinite(bv)
    err = np.abs(av[fin] - bv[fin]) / np.maximum(1.0, np.abs(bv[fin]))
    inf_ok = bool(np.array_equal(inf_a, inf_b) and np.array_equal(av[inf_a], bv[inf_b]))
    r = {"kind": "num", "n": int(len(b)), "n_finite": int(fin.sum()), "nan_mismatch": int((na_a != na_b).sum()),
         "inf_ok": inf_ok, "max_scaled_err": float(err.max()) if len(err) else 0.0,
         "n_exact": int((av[fin] == bv[fin]).sum())}
    r["ok"] = r["nan_mismatch"] == 0 and inf_ok and r["max_scaled_err"] <= TOL
    return r


def main() -> None:
    res = json.loads(out.read_text()) if out.exists() else {}
    for g in groups:
        t0 = time.time()
        ref_path, sub = REF[g]
        ref = pd.read_parquet(ref_path)
        ref.index = ref.index.set_levels(pd.to_datetime(ref.index.levels[0]), level="as_of")
        dates = sorted(ref.index.get_level_values("as_of").unique())
        files = sorted(str(p) for p in (G / sub).glob("*.parquet"))
        if not files:
            res[g] = {"ok": False, "error": f"no daily files in {G / sub}"}
            continue
        t = ds.dataset(files, format="parquet").to_table(
            filter=ds.field("as_of").isin(pd.DatetimeIndex(dates).as_unit("ns").to_numpy()),
            columns=["as_of", "ticker", *ref.columns]).to_pandas()
        if "as_of" not in t.columns:                  # pandas 元数据会把索引列还原成索引
            t = t.reset_index()
        new = t.set_index(["as_of", "ticker"]).sort_index()
        if g == "sh":                                 # 日频 sh 按全部有行情的票取样；v6_1 参照与训练矩阵都只用面板行
            bt = ds.dataset(sorted(str(p) for p in (G / "base").glob("*.parquet")), format="parquet").to_table(
                filter=ds.field("as_of").isin(pd.DatetimeIndex(dates).as_unit("ns").to_numpy()),
                columns=["as_of", "ticker"]).to_pandas()
            if "as_of" not in bt.columns:
                bt = bt.reset_index()
            panel_idx = pd.MultiIndex.from_frame(bt[["as_of", "ticker"]])
            sh_rows_all = int(len(new))
            new = new.loc[new.index.isin(panel_idx)]
        if PARTIAL:                                   # 仅早期自检：只比日频已建成月份覆盖的日期，结果不可作为 P0 结论
            have = set(new.index.get_level_values("as_of").unique())
            ref = ref.loc[ref.index.get_level_values("as_of").isin(have)]
            dates = sorted(have & set(dates))
        miss = ~ref.index.isin(new.index)
        missing_rows = int(miss.sum())
        missing_nonnull = int(ref.loc[miss].notna().any(axis=1).sum())   # fnd 参照被 reindex 到面板：缺行若全 NaN 等价
        new_rows_at_weekly = int(len(new))
        newr = new.reindex(ref.index)
        cols = {c: compare_col(newr[c], ref[c]) for c in ref.columns}
        r = {"ref": str(ref_path), "n_ref_rows": int(len(ref)), "n_dates": len(dates),
             "daily_rows_on_weekly_dates": new_rows_at_weekly, "ref_rows_missing_in_daily": missing_rows,
             "ref_rows_missing_nonnull": missing_nonnull,
             "n_cols": len(cols), "cols_failed": [c for c, v in cols.items() if not v["ok"]],
             "n_exact_total": int(sum(v.get("n_exact", 0) for v in cols.values())),
             "n_finite_total": int(sum(v.get("n_finite", 0) for v in cols.values())),
             "max_scaled_err": max([v.get("max_scaled_err", 0.0) for v in cols.values()]), "cols": cols,
             "seconds": round(time.time() - t0, 1)}
        if g == "sh":
            r["daily_rows_all_priced_tickers"] = sh_rows_all
            r["row_rule"] = "daily sh restricted to daily base panel rows before comparison"
        if g in ("base", "labels", "sh", "sector", "event"):     # 这些组的行 = 面板 PIT 票池，日频在周频日上的行数须完全相同
            r["row_set_equal"] = bool(missing_rows == 0 and new_rows_at_weekly == len(ref))
        r["ok"] = bool(missing_nonnull == 0 and not r["cols_failed"] and r.get("row_set_equal", True))
        res[g] = r
        print(g, {k: v for k, v in r.items() if k != "cols"}, flush=True)
        del ref, new, newr, t
    res["all_ok"] = all(res.get(g, {}).get("ok", False) for g in REF) and not PARTIAL
    res["partial"] = PARTIAL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
