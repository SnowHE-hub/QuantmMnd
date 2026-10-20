"""T-036：重叠期一致性检查（规则写在 docs/work/tasks/T-036.md，运行前提交）.

在 [2026-09-01, 2026-10-09] 的 (as_of, ticker) 上，逐组比较影子数据根的重算结果与冻结产物：
  panel/{base, fnd, sh, sector, event, a158}、seq（序列数组）、panel/market/market_daily、model/v1（X、meta、market）。
要求：行索引相同；数值列相等且 NaN 位置相同；字符列相等。只允许三类差异，逐类计数：
  1. 标签成熟：远期收益列（base 的 forward_return_*、meta 的 y20 / y63）冻结时为 NaN、现在有值；
  2. 迟到财报：基本面组（fnd 和依赖它的 sector），且该股票（或 sector 中同一申万一级行业里的某只股票）
     在 2026-10-19 的财报更新中新入湖了 ann_date ≤ 2026-10-09 的记录，as_of ≥ 该记录的 ann_date；
     model/v1 的 X 是按日截面 gauss-rank，原始值变一个，同日该列其他股票的秩也会变，
     所以 F2（fnd）/ F4（sector）列在"该日该列存在被解释的原始差异"时，该日该列的 X 差异视为被解释；
  3. 序列下标重排：meta.seq_idx 不同但指向同一只股票（新上市股票插入后下标重排）。
其余任何差异都判为不通过。输出 JSON；终端只打印每组的结论和计数。
用法：python overlap_check.py <data_root> <fwd_main_root> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W0, W1 = pd.Timestamp("2026-09-01"), pd.Timestamp("2026-10-09")
MONTHS = ["2026-09", "2026-10"]
FND_TABLES = ["income", "balancesheet", "cashflow", "fina_indicator_full"]
KEY = ["ts_code", "end_date", "ann_date"]


def late_fundamentals(lake: Path, backup: Path) -> dict[str, pd.Timestamp]:
    """2026-10-19 更新新入湖、ann_date ≤ 20261009 的记录：股票 → 最早的 ann_date。"""
    out: dict[str, pd.Timestamp] = {}
    for t in FND_TABLES:
        cur = pd.read_parquet(lake / f"{t}.parquet", columns=KEY).astype(str)
        old = pd.read_parquet(backup / f"{t}.parquet", columns=KEY).astype(str)
        new = cur.merge(old.drop_duplicates(), on=KEY, how="left", indicator=True)
        new = new[(new["_merge"] == "left_only") & (new["ann_date"] <= "20261009") & (new["ann_date"] >= "20000101")]
        for tk, d in new.groupby("ts_code")["ann_date"].min().items():
            ts = pd.Timestamp(d)
            out[tk] = min(out.get(tk, ts), ts)
    return out


def _window(df: pd.DataFrame) -> pd.DataFrame:
    d = df.index.get_level_values("as_of")
    return df[(d >= W0) & (d <= W1)].sort_index()


def read_group(root: Path, g: str) -> pd.DataFrame:
    return _window(pd.concat([pd.read_parquet(root / "data/gpu_v1/panel" / g / f"{m}.parquet") for m in MONTHS]))


def compare(fz: pd.DataFrame, sh: pd.DataFrame, *, label_cols=(), explain=None) -> dict:
    res = {"rows_frozen": int(len(fz)), "rows_shadow": int(len(sh)),
           "rows_missing_in_shadow": int(len(fz.index.difference(sh.index))),
           "rows_extra_in_shadow": int(len(sh.index.difference(fz.index))),
           "columns_equal": list(fz.columns) == list(sh.columns), "columns": {}}
    idx = fz.index.intersection(sh.index)
    a, b = fz.loc[idx], sh.loc[idx, fz.columns.intersection(sh.columns)]
    unexplained_total = 0
    for c in b.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xa, ya = x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64)
            nx, ny = np.isnan(xa), np.isnan(ya)
            mism = ~((xa == ya) | (nx & ny))
            allowed = np.zeros(len(xa), dtype=bool)
            if c in label_cols:
                allowed |= nx & ~ny
        else:
            mism = ~((x.astype(object) == y.astype(object)) | (x.isna() & y.isna())).to_numpy()
            allowed = np.zeros(len(x), dtype=bool)
        if explain is not None and mism.any():
            allowed |= explain(c, idx, mism)
        unexpl = mism & ~allowed
        n_m, n_u = int(mism.sum()), int(unexpl.sum())
        unexplained_total += n_u
        if n_m:
            ex = [list(map(str, idx[i])) for i in np.flatnonzero(unexpl)[:5]]
            res["columns"][c] = {"mismatch": n_m, "explained": n_m - n_u, "unexplained": n_u, "examples": ex}
    res["unexplained_total"] = unexplained_total
    res["pass"] = (unexplained_total == 0 and res["rows_missing_in_shadow"] == 0 and res["rows_extra_in_shadow"] == 0
                   and res["columns_equal"])
    return res


def main() -> None:
    real, shadow, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    late = late_fundamentals(real / "data/lake", real / "data/lake/_backup_20261019")
    late_ser = pd.Series(late, dtype="datetime64[ns]")
    report: dict = {"window": [str(W0.date()), str(W1.date())], "late_fundamental_tickers": len(late), "groups": {}}

    def fnd_explain(c, idx, mism):
        tk = idx.get_level_values("ticker")
        d = idx.get_level_values("as_of")
        first = late_ser.reindex(tk).to_numpy()
        return mism & ~pd.isna(first) & (d.to_numpy() >= first)

    sector_sh = read_group(shadow, "sector")
    def sector_explain(c, idx, mism):
        sw = sector_sh["sw_l1"].reindex(idx)
        tk = pd.Series(idx.get_level_values("ticker"), index=idx)
        first = late_ser.reindex(tk.to_numpy())
        flag = pd.Series(~pd.isna(first.to_numpy()) & (idx.get_level_values("as_of").to_numpy() >= first.to_numpy()),
                         index=idx)
        hit = flag.groupby([idx.get_level_values("as_of"), sw.to_numpy()]).transform("any").to_numpy()
        return mism & hit

    explained_raw: dict[str, set] = {}
    for g in ("base", "fnd", "sh", "sector", "event", "a158"):
        fz, sh = read_group(real, g), read_group(shadow, g)
        labels = [c for c in fz.columns if c.startswith("forward_return_")]
        ex = fnd_explain if g == "fnd" else (sector_explain if g == "sector" else None)
        r = compare(fz, sh, label_cols=labels, explain=ex)
        report["groups"][g] = r
        if g in ("fnd", "sector"):
            for c, v in r["columns"].items():
                if v["explained"]:
                    explained_raw.setdefault(c, set())
        if ex is not None:   # 记录被解释的 (日期, 列) 供 X 使用
            idx = fz.index.intersection(sh.index)
            for c in r["columns"]:
                x = fz.loc[idx, c].to_numpy(dtype=np.float64)
                y = sh.loc[idx, c].to_numpy(dtype=np.float64)
                m = ~((x == y) | (np.isnan(x) & np.isnan(y)))
                explained_raw.setdefault(c, set()).update(map(str, idx.get_level_values("as_of")[m].date))

    # ---- seq array (common tickers, days in window)
    fm, sm = (json.loads((p / "data/gpu_v1/seq/meta.json").read_text()) for p in (real, shadow))
    fa = np.load(real / "data/gpu_v1/seq/seq_raw_f32.npy", mmap_mode="r")
    sa = np.load(shadow / "data/gpu_v1/seq/seq_raw_f32.npy", mmap_mode="r")
    days = [d for d in fm["days"] if str(W0.date()) <= d <= str(W1.date())]
    fdi, sdi = [fm["days"].index(d) for d in days], [sm["days"].index(d) for d in days if d in sm["days"]]
    common = sorted(set(fm["tickers"]) & set(sm["tickers"]))
    fti, sti = [fm["tickers"].index(t) for t in common], [sm["tickers"].index(t) for t in common]
    seq_ok = fm["channels"] == sm["channels"] and len(fdi) == len(sdi)
    if seq_ok:
        x, y = fa[np.ix_(fti, fdi)], sa[np.ix_(sti, sdi)]
        mism = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        report["groups"]["seq"] = {"days": len(days), "tickers": len(common), "mismatch_cells": int(mism.sum()),
                                   "mismatch_by_channel": {fm["channels"][k]: int(mism[:, :, k].sum())
                                                           for k in range(len(fm["channels"])) if mism[:, :, k].any()},
                                   "pass": bool(not mism.any())}
    else:
        report["groups"]["seq"] = {"pass": False, "reason": "channels or days differ"}
    mk_f = pd.read_parquet(real / "data/gpu_v1/panel/market/market_daily.parquet")
    mk_s = pd.read_parquet(shadow / "data/gpu_v1/panel/market/market_daily.parquet")
    win = lambda m: m[(m.index >= W0) & (m.index <= W1)]  # noqa: E731
    report["groups"]["market_daily"] = compare(win(mk_f), win(mk_s))

    # ---- model/v1
    F = json.loads((real / "data/gpu_v1/model/v1/features.json").read_text())
    cols = F["stock_features"]
    src = {c: "F2" for c in F["groups"]["F2"]} | {c: "F4" for c in F["groups"]["F4"]}
    Xf, Xs, Mf, Ms = [], [], [], []
    for m in MONTHS:
        Xf.append(np.load(real / f"data/gpu_v1/model/v1/{m}.X.npy"))
        Xs.append(np.load(shadow / f"data/gpu_v1/model/v1/{m}.X.npy"))
        Mf.append(pd.read_parquet(real / f"data/gpu_v1/model/v1/{m}.meta.parquet"))
        Ms.append(pd.read_parquet(shadow / f"data/gpu_v1/model/v1/{m}.meta.parquet"))
    mf, ms = pd.concat(Mf, ignore_index=True), pd.concat(Ms, ignore_index=True)
    xf, xs = np.concatenate(Xf), np.concatenate(Xs)
    kf = pd.MultiIndex.from_arrays([pd.to_datetime(mf["as_of"]), mf["ticker"]])
    ks = pd.MultiIndex.from_arrays([pd.to_datetime(ms["as_of"]), ms["ticker"]])
    in_w = (kf.get_level_values(0) >= W0) & (kf.get_level_values(0) <= W1)
    pos_s = pd.Series(np.arange(len(ks)), index=ks)
    kfw = kf[in_w]
    have = kfw.isin(ks)
    fi = np.flatnonzero(in_w)[have]
    si = pos_s.reindex(kfw[have]).to_numpy()
    xa, xb = xf[fi].astype(np.float32), xs[si].astype(np.float32)
    mism = ~((xa == xb) | (np.isnan(xa) & np.isnan(xb)))
    dates = kf.get_level_values(0)[fi]
    x_cols = {}
    x_unexpl = 0
    for j, c in enumerate(cols):
        if not mism[:, j].any():
            continue
        ok_days = explained_raw.get(c, set()) if src.get(c) in ("F2", "F4") else set()
        allowed = np.isin(np.array([str(d.date()) for d in dates]), list(ok_days)) if ok_days else np.zeros(len(fi), bool)
        u = int((mism[:, j] & ~allowed).sum())
        x_unexpl += u
        x_cols[c] = {"mismatch": int(mism[:, j].sum()), "unexplained": u, "group": src.get(c, "other")}
    meta_res = {}
    for c in ("y20", "y63", "top1500", "sw_l1"):
        a, b = mf.iloc[fi][c].to_numpy(dtype=np.float64), ms.iloc[si][c].to_numpy(dtype=np.float64)
        mm = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        allowed = (np.isnan(a) & ~np.isnan(b)) if c in ("y20", "y63") else np.zeros(len(a), bool)
        meta_res[c] = {"mismatch": int(mm.sum()), "unexplained": int((mm & ~allowed).sum())}
    tf = np.array(fm["tickers"])[mf.iloc[fi]["seq_idx"].to_numpy()]
    ts = np.array(sm["tickers"])[ms.iloc[si]["seq_idx"].to_numpy()]
    meta_res["seq_idx_same_ticker"] = bool((tf == ts).all())
    mkf = pd.read_parquet(real / "data/gpu_v1/model/v1/market.parquet")
    mks = pd.read_parquet(shadow / "data/gpu_v1/model/v1/market.parquet")
    report["groups"]["model_v1"] = {
        "rows_frozen_window": int(in_w.sum()), "rows_matched": int(len(fi)), "rows_missing_in_shadow": int((~have).sum()),
        "rows_extra_in_shadow": int(((ks.get_level_values(0) >= W0) & (ks.get_level_values(0) <= W1)).sum() - len(fi)),
        "X_columns_with_mismatch": x_cols, "X_unexplained": x_unexpl, "meta": meta_res,
        "market": compare(win(mkf), win(mks)),
    }
    mv = report["groups"]["model_v1"]
    mv["pass"] = (x_unexpl == 0 and mv["rows_missing_in_shadow"] == 0 and mv["rows_extra_in_shadow"] == 0
                  and all(v["unexplained"] == 0 for k, v in meta_res.items() if isinstance(v, dict))
                  and meta_res["seq_idx_same_ticker"] and mv["market"]["pass"])
    report["pass"] = all(v.get("pass") for v in report["groups"].values())
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for g, v in report["groups"].items():
        print(f"{g:12s} pass={v.get('pass')} unexplained={v.get('unexplained_total', v.get('X_unexplained', v.get('mismatch_cells')))}")
    print("OVERALL", "PASS" if report["pass"] else "FAIL", "late_fundamental_tickers", len(late))


if __name__ == "__main__":
    main()
