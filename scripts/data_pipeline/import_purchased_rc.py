"""scripts/data_pipeline/import_purchased_rc.py — 导入用户购买的 report_rc / research_report 数据包并按登记的规则验收（T-038）.

步骤：
1. 原样复制到 data/lake/events_purchased/xianyu_20261022/raw/，核对 sha256；
2. 按"全部列都是字符串"读 CSV（字段里有换行），再把数值列转成 float64（转不了的非空值另存 <列>__raw），
   写 parquet：report_rc.parquet、research_report.parquet；
3. 概况：行数、row_hash 唯一数、日期范围、每年行数、同 KEY 多行的情况、create_time 与 report_date 的关系；
4. report_rc 与官方 13 个窗口逐行比对（规范化规则与中转数据相同：fetch_report_rc_relay._canon）。
数据包只当数据读，不执行其中任何内容。
用法：python -I scripts/data_pipeline/import_purchased_rc.py --data-root <root> --src <windows 目录>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data_pipeline.fetch_report_rc_relay import KEY, NUMERIC, _canon  # noqa: E402

FILES = {"report_rc": ("tushare_report_rc.csv", "53ba1da9384bfdbeb93feb0025e5850d4fadba4d4ed1d6700200fab7d39315a1"),
         "research_report": ("tushare_research_report.csv",
                             "1550b21ab996aec990ab99c9f7cf2fa51873775f16cbf099bcc9c3211d752cc2")}


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read_strings(p: Path) -> pa.Table:
    with open(p, "rb") as f:
        head = f.readline().decode("utf-8-sig").strip()       # 表头是不带引号的逗号分隔；去掉 BOM
    cols = head.split(",")
    return pacsv.read_csv(p, read_options=pacsv.ReadOptions(column_names=cols, skip_rows=1, block_size=1 << 26),
                          parse_options=pacsv.ParseOptions(newlines_in_values=True),
                          convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in cols},
                                                               strings_can_be_null=False))


def _typed(t: pa.Table, numeric: tuple) -> tuple[pa.Table, dict]:
    bad = {}
    for c in numeric:
        if c not in t.column_names:
            continue
        s = t[c].to_pandas()
        v = pd.to_numeric(s.where(s.str.strip() != ""), errors="coerce")
        nb = s.notna() & (s.str.strip() != "") & v.isna()
        if nb.any():
            bad[c] = {"n": int(nb.sum()), "examples": s[nb].head(5).tolist()}
            t = t.append_column(f"{c}__raw", pa.array(s.where(nb), type=pa.string()))
        t = t.set_column(t.column_names.index(c), c, pa.array(v, type=pa.float64()))
    return t, bad


def profile_rc(df: pd.DataFrame) -> dict:
    y = df["report_date"].str[:4]
    dup_key = df.duplicated(KEY, keep=False)
    ct = pd.to_datetime(df["create_time"], errors="coerce")
    rd = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
    lag = (ct.dt.normalize() - rd).dt.days
    return {"rows": int(len(df)), "row_hash_unique": int(df["row_hash"].nunique()),
            "exact_dup_rows_excl_hash": int(df.drop(columns=["row_hash"]).duplicated().sum()),
            "report_date_min": df["report_date"].min(), "report_date_max": df["report_date"].max(),
            "tickers": int(df["ts_code"].nunique()), "by_year": y.value_counts().sort_index().to_dict(),
            "rows_sharing_KEY": int(dup_key.sum()),
            "create_time_null": int(ct.isna().sum()),
            "create_lag_days_quantiles": {str(q): float(lag.quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
            "create_before_report_date": int((lag < 0).sum())}


def validate(df: pd.DataFrame, official: Path) -> dict:
    res = {"windows": [], "column_diffs": {}}
    for f in sorted(official.glob("*.parquet")):
        s, e = f.stem.split("_")
        a = pd.read_parquet(f)
        b = df[(df["report_date"] >= s) & (df["report_date"] <= e)]
        cols = [c for c in a.columns if c in b.columns]
        ka, kb = _canon(a[cols]), _canon(b[cols])
        m = ka.drop_duplicates().merge(kb.drop_duplicates(), on=cols, how="outer", indicator=True)
        res["windows"].append({"window": f.stem, "official_rows": int(len(a)), "official_unique": int(len(ka.drop_duplicates())),
                               "pack_rows": int(len(b)), "both": int((m["_merge"] == "both").sum()),
                               "only_official": int((m["_merge"] == "left_only").sum()),
                               "only_pack": int((m["_merge"] == "right_only").sum())})
        j = ka.drop_duplicates(KEY).merge(kb.drop_duplicates(KEY), on=KEY, suffixes=("_o", "_p"))
        for c in cols:
            if c not in KEY:
                res["column_diffs"][c] = res["column_diffs"].get(c, 0) + int((j[f"{c}_o"] != j[f"{c}_p"]).sum())
    w = res["windows"]
    res["identical"] = bool(w) and all(x["only_official"] == 0 and x["only_pack"] == 0 for x in w)
    res["column_diffs"] = {c: n for c, n in res["column_diffs"].items() if n}
    res["totals"] = {k: sum(x[k] for x in w) for k in ("official_rows", "official_unique", "pack_rows", "both",
                                                      "only_official", "only_pack")}
    return res


def build_clean(out: Path, official: Path) -> dict:
    """清洗版（2026-10-22，看到验收结果之后定的规则，与原始判定一起记在 T-038）：
    - 官方已有的 13 个窗口（2018-10-01～2019-04-13）用官方数据替换数据包；
    - 其余时段用数据包；同一 KEY 有多行的整组标 version_conflict = True（数据包合并了不同时间的抓取，
      Tushare 后来改过的记录新旧版本都在），下游算特征时排除这些行；
    - 加 source 列（official / pack）；原始 parquet 不动。"""
    cols = [c for c in pq.read_schema(out / "report_rc.parquet").names if not c.endswith("__raw")]
    df = pd.read_parquet(out / "report_rc.parquet", columns=cols)
    off = pd.concat([pd.read_parquet(f) for f in sorted(official.glob("*.parquet"))], ignore_index=True)
    lo, hi = min(f.stem.split("_")[0] for f in official.glob("*.parquet")), max(f.stem.split("_")[1] for f in official.glob("*.parquet"))
    keep = df[(df["report_date"] < lo) | (df["report_date"] > hi)].assign(source="pack")
    o = off.assign(source="official")
    for c in o.columns:
        if c in NUMERIC:
            o[c] = pd.to_numeric(o[c], errors="coerce").astype("float64")
        elif c != "source":
            o[c] = o[c].astype("string").fillna("")
    clean = pd.concat([keep, o], ignore_index=True)
    k = clean[KEY].astype(str)
    clean["version_conflict"] = k.duplicated(keep=False)
    clean = clean.sort_values(["report_date", "ts_code", "org_name", "quarter"]).reset_index(drop=True)
    clean.to_parquet(out / "report_rc_clean.parquet", compression="zstd")
    info = {"rows": int(len(clean)), "from_official": int((clean["source"] == "official").sum()),
            "official_window": [lo, hi], "version_conflict_rows": int(clean["version_conflict"].sum()),
            "report_date_min": clean["report_date"].min(), "report_date_max": clean["report_date"].max()}
    (out / "_clean.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--clean-only", action="store_true", help="只生成清洗版（导入和验收已做过）")
    a = ap.parse_args()
    out = a.data_root / "data/lake/events_purchased/xianyu_20261022"
    if a.clean_only:
        print("CLEAN", json.dumps(build_clean(out, a.data_root / "data/lake/events/_parts/report_rc"), ensure_ascii=False))
        return 0
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rep = {"source": "third-party CSV pack bought by the user on Xianyu (resold Tushare data; provenance unverifiable)",
           "files": {}}
    for name, (fn, sha) in FILES.items():
        dst = raw / fn
        if not dst.exists():
            shutil.copy2(a.src / fn, dst)
        got = _sha_file(dst)
        if got != sha:
            raise SystemExit(f"{fn}: sha256 {got} does not match the registered {sha}")
        t = _read_strings(dst)
        t, bad = _typed(t, NUMERIC if name == "report_rc" else ())
        pq.write_table(t, out / f"{name}.parquet", compression="zstd")
        rep["files"][name] = {"csv": fn, "sha256": sha, "rows": t.num_rows, "columns": t.column_names,
                              "non_numeric_values": bad}
        print(name, t.num_rows, "rows", flush=True)
        del t
    (out / "_provenance.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = [c for c in pq.read_schema(out / "report_rc.parquet").names if not c.endswith("__raw")]
    df = pd.read_parquet(out / "report_rc.parquet", columns=cols)
    prof = profile_rc(df)
    print("PROFILE", json.dumps(prof, ensure_ascii=False), flush=True)
    val = validate(df, a.data_root / "data/lake/events/_parts/report_rc")
    (out / "_validation_official.json").write_text(json.dumps({"profile": prof, **val}, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    print("VALIDATION", json.dumps({k: val[k] for k in ("identical", "totals", "column_diffs")}, ensure_ascii=False))
    rr = pd.read_parquet(out / "research_report.parquet", columns=["trade_date", "ts_code", "row_hash"])
    print("RESEARCH", json.dumps({"rows": int(len(rr)), "row_hash_unique": int(rr["row_hash"].nunique()),
                                  "date_min": rr["trade_date"].min(), "date_max": rr["trade_date"].max(),
                                  "with_ts_code": int((rr["ts_code"] != "").sum()),
                                  "by_year": rr["trade_date"].str[:4].value_counts().sort_index().to_dict()},
                                 ensure_ascii=False))
    return 0 if val["identical"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
