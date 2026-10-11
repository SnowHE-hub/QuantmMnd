"""scripts/data_pipeline/verify_lake_update.py — 核验 lake 增量更新（T-015），只读.

对每个序列，以下任一项不满足即 FAILED（退出码 1）：
  1. 备份（更新前）中的每一行在新表中逐值相同（含 null；主键按 null 安全比较），列与 dtype 不变
  2. 主键无重复；新增行的字符串列里没有字面 'None' / 'nan' / 'NaN' / 'NaT'
  3. 按日序列：与 trade_cal_sse 相比，首日之后到 --end 没有缺失交易日，也没有非交易日、晚于 --end 的日期。
     **唯一豁免**是 north_bound（港股通休市日不披露）：缺失日必须逐日出现在 lake 更新运行报告
     （``data/lake/_checkpoints/update_latest_*.json``，非 plan 运行）的 north_bound ``empty`` 列表中；
     未被记录的缺失日照样失败。其他序列没有任何豁免
  4. 财报序列：ann_date 不晚于 --end；给了 ``--fund-checkpoint`` 时，检查点必须 ``summary.complete`` 为真、
     failed 与待重试为零
用法：python scripts/data_pipeline/verify_lake_update.py --backup data/lake/_backup_20261010 --end 2026-10-09
      [--data-root ...] [--series ...] [--fund-checkpoint data/lake/_checkpoints/fundamentals_update_20261009.v2.json]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

DAILY_KEYED = {"daily": ["ts_code", "trade_date"], "adj_factor": ["ts_code", "trade_date"],
               "daily_basic": ["ts_code", "trade_date"], "margin": ["ts_code", "trade_date"],
               "index_daily": ["ts_code", "trade_date"], "north_bound": ["trade_date"]}
FUND_KEYED = {t: ["ts_code", "end_date", "ann_date"]
              for t in ("fina_indicator", "fina_indicator_full", "income", "balancesheet", "cashflow")}
EXEMPT_FROM_RUN_REPORTS = {"north_bound"}
NULL_LITERALS = ("None", "nan", "NaN", "NaT")
_NULL = "\x00<null>"


def _key(df: pd.DataFrame, key: list[str]) -> pd.Series:
    parts = [df[c].astype(object).where(df[c].notna(), _NULL).astype(str) for c in key]
    out = parts[0]
    for p in parts[1:]:
        out = out + "\x1f" + p
    return out.reset_index(drop=True)


def old_rows_unchanged(old: pd.DataFrame, new: pd.DataFrame, key: list[str]) -> dict:
    """旧表每一行（按 null 安全主键）在新表中存在且逐列相同（null 对 null 视为相同）。"""
    out = {"old_rows": int(len(old)), "missing_in_new": 0, "value_mismatch": {},
           "column_change": sorted(set(old.columns) ^ set(new.columns)),
           "dtype_change": {c: [str(old[c].dtype), str(new[c].dtype)] for c in old.columns
                            if c in new.columns and str(old[c].dtype) != str(new[c].dtype)}}
    cols = [c for c in old.columns if c in new.columns]
    vals = [c for c in cols if c not in key]
    if not any(old[c].isna().any() or new[c].isna().any() for c in key):
        # 主键无 null（行情表，数百万行）：merge 路径，内存可控
        m = old[cols].merge(new[cols].drop_duplicates(key), on=key, how="left", suffixes=("_o", "_n"),
                            indicator=True)
        out["missing_in_new"] = int((m["_merge"] == "left_only").sum())
        m = m[m["_merge"] == "both"]
        pairs = {c: (m[f"{c}_o"], m[f"{c}_n"]) for c in vals}
    else:
        # 主键含 null（财报表）：null 安全的字符串主键
        old, new = old.reset_index(drop=True), new.reset_index(drop=True)
        ko, kn = _key(old, key), _key(new, key)
        first = ~kn.duplicated()
        n_first = new[first.values].set_index(kn[first].values)
        present = ko.isin(set(n_first.index))
        out["missing_in_new"] = int((~present).sum())
        o = old[present.values].set_index(ko[present].values)
        n = n_first.loc[o.index]
        pairs = {c: (o[c], n[c]) for c in vals}
    for c, (a, b) in pairs.items():
        try:
            eq = (a == b) if a.dtype == b.dtype else (a.astype(object) == b.astype(object))
        except TypeError:
            eq = a.astype(object) == b.astype(object)
        neq = ~((a.isna() & b.isna()) | eq)
        if int(neq.sum()):
            out["value_mismatch"][c] = int(neq.sum())
    return out


def recorded_empty_days(lake_dir: Path, series: str) -> set[str]:
    """lake 更新运行报告（非 plan）里该序列 API 返回空表的日期 = 有记录的不披露日。"""
    days: set[str] = set()
    for f in glob.glob(str(lake_dir / "_checkpoints" / "update_latest_*.json")):
        d = json.loads(Path(f).read_text())
        if d.get("plan"):
            continue
        for r in d.get("reports", []):
            if r.get("series") == series:
                days.update(r.get("empty", []))
    return days


def check_series(s: str, old: pd.DataFrame, new: pd.DataFrame, cal: pd.DatetimeIndex, end: str,
                 lake_dir: Path) -> tuple[dict, bool]:
    key = DAILY_KEYED.get(s) or FUND_KEYED[s]
    r = old_rows_unchanged(old, new, key)
    r["new_rows"] = int(len(new) - len(old))
    r["duplicate_keys"] = int(new.duplicated(key).sum())   # pandas 视 null 与 null 为重复
    # 字面空值：新表比旧表多出来的数量（旧行已由第 1 项保证逐值不变，多出的只能来自新增行）
    lit = {c: int(new[c].isin(NULL_LITERALS).sum()) - (int(old[c].isin(NULL_LITERALS).sum()) if c in old else 0)
           for c in new.columns if new[c].dtype == object}
    r["null_literals_in_new_rows"] = {c: n for c, n in lit.items() if n > 0}
    bad = bool(r["missing_in_new"] or r["value_mismatch"] or r["duplicate_keys"] or r["column_change"]
               or r["dtype_change"] or r["null_literals_in_new_rows"])
    if s in DAILY_KEYED:
        d = pd.DatetimeIndex(pd.to_datetime(new["trade_date"]).unique()).sort_values()
        lo = d.min()
        want = cal[(cal >= lo) & (cal <= pd.Timestamp(end))]
        r["first"], r["last"] = str(lo.date()), str(d.max().date())
        missing = sorted(x.strftime("%Y%m%d") for x in want.difference(d))
        r["non_trading_days"] = [x.strftime("%Y%m%d") for x in d.difference(cal)]
        r["after_end"] = int((d > pd.Timestamp(end)).sum())
        if s in EXEMPT_FROM_RUN_REPORTS:
            rec = recorded_empty_days(lake_dir, s)
            r["exempt_recorded_empty_days"] = sorted(set(missing) & rec)
            missing = sorted(set(missing) - rec)
        r["missing_trading_days"] = missing
        bad = bad or bool(missing or r["non_trading_days"] or r["after_end"])
    else:
        a = new["ann_date"].astype(object)
        r["ann_after_end"] = int((a.notna() & (a.astype(str) > end)).sum())
        bad = bad or bool(r["ann_after_end"])
    return r, not bad


def check_fund_checkpoint(path: Path) -> tuple[dict, bool]:
    ck = json.loads(path.read_text())
    s = ck.get("summary", {})
    r = {k: s.get(k) for k in ("target", "done", "partial", "no_data", "failed", "pending", "complete")}
    ok = bool(s.get("complete")) and not s.get("failed") and not s.get("pending")
    return r, ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", type=Path, required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--series", nargs="*", default=list(DAILY_KEYED) + list(FUND_KEYED))
    ap.add_argument("--fund-checkpoint", type=Path)
    a = ap.parse_args(argv)
    root = a.data_root.resolve()
    lake_dir = root / "data" / "lake"
    backup = a.backup if a.backup.is_absolute() else root / a.backup
    end = pd.Timestamp(a.end).strftime("%Y%m%d")
    cal = pd.DatetimeIndex(pd.read_parquet(lake_dir / "trade_cal_sse.parquet")["cal_date"])
    cal = cal[cal <= pd.Timestamp(end)]
    ok = True
    for s in a.series:
        new_p, old_p = lake_dir / f"{s}.parquet", backup / f"{s}.parquet"
        if not old_p.exists():
            print(s, json.dumps({"skipped": "no backup (series not updated)"}, ensure_ascii=False), flush=True)
            continue
        r, good = check_series(s, pd.read_parquet(old_p), pd.read_parquet(new_p), cal, end, lake_dir)
        ok = ok and good
        print(s, "OK" if good else "BAD", json.dumps(
            {k: (v if not isinstance(v, list) or len(v) <= 10 else v[:10] + [f"...{len(v)}"]) for k, v in r.items()},
            ensure_ascii=False), flush=True)
    if a.fund_checkpoint:
        p = a.fund_checkpoint if a.fund_checkpoint.is_absolute() else root / a.fund_checkpoint
        r, good = check_fund_checkpoint(p)
        ok = ok and good
        print("fund_checkpoint", "OK" if good else "BAD", json.dumps(r, ensure_ascii=False), flush=True)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
