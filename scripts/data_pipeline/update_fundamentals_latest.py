"""scripts/data_pipeline/update_fundamentals_latest.py — 财报类 lake 表增量补到最新（T-015）.

lake 中的 fina_indicator / fina_indicator_full / income / balancesheet / cashflow 由
scripts/survivorship/p63_1*.py 按 ts_code 全量拉取，止于 2026-05/06。本脚本按同样字段、同样主键
(ts_code, end_date, ann_date) 增量补入 ``--since``～``--end`` 公告的记录：

- 股票池：先刷新 stock_basic（SH+SZ，L+D+P），另存 ``data/lake/stock_basic_full_<END>.parquet``（不覆盖原文件）；
  只拉在市（L/P）或 ``--since`` 之后才退市的股票
- **旧表一行不动**：新数据先按旧表的列与 dtype 对齐，主键（null 安全比较）已在旧表中的新行直接丢弃，
  其余追加在旧表之后；写入前断言旧表部分与读入时逐值相同（含 null）。不对旧表做任何类型转换
- ann_date 晚于 ``--end`` 的记录不写入（固定数据截止日，便于复现）
- fina_indicator 是 fina_indicator_full 的字段子集：一次请求同时补两张表
- 每只股票的终态（检查点 ``data/lake/_checkpoints/fundamentals_update_<END>.v2.json``）：
    done     四个接口都返回 ≥1 行
    partial  部分接口持续为空（尝试 ``--empty-retries`` 次后仍为空），非空部分已写入，空接口逐只记录
    no_data  四个接口都持续为空（同上），记录每次尝试的时间与上市状态，供人工复核
    failed   请求异常；不算完成，本次运行末尾与下次运行都会重试
  空返回在达到尝试次数前**不算完成**；同一次运行的末尾会对它们再试（间隔 ``--retry-wait`` 秒）
- 旧版检查点（v1，只有 done 列表、不区分空返回）不被信任：其中的股票全部重新拉取（合并幂等）
- 每张表第一次写入前备份到 ``data/lake/_backup_<YYYYMMDD>/``
- ``--repair-from-backup DIR``：用备份中的旧表 + 当前表中主键不在备份里的行重建（修复旧行被改的情况），
  被替换的当前表复制到 ``data/lake/_quarantine_<YYYYMMDD>/``，不删除；``--plan`` 只报告不写

token 走 backfill_tushare 同一套（api_key.txt，官方端点），loguru 全部移除，不回显不落盘。

用法：python scripts/data_pipeline/update_fundamentals_latest.py --end 2026-10-09 --since 2026-01-01 [--workers 3] [--data-root ...]
      python scripts/data_pipeline/update_fundamentals_latest.py --end 2026-10-09 --repair-from-backup data/lake/_backup_20261010 [--plan]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

KEY = ["ts_code", "end_date", "ann_date"]
FINA_FULL = ("ts_code,ann_date,end_date,roe,roa,roe_yearly,roa_yearly,grossprofit_margin,netprofit_margin,"
             "assets_turn,current_ratio,quick_ratio,debt_to_assets,assets_to_eqt,ebit_ps,bps,eps,ocfps,fcff,"
             "or_yoy,tr_yoy,netprofit_yoy,dt_netprofit_yoy,roe_yoy,op_yoy,q_sales_yoy,q_op_qoq,q_gr_yoy,q_profit_yoy")
# api → (lake 表, 字段)
SPEC = {
    "fina_indicator": ("fina_indicator_full", FINA_FULL),
    "income": ("income", "ts_code,ann_date,end_date,total_revenue,revenue,n_income,n_income_attr_p,operate_profit"),
    "balancesheet": ("balancesheet", "ts_code,ann_date,end_date,total_assets,total_liab,"
                                     "total_hldr_eqy_exc_min_int,total_cur_assets,total_cur_liab"),
    "cashflow": ("cashflow", "ts_code,ann_date,end_date,n_cashflow_act,c_paid_invest,"
                             "c_pay_acq_const_fiolta,free_cashflow"),
}
DERIVED = {"fina_indicator": "fina_indicator_full"}   # 子集表 ← 来源表
TABLES = [t for t, _ in SPEC.values()] + list(DERIVED)
NULL_LITERALS = ("None", "nan", "NaN", "NaT")
_NULL = "\x00<null>"


def _is_sh_sz(ts_code: str) -> bool:
    s = str(ts_code)
    if s.endswith(".BJ"):
        return False
    code = s.split(".")[0]
    return code[:1] in ("0", "3", "6") and not code.startswith(("8", "4", "92"))


def key_strings(df: pd.DataFrame) -> pd.Series:
    """主键的 null 安全字符串表示：null 与任何字符串（含字面 'None'）都不相等，null 与 null 相等。"""
    parts = [df[c].astype(object).where(df[c].notna(), _NULL).astype(str) for c in KEY]
    out = parts[0]
    for p in parts[1:]:
        out = out + "\x1f" + p
    return out


def align_new(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """把新数据对齐到旧表的列顺序与 dtype；主键列非空值转 str、null 保持 None。旧表不做任何改动。"""
    for c in KEY:
        if old[c].dtype != object:
            raise TypeError(f"旧表主键列 {c} 的 dtype 是 {old[c].dtype}，期望 object（字符串）")
    new = new.reindex(columns=old.columns).copy()
    for c in old.columns:
        if c in KEY:
            s = new[c].astype(object)
            new[c] = s.where(s.isna(), s.astype(str)).where(s.notna(), None)
        elif new[c].dtype != old[c].dtype:
            new[c] = new[c].astype(old[c].dtype)
    return new


def _append_new(old: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """旧表在前原样保留 + 主键不在旧表、且彼此不重复的新行；断言旧表部分逐值不变。"""
    k = key_strings(add)
    add = add[(~k.isin(set(key_strings(old))) & ~k.duplicated()).values]
    if add.empty:
        return old
    merged = pd.concat([old, add], ignore_index=True)
    for c in old.columns:   # concat 不得改变旧表 dtype
        if merged[c].dtype != old[c].dtype:
            merged[c] = merged[c].astype(old[c].dtype)
    pd.testing.assert_frame_equal(merged.iloc[: len(old)].reset_index(drop=True), old.reset_index(drop=True),
                                  check_exact=True)
    return merged


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def merge_keep_old(path: Path, new: pd.DataFrame | list[pd.DataFrame], backup_dir: Path) -> tuple[int, int]:
    """旧表原样保留，只追加主键（null 安全）不在旧表中的新行。返回 (新增行, 总行)."""
    old = pd.read_parquet(path)
    parts = [new] if isinstance(new, pd.DataFrame) else list(new)
    parts = [align_new(p, old) for p in parts if p is not None and len(p)]
    if not parts:
        return 0, len(old)
    merged = _append_new(old, pd.concat(parts, ignore_index=True))
    added = len(merged) - len(old)
    if not added:
        return 0, len(old)
    if not (backup_dir / path.name).exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)
    _atomic_write(merged, path)
    return added, len(merged)


def clip_ann(df: pd.DataFrame | None, end: str) -> pd.DataFrame:
    """去掉 ann_date 晚于截止日的记录（ann_date 为空的保留，与旧表一致）。"""
    if df is None or not len(df):
        return pd.DataFrame()
    a = df["ann_date"].astype(object)
    late = a.notna() & (a.astype(str) > end)
    return df[~late.values]


# ---------------------------------------------------------------- 状态机
def load_checkpoint(ck_path: Path, legacy_path: Path) -> dict:
    if ck_path.exists():
        return json.loads(ck_path.read_text())
    ck = {"version": 2, "done": {}, "partial": {}, "no_data": {}, "empty_attempts": {}, "failed": {}}
    if legacy_path.exists():   # v1 检查点不区分空返回：只记数量，不信任，全部重拉
        ck["legacy_v1_done_ignored"] = len(json.loads(legacy_path.read_text()).get("done", []))
    return ck


def classify(ck: dict, ts: str, counts: dict[str, int] | None, err: str | None, *,
             empty_retries: int, info: dict, now: str) -> str:
    """根据一次拉取结果更新检查点，返回该股票当前状态。counts = 每个接口（截止日内）返回的行数。"""
    for k in ("done", "partial", "no_data", "failed"):
        ck[k].pop(ts, None)
    if err is not None:
        ck["failed"][ts] = {"error": err[:160], "at": now}
        return "failed"
    if all(n > 0 for n in counts.values()):
        ck["empty_attempts"].pop(ts, None)
        ck["done"][ts] = counts
        return "done"
    att = ck["empty_attempts"].setdefault(ts, [])
    att.append({"at": now, "counts": counts})
    if len(att) < empty_retries:
        return "pending"
    empty_apis = sorted(a for a, n in counts.items() if n == 0)
    if len(empty_apis) == len(counts):
        ck["no_data"][ts] = {"attempts": att, **info}
        return "no_data"
    ck["partial"][ts] = {"empty_apis": empty_apis, "counts": counts, "attempts": att, **info}
    return "partial"


def terminal(ck: dict) -> set[str]:
    return set(ck["done"]) | set(ck["partial"]) | set(ck["no_data"])


# ---------------------------------------------------------------- 修复
def repair_from_backup(lake_dir: Path, backup: Path, end: str, *, plan: bool) -> dict:
    """备份旧表 + 当前表中主键不在备份里的行（主键里的字面 'None' 等视为 null）→ 重建；当前表复制进隔离目录。"""
    quarantine = lake_dir / f"_quarantine_{datetime.now():%Y%m%d}"
    out = {}
    for t in TABLES:
        cur_p, old_p = lake_dir / f"{t}.parquet", backup / f"{t}.parquet"
        if not old_p.exists():
            out[t] = {"skipped": "no backup"}
            continue
        old, cur = pd.read_parquet(old_p), pd.read_parquet(cur_p)
        fixed = cur.copy()
        lit = {c: int(fixed[c].isin(NULL_LITERALS).sum()) for c in KEY}
        for c in KEY:
            fixed[c] = fixed[c].where(~fixed[c].isin(NULL_LITERALS), None)
        extra = fixed[~key_strings(fixed).isin(set(key_strings(old))).values]
        kept = clip_ann(extra, end)
        rebuilt = _append_new(old, align_new(kept, old)) if len(kept) else old
        same = len(rebuilt) == len(cur) and rebuilt.equals(cur)
        out[t] = {"backup_rows": len(old), "current_rows": len(cur), "rebuilt_rows": len(rebuilt),
                  "null_literals_in_current_keys": lit, "dropped_ann_after_end": int(len(extra) - len(kept)),
                  "unchanged": bool(same)}
        if plan or same:
            continue
        quarantine.mkdir(parents=True, exist_ok=True)
        if not (quarantine / cur_p.name).exists():
            shutil.copy2(cur_p, quarantine / cur_p.name)
        _atomic_write(rebuilt, cur_p)
        out[t]["quarantined"] = str(quarantine / cur_p.name)
    return out


# ---------------------------------------------------------------- 主流程
def target_tickers(tp, lake_dir: Path, since: str, tag: str) -> tuple[list[str], pd.DataFrame]:
    sb_path = lake_dir / f"stock_basic_full_{tag}.parquet"
    if sb_path.exists():
        sb = pd.read_parquet(sb_path)
    else:
        sb = tp._raw_stock_basic_all()
        sb["ts_code"] = sb["ts_code"].astype(str)
        sb = sb[sb["ts_code"].map(_is_sh_sz)].reset_index(drop=True)
        sb.to_parquet(sb_path, index=False)
    alive = (sb["list_status"].isin(["L", "P"])) | (sb["delist_date"].fillna("").astype(str) >= since)
    return sorted(sb.loc[alive, "ts_code"].unique()), sb


def run(tp, lake_dir: Path, *, since: str, end: str, tag: str, workers: int = 1, flush_every: int = 200,
        empty_retries: int = 2, retry_wait: float = 120.0, max_tickers: int = 0,
        tickers: list[str] | None = None, ck_tag: str | None = None) -> int:
    """tickers 给定时（三队列模式）只拉这些股票，股票信息取最新的 stock_basic_full_<日期>；检查点用 ck_tag 命名。"""
    if tickers is None:
        tickers, sb = target_tickers(tp, lake_dir, since, tag)
    else:
        sb = pd.read_parquet(sorted(lake_dir.glob("stock_basic_full_2*.parquet"))[-1])
    if max_tickers:
        tickers = tickers[:max_tickers]
    sb_info = sb.drop_duplicates("ts_code").set_index("ts_code")
    sb_info = sb_info[[c for c in ("list_status", "list_date", "delist_date") if c in sb_info.columns]]
    ck_dir = lake_dir / "_checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck_path = ck_dir / f"fundamentals_update_{ck_tag or tag}.v2.json"
    ck = load_checkpoint(ck_path, ck_dir / f"fundamentals_update_{ck_tag or tag}.json")
    ck.update({"since": since, "end": end, "target": len(tickers), "empty_retries": empty_retries})
    backup_dir = lake_dir / f"_backup_{datetime.now():%Y%m%d}"
    buf: dict[str, list[pd.DataFrame]] = {t: [] for t, _ in SPEC.values()}
    added_total = dict.fromkeys(TABLES, 0)
    t0 = time.time()
    print(f"[fundamentals] 目标 {len(tickers)} 只；公告期 {since}–{end}；已终态 {len(terminal(ck) & set(tickers))}；"
          f"旧版检查点忽略 {ck.get('legacy_v1_done_ignored', 0)} 只", flush=True)

    def fetch(ts: str):
        parts: dict[str, list[pd.DataFrame]] = {}
        counts = {}
        for api, (table, fields) in SPEC.items():
            try:
                df = tp._call(api, ts_code=ts, start_date=since, end_date=end, fields=fields)
            except Exception as e:  # noqa: BLE001
                return ts, None, repr(e), {}
            df = clip_ann(df, end)
            counts[api] = int(len(df))
            if len(df):
                parts.setdefault(table, []).append(df)
        return ts, counts, None, parts

    def flush() -> None:
        for table, parts in buf.items():
            if not parts:
                continue
            added, _ = merge_keep_old(lake_dir / f"{table}.parquet", parts, backup_dir)
            added_total[table] += added
            for sub, src in DERIVED.items():
                if src == table:
                    sub_cols = list(pd.read_parquet(lake_dir / f"{sub}.parquet").columns)
                    a2, _ = merge_keep_old(lake_dir / f"{sub}.parquet", [p.reindex(columns=sub_cols) for p in parts],
                                           backup_dir)
                    added_total[sub] += a2
            parts.clear()
        ck["updated"] = datetime.now().isoformat(timespec="seconds")
        ck["added_rows_this_run"] = dict(added_total)
        tmp = ck_path.with_name(ck_path.name + ".tmp")
        tmp.write_text(json.dumps(ck, ensure_ascii=False))
        tmp.replace(ck_path)

    def one_pass(todo: list[str], label: str) -> dict[str, int]:
        stat: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = [ex.submit(fetch, ts) for ts in todo]
            for i, f in enumerate(as_completed(futs), 1):
                ts, counts, err, parts = f.result()
                for table, dfs in parts.items():
                    buf[table].extend(dfs)
                info = ({k: (None if pd.isna(v) else str(v)) for k, v in sb_info.loc[ts].items()}
                        if ts in sb_info.index else {})
                s = classify(ck, ts, counts, err, empty_retries=empty_retries, info=info,
                             now=datetime.now().isoformat(timespec="seconds"))
                stat[s] = stat.get(s, 0) + 1
                if i % flush_every == 0 or i == len(todo):
                    flush()
                    print(f"  [{label}] {i}/{len(todo)} {stat} 新增行={added_total} ({time.time() - t0:.0f}s)",
                          flush=True)
        return stat

    todo = [t for t in tickers if t not in terminal(ck)]
    if todo:
        one_pass(todo, "main")
    for r in range(1, empty_retries + 2):   # 空返回 / 异常的重试轮
        todo = [t for t in tickers if t not in terminal(ck)]
        if not todo:
            break
        print(f"[fundamentals] 重试轮 {r}：{len(todo)} 只（等待 {retry_wait:.0f}s）", flush=True)
        time.sleep(retry_wait)
        one_pass(todo, f"retry{r}")
    tset = set(tickers)
    pend = sorted(tset - terminal(ck))
    ck["summary"] = {"target": len(tickers), "done": len(set(ck["done"]) & tset),
                     "partial": len(set(ck["partial"]) & tset), "no_data": len(set(ck["no_data"]) & tset),
                     "failed": len(set(ck["failed"]) & tset), "pending": len(pend),
                     "complete": bool(tickers) and not pend, "added_rows_this_run": dict(added_total)}
    flush()
    print(f"[fundamentals] 结束：{ck['summary']}", flush=True)
    return 0 if ck["summary"]["complete"] else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--end", required=True)
    ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--data-root", type=Path, default=REPO)
    ap.add_argument("--rate-interval", type=float, default=0.5)
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--empty-retries", type=int, default=2, help="空返回第几次才认定 no_data / partial")
    ap.add_argument("--retry-wait", type=float, default=120.0)
    ap.add_argument("--max-tickers", type=int, default=0, help="只处理前 N 只（试跑）")
    ap.add_argument("--queues-since", help="三队列模式：触发事件窗口的起点（不含），窗口终点 = --end")
    ap.add_argument("--recon-buckets", type=int, default=20)
    ap.add_argument("--repair-from-backup", type=Path)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    root = a.data_root.resolve()
    lake_dir = root / "data" / "lake"
    tag = pd.Timestamp(a.end).strftime("%Y%m%d")
    since, end = pd.Timestamp(a.since).strftime("%Y%m%d"), tag

    if a.repair_from_backup:
        bk = a.repair_from_backup if a.repair_from_backup.is_absolute() else root / a.repair_from_backup
        print(json.dumps(repair_from_backup(lake_dir, bk, end, plan=a.plan), ensure_ascii=False, indent=1))
        return 0

    import scripts.backfill_tushare as bf  # 只在联网路径导入
    bf.REPO_ROOT = str(root)
    bf._load_token_into_env()
    from quantmind.data import tushare_provider as tp
    try:
        from loguru import logger as _lg
        _lg.remove()
    except Exception:  # noqa: BLE001
        pass
    tp._MIN_INTERVAL = max(tp._MIN_INTERVAL, a.rate_interval)
    if a.queues_since:
        return run_queues(tp, lake_dir, a, end, tag)
    return run(tp, lake_dir, since=since, end=end, tag=tag, workers=a.workers, flush_every=a.flush_every,
               empty_retries=a.empty_retries, retry_wait=a.retry_wait, max_tickers=a.max_tickers)


def run_queues(tp, lake_dir: Path, a, end: str, tag: str) -> int:
    """三队列模式（quantmind.line_a.fin_queue）：按触发事件和轮换对账决定拉哪些股票，每个起点一组，各自一个检查点。"""
    from quantmind.line_a.fin_queue import build_queues
    sb = pd.read_parquet(sorted(lake_dir.glob("stock_basic_full_2*.parquet"))[-1])
    lo = pd.Timestamp(a.queues_since).strftime("%Y%m%d")
    alive = sb["list_status"].isin(["L", "P"]) | (sb["delist_date"].fillna("").astype(str) >= lo)
    universe = sorted(sb.loc[alive, "ts_code"].astype(str).unique())
    cal = pd.to_datetime(pd.read_parquet(lake_dir / "trade_cal_sse.parquet")["cal_date"].astype(str))
    run_index = int((cal <= pd.Timestamp(end)).sum())
    q = build_queues(lake_dir, lo, end, universe, run_index=run_index, buckets=a.recon_buckets)
    qpath = lake_dir / "_checkpoints" / f"fin_queue_{lo}_{end}.parquet"
    q.to_parquet(qpath, index=False)
    by_q = q["queues"].str.split(",").explode().value_counts().to_dict() if len(q) else {}
    print(f"[queues] 窗口 ({lo}, {end}]：{len(q)} 只；按队列 {by_q}；清单 {qpath}", flush=True)
    rc = 0
    for fs, g in q.groupby("fetch_since"):
        r = run(tp, lake_dir, since=fs, end=end, tag=tag, workers=a.workers, flush_every=a.flush_every,
                empty_retries=a.empty_retries, retry_wait=a.retry_wait, tickers=sorted(g["ts_code"]),
                ck_tag=f"q_{lo}_{end}_{fs}")
        rc = rc or r
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
