"""scripts/data_pipeline/fetch_report_rc_relay.py — 通过用户购买的第三方中转接口拉取 report_rc（2026-10-21，用户要求试用）.

为什么单独写：
- 中转接口不是 Tushare 官方，来源不同，数据**单独存放**并标明来源，只有通过官方重叠校验才能使用；
- token 只从 ~/.config/quantmind/relay_token 读取（权限 600），不写进代码、日志或提交，也不打印。

数据：
- 中转接口的 report_rc 必须带 ts_code、单次区间约一年以内，所以按"股票 × 自然年"分段（--by-ticker）；
  每段内沿用 download_events 的分页（单次上限 5000 行，按 offset 翻页），区间过大时对半拆分；
- 输出：data/lake/events_relay/report_rc/by_ticker/part_<序号>.parquet；检查点 by_ticker/_checkpoint.json；
  来源说明 _provenance.json；校验时按官方窗口切出 <窗口>.parquet 并写 _validation.json。

用法：
  python scripts/data_pipeline/fetch_report_rc_relay.py --data-root <root> --probe     # 小测试
  python scripts/data_pipeline/fetch_report_rc_relay.py --data-root <root> --by-ticker [--limit N]
  python scripts/data_pipeline/fetch_report_rc_relay.py --data-root <root> --validate  # 与官方已下载窗口逐行比对
验收规则见 docs/work/tasks/T-038.md。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.data_pipeline.download_events import SPECS, fetch_adaptive  # noqa: E402

RELAY_URL = "http://tus.sxdata.top/tu_share/"
TOKEN_FILE = Path.home() / ".config/quantmind/relay_token"
KEY = ["ts_code", "report_date", "org_name", "author_name", "quarter", "report_title"]
RELAY_START = "2019-01-01"


class RangeTooLargeError(RuntimeError):
    pass


def make_call(min_interval: float, max_attempts: int = 15):
    """带自适应限速的调用。错误分三类处理：
    - "上游账号池暂时用完"（upstream_pool_exhausted）：2026-10-21 实测 report_rc 有 30–70% 的请求这样失败，
      同一时间 daily 全部成功，而且成功和失败交替出现，所以只等 4 秒就重试；
    - 频率超限（rate_limited / ip_rate_limited）：等 35 秒，并把间隔加 0.5 秒（最多 10 秒），连续成功 50 次后减 0.25 秒；
    - 其他错误（超时、连接断开）：等 30 秒。
    "区间过大"抛 RangeTooLargeError 交给上层拆分；"参数不合法"直接抛出，不重试。"""
    from tushare.pro.client import DataApi
    api = DataApi(token=TOKEN_FILE.read_text().strip(), timeout=60)
    api._DataApi__http_url = RELAY_URL
    st = {"last": 0.0, "gap": min_interval, "ok": 0, "attempts": 0, "pool": 0, "rate": 0, "other": 0}

    def call(name: str, **kw):
        for attempt in range(max_attempts):
            wait = st["gap"] - (time.time() - st["last"])
            if wait > 0:
                time.sleep(wait)
            try:
                st["last"] = time.time()
                st["attempts"] += 1
                df = api.query(name, **kw)
                st["ok"] += 1
                if st["ok"] % 50 == 0 and st["gap"] > min_interval:
                    st["gap"] = max(min_interval, st["gap"] - 0.25)
                return df
            except Exception as e:  # noqa: BLE001  按错误类型等待后重试（中转方建议的补偿机制）
                msg = str(e)[:160]
                if "date_range_too_large" in msg:
                    raise RangeTooLargeError(msg) from None
                if "invalid_params" in msg or attempt == max_attempts - 1:
                    raise
                if "upstream_pool_exhausted" in msg:
                    st["pool"] += 1
                    time.sleep(4)
                elif "rate_limited" in msg:
                    st["rate"] += 1
                    st["gap"] = min(10.0, st["gap"] + 0.5)
                    time.sleep(35)
                else:
                    st["other"] += 1
                    print(f"    retry {attempt + 1} after error: {msg}", flush=True)
                    time.sleep(30)
        return None
    call.state = st
    return call


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    # 官方已有 2018-10-01～2019-04-13，所以中转从 2019 年开始；2019 年 1–4 月与官方重叠，用来校验
    ap.add_argument("--start", default=RELAY_START)
    ap.add_argument("--end", default="2026-10-16")
    ap.add_argument("--min-interval", type=float, default=3.0)
    # 2026-10-21 23:50 实测：中转的请求本身要 6–20 秒，单线程约 50 秒一段；4 线程每分钟成功约 7 段，没有触发频率限制
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只拉前 N 段（试跑）")
    ap.add_argument("--by-ticker", action="store_true",
                    help="中转接口的 report_rc 必须带 ts_code：按股票 × 自然年分段拉取")
    a = ap.parse_args(argv)
    out = a.data_root / "data/lake/events_relay/report_rc"
    out.mkdir(parents=True, exist_ok=True)
    official = a.data_root / "data/lake/events/_parts/report_rc"
    if a.validate:
        return validate(out, official)
    call = make_call(a.min_interval)
    if a.probe:
        t0 = time.time()
        st, df, pages = fetch_adaptive(call, "report_rc", {"ts_code": "600519.SH", "start_date": "20190101",
                                                           "end_date": "20191231"}, SPECS["report_rc"], sleep=lambda s: None)
        print("report_rc 600519.SH 2019:", st, len(df), "rows,", pages, "pages", f"{time.time() - t0:.1f}s", flush=True)
        return 0
    if not a.by_ticker:
        print("中转接口的 report_rc 必须带 ts_code，只支持 --by-ticker", flush=True)
        return 2
    prov = out / "_provenance.json"
    if not prov.exists():
        prov.write_text(json.dumps({
            "source": "third-party relay purchased by the user (not official Tushare)", "url": RELAY_URL,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "api": "report_rc",
            "layout": "by_ticker: one request series per (ts_code, calendar year); see by_ticker/_checkpoint.json",
            "rule": "use only after row-level validation against official windows (docs/work/tasks/T-038.md)"}, indent=1))
    return by_ticker(a, out)


# 拉取顺序（2026-10-21：中转的 report_rc 有一半请求要重试，按现在的速度 7 天拉不完全部 4 万段）：
# 1. 校验集：官方 2019 年窗口里有记录的股票抽 200 只，没有记录的抽 50 只，先拉它们的 2019 段；
# 2. 其余按"层"拉：每年按年初总市值排名（年内上市的按上市后第一个交易日），1–1500 名为第一层，
#    1501–3000 名为第二层，其余第三层；每层内按年份从近到远。
#    这样中途停下时，已拉到的范围是"每年市值前 N 名"，各年口径一致，便于在下游说明覆盖范围。
VAL_COVERED, VAL_UNCOVERED, VAL_SEED = 200, 50, 20261021
TIERS = (1500, 3000)


def _segments(t: str, list_d: str, delist_d: str, lo: str, hi: str) -> list[tuple[str, str]]:
    """按自然年切段，并裁到 [上市日, 退市日] ∩ [lo, hi]。中转接口单次最多约一年。"""
    s = max(lo, list_d or lo)
    e = min(hi, delist_d or hi)
    out = []
    for y in range(int(s[:4]), int(e[:4]) + 1):
        a, b = max(s, f"{y}0101"), min(e, f"{y}1231")
        if a <= b:
            out.append((a, b))
    return out


def _fetch_split(call, spec, params: dict, depth: int = 0):
    """区间过大时对半拆分重取（最多拆 4 层）。"""
    try:
        return fetch_adaptive(call, "report_rc", params, spec, sleep=lambda s: None)
    except RangeTooLargeError:
        if depth >= 4:
            raise
        s, e = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"])
        mid = s + (e - s) // 2
        r1 = _fetch_split(call, spec, {**params, "end_date": mid.strftime("%Y%m%d")}, depth + 1)
        r2 = _fetch_split(call, spec, {**params, "start_date": (mid + pd.Timedelta(days=1)).strftime("%Y%m%d")}, depth + 1)
        st = "done" if r1[0] == r2[0] == "done" else (r1[0] if r1[0] != "done" else r2[0])
        return st, pd.concat([r1[1], r2[1]], ignore_index=True), r1[2] + r2[2]


def _mv_rank(lake: Path, lo: str) -> pd.DataFrame:
    """每只股票每年第一个交易日（年内上市的就是上市后第一个交易日）的总市值在当年的排名。只用于决定拉取顺序。"""
    import pyarrow.dataset as ds
    t = ds.dataset(lake / "daily_basic.parquet").to_table(
        columns=["ts_code", "trade_date", "total_mv"], filter=ds.field("trade_date") >= pd.Timestamp(lo))
    mv = t.to_pandas(strings_to_categorical=True)
    mv["year"] = mv["trade_date"].dt.year.astype(str)
    first = mv.sort_values("trade_date").drop_duplicates(["ts_code", "year"]).astype({"ts_code": str})
    first["rank"] = first.groupby("year")["total_mv"].rank(ascending=False, method="first")
    return first[["ts_code", "year", "rank"]]


def _order(segs: list[tuple[str, str, str]], lake: Path, official: Path, lo: str) -> tuple[list, set]:
    """返回 (拉取顺序, 校验集股票)。顺序规则见上方注释。"""
    import random
    covered = set()
    for f in sorted(official.glob("*.parquet")):
        if f.stem.split("_")[0] >= "20190101":
            covered |= set(pd.read_parquet(f, columns=["ts_code"])["ts_code"].astype(str))
    in2019 = sorted({t for t, s, _ in segs if s[:4] == "2019" and s <= "20190114"})   # 2019 年初就已上市
    rng = random.Random(VAL_SEED)
    cov = sorted(set(in2019) & covered)
    unc = sorted(set(in2019) - covered)
    val = set(rng.sample(cov, min(VAL_COVERED, len(cov)))) | set(rng.sample(unc, min(VAL_UNCOVERED, len(unc))))
    first = sorted(x for x in segs if x[0] in val and x[1][:4] == "2019")
    rk = _mv_rank(lake, lo).set_index(["ts_code", "year"])["rank"].to_dict()

    def tier(x):
        r = rk.get((x[0], x[1][:4]), float("inf"))
        return 0 if r <= TIERS[0] else (1 if r <= TIERS[1] else 2)
    fs = set(first)
    rest = sorted((x for x in segs if x not in fs), key=lambda x: (tier(x), -int(x[1][:4]), x[0]))
    return first + rest, val


def by_ticker(a, out: Path) -> int:
    """逐只股票、逐年分段拉取，顺序见 _order。
    每 100 段合并写一个分片 by_ticker/part_<序号>.parquet，同时写检查点（"股票:起始日" → 状态），两者一起落盘。"""
    lake = a.data_root / "data/lake"
    sb = pd.read_parquet(sorted(lake.glob("stock_basic_full_merged_2*.parquet"))[-1],
                         columns=["ts_code", "list_date", "delist_date"])
    lo, hi = pd.Timestamp(a.start).strftime("%Y%m%d"), pd.Timestamp(a.end).strftime("%Y%m%d")
    sb = sb.assign(ts_code=sb["ts_code"].astype(str), list_date=sb["list_date"].astype(str).str.slice(0, 8),
                   delist_date=sb["delist_date"].astype("string").fillna("").str.slice(0, 8))
    sb = sb[(sb["delist_date"] == "") | (sb["delist_date"] >= lo)].drop_duplicates("ts_code").sort_values("ts_code")
    segs = [(t, s, e) for t, ld, dd in sb[["ts_code", "list_date", "delist_date"]].itertuples(index=False)
            for s, e in _segments(t, ld, dd, lo, hi)]
    order, val = _order(segs, lake, a.data_root / "data/lake/events/_parts/report_rc", lo)
    d = out / "by_ticker"
    d.mkdir(exist_ok=True)
    (d / "_validation_tickers.json").write_text(json.dumps({"seed": VAL_SEED, "tickers": sorted(val)}))
    ck_path = d / "_checkpoint.json"
    ck = json.loads(ck_path.read_text()) if ck_path.exists() else {"segments": {}}
    todo = [x for x in order if ck["segments"].get(f"{x[0]}:{x[1]}", {}).get("state") not in ("done", "empty")]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[relay report_rc by ticker] {len(sb)} tickers, {len(segs)} segments, {len(todo)} to fetch "
          f"(validation tickers {len(val)})", flush=True)
    spec = SPECS["report_rc"]
    buf, t0 = [], time.time()
    part = len(list(d.glob("part_*.parquet")))
    lock = threading.Lock()
    it = iter(todo)
    n_done = [0]
    calls = [make_call(a.min_interval) for _ in range(a.workers)]   # 每个线程一个连接，各自限速

    def flush():
        nonlocal buf, part
        if buf:
            df = _normalize(pd.concat(buf, ignore_index=True))
            df.to_parquet(d / f"part_{part:05d}.parquet")      # 先写数据、再写检查点：写盘失败时不会把段记成已完成
            part += 1
            buf = []
        tmp = ck_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ck))
        tmp.replace(ck_path)

    def report():
        sv = ck["segments"].values()
        n = sum(v.get("rows", 0) for v in sv)
        nf = sum(v.get("state") == "failed" for v in sv)
        el = (time.time() - t0) / 60
        c = {k: sum(x.state[k] for x in calls) for k in ("attempts", "ok", "pool", "rate", "other")}
        i = n_done[0]
        print(f"  [{i}/{len(todo)}] {time.strftime('%m-%d %H:%M')} rows so far {n}; failed {nf}; "
              f"attempts {c['attempts']} ok {c['ok']} pool {c['pool']} rate {c['rate']} other {c['other']}; "
              f"{el:.0f} min, eta {el / max(1, i) * (len(todo) - i) / 60:.1f} h", flush=True)

    def work(call):
        while True:
            with lock:
                x = next(it, None)
            if x is None:
                return
            t, s, e = x
            try:
                st, df, pages = _fetch_split(call, spec, {"ts_code": t, "start_date": s, "end_date": e})
                rec = {"state": st, "end": e, "rows": int(len(df)), "pages": pages}
            except Exception as ex:  # noqa: BLE001  单段失败记下来，下次重试
                df, rec = None, {"state": "failed", "end": e, "error": str(ex)[:160]}
            with lock:
                if df is not None and len(df):
                    buf.append(df)
                ck["segments"][f"{t}:{s}"] = rec
                n_done[0] += 1
                if n_done[0] % 100 == 0:
                    flush()
                    report()
    threads = [threading.Thread(target=work, args=(c,), daemon=True) for c in calls]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    flush()
    report()
    return 0


NUMERIC = ("op_rt", "op_pr", "tp", "np", "eps", "pe", "rd", "roe", "ev_ebitda", "max_price", "min_price")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """列类型与官方一致：数值列转 float（中转在整段为空时返回 object 列，拼起来会混型导致写 parquet 失败），其余转字符串。
    数值列里如果有转不了的非空值，原样另存一列 <列>__raw，不丢信息。"""
    df = df.copy()
    for c in df.columns:
        if c in NUMERIC:
            v = pd.to_numeric(df[c], errors="coerce")
            bad = df[c].notna() & v.isna() & (df[c].astype(str).str.strip() != "")
            if bad.any():
                df[f"{c}__raw"] = df[c].where(bad).astype("string")
            df[c] = v.astype("float64")
        else:
            df[c] = df[c].astype("string")
    return df


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize(df)
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c in NUMERIC:
            out[c] = df[c].map(lambda x: "" if pd.isna(x) else repr(float(x)))
        else:
            out[c] = df[c].fillna("").astype(str)
    return out


def validate(relay: Path, official: Path) -> int:
    """与官方已下载的窗口逐行比对（验收规则见 T-038）。
    - 只比完全落在中转起始日之后的官方窗口（2019-01-14 起的 6 个窗口），只比 2019 段已拉完的股票；
    - 两边先按同一规则转成规范字符串（_canon：空值一律为空串，数值列用 float 的 repr），再整行比对；
      不一致时按 KEY 对齐，统计每一列有多少行取值不同，便于判断是格式还是数据的差异。
      （规范化规则在做任何比对之前定下：中转对整段为空的数值列返回 object 类型，官方是 float，直接转字符串会造成假差异。）"""
    bt = relay / "by_ticker"
    segs = json.loads((bt / "_checkpoint.json").read_text())["segments"]
    y0 = RELAY_START.replace("-", "")
    fetched = {k.split(":")[0] for k, v in segs.items()
               if k.split(":")[1][:4] == y0[:4] and v.get("state") in ("done", "empty")}
    allr = pd.concat([pd.read_parquet(p) for p in sorted(bt.glob("part_*.parquet"))], ignore_index=True)
    allr = allr[allr["ts_code"].astype(str).isin(fetched)]
    vdir = relay / "_validation_windows"
    vdir.mkdir(exist_ok=True)
    res = {"n_tickers_compared": len(fetched), "windows": [], "column_diffs": {}}
    for f in sorted(official.glob("*.parquet")):
        s, e = f.stem.split("_")
        if s < y0:
            continue
        a = pd.read_parquet(f)
        a = a[a["ts_code"].astype(str).isin(fetched)]
        b = allr[(allr["report_date"].astype(str) >= s) & (allr["report_date"].astype(str) <= e)]
        b.to_parquet(vdir / f.name)
        cols = [c for c in a.columns if c in b.columns]
        ka = _canon(a[cols]).drop_duplicates()
        kb = _canon(b[cols]).drop_duplicates()
        m = ka.merge(kb, on=cols, how="outer", indicator=True)
        res["windows"].append({"window": f.stem, "official_rows": int(len(a)), "relay_rows": int(len(b)),
                               "both": int((m["_merge"] == "both").sum()),
                               "only_official": int((m["_merge"] == "left_only").sum()),
                               "only_relay": int((m["_merge"] == "right_only").sum())})
        j = ka.drop_duplicates(KEY).merge(kb.drop_duplicates(KEY), on=KEY, suffixes=("_o", "_r"))
        for c in cols:
            if c not in KEY:
                res["column_diffs"][c] = res["column_diffs"].get(c, 0) + int((j[f"{c}_o"] != j[f"{c}_r"]).sum())
    w = res["windows"]
    res["identical"] = bool(w) and all(x["only_official"] == 0 and x["only_relay"] == 0 for x in w)
    res["column_diffs"] = {c: n for c, n in res["column_diffs"].items() if n}
    (relay / "_validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res["identical"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
