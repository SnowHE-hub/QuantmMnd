"""scripts/data_pipeline/fetch_report_rc_relay.py — 通过用户购买的第三方中转接口拉取 report_rc（2026-10-21，用户要求试用）.

为什么单独写：
- 中转接口不是 Tushare 官方，来源不同，数据**单独存放**并标明来源，只有通过官方重叠校验才能使用；
- token 只从 ~/.config/quantmind/relay_token 读取（权限 600），不写进代码、日志或提交，也不打印。

数据：
- 分块与分页沿用 download_events 的 report_rc 规则：15 天窗口，单次上限 5000 行，按 offset 翻页，满页截断时对半拆分；
- 输出：data/lake/events_relay/report_rc/<窗口>.parquet；检查点 _checkpoint.json；来源说明 _provenance.json。

用法：
  python scripts/data_pipeline/fetch_report_rc_relay.py --data-root <root> --probe     # 小测试
  python scripts/data_pipeline/fetch_report_rc_relay.py --data-root <root> --start 2018-10-01 --end 2026-10-16
  python scripts/data_pipeline/fetch_report_rc_relay.py --data-root <root> --validate  # 与官方已下载窗口逐行比对
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.data_pipeline.download_events import SPECS, chunks_for, fetch_adaptive  # noqa: E402

RELAY_URL = "http://tus.sxdata.top/tu_share/"
TOKEN_FILE = Path.home() / ".config/quantmind/relay_token"
KEY = ["ts_code", "report_date", "org_name", "author_name", "quarter", "report_title"]


class RangeTooLargeError(RuntimeError):
    pass


def make_call(min_interval: float):
    """带自适应限速的调用：频率超限时等 35 秒并把间隔加 0.5 秒（最多 10 秒），连续成功 50 次后减 0.25 秒。
    "区间过大"抛 RangeTooLargeError 交给上层拆分；"参数不合法"直接抛出，不重试。"""
    from tushare.pro.client import DataApi
    api = DataApi(token=TOKEN_FILE.read_text().strip(), timeout=60)
    api._DataApi__http_url = RELAY_URL
    st = {"last": 0.0, "gap": min_interval, "ok": 0}

    def call(name: str, **kw):
        for attempt in range(6):
            wait = st["gap"] - (time.time() - st["last"])
            if wait > 0:
                time.sleep(wait)
            try:
                st["last"] = time.time()
                df = api.query(name, **kw)
                st["ok"] += 1
                if st["ok"] % 50 == 0 and st["gap"] > min_interval:
                    st["gap"] = max(min_interval, st["gap"] - 0.25)
                return df
            except Exception as e:  # noqa: BLE001  超时或频率超限：等待后重试（中转方建议的补偿机制）
                msg = str(e)[:160]
                if "date_range_too_large" in msg:
                    raise RangeTooLargeError(msg) from None
                if "invalid_params" in msg or attempt == 5:
                    raise
                if "rate_limited" in msg:
                    st["gap"] = min(10.0, st["gap"] + 0.5)
                    time.sleep(35)
                else:
                    print(f"    retry {attempt + 1} after error: {msg}", flush=True)
                    time.sleep(60)
        return None
    call.state = st
    return call


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--start", default="2018-10-01")
    ap.add_argument("--end", default="2026-10-16")
    ap.add_argument("--min-interval", type=float, default=3.0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只拉前 N 只（试跑）")
    ap.add_argument("--by-ticker", action="store_true",
                    help="中转接口的 report_rc 必须带 ts_code：按股票逐只拉 [start, end] 的全部记录")
    a = ap.parse_args(argv)
    out = a.data_root / "data/lake/events_relay/report_rc"
    out.mkdir(parents=True, exist_ok=True)
    official = a.data_root / "data/lake/events/_parts/report_rc"
    if a.validate:
        return validate(out, official)
    call = make_call(a.min_interval)
    if a.probe:
        t0 = time.time()
        cal = call("trade_cal", exchange="SSE", start_date="20261001", end_date="20261021")
        print("trade_cal rows", 0 if cal is None else len(cal), f"{time.time() - t0:.1f}s", flush=True)
        # 官方规格里每页之后等 65 秒（官方每分钟 1 次）；中转接口的间隔由 call() 自己控制，这里不再额外等待
        st, df, pages = fetch_adaptive(call, "report_rc", {"start_date": "20181001", "end_date": "20181015"},
                                       SPECS["report_rc"], sleep=lambda s: None)
        print("report_rc 20181001-20181015:", st, len(df), "rows,", pages, "pages", flush=True)
        df.to_parquet(out / "_probe_20181001_20181015.parquet")
        return 0
    if a.by_ticker:
        return by_ticker(a, out, call)
    spec = SPECS["report_rc"]
    ck_path = out / "_checkpoint.json"
    ck = json.loads(ck_path.read_text()) if ck_path.exists() else {"chunks": {}}
    (out / "_provenance.json").write_text(json.dumps({
        "source": "third-party relay purchased by the user (not official Tushare)", "url": RELAY_URL,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "api": "report_rc",
        "rule": "use only after row-level validation against official windows (see --validate)"}, indent=1))
    todo = [(k, p) for k, p in chunks_for(spec, a.start, a.end) if ck["chunks"].get(k, {}).get("state") != "done"]
    print(f"[relay report_rc] {len(todo)} windows to fetch", flush=True)
    t0 = time.time()
    for i, (k, p) in enumerate(todo, 1):
        st, df, pages = fetch_adaptive(call, "report_rc", p, spec, sleep=lambda s: None)
        if len(df):
            df.to_parquet(out / f"{k}.parquet")
        ck["chunks"][k] = {"state": st, "rows": int(len(df)), "pages": pages}
        tmp = ck_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ck))
        tmp.replace(ck_path)
        print(f"  [{i}/{len(todo)}] {k} {st} rows={len(df)} pages={pages} ({time.time() - t0:.0f}s)", flush=True)
    return 0


VALIDATION_TICKERS = 300          # 先拉这些股票的 2018–2019 段，尽早与官方窗口比对


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


def by_ticker(a, out: Path, call) -> int:
    """逐只股票、逐年分段拉取。顺序：先 VALIDATION_TICKERS 只股票的 2018–2019 段（用于校验），再按年份从近到远。
    每 200 段合并写一个分片 by_ticker/part_<序号>.parquet；检查点按 "股票:起始日" 记录状态。"""
    lake = a.data_root / "data/lake"
    sb = pd.read_parquet(sorted(lake.glob("stock_basic_full_merged_2*.parquet"))[-1],
                         columns=["ts_code", "list_date", "delist_date"])
    lo, hi = pd.Timestamp(a.start).strftime("%Y%m%d"), pd.Timestamp(a.end).strftime("%Y%m%d")
    sb = sb.assign(ts_code=sb["ts_code"].astype(str), list_date=sb["list_date"].astype(str).str.slice(0, 8),
                   delist_date=sb["delist_date"].astype("string").fillna("").str.slice(0, 8))
    sb = sb[(sb["delist_date"] == "") | (sb["delist_date"] >= lo)].drop_duplicates("ts_code").sort_values("ts_code")
    segs = [(t, s, e) for t, ld, dd in sb[["ts_code", "list_date", "delist_date"]].itertuples(index=False)
            for s, e in _segments(t, ld, dd, lo, hi)]
    val = set(sb["ts_code"].head(VALIDATION_TICKERS))
    first = [x for x in segs if x[0] in val and x[1] < "20200101"]
    rest = sorted((x for x in segs if x not in set(first)), key=lambda x: (x[1][:4], x[0]), reverse=True)
    d = out / "by_ticker"
    d.mkdir(exist_ok=True)
    ck_path = d / "_checkpoint.json"
    ck = json.loads(ck_path.read_text()) if ck_path.exists() else {"segments": {}}
    todo = [x for x in first + rest if ck["segments"].get(f"{x[0]}:{x[1]}", {}).get("state") not in ("done", "empty")]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[relay report_rc by ticker] {len(sb)} tickers, {len(segs)} segments, {len(todo)} to fetch", flush=True)
    spec = SPECS["report_rc"]
    buf, t0 = [], time.time()
    part = len(list(d.glob("part_*.parquet")))

    def flush():
        nonlocal buf, part
        if buf:
            pd.concat(buf, ignore_index=True).to_parquet(d / f"part_{part:05d}.parquet")
            part += 1
            buf = []
        tmp = ck_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ck))
        tmp.replace(ck_path)
    for i, (t, s, e) in enumerate(todo, 1):
        key = f"{t}:{s}"
        try:
            st, df, pages = _fetch_split(call, spec, {"ts_code": t, "start_date": s, "end_date": e})
        except Exception as ex:  # noqa: BLE001  单段失败记下来，下次重试
            ck["segments"][key] = {"state": "failed", "end": e, "error": str(ex)[:160]}
            continue
        if len(df):
            buf.append(df)
        ck["segments"][key] = {"state": st, "end": e, "rows": int(len(df)), "pages": pages}
        if i % 200 == 0 or i == len(todo):
            flush()
            n = sum(v.get("rows", 0) for v in ck["segments"].values())
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(todo)}] rows so far {n}; gap {call.state['gap']:.2f}s; {el:.0f} min, "
                  f"eta {el / i * (len(todo) - i) / 60:.1f} h", flush=True)
    flush()
    return 0


def validate(relay: Path, official: Path) -> int:
    """与官方已下载的窗口逐行比对。按股票拉取的数据先按 report_date 切到官方窗口的日期范围再比。"""
    bt = relay / "by_ticker"
    if bt.exists() and list(bt.glob("part_*.parquet")):
        allr = pd.concat([pd.read_parquet(p) for p in sorted(bt.glob("part_*.parquet"))], ignore_index=True)
        for f in sorted(official.glob("*.parquet")):
            s, e = f.stem.split("_")
            sub = allr[(allr["report_date"].astype(str) >= s) & (allr["report_date"].astype(str) <= e)]
            sub.to_parquet(relay / f.name)
    res = {"windows": []}
    fetched = None
    if (bt / "_checkpoint.json").exists():
        segs = json.loads((bt / "_checkpoint.json").read_text())["segments"]
        early: dict[str, list[str]] = {}
        for k, v in segs.items():                  # 只看覆盖官方窗口（2018-10～2019-04）的分段
            t, s = k.split(":")
            if s < "20200101":
                early.setdefault(t, []).append(v.get("state"))
        fetched = {t for t, sts in early.items() if all(x in ("done", "empty") for x in sts)}
        res["n_tickers_fetched"] = len(fetched)
    for f in sorted(official.glob("*.parquet")):
        r = relay / f.name
        if not r.exists():
            continue
        a, b = pd.read_parquet(f), pd.read_parquet(r)
        if fetched is not None:                  # 只比较中转接口已经拉过的股票
            a = a[a["ts_code"].astype(str).isin(fetched)]
        cols = [c for c in a.columns if c in b.columns]
        ka = a[cols].astype(str).drop_duplicates()
        kb = b[cols].astype(str).drop_duplicates()
        m = ka.merge(kb, on=cols, how="outer", indicator=True)
        res["windows"].append({"window": f.stem, "official_rows": int(len(a)), "relay_rows": int(len(b)),
                               "both": int((m["_merge"] == "both").sum()), "only_official": int((m["_merge"] == "left_only").sum()),
                               "only_relay": int((m["_merge"] == "right_only").sum())})
    w = res["windows"]
    res["identical"] = bool(w) and all(x["only_official"] == 0 and x["only_relay"] == 0 for x in w)
    (relay / "_validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res["identical"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
