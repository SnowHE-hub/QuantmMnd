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


def make_call(min_interval: float):
    from tushare.pro.client import DataApi
    api = DataApi(token=TOKEN_FILE.read_text().strip(), timeout=60)
    api._DataApi__http_url = RELAY_URL
    last = [0.0]

    def call(name: str, **kw):
        for attempt in range(4):
            wait = min_interval - (time.time() - last[0])
            if wait > 0:
                time.sleep(wait)
            try:
                last[0] = time.time()
                return api.query(name, **kw)
            except Exception as e:  # noqa: BLE001  超时或频率超限：等待后重试（中转方建议的补偿机制）
                msg = str(e)[:160]
                if attempt == 3 or "invalid_params" in msg:
                    raise
                print(f"    retry {attempt + 1} after error: {msg}", flush=True)
                time.sleep(35 if "rate_limited" in msg else 60)
        return None
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


def by_ticker(a, out: Path, call) -> int:
    """逐只股票拉取；每 100 只合并写一个分片 by_ticker/part_<序号>.parquet；检查点记录每只的状态和行数。"""
    lake = a.data_root / "data/lake"
    sb = pd.read_parquet(sorted(lake.glob("stock_basic_full_merged_2*.parquet"))[-1], columns=["ts_code", "delist_date"])
    lo = pd.Timestamp(a.start).strftime("%Y%m%d")
    live = sb[sb["delist_date"].isna() | (sb["delist_date"].astype(str) >= lo)]
    tickers = sorted(live["ts_code"].astype(str).unique())
    bse = lake / "bse"
    if bse.exists():
        fs = sorted(bse.glob("stock_basic_bse_*.parquet"))
        if fs:
            tickers = sorted(set(tickers) | set(pd.read_parquet(fs[-1])["ts_code"].astype(str)))
    d = out / "by_ticker"
    d.mkdir(exist_ok=True)
    ck_path = d / "_checkpoint.json"
    ck = json.loads(ck_path.read_text()) if ck_path.exists() else {"tickers": {}}
    todo = [t for t in tickers if ck["tickers"].get(t, {}).get("state") not in ("done", "empty")]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[relay report_rc by ticker] {len(tickers)} tickers, {len(todo)} to fetch", flush=True)
    spec = SPECS["report_rc"]
    params = {"start_date": lo, "end_date": pd.Timestamp(a.end).strftime("%Y%m%d")}
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
    for i, t in enumerate(todo, 1):
        try:
            st, df, pages = fetch_adaptive(call, "report_rc", {**params, "ts_code": t}, spec, sleep=lambda s: None)
        except Exception as e:  # noqa: BLE001  单只失败记下来，下次重试
            ck["tickers"][t] = {"state": "failed", "error": str(e)[:160]}
            continue
        if len(df):
            buf.append(df)
        ck["tickers"][t] = {"state": st, "rows": int(len(df)), "pages": pages}
        if i % 100 == 0 or i == len(todo):
            flush()
            n = sum(v.get("rows", 0) for v in ck["tickers"].values())
            print(f"  [{i}/{len(todo)}] rows so far {n} ({(time.time() - t0) / 60:.0f} min)", flush=True)
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
        ck = json.loads((bt / "_checkpoint.json").read_text())["tickers"]
        fetched = {t for t, v in ck.items() if v.get("state") in ("done", "empty")}
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
