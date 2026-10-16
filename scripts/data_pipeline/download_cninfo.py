"""T-031：巨潮资讯网（CNINFO）公告元数据下载（标题、发布时间、PDF 地址），按自然日，只写新文件，可续跑。

接口：POST http://www.cninfo.com.cn/new/hisAnnouncement/query（法定信息披露平台的公开检索接口；robots.txt 不存在）。
实测（2026-10-17）：
  - 每页最多 30 条（pageSize 再大也只回 30）；单个查询最多翻 100 页 = 3,000 条，第 101 页起**静默返回第 1 页**；
  - 不带板块时返回深沪京全部；plate = szmb / szcy / shmb / shkcp / bj 的合计等于全部（2019-04-26：3979 + 5470 + 4318 = 13767）；
  - category=… 可在板块内再拆，类别之间有少量重叠（同日深市各类别合计 9482 > 板块总数 9449）；
  - announcementTime 是带分钟的毫秒时间戳；另存北京时间 publish_ts_bj。
拆分树：日 → 总数 ≤ 3000 直接翻页；否则按板块 → 板块内仍 > 3000 按类别。每个节点用去重后的 announcementId 数
       与该节点的 totalAnnouncement 对账：对不上（或叶子超过 3000、出现回绕页）记 truncated，数据照存，续跑重试。
产物：<root>/data/lake/text/cninfo/_parts/<YYYYMMDD>.parquet；_checkpoints/cninfo_meta.json（每日终态、总数、行数、请求数）。
限速：请求间隔 --interval 秒（默认 1.5）；HTTP / JSON 错误指数退避，最多 5 次；--max-seconds 到时在两天之间退出（定时任务续跑）。
用法：python download_cninfo.py --data-root <root> [--start 2019-01-01] [--end 2026-10-16] [--max-seconds 3300]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
PAGE_SIZE = 30
MAX_PAGES = 100
LEAF_CAP = PAGE_SIZE * MAX_PAGES
PLATES = ("szmb", "szcy", "shmb", "shkcp", "bj")
CATEGORIES = (
    "category_ndbg_szsh", "category_bndbg_szsh", "category_yjdbg_szsh", "category_sjdbg_szsh",
    "category_yjygjxz_szsh", "category_qyfpxzcs_szsh", "category_dshgg_szsh", "category_jshgg_szsh",
    "category_gddh_szsh", "category_rcjy_szsh", "category_gszl_szsh", "category_zj_szsh", "category_sf_szsh",
    "category_zf_szsh", "category_gqjl_szsh", "category_pg_szsh", "category_jj_szsh", "category_gszq_szsh",
    "category_kzzq_szsh", "category_qtrz_szsh", "category_gqbd_szsh", "category_bcgz_szsh",
    "category_cqdq_szsh", "category_fxts_szsh", "category_tbclts_szsh", "category_tszlq_szsh",
)
KEEP = ("announcementId", "secCode", "secName", "orgId", "announcementTitle", "announcementTime", "adjunctUrl",
        "adjunctSize", "adjunctType", "announcementType", "columnId", "pageColumn", "important", "storageTime")
BJ = timezone(timedelta(hours=8))
DONE, EMPTY, FAILED, TRUNCATED = "done", "empty", "failed", "truncated"
TERMINAL_OK = {DONE, EMPTY}


class Client:
    """一次请求 = 一页。post(data) → dict 可注入（测试用假接口）；默认用 requests。"""

    def __init__(self, post=None, interval: float = 1.5, sleep=time.sleep, retries: int = 5):
        self.post = post or _requests_post()
        self.interval, self.sleep, self.retries = interval, sleep, retries
        self.requests = 0

    def page(self, day: str, page: int, plate: str = "", category: str = "") -> dict:
        data = {"pageNum": page, "pageSize": PAGE_SIZE, "column": "szse", "tabName": "fulltext", "plate": plate,
                "stock": "", "searchkey": "", "secid": "", "category": category, "trade": "",
                "seDate": f"{day}~{day}", "sortName": "", "sortType": "", "isHHTime": "true"}
        for attempt in range(self.retries):
            try:
                self.requests += 1
                out = self.post(data)
                self.sleep(self.interval)
                if not isinstance(out, dict) or "totalAnnouncement" not in out:
                    raise ValueError("unexpected response")
                return out
            except Exception:  # noqa: BLE001
                if attempt == self.retries - 1:
                    raise
                self.sleep(min(300, 5 * 2 ** attempt))
        raise AssertionError("unreachable")


def _requests_post():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (research; QuantMind T-031)", "X-Requested-With": "XMLHttpRequest"})

    def post(data):
        r = s.post(URL, data=data, timeout=30)
        r.raise_for_status()
        return r.json()
    return post


def fetch_leaf(client: Client, day: str, plate: str, category: str, first: dict) -> tuple[dict, bool]:
    """翻完一个查询的全部页（≤ 100 页）。返回 ({announcementId: 行}, 是否与总数对上)。"""
    total = int(first.get("totalAnnouncement") or 0)
    rows: dict[str, dict] = {}
    pages = min(MAX_PAGES, math.ceil(total / PAGE_SIZE))
    for p in range(1, pages + 1):
        anns = (first if p == 1 else client.page(day, p, plate, category)).get("announcements") or []
        new = [a for a in anns if str(a["announcementId"]) not in rows]
        if anns and not new:                        # 回绕页（第 101 页起返回第 1 页）或重复页：停止
            break
        for a in new:
            rows[str(a["announcementId"])] = {**{k: a.get(k) for k in KEEP}, "plate": plate, "category": category}
        if len(anns) < PAGE_SIZE:
            break
    return rows, (len(rows) == total and total <= LEAF_CAP)


def fetch_node(client: Client, day: str, plate: str = "", category: str = "") -> tuple[dict, bool, int]:
    """拆分树的一个节点。返回 (行, 是否对账通过, 本节点 totalAnnouncement)。"""
    first = client.page(day, 1, plate, category)
    total = int(first.get("totalAnnouncement") or 0)
    if total == 0:
        return {}, True, 0
    if total <= LEAF_CAP or category:
        rows, ok = fetch_leaf(client, day, plate, category, first)
        return rows, ok, total
    children = PLATES if not plate else CATEGORIES
    rows, ok = {}, True
    for c in children:
        sub, sub_ok, _ = fetch_node(client, day, c, "") if not plate else fetch_node(client, day, plate, c)
        for k, v in sub.items():
            rows.setdefault(k, v)
        ok &= sub_ok
    return rows, ok and len(rows) == total, total


def to_frame(rows: dict, fetched_at: str) -> pd.DataFrame:
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df
    df["announcementId"] = df["announcementId"].astype(str)
    df["publish_ts_bj"] = [datetime.fromtimestamp(int(t) / 1000, BJ).isoformat() for t in df["announcementTime"]]
    df["fetched_at"] = fetched_at
    return df.sort_values(["announcementTime", "announcementId"]).reset_index(drop=True)


def load_ckpt(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "days": {}}


def save_ckpt(path: Path, ck: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ck, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, path)


def run(out_dir: Path, days: list[str], client: Client, max_seconds: float = 0, log=print,
        clock=time.monotonic) -> dict:
    parts, ck_path = out_dir / "_parts", out_dir / "_checkpoints/cninfo_meta.json"
    parts.mkdir(parents=True, exist_ok=True)
    ck = load_ckpt(ck_path)
    todo = [d for d in days if ck["days"].get(d, {}).get("state") not in TERMINAL_OK]
    log(f"[cninfo] days={len(days)} todo={len(todo)}")
    t0 = clock()
    for i, day in enumerate(todo, 1):
        if max_seconds and clock() - t0 > max_seconds:
            log(f"[cninfo] time budget reached before {day}")
            break
        r0 = client.requests
        rec: dict = {}
        try:
            rows, ok, total = fetch_node(client, day)
            df = to_frame(rows, datetime.now(BJ).isoformat())
            if len(df):
                tmp = parts / f"{day.replace('-', '')}.parquet.tmp"
                df.to_parquet(tmp, index=False)
                os.replace(tmp, parts / f"{day.replace('-', '')}.parquet")
            state = EMPTY if total == 0 else (DONE if ok else TRUNCATED)
            rec = {"state": state, "total": total, "rows": int(len(df))}
        except Exception as e:  # noqa: BLE001
            rec = {"state": FAILED, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        rec["requests"] = client.requests - r0
        ck["days"][day] = rec
        save_ckpt(ck_path, ck)
        if i % 20 == 0 or rec["state"] not in TERMINAL_OK:
            log(f"[cninfo] {i}/{len(todo)} {day} {rec} ({clock() - t0:.0f}s)")
    states = pd.Series([v["state"] for v in ck["days"].values()]).value_counts().to_dict() if ck["days"] else {}
    log(f"[cninfo] states={states} requests_this_run={client.requests}")
    return states


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-10-16")
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--max-seconds", type=float, default=0)
    a = ap.parse_args(argv)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(a.start, a.end, freq="D")]
    states = run(Path(a.data_root) / "data/lake/text/cninfo", days, Client(interval=a.interval),
                 max_seconds=a.max_seconds, log=lambda m: print(m, flush=True))
    return 2 if states.get(FAILED) or states.get(TRUNCATED) else 0


if __name__ == "__main__":
    sys.exit(main())
