"""T-031：巨潮资讯网（CNINFO）公告元数据下载（标题、发布时间、PDF 地址），按自然日，只写新文件，可续跑。

接口：POST http://www.cninfo.com.cn/new/hisAnnouncement/query（法定信息披露平台的公开检索接口；robots.txt 不存在）。
实测（2026-10-17）：
  - 每页最多 30 条（pageSize 再大也只回 30）；单个查询最多翻 100 页 = 3,000 条，第 101 页起**静默返回第 1 页**；
  - 不带板块时返回深沪京全部；plate = szmb / szcy / shmb / shkcp / bj 的合计等于全部（2019-04-26：3979 + 5470 + 4318 = 13767）；
  - category=… 可在板块内再拆，类别之间有少量重叠，且有个别公告不属于任何类别（同日沪主板按类别合并只有 4317 / 4318）；
  - sortName=code 的 asc / desc 有效，但代码有并列，分页边界会漏行（单独用正倒序两遍：13719 / 13767）；
    与类别拆分的结果合并后恰好 13767 / 13767；
  - announcementTime 是带分钟的毫秒时间戳；另存北京时间 publish_ts_bj。
拆分树：节点总数 ≤ 3000 → 直接翻页（默认顺序分页稳定）；更大 → 日按板块拆、板块按类别拆；
       板块按类别合并后仍少于总数 → 再按代码正倒序两遍补漏；类别叶子 > 3000（少见）→ 正倒序两遍兜底。
       每个节点用去重后的 announcementId 数与该节点的 totalAnnouncement 对账。对不上记 truncated，数据照存，下次运行重试；
       连续两次 truncated 且总数、行数都不变 → 记 partial（终态，缺口写进检查点，不再重试）。
产物：<root>/data/lake/text/cninfo/_parts/<YYYYMMDD>.parquet；_checkpoints/cninfo_meta.json（每日终态、总数、行数、请求数）。
限速：请求间隔 --interval 秒（默认 1.5）；HTTP / JSON 错误指数退避，最多 5 次；--max-seconds 到时在两天之间退出（定时任务续跑）。
用法：python download_cninfo.py --data-root <root> [--start 2019-01-01] [--end 2026-10-16] [--max-seconds 3300] [--refresh-recent N]
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
DONE, EMPTY, FAILED, TRUNCATED, PARTIAL = "done", "empty", "failed", "truncated", "partial"
TERMINAL_OK = {DONE, EMPTY, PARTIAL}
NO_SORT = ("", "")


class Client:
    """一次请求 = 一页。post(data) → dict 可注入（测试用假接口）；默认用 requests。"""

    def __init__(self, post=None, interval: float = 1.5, sleep=time.sleep, retries: int = 5):
        self.post = post or _requests_post()
        self.interval, self.sleep, self.retries = interval, sleep, retries
        self.requests = 0

    def page(self, day: str, page: int, plate: str = "", category: str = "", sort: tuple = NO_SORT) -> dict:
        data = {"pageNum": page, "pageSize": PAGE_SIZE, "column": "szse", "tabName": "fulltext", "plate": plate,
                "stock": "", "searchkey": "", "secid": "", "category": category, "trade": "",
                "seDate": f"{day}~{day}", "sortName": sort[0], "sortType": sort[1], "isHHTime": "true"}
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


def _pages(client: Client, day: str, plate: str, category: str, sort: tuple, total: int, rows: dict,
           first: dict | None = None) -> None:
    """按一种排序翻页（≤ 100 页），新行并入 rows；凑满 total、短页、或整页都是**本遍**已见过的（回绕 / 重复）就停。
    回绕只和本遍比较：补漏时 rows 里已有其他拆分取到的行，整页"不新"是正常的。"""
    seen: set[str] = set()
    for p in range(1, min(MAX_PAGES, math.ceil(total / PAGE_SIZE)) + 1):
        if len(rows) >= total:
            return
        anns = ((first if (p == 1 and first is not None) else client.page(day, p, plate, category, sort))
                .get("announcements") or [])
        ids = [str(a["announcementId"]) for a in anns]
        if ids and seen.issuperset(ids):
            return
        seen.update(ids)
        for i, a in zip(ids, anns, strict=True):
            if i not in rows:
                rows[i] = {**{k: a.get(k) for k in KEEP}, "plate": plate, "category": category}
        if len(anns) < PAGE_SIZE:
            return


def fetch_node(client: Client, day: str, plate: str = "", category: str = "") -> tuple[dict, bool, int]:
    """拆分树的一个节点。返回 (行, 是否对账通过, 本节点 totalAnnouncement)。"""
    first = client.page(day, 1, plate, category)
    total = int(first.get("totalAnnouncement") or 0)
    rows: dict[str, dict] = {}
    if total == 0:
        return rows, True, 0
    if total <= LEAF_CAP:
        _pages(client, day, plate, category, NO_SORT, total, rows, first)
    elif category:                                       # 类别叶子 > 3000（少见）：只能正倒序各取前 3000
        _two_way(client, day, plate, category, total, rows)
    else:
        for c in (PLATES if not plate else CATEGORIES):
            sub, _, _ = fetch_node(client, day, c, "") if not plate else fetch_node(client, day, plate, c)
            for k, v in sub.items():
                rows.setdefault(k, v)
    if len(rows) < total:                                # 任何节点不够数：按代码、时间的正倒序补漏
        _supplement(client, day, plate, category, total, rows)
    return rows, len(rows) >= total, total


def _supplement(client: Client, day: str, plate: str, category: str, total: int, rows: dict) -> None:
    """2026-10-19 实测：总数 ≤ 3000 的日子直接翻页也会少几条（如 2021-09-01 1175 / 1211），默认排序分页不稳定；
    2021-08 前后板块合计也少于全部（北交所前身精选层等不在 PLATES 里 `[推断]`）。所以任何节点不够数时都补四遍。"""
    for sort in (("code", "asc"), ("code", "desc"), ("time", "asc"), ("time", "desc")):
        if len(rows) >= total:
            return
        _pages(client, day, plate, category, sort, total, rows)


def _two_way(client: Client, day: str, plate: str, category: str, total: int, rows: dict) -> None:
    """按代码正序、倒序各翻 ≤ 100 页，新行并入 rows。代码有并列（同一公司一天多条），分页边界可能漏行，
    所以只用作补漏或兜底，结果仍以与总数对账为准（2019-04-26 实测：单独用会漏 48 条，与类别拆分合并后恰好对上）。"""
    _pages(client, day, plate, category, ("code", "asc"), total, rows)
    _pages(client, day, plate, category, ("code", "desc"), total, rows)


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


def _merge_existing(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """重抓时与已有的行合并（只增不删）：同一 announcementId 用新抓的版本，旧文件里有、这次没抓到的行保留。"""
    if not path.exists():
        return df
    old = pd.read_parquet(path)
    if df.empty:
        return old
    keep = old[~old["announcementId"].astype(str).isin(set(df["announcementId"]))]
    return pd.concat([df, keep], ignore_index=True).sort_values(["announcementTime", "announcementId"]) \
        .reset_index(drop=True)


def run(out_dir: Path, days: list[str], client: Client, max_seconds: float = 0, log=print,
        clock=time.monotonic, refresh: set[str] = frozenset()) -> dict:
    """refresh：这些日子即使已是终态也重抓（每日运行补迟到的公告），结果与已有行合并。"""
    parts, ck_path = out_dir / "_parts", out_dir / "_checkpoints/cninfo_meta.json"
    parts.mkdir(parents=True, exist_ok=True)
    ck = load_ckpt(ck_path)
    todo = [d for d in days if d in refresh or ck["days"].get(d, {}).get("state") not in TERMINAL_OK]
    log(f"[cninfo] days={len(days)} todo={len(todo)}")
    t0 = clock()
    for i, day in enumerate(todo, 1):
        if max_seconds and clock() - t0 > max_seconds:
            log(f"[cninfo] time budget reached before {day}")
            break
        r0, prev = client.requests, ck["days"].get(day, {})
        try:
            rows, ok, total = fetch_node(client, day)
            df = to_frame(rows, datetime.now(BJ).isoformat())
            dest = parts / f"{day.replace('-', '')}.parquet"
            if day in refresh:
                df = _merge_existing(dest, df)
            if len(df):
                tmp = parts / f"{day.replace('-', '')}.parquet.tmp"
                df.to_parquet(tmp, index=False)
                os.replace(tmp, dest)
            state = EMPTY if total == 0 else (DONE if ok else TRUNCATED)
            rec = {"state": state, "total": total, "rows": int(len(df))}
            if total and len(df) > total:
                rec["excess"] = int(len(df) - total)                   # 抓取期间总数变了（如 2021-04-30 15388 / 15387）
            if day in refresh:
                rec["refreshed_at"] = datetime.now(BJ).isoformat(timespec="seconds")
            if state == TRUNCATED and prev.get("state") == TRUNCATED and \
                    (prev.get("total"), prev.get("rows")) == (total, rec["rows"]):
                rec.update(state=PARTIAL, gap=total - rec["rows"])     # 重试结果不变：缺口是接口本身的，不再重试
        except Exception as e:  # noqa: BLE001
            rec = {"state": FAILED, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        rec["requests"] = client.requests - r0
        ck["days"][day] = rec
        save_ckpt(ck_path, ck)
        if i % 20 == 0 or rec["state"] not in (DONE, EMPTY):
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
    ap.add_argument("--refresh-recent", type=int, default=0,
                    help="范围内最后 N 天即使已是终态也重抓并与已有行合并（每日运行补迟到的公告，T-040）")
    a = ap.parse_args(argv)
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(a.start, a.end, freq="D")]
    refresh = set(days[-a.refresh_recent:]) if a.refresh_recent > 0 else set()
    states = run(Path(a.data_root) / "data/lake/text/cninfo", days, Client(interval=a.interval),
                 max_seconds=a.max_seconds, log=lambda m: print(m, flush=True), refresh=refresh)
    return 2 if states.get(FAILED) or states.get(TRUNCATED) else 0


if __name__ == "__main__":
    sys.exit(main())
