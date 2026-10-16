"""T-031 L4：前瞻新闻采集（7×24 快讯）。免费源没有历史，只能从现在起积累；时点以"首次看到的时间"为准。

来源（AKShare，2026-10-17 实测可用，每次只返回最近约 20 条 → 每 10 分钟拉一次）：
  sina_global  新浪全球财经快讯（列：时间、内容）
  ths_global   同花顺全球财经直播（列：标题、内容、发布时间、链接）
记录：source, item_id（来源 + 发布时间 + 内容前 200 字的 sha1）, publish_time_bj（来源给的时间，按北京时间解释）,
      title, content, url, first_seen_ts_bj（本次采集时间）。
产物：<root>/data/lake/text/news_fwd/<source>/<YYYYMMDD>.parquet（按首次看到的北京日期分文件，只追加新 item_id）；
      <root>/data/lake/text/news_fwd/_seen/<source>.json（最近 20,000 个 item_id，跨天去重）。
每个来源单独 45 秒超时，一个来源失败不影响另一个。用法：python collect_news_fwd.py --data-root <root>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BJ = timezone(timedelta(hours=8))
SEEN_KEEP = 20_000
COLS = ("source", "item_id", "publish_time_bj", "title", "content", "url", "first_seen_ts_bj")


def _ak_sina():
    import akshare as ak
    df = ak.stock_info_global_sina()
    return [{"publish_time": str(r["时间"]), "title": "", "content": str(r["内容"]), "url": ""} for _, r in df.iterrows()]


def _ak_ths():
    import akshare as ak
    df = ak.stock_info_global_ths()
    return [{"publish_time": str(r["发布时间"]), "title": str(r["标题"]), "content": str(r["内容"]),
             "url": str(r["链接"])} for _, r in df.iterrows()]


SOURCES = {"sina_global": _ak_sina, "ths_global": _ak_ths}


def item_id(source: str, publish_time: str, content: str) -> str:
    return hashlib.sha1(f"{source}|{publish_time}|{content[:200]}".encode()).hexdigest()


class _TimeoutError(Exception):
    pass


def _with_timeout(fn, seconds: int):
    def _alarm(*_):
        raise _TimeoutError(f"timeout {seconds}s")
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def collect(out_dir: Path, sources: dict, now: datetime | None = None, timeout: int = 45, log=print) -> dict:
    now = now or datetime.now(BJ)
    seen_dir = out_dir / "_seen"
    seen_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, fetch in sources.items():
        try:
            items = _with_timeout(fetch, timeout)
        except Exception as e:  # noqa: BLE001
            report[name] = f"error {type(e).__name__}: {str(e)[:100]}"
            log(f"[news] {name} {report[name]}")
            continue
        seen_path = seen_dir / f"{name}.json"
        seen = json.loads(seen_path.read_text(encoding="utf-8")) if seen_path.exists() else []
        seen_set = set(seen)
        new = []
        for it in items:
            iid = item_id(name, it["publish_time"], it["content"])
            if iid in seen_set:
                continue
            seen_set.add(iid)
            seen.append(iid)
            new.append({"source": name, "item_id": iid, "publish_time_bj": it["publish_time"], "title": it["title"],
                        "content": it["content"], "url": it["url"], "first_seen_ts_bj": now.isoformat()})
        if new:
            day_file = out_dir / name / f"{now.strftime('%Y%m%d')}.parquet"
            day_file.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(new, columns=list(COLS))
            if day_file.exists():
                df = pd.concat([pd.read_parquet(day_file), df], ignore_index=True)
            tmp = day_file.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, day_file)
        tmp = seen_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(seen[-SEEN_KEEP:]), encoding="utf-8")
        os.replace(tmp, seen_path)
        report[name] = f"fetched {len(items)} new {len(new)}"
    log(f"[news] {now.isoformat()} {report}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True)
    a = ap.parse_args(argv)
    rep = collect(Path(a.data_root) / "data/lake/text/news_fwd", SOURCES, log=lambda m: print(m, flush=True))
    return 0 if any(v.startswith("fetched") for v in rep.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
