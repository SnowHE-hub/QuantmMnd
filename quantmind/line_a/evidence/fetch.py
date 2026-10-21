"""quantmind.line_a.evidence.fetch — 公告 PDF 按需获取（03_final_plan_v2 第 5 节 F5(b)1、第 8.4 节工具层安全）.

- 只接受登记过的来源：巨潮资讯静态站 static.cninfo.com.cn，路径必须是元数据里的 `finalpage/<日期>/<编号>.PDF`；
- 不跟随跳转（有跳转就拒绝）；检查响应类型、`%PDF` 文件头和大小上限；
- 每次请求之间至少间隔 1 秒；同一份文件只取一次，缓存在 <cache>/<年-月>/<公告编号>.pdf，并记录 sha256；
- 这里只取公开披露文件，不带任何账户或个人信息。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

HOST = "https://static.cninfo.com.cn/"
_PATH = re.compile(r"^finalpage/\d{4}-\d{2}-\d{2}/\d{6,14}\.(?:PDF|pdf)$")
MAX_BYTES = 30 * 1024 * 1024
MIN_INTERVAL = 1.0
_last = [0.0]


class FetchRefusedError(ValueError):
    pass


def cache_path(cache_dir: Path, ann_id: str, adjunct_url: str) -> Path:
    return Path(cache_dir) / adjunct_url.split("/")[1][:7] / f"{ann_id}.pdf"


def fetch_pdf(ann_id: str, adjunct_url: str, cache_dir: Path, *, transport=None) -> dict:
    """返回 {path, sha256, bytes, cached}。transport：测试注入 callable(url) -> (status, headers, body)。"""
    if not re.fullmatch(r"\d{6,14}", str(ann_id)) or not _PATH.match(adjunct_url or ""):
        raise FetchRefusedError(f"unregistered or malformed source: {adjunct_url!r}")
    if not adjunct_url.split("/")[-1].startswith(str(ann_id)):
        raise FetchRefusedError("announcement id does not match the file name")
    p = cache_path(cache_dir, ann_id, adjunct_url)
    meta = p.with_suffix(".json")
    if p.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        return {**m, "path": str(p), "cached": True}
    wait = MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    url = HOST + adjunct_url
    if transport is None:
        import httpx
        r = httpx.get(url, follow_redirects=False, timeout=60, headers={"User-Agent": "Mozilla/5.0 (research)"})
        status, headers, body = r.status_code, dict(r.headers), r.content
    else:
        status, headers, body = transport(url)
    _last[0] = time.time()
    if 300 <= status < 400:
        raise FetchRefusedError(f"redirect refused ({status})")
    if status != 200:
        raise RuntimeError(f"HTTP {status} for {url}")
    ctype = (headers.get("content-type") or headers.get("Content-Type") or "").lower()
    if ctype and not any(x in ctype for x in ("pdf", "octet-stream")):
        raise FetchRefusedError(f"unexpected content type {ctype}")
    if len(body) > MAX_BYTES:
        raise FetchRefusedError(f"file too large ({len(body)} bytes)")
    if not body.startswith(b"%PDF"):
        raise FetchRefusedError("not a PDF file")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, p)
    m = {"ann_id": str(ann_id), "url": url, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
         "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return {**m, "path": str(p), "cached": False}
