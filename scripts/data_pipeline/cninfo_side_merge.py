"""scripts/data_pipeline/cninfo_side_merge.py — 把旁路下载的巨潮公告元数据合并进主目录（T-038，加快 X4 的数据准备）.

为什么要旁路：
- 主下载（T-031 的定时任务）是单线程、单一检查点文件。两个进程同时写这个检查点会互相覆盖；
- 所以第二个进程用同一份下载代码，把另一段日期下到单独的 data-root，下完后再合并。

合并规则：
- 只合并旁路检查点里已是终态（done / empty / partial）、而主检查点里还不是终态的日子；
- 每天一个分片文件，先写临时文件再改名；主检查点合并前先备份（cninfo_meta.json.bak_<时间>），合并后整体原子替换；
- 被合并的日子在检查点里加 "merged_from": "side"，以后能查出来源；
- 调用方负责先拿住 /tmp/qm_cninfo_meta.lock（与定时任务同一把锁），保证合并时主下载没有在运行。
用法：python scripts/data_pipeline/cninfo_side_merge.py <side_root> <main_root>
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

TERMINAL = {"done", "empty", "partial"}


def merge(side_root: Path, main_root: Path) -> dict:
    s = Path(side_root) / "data/lake/text/cninfo"
    m = Path(main_root) / "data/lake/text/cninfo"
    sck = json.loads((s / "_checkpoints/cninfo_meta.json").read_text(encoding="utf-8"))
    mck_path = m / "_checkpoints/cninfo_meta.json"
    mck = json.loads(mck_path.read_text(encoding="utf-8"))
    backup = mck_path.with_name(f"cninfo_meta.json.bak_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(mck_path, backup)
    merged, skipped = [], 0
    for day, rec in sorted(sck["days"].items()):
        if rec.get("state") not in TERMINAL or mck["days"].get(day, {}).get("state") in TERMINAL:
            skipped += 1
            continue
        f = day.replace("-", "") + ".parquet"
        if (s / "_parts" / f).exists():
            tmp = m / "_parts" / (f + ".tmp")
            shutil.copy2(s / "_parts" / f, tmp)
            os.replace(tmp, m / "_parts" / f)
        mck["days"][day] = {**rec, "merged_from": "side"}
        merged.append(day)
    tmp = mck_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mck, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, mck_path)
    return {"merged": len(merged), "first": merged[0] if merged else None, "last": merged[-1] if merged else None,
            "skipped": skipped, "backup": str(backup)}


if __name__ == "__main__":
    print(json.dumps(merge(Path(sys.argv[1]), Path(sys.argv[2])), ensure_ascii=False))
