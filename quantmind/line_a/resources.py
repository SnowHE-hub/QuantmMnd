"""quantmind.line_a.resources — 磁盘预算.

WSL 的根文件系统是 D:\\WSL\\Ubuntu 下的虚拟盘：`df /` 显示的是虚拟盘的剩余容量（约 830 GB），
但虚拟盘的实际增长受 D 盘剩余空间限制（2026-10-21 实测约 240 GB）。有效剩余空间取两者的较小值。
"""
from __future__ import annotations

import shutil
from pathlib import Path


def dir_size_bytes(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def effective_free_bytes(vhd_mount: str = "/", host_mount: str = "/mnt/d") -> dict:
    vhd = shutil.disk_usage(vhd_mount).free
    host = shutil.disk_usage(host_mount).free if Path(host_mount).exists() else None
    eff = vhd if host is None else min(vhd, host)
    return {"vhd_free": vhd, "host_free": host, "effective_free": eff}


def check_cache_budget(cache_dir: str | Path, cap_bytes: int, *, min_effective_free: int,
                       vhd_mount: str = "/", host_mount: str = "/mnt/d") -> dict:
    """缓存目录是否还能写：缓存大小不超过上限，且有效剩余空间不低于下限。"""
    used = dir_size_bytes(cache_dir)
    free = effective_free_bytes(vhd_mount, host_mount)
    ok = used < cap_bytes and free["effective_free"] >= min_effective_free
    return {"cache_used": used, "cache_cap": cap_bytes, **free, "min_effective_free": min_effective_free, "ok": ok}
