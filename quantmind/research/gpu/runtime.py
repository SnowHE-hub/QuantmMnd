"""运行时：显存上限、空闲显存查询、原子写、checkpoint 与完成单元台账（任何进程被杀都能续跑）."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def free_vram_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0])
    except Exception:  # noqa: BLE001
        return None


def wait_for_vram(min_free_mb: int, poll_s: int = 60, log=print) -> int | None:
    """空闲显存不足（例如日更 Step7a 的 ollama 占着）时等待，不杀任何进程。"""
    while True:
        f = free_vram_mb()
        if f is None or f >= min_free_mb:
            return f
        log(f"[vram] free {f} MiB < {min_free_mb} MiB, waiting {poll_s}s")
        time.sleep(poll_s)


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("MemAvailable missing in /proc/meminfo")


RAM_GUARD_EXIT = 75


def start_ram_guard(min_avail_mb: int = 1500, poll_s: float = 2.0) -> None:
    """可用内存低于阈值就立刻退出本进程（退出码 75），把内存还给同机的生产定时任务；台账 / checkpoint 负责续跑。"""
    import sys
    import threading

    def loop() -> None:
        while True:
            a = mem_available_mb()
            if a < min_avail_mb:
                print(f"[ram_guard] MemAvailable {a} MiB < {min_avail_mb} MiB, exiting", file=sys.stderr, flush=True)
                os._exit(RAM_GUARD_EXIT)
            time.sleep(poll_s)

    threading.Thread(target=loop, daemon=True).start()


def cap_torch_memory(total_mb: int = 8151, budget_mb: int = 6500) -> None:
    """WSL 上显存超额时会静默溢出到共享内存（不报 OOM、只是变慢）。设上限让超额变成真正的 OOM，由调用方降 batch。"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget_mb / total_mb), 0)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(Path(path), json.dumps(obj, ensure_ascii=False, indent=1, default=str).encode("utf-8"))


def save_checkpoint(path: Path, state: dict) -> None:
    import io

    import torch
    buf = io.BytesIO()
    torch.save(state, buf)
    atomic_write_bytes(Path(path), buf.getvalue())


def load_checkpoint(path: Path) -> dict | None:
    import torch
    path = Path(path)
    return torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None


class UnitLedger:
    """每个完成的训练单元（族 / 期限 / 折 / 配置 / 种子）一个 JSON；存在即视为完成，续跑时跳过。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, unit: str) -> Path:
        return self.root / f"{unit}.json"

    def done(self, unit: str) -> bool:
        return self.path(unit).exists()

    def get(self, unit: str) -> dict | None:
        p = self.path(unit)
        return json.loads(p.read_text()) if p.exists() else None

    def mark(self, unit: str, record: dict) -> None:
        atomic_write_json(self.path(unit), record)

    def all(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.root.glob("*.json"))]
