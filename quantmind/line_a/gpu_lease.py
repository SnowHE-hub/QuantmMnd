"""quantmind.line_a.gpu_lease — GPU 租约（03_final_plan_v2 第 7.3 节）.

本机只有一块 8 GB 显卡，而 Windows 与 WSL 两边各有一个 Ollama（另有 Windows 侧的 llama.cpp）。
租约保证 QuantMind 自己的进程同一时刻只有一个持有显卡；拿到租约之前：
1. 请两边的 Ollama 卸载已加载的模型（WSL 侧走 HTTP，Windows 侧通过 WSL interop 调 ollama.exe）；
2. 等显存空闲量达到本次任务声明的需求，超时就失败（不会挤爆显存，也不会去杀别的进程）。
不受本租约管理的外部进程（例如手动启动的 llama.cpp），只能通过空闲显存检查发现。
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from filelock import FileLock, Timeout

DEFAULT_LOCK = Path.home() / ".cache" / "quantmind" / "gpu.lock"
WSL_OLLAMA = "http://127.0.0.1:11434"
WIN_OLLAMA_EXE = "/mnt/c/Users/lenovo/AppData/Local/Programs/Ollama/ollama.exe"


def nvidia_free_mib(run: Callable = subprocess.run) -> int:
    out = run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
              capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def unload_wsl_ollama(base: str = WSL_OLLAMA, timeout: float = 10.0) -> list[str]:
    """卸载 WSL 侧 Ollama 已加载的模型；服务没开就返回空列表。"""
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=timeout) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []
    for name in models:
        body = json.dumps({"model": name, "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{base}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with contextlib.suppress(urllib.error.URLError, OSError):
            urllib.request.urlopen(req, timeout=timeout).read()
    return models


def unload_windows_ollama(exe: str = WIN_OLLAMA_EXE, run: Callable = subprocess.run) -> list[str]:
    """卸载 Windows 侧 Ollama 已加载的模型（`ollama ps` → `ollama stop`）；不可用就返回空列表。"""
    if not os.path.exists(exe):
        return []
    try:
        out = run([exe, "ps"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    names = [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]
    for name in names:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            run([exe, "stop", name], capture_output=True, text=True, timeout=60)
    return names


class GpuLeaseError(RuntimeError):
    pass


class GpuLease:
    """用法：
        with GpuLease(holder="T-032 VT scoring", need_mib=4500):
            ...  # 加载模型、推理、卸载
    """

    def __init__(self, holder: str, need_mib: int, *, lock_path: Path = DEFAULT_LOCK, lock_timeout: float = 3600,
                 free_timeout: float = 120, poll: float = 5.0, free_servers: bool = True,
                 free_mib: Callable[[], int] = nvidia_free_mib,
                 unloaders: tuple[Callable[[], list[str]], ...] = (unload_wsl_ollama, unload_windows_ollama),
                 sleep: Callable[[float], None] = time.sleep):
        self.holder, self.need_mib = holder, need_mib
        self.lock_path = Path(lock_path)
        self.lock_timeout, self.free_timeout, self.poll = lock_timeout, free_timeout, poll
        self.free_servers, self.free_mib, self.unloaders, self.sleep = free_servers, free_mib, unloaders, sleep
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.lock_path))
        self.info: dict = {}

    @property
    def holder_file(self) -> Path:
        return self.lock_path.with_suffix(".holder.json")

    def __enter__(self) -> GpuLease:
        try:
            self._lock.acquire(timeout=self.lock_timeout)
        except Timeout as e:
            raise GpuLeaseError(f"GPU lease busy for {self.lock_timeout}s") from e
        try:
            unloaded = []
            if self.free_servers:
                for f in self.unloaders:
                    unloaded += f()
            waited = 0.0
            free = self.free_mib()
            while free < self.need_mib:
                if waited >= self.free_timeout:
                    raise GpuLeaseError(f"only {free} MiB free after {waited:.0f}s; need {self.need_mib} MiB")
                self.sleep(self.poll)
                waited += self.poll
                free = self.free_mib()
            self.info = {"holder": self.holder, "pid": os.getpid(), "need_mib": self.need_mib, "free_mib": free,
                         "unloaded": unloaded, "acquired_at": time.time()}
            self.holder_file.write_text(json.dumps(self.info, ensure_ascii=False), encoding="utf-8")
            return self
        except BaseException:
            self._lock.release()
            raise

    def __exit__(self, *exc) -> None:
        try:
            self.holder_file.unlink(missing_ok=True)
        finally:
            self._lock.release()
