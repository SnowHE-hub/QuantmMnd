"""quantmind.line_a.atomic — 原子写文件与哈希.

先写到同目录的临时文件，fsync 后用 os.replace 改名：读者要么看到旧文件，要么看到完整的新文件。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> str:
    """原子写入，返回内容的 sha256。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return sha256_bytes(data)


def canonical_json(obj: Any) -> bytes:
    """确定性的 JSON 序列化（键排序、无多余空白），用于哈希和任务键。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def atomic_write_json(path: str | os.PathLike, obj: Any, *, indent: int | None = 1) -> str:
    data = json.dumps(obj, ensure_ascii=False, indent=indent, default=str).encode("utf-8")
    return atomic_write_bytes(path, data)
