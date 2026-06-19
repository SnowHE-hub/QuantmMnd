"""F-13：轻量 run manifest —— 训练/评估脚本结束时落运行元数据 JSON，便于溯源复现。

仅用 stdlib（可在 qlib_bakeoff / quantmind 任一 env 使用，不引 quantmind 重依赖）。
字段：git_sha, env, python, data_version, feature_set, label, horizon, timestamp, kind, extra。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """文件 sha256（流式，省内存）。文件不存在返回 'missing'。"""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()[:12]
    except Exception:
        return "unknown"


def _env_name() -> str:
    e = os.environ.get("CONDA_DEFAULT_ENV")
    if e:
        return e
    # 退路：从解释器路径推断 env 名（…/envs/<name>/bin/python）
    p = Path(sys.executable)
    return p.parent.parent.name if "envs" in str(p) else "system"


def write_run_manifest(
    out_dir: str | Path,
    *,
    kind: str,                       # "train" | "eval"
    data_version: str | None = None,
    feature_set: str | None = None,
    label: str | None = None,
    horizon: str | int | None = None,
    extra: dict | None = None,
    artifacts: list[str | Path] | None = None,
    repo: str | Path | None = None,
) -> Path:
    """写运行元数据 JSON 到 out_dir，返回路径。自动采集 git_sha/env/python/timestamp。

    artifacts: 落盘产物路径列表 → 记录各自 sha256（纪律：v* 面板首次落盘即预录哈希，
    防止后续静默改写无从察觉）。
    """
    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    manifest = {
        "kind": kind,
        "git_sha": _git_sha(repo),
        "env": _env_name(),
        "python": sys.version.split()[0],
        "data_version": data_version,
        "feature_set": feature_set,
        "label": label,
        "horizon": str(horizon) if horizon is not None else None,
        "timestamp": ts,
        "artifacts_sha256": {str(Path(a).name): sha256_file(a) for a in (artifacts or [])},
        "extra": extra or {},
    }
    stamp = ts.replace(":", "").replace("-", "").replace("T", "_")
    fn = out_dir / f"run_manifest_{kind}_{stamp}.json"
    fn.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # ModelRegistry 联动：训练 manifest 自动注册一条研究候选（懒导入，保持 stdlib-only 可用性）。
    if kind == "train":
        try:
            from quantmind.contracts.model_registry import register_from_manifest
            register_from_manifest(manifest)
        except Exception:
            pass
    return fn


if __name__ == "__main__":
    # 自测
    p = write_run_manifest("/tmp/_rm_test", kind="train", data_version="v5",
                           feature_set="35+16+158", label="forward_return_12d", horizon="12d",
                           extra={"model": "ridge"})
    print("wrote", p)
    print(p.read_text())
