"""Pre-download and smoke-test BGE-M3 via EmbeddingService (no Chroma / KB / AkShare).

Environment:
    HF_ENDPOINT=https://hf-mirror.com  (optional mirror)

Note:
    Some mirrors may return 403 on certain repo artifacts (e.g. BAAI/bge-m3). If so,
    run once without HF_ENDPOINT to use huggingface.co, then cached weights load offline.

Logs:
    Use --log-file (e.g. reports/prefetch_bge_m3.log); stdout/stderr can be tee'd separately.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

DEFAULT_TEXTS = (
    "贵州茅台是一家白酒公司",
    "宁德时代是一家新能源电池公司",
)


def _hf_hub_cache_dir() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return str(HF_HUB_CACHE)
    except Exception:
        home = Path.home()
        return str(home / ".cache" / "huggingface" / "hub")


def parse_prefetch_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prefetch BAAI/bge-m3 and run a tiny embed_batch smoke test.",
    )
    p.add_argument("--model-name", default="BAAI/bge-m3")
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="EmbeddingService device: auto uses cuda if available else cpu.",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--log-file",
        type=Path,
        default=Path("reports/prefetch_bge_m3.log"),
    )
    p.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=None,
        help="Override EmbeddingService on-disk embedding JSON cache (default .cache/embeddings).",
    )
    return p.parse_args(argv)


def _append_log(log_file: Path, line: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run_prefetch_smoke(
    model_name: str,
    device_mode: str,
    batch_size: int,
    texts: list[str],
    log_file: Path,
    *,
    embedding_cache_dir: Path | None = None,
    service_factory: Callable[..., Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run download/load + embed_batch. Returns (exit_code, metrics).

    service_factory: for tests; default constructs quantmind.kb.embedding_service.EmbeddingService.
    """
    from quantmind.kb.embedding_service import EmbeddingService

    factory = service_factory or EmbeddingService
    resolved_device: str | None = None if device_mode == "auto" else device_mode
    vec_cache = embedding_cache_dir if embedding_cache_dir is not None else Path(".cache/embeddings")

    metrics: dict[str, Any] = {
        "model_name": model_name,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "hf_hub_cache": _hf_hub_cache_dir(),
        "embedding_vector_cache": str(vec_cache.resolve()),
        "test_text_count": len(texts),
    }

    _append_log(log_file, f"[prefetch] start {_now_iso()}")

    try:
        svc = factory(
            model_name=model_name,
            cache_dir=vec_cache,
            batch_size=batch_size,
            device=resolved_device,
            use_cache=True,
        )
    except Exception as e:
        msg = f"[prefetch] EmbeddingService 构造失败: {e}"
        logger.error(msg)
        _append_log(log_file, msg)
        metrics["error"] = "service_init"
        return 1, metrics

    dim = int(svc.dim)
    metrics["embedding_dimension"] = dim

    _append_log(
        log_file,
        f"model_name={model_name} device_mode={device_mode} "
        f"batch_size={batch_size} dim={dim} texts={len(texts)}",
    )
    _append_log(
        log_file,
        f"HF_ENDPOINT={metrics['hf_endpoint'] or '(unset)'} "
        f"hf_hub_cache={metrics['hf_hub_cache']} "
        f"embedding_vector_cache={metrics['embedding_vector_cache']}",
    )

    load_start = time.perf_counter()
    load_start_iso = _now_iso()
    _append_log(log_file, f"[prefetch] first embed_batch start (includes HF download + model load): {load_start_iso}")
    print(f"[prefetch] first embed_batch start: {load_start_iso}", flush=True)

    try:
        vectors = svc.embed_batch(list(texts))
    except ImportError as e:
        msg = f"[prefetch] FlagEmbedding 缺失或依赖不完整: {e}"
        logger.error(msg)
        _append_log(log_file, msg)
        metrics["error"] = "flagembedding_import"
        return 1, metrics
    except OSError as e:
        msg = f"[prefetch] 网络或磁盘错误（含下载失败可能）: {e}"
        logger.error(msg)
        _append_log(log_file, msg)
        metrics["error"] = "network_or_os"
        return 1, metrics
    except Exception as e:
        err = str(e).lower()
        if any(x in err for x in ("connection", "timeout", "443", "ssl", "resolve", "unreachable")):
            kind = "network_download"
            msg = f"[prefetch] 网络下载失败: {e}"
        else:
            kind = "model_or_embed"
            msg = f"[prefetch] 模型加载或推理失败: {e}"
        logger.error(msg)
        _append_log(log_file, msg)
        metrics["error"] = kind
        return 1, metrics

    load_done = time.perf_counter()
    load_done_iso = _now_iso()
    total_sec = load_done - load_start
    metrics["load_window_seconds"] = round(total_sec, 3)
    metrics["success"] = True

    _append_log(
        log_file,
        f"[prefetch] first embed_batch done: {load_done_iso} total_seconds={total_sec:.3f}",
    )
    print(
        f"[prefetch] first embed_batch done: {load_done_iso} total_seconds={total_sec:.3f}",
        flush=True,
    )

    if len(vectors) != len(texts):
        msg = f"[prefetch] 输出条数异常: want {len(texts)} got {len(vectors)}"
        logger.error(msg)
        _append_log(log_file, msg)
        metrics["error"] = "bad_output_len"
        metrics["success"] = False
        return 1, metrics

    for v in vectors:
        if len(v) != dim:
            msg = f"[prefetch] 向量维度异常: want {dim} got {len(v)}"
            logger.error(msg)
            _append_log(log_file, msg)
            metrics["error"] = "bad_dim"
            metrics["success"] = False
            return 1, metrics

    _append_log(log_file, "[prefetch] OK success=true")
    print("[prefetch] OK success=true", flush=True)
    return 0, metrics


def main() -> None:
    args = parse_prefetch_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text("", encoding="utf-8")
    sink_id = logger.add(str(args.log_file), level="INFO", encoding="utf-8")
    try:
        code, metrics = run_prefetch_smoke(
            model_name=args.model_name,
            device_mode=args.device,
            batch_size=args.batch_size,
            texts=list(DEFAULT_TEXTS),
            log_file=args.log_file,
            embedding_cache_dir=args.embedding_cache_dir,
        )
    finally:
        logger.remove(sink_id)
    print(
        f"[prefetch] summary: exit={code} dim={metrics.get('embedding_dimension')} "
        f"seconds={metrics.get('load_window_seconds')} "
        f"error={metrics.get('error', '')}",
        flush=True,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
