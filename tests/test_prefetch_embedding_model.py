"""Tests for scripts/prefetch_embedding_model.py (no real model download)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "prefetch_embedding_model",
    _ROOT / "scripts" / "prefetch_embedding_model.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_mod)


def test_parse_prefetch_args_defaults():
    args = _mod.parse_prefetch_args([])
    assert args.model_name == "BAAI/bge-m3"
    assert args.device == "auto"
    assert args.batch_size == 2
    assert args.log_file == Path("reports/prefetch_bge_m3.log")
    assert args.embedding_cache_dir is None


def test_parse_prefetch_args_custom():
    args = _mod.parse_prefetch_args(
        [
            "--model-name",
            "BAAI/bge-m3",
            "--device",
            "cpu",
            "--batch-size",
            "8",
            "--log-file",
            "/tmp/x.log",
        ]
    )
    assert args.device == "cpu"
    assert args.batch_size == 8
    assert args.log_file == Path("/tmp/x.log")


def test_run_prefetch_smoke_mock_success(tmp_path):
    log_f = tmp_path / "p.log"

    class _FakeSvc:
        dim = 1024

        def embed_batch(self, texts: list[str]):
            return [[0.1] * 1024 for _ in texts]

    code, metrics = _mod.run_prefetch_smoke(
        "BAAI/bge-m3",
        "cpu",
        2,
        ["a", "b"],
        log_f,
        embedding_cache_dir=tmp_path / "ec",
        service_factory=lambda **kwargs: _FakeSvc(),
    )
    assert code == 0
    assert metrics.get("success") is True
    assert metrics["embedding_dimension"] == 1024
    assert metrics["test_text_count"] == 2
    text = log_f.read_text(encoding="utf-8")
    assert "OK success=true" in text


def test_run_prefetch_smoke_service_factory_raises(tmp_path):
    log_f = tmp_path / "p.log"

    def _boom(**kwargs):
        raise ImportError("cannot init")

    code, metrics = _mod.run_prefetch_smoke(
        "BAAI/bge-m3",
        "cpu",
        2,
        ["x"],
        log_f,
        embedding_cache_dir=tmp_path / "ec",
        service_factory=_boom,
    )
    assert code == 1
    assert metrics.get("error") == "service_init"


def test_run_prefetch_smoke_embed_import_error(tmp_path):
    log_f = tmp_path / "p.log"

    class _Svc:
        dim = 1024

        def embed_batch(self, texts):
            raise ImportError("FlagEmbedding missing")

    code, metrics = _mod.run_prefetch_smoke(
        "BAAI/bge-m3",
        "cpu",
        2,
        ["x"],
        log_f,
        embedding_cache_dir=tmp_path / "ec2",
        service_factory=lambda **kwargs: _Svc(),
    )
    assert code == 1
    assert metrics.get("error") == "flagembedding_import"


def test_run_prefetch_smoke_bad_vector_len(tmp_path):
    log_f = tmp_path / "p.log"

    class _Bad:
        dim = 1024

        def embed_batch(self, texts):
            return [[0.0] * 3]  # wrong dim

    code, metrics = _mod.run_prefetch_smoke(
        "BAAI/bge-m3",
        "cpu",
        2,
        ["a"],
        log_f,
        service_factory=lambda **kwargs: _Bad(),
    )
    assert code == 1
    assert metrics.get("error") == "bad_dim"


def test_main_exit_code_failure(monkeypatch, tmp_path):
    log_f = tmp_path / "x.log"

    def _fail(*a, **k):
        return 1, {"embedding_dimension": None, "error": "x"}

    monkeypatch.setattr(_mod, "parse_prefetch_args", lambda: MagicMock(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=2,
        log_file=log_f,
        embedding_cache_dir=None,
    ))
    monkeypatch.setattr(_mod, "run_prefetch_smoke", _fail)
    with pytest.raises(SystemExit) as ei:
        _mod.main()
    assert ei.value.code == 1
