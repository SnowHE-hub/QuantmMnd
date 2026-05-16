"""HybridRetriever: ensure Chroma init does not deadlock (nested lock)."""

from __future__ import annotations

from pathlib import Path

from quantmind.kb.retriever import HybridRetriever


def test_get_collection_no_deadlock(tmp_path: Path) -> None:
    r = HybridRetriever(
        collection_name="default",
        chroma_dir=str(tmp_path / "chroma"),
        use_reranker=False,
    )
    coll = r._get_collection()
    assert coll is not None
    assert coll.name == "quantmind_kb_default"
