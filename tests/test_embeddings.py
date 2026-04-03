"""Tests for src/embeddings.py — FAISS index build, persistence, and search."""

import json

import faiss
import numpy as np
import pytest

from src.embeddings import build_faiss_index, save_index, load_index, search


# ---------------------------------------------------------------------------
# Mock encoder — avoids downloading / loading the real SentenceTransformer model
# ---------------------------------------------------------------------------

DIM = 8  # tiny embedding dimension for tests


class _MockEncoder:
    """Deterministic mock replacing SentenceTransformer."""

    def __init__(self, model_name: str) -> None:
        self._rng = np.random.default_rng(42)

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return a fixed-seed random embedding of shape (len(texts), DIM)."""
        return self._rng.random((len(texts), DIM)).astype(np.float32)


@pytest.fixture(autouse=False)
def mock_encoder(monkeypatch):
    """Patch SentenceTransformer in src.embeddings with the mock encoder."""
    monkeypatch.setattr("src.embeddings.SentenceTransformer", _MockEncoder)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_faiss_index(mock_encoder, sample_corpus_for_embed):
    """EMBED-01 + EMBED-02: build_faiss_index returns correct index and page_ids."""
    index, page_ids = build_faiss_index(sample_corpus_for_embed)

    # Correct return types
    assert isinstance(index, faiss.IndexFlatIP)
    assert isinstance(page_ids, list)

    # One vector per corpus page
    assert index.ntotal == len(sample_corpus_for_embed)

    # page_ids aligned with corpus titles
    expected_ids = [p["title"] for p in sample_corpus_for_embed]
    assert page_ids == expected_ids


def test_save_load_roundtrip(mock_encoder, sample_corpus_for_embed, tmp_path):
    """EMBED-03: save_index + load_index preserves index.ntotal and page_ids."""
    index, page_ids = build_faiss_index(sample_corpus_for_embed)
    save_index(index, page_ids, tmp_path)

    # Files written
    assert (tmp_path / "faiss_index.bin").exists()
    assert (tmp_path / "faiss_ids.json").exists()

    # Load back
    loaded_index, loaded_ids = load_index(tmp_path)

    assert isinstance(loaded_index, faiss.IndexFlatIP)
    assert loaded_index.ntotal == index.ntotal
    assert loaded_ids == page_ids


def test_search_returns_pages(mock_encoder, sample_corpus_for_embed):
    """search() returns k page dicts, each with a 'score' key."""
    index, page_ids = build_faiss_index(sample_corpus_for_embed)

    k = 2
    results = search(
        query="physicist quantum",
        index=index,
        page_ids=page_ids,
        corpus=sample_corpus_for_embed,
        k=k,
    )

    # Correct number of results
    assert len(results) == k

    # Each result is a page dict augmented with 'score'
    for result in results:
        assert "title" in result
        assert "text" in result
        assert "html" in result
        assert "links" in result
        assert "score" in result
        assert isinstance(result["score"], float)

    # Titles come from the corpus
    corpus_titles = {p["title"] for p in sample_corpus_for_embed}
    for result in results:
        assert result["title"] in corpus_titles
