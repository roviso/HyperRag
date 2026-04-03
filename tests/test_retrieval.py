"""Tests for src/retrieval.py — B1, B2 retrievers and EM/F1 metrics."""

import numpy as np
import pytest

import faiss
from src.embeddings import build_faiss_index
from src.retrieval import naive_rag, htmlrag_style, compute_em, compute_f1


# ---------------------------------------------------------------------------
# Mock encoder — avoids downloading the real SentenceTransformer model
# ---------------------------------------------------------------------------

DIM = 8


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
        """Return fixed-seed random embeddings of shape (len(texts), DIM)."""
        return self._rng.random((len(texts), DIM)).astype(np.float32)


@pytest.fixture(autouse=False)
def mock_encoder(monkeypatch):
    """Patch SentenceTransformer in src.embeddings with the mock encoder."""
    monkeypatch.setattr("src.embeddings.SentenceTransformer", _MockEncoder)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def retrieval_setup(mock_encoder, sample_corpus_for_embed):
    """Build a FAISS index over sample_corpus_for_embed and return (index, corpus)."""
    index, _page_ids = build_faiss_index(sample_corpus_for_embed)
    return index, sample_corpus_for_embed


# ---------------------------------------------------------------------------
# B1 — naive_rag tests
# ---------------------------------------------------------------------------


def test_naive_rag_returns_text(retrieval_setup):
    """B1-01 + B1-02: naive_rag returns a non-empty string."""
    index, corpus = retrieval_setup
    result = naive_rag("Who studied quantum mechanics?", index, corpus, k=2)
    assert isinstance(result, str)
    assert len(result) > 0


def test_naive_rag_context_contains_corpus_text(retrieval_setup):
    """B1-02: naive_rag output contains plain text from retrieved pages."""
    index, corpus = retrieval_setup
    result = naive_rag("physicist", index, corpus, k=1)
    # At least one corpus page's text should appear in the result
    corpus_texts = [p["text"] for p in corpus]
    assert any(t in result for t in corpus_texts)


# ---------------------------------------------------------------------------
# B2 — htmlrag_style tests
# ---------------------------------------------------------------------------


def test_htmlrag_returns_string(retrieval_setup):
    """B2-01 + B2-02: htmlrag_style returns a non-empty string."""
    index, corpus = retrieval_setup
    result = htmlrag_style("Who studied quantum mechanics?", index, corpus, k=2)
    assert isinstance(result, str)
    assert len(result) > 0


def test_htmlrag_preserves_html_structure(retrieval_setup):
    """B2-02: htmlrag_style output contains at least one structural HTML tag."""
    index, corpus = retrieval_setup
    result = htmlrag_style("physicist", index, corpus, k=2)
    # sample_corpus_for_embed pages have <p> tags in their html field
    assert "<p>" in result


# ---------------------------------------------------------------------------
# EM metric tests (B1-03, B2-03)
# ---------------------------------------------------------------------------


def test_compute_em_match():
    """compute_em returns 1.0 when gold appears in prediction (case-insensitive)."""
    assert compute_em("The answer is France", "France") == 1.0
    assert compute_em("albert einstein was born in germany", "Albert Einstein") == 1.0


def test_compute_em_no_match():
    """compute_em returns 0.0 when gold is absent from prediction."""
    assert compute_em("nothing relevant here about quantum physics", "France") == 0.0


# ---------------------------------------------------------------------------
# F1 metric tests (B1-03, B2-03)
# ---------------------------------------------------------------------------


def test_compute_f1_exact():
    """compute_f1 returns 1.0 for identical strings."""
    assert compute_f1("France", "France") == 1.0


def test_compute_f1_partial():
    """compute_f1 returns a value in (0, 1) for partial token overlap."""
    score = compute_f1("Albert Einstein physicist germany", "Albert Einstein")
    assert 0.0 < score < 1.0


def test_compute_f1_no_overlap():
    """compute_f1 returns 0.0 when there is no token overlap."""
    assert compute_f1("nothing here", "France quantum") == 0.0
