"""Tests for src/evaluate.py — run_evaluation output structure and correctness."""

import numpy as np
import pytest

import faiss
from src.embeddings import build_faiss_index
from src.evaluate import run_evaluation
from src.retrieval import naive_rag


# ---------------------------------------------------------------------------
# Mock encoder — same pattern as test_retrieval.py
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
        return self._rng.random((len(texts), DIM)).astype(np.float32)


@pytest.fixture(autouse=False)
def mock_encoder(monkeypatch):
    """Patch SentenceTransformer in src.embeddings with the mock encoder."""
    monkeypatch.setattr("src.embeddings.SentenceTransformer", _MockEncoder)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_evaluation_returns_scores(mock_encoder, sample_corpus_for_embed, sample_qa_items):
    """run_evaluation returns dict with avg_em, avg_f1 floats and per_question list."""
    index, _page_ids = build_faiss_index(sample_corpus_for_embed)

    results = run_evaluation(
        naive_rag,
        sample_qa_items[:2],
        k=2,
        index=index,
        corpus=sample_corpus_for_embed,
    )

    assert "avg_em" in results
    assert "avg_f1" in results
    assert "per_question" in results

    assert isinstance(results["avg_em"], float)
    assert isinstance(results["avg_f1"], float)
    assert 0.0 <= results["avg_em"] <= 1.0
    assert 0.0 <= results["avg_f1"] <= 1.0
    assert len(results["per_question"]) == 2


def test_run_evaluation_per_question_keys(mock_encoder, sample_corpus_for_embed, sample_qa_items):
    """Each per_question entry has question_id, em, and f1 keys."""
    index, _page_ids = build_faiss_index(sample_corpus_for_embed)

    results = run_evaluation(
        naive_rag,
        sample_qa_items[:1],
        k=2,
        index=index,
        corpus=sample_corpus_for_embed,
    )

    entry = results["per_question"][0]
    assert "question_id" in entry
    assert "em" in entry
    assert "f1" in entry
    assert isinstance(entry["em"], float)
    assert isinstance(entry["f1"], float)
