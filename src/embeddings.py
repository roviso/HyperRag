"""Embeddings module: sentence-transformers encoding and FAISS index management."""

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INDEX_DIR = Path("data")


def build_faiss_index(
    corpus: list[dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple:
    """Build a FAISS index from corpus page texts.

    Args:
        corpus: List of page dicts with 'text' key.
        model_name: Sentence-transformers model to use.

    Returns:
        Tuple of (faiss_index, page_ids list).
    """
    raise NotImplementedError("Implemented in Phase 4")


def save_index(index: Any, page_ids: list[str], path: str | Path = DEFAULT_INDEX_DIR) -> None:
    """Save FAISS index and page ID mapping to disk.

    Args:
        index: FAISS index object.
        page_ids: List of page IDs corresponding to index rows.
        path: Directory to save files in.
    """
    raise NotImplementedError("Implemented in Phase 4")


def load_index(path: str | Path = DEFAULT_INDEX_DIR) -> tuple:
    """Load FAISS index and page ID mapping from disk.

    Args:
        path: Directory containing saved index files.

    Returns:
        Tuple of (faiss_index, page_ids list).
    """
    raise NotImplementedError("Implemented in Phase 4")


def search(
    query: str,
    index: Any,
    page_ids: list[str],
    corpus: list[dict[str, Any]],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Search FAISS index for top-k pages matching query.

    Args:
        query: Query string to search for.
        index: FAISS index.
        page_ids: List of page IDs.
        corpus: Full corpus for retrieving page data.
        k: Number of results to return.

    Returns:
        List of top-k page dicts with similarity scores.
    """
    raise NotImplementedError("Implemented in Phase 4")
