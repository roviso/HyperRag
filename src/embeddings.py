"""Embeddings module: sentence-transformers encoding and FAISS index management."""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_INDEX_DIR = Path("data")


def build_faiss_index(
    corpus: list[dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[Any, list[str]]:
    """Build a FAISS flat inner-product index from corpus page texts.

    Embeddings are L2-normalised so inner-product equals cosine similarity.

    Args:
        corpus: List of page dicts with 'title' and 'text' keys.
        model_name: Sentence-transformers model identifier.

    Returns:
        Tuple of (faiss.IndexFlatIP, page_ids) where page_ids is a list of
        page titles aligned with the index rows.
    """
    model = SentenceTransformer(model_name)
    texts = [page["text"] for page in corpus]
    page_ids = [page["title"] for page in corpus]

    print(f"Encoding {len(texts)} pages with {model_name} ...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)  # in-place; makes IP == cosine similarity

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, page_ids


def save_index(
    index: Any,
    page_ids: list[str],
    path: str | Path = DEFAULT_INDEX_DIR,
) -> None:
    """Save FAISS index and page ID list to a directory.

    Writes two files:
        <path>/faiss_index.bin  — binary FAISS index
        <path>/faiss_ids.json   — JSON list of page title IDs

    Args:
        index: FAISS index object (must support faiss.write_index).
        page_ids: List of page title strings aligned with index rows.
        path: Directory to save into. Created if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path / "faiss_index.bin"))
    with open(path / "faiss_ids.json", "w", encoding="utf-8") as f:
        json.dump(page_ids, f)
    print(f"Index saved to {path} ({index.ntotal} vectors)")


def load_index(path: str | Path = DEFAULT_INDEX_DIR) -> tuple[Any, list[str]]:
    """Load FAISS index and page ID list from a directory.

    Reads:
        <path>/faiss_index.bin  — binary FAISS index
        <path>/faiss_ids.json   — JSON list of page title IDs

    Args:
        path: Directory containing the saved index files.

    Returns:
        Tuple of (faiss_index, page_ids list).
    """
    path = Path(path)
    index = faiss.read_index(str(path / "faiss_index.bin"))
    with open(path / "faiss_ids.json", encoding="utf-8") as f:
        page_ids = json.load(f)
    print(f"Index loaded from {path} ({index.ntotal} vectors)")
    return index, page_ids


def search(
    query: str,
    index: Any,
    page_ids: list[str],
    corpus: list[dict[str, Any]],
    k: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict[str, Any]]:
    """Search the FAISS index for top-k pages matching a query.

    Encodes the query with the same model used to build the index, normalises
    the vector, and returns the top-k closest corpus pages.

    Args:
        query: Natural-language question or search string.
        index: FAISS index (built with build_faiss_index).
        page_ids: List of page titles aligned with index rows.
        corpus: Full corpus for retrieving page data by title.
        k: Number of top results to return.
        model_name: Sentence-transformers model (must match index build model).

    Returns:
        List of up to k page dicts, each augmented with a 'score' float key
        (cosine similarity in [0, 1]).
    """
    model = SentenceTransformer(model_name)
    title_to_page: dict[str, dict[str, Any]] = {p["title"]: p for p in corpus}

    q_emb = model.encode([query]).astype(np.float32)
    faiss.normalize_L2(q_emb)

    distances, indices = index.search(q_emb, k)
    results: list[dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(page_ids):
            continue
        title = page_ids[idx]
        page = title_to_page.get(title)
        if page is not None:
            results.append({**page, "score": float(dist)})
    return results


def build_and_save_index(
    corpus: list[dict[str, Any]],
    output_dir: str | Path = DEFAULT_INDEX_DIR,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    force: bool = False,
) -> tuple[Any, list[str]]:
    """Build and persist the FAISS index with cache-first behaviour.

    Skips the embedding step if both faiss_index.bin and faiss_ids.json
    already exist in output_dir (and force is False).

    Args:
        corpus: List of page dicts with 'title' and 'text' keys.
        output_dir: Directory to save/load index files.
        model_name: Sentence-transformers model identifier.
        force: If True, rebuild even if cached files exist.

    Returns:
        Tuple of (faiss_index, page_ids).
    """
    output_dir = Path(output_dir)
    index_file = output_dir / "faiss_index.bin"
    ids_file = output_dir / "faiss_ids.json"

    if index_file.exists() and ids_file.exists() and not force:
        print(f"Loading cached FAISS index from {output_dir}")
        return load_index(output_dir)

    index, page_ids = build_faiss_index(corpus, model_name=model_name)
    save_index(index, page_ids, output_dir)
    return index, page_ids
