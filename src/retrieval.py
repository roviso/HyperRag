"""Retrieval module: B1 (Naive RAG), B2 (HtmlRAG-style), and HyperRAG retrieval functions."""

from collections import Counter
from typing import Any

import networkx as nx
from bs4 import BeautifulSoup

from src.embeddings import search


def _clean_html(html: str) -> str:
    """Clean raw HTML, preserving structural tags for context.

    Removes script, style, and nav elements. Returns cleaned HTML string
    with h1, h2, p, table, and li tags preserved.

    Args:
        html: Raw HTML string from corpus page.

    Returns:
        Cleaned HTML string with structural tags intact.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav"]):
        tag.decompose()
    return str(soup)


def naive_rag(query: str, index: Any, corpus: list[dict[str, Any]], k: int = 5) -> str:
    """B1 - Naive RAG: retrieve top-K pages via FAISS, return plain text context.

    Derives page_ids from corpus titles, calls search(), and concatenates
    plain text of the top-K retrieved pages.

    Args:
        query: User query string.
        index: FAISS index (faiss.IndexFlatIP, built with build_faiss_index).
        corpus: Full corpus list of page dicts.
        k: Number of pages to retrieve.

    Returns:
        Concatenated plain text context string (pages separated by double newline).
    """
    page_ids = [p["title"] for p in corpus]
    pages = search(query, index, page_ids, corpus, k=k)
    return "\n\n".join(p["text"] for p in pages)


def htmlrag_style(query: str, index: Any, corpus: list[dict[str, Any]], k: int = 5) -> str:
    """B2 - HtmlRAG-style: retrieve top-K pages, return cleaned HTML context.

    Preserves structural HTML tags (h1, h2, p, table, li) while removing
    scripts, styles, and nav elements. Same retrieval as naive_rag.

    Args:
        query: User query string.
        index: FAISS index (faiss.IndexFlatIP, built with build_faiss_index).
        corpus: Full corpus list of page dicts.
        k: Number of pages to retrieve.

    Returns:
        Cleaned HTML context string with structural tags preserved
        (pages separated by double newline).
    """
    page_ids = [p["title"] for p in corpus]
    pages = search(query, index, page_ids, corpus, k=k)
    return "\n\n".join(_clean_html(p["html"]) for p in pages)


def compute_em(prediction: str, gold: str) -> float:
    """Compute Exact Match score for retrieval quality.

    Checks whether the gold answer string appears (case-insensitive)
    anywhere in the prediction context. This measures retrieval quality,
    not generation quality — no LLM is needed.

    Args:
        prediction: Retrieved context string (output of naive_rag or htmlrag_style).
        gold: Gold answer string from HotpotQA.

    Returns:
        1.0 if gold appears in prediction (case-insensitive), else 0.0.
    """
    return 1.0 if gold.lower() in prediction.lower() else 0.0


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score for retrieval quality.

    Tokenizes both strings by whitespace (after lowercasing) and computes
    precision, recall, and F1 based on token multiset overlap.

    Args:
        prediction: Retrieved context string (output of naive_rag or htmlrag_style).
        gold: Gold answer string from HotpotQA.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = prediction.lower().split()
    gold_tokens = gold.lower().split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # Multiset intersection: sum of min counts for shared tokens
    overlap = sum((pred_counter & gold_counter).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hyperrag(
    query: str,
    index: Any,
    corpus: list[dict[str, Any]],
    graph: nx.DiGraph,
    k: int = 5,
    expand_k: int = 3,
) -> str:
    """HyperRAG: dense retrieval + 1-hop graph expansion + re-ranking.

    1. Dense retrieval: top-K pages from FAISS
    2. Graph expansion: add 1-hop neighbors from NetworkX graph
    3. Deduplicate candidates
    4. Re-rank by cosine similarity to query
    5. Return top-N context string

    Args:
        query: User query string.
        index: FAISS index.
        corpus: Full corpus.
        graph: NetworkX DiGraph of hyperlinks.
        k: Number of initial pages to retrieve.
        expand_k: Number of final pages after re-ranking.

    Returns:
        Context string from re-ranked expanded candidate set.
    """
    raise NotImplementedError("Implemented in Phase 6")
