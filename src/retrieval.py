"""Retrieval module: B1 (Naive RAG), B2 (HtmlRAG-style), and HyperRAG retrieval functions."""

from collections import Counter
from typing import Any

import networkx as nx
from bs4 import BeautifulSoup

from src.embeddings import search, build_faiss_index


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


def _strip_html_for_metrics(text: str) -> str:
    """Strip HTML tags from text before EM/F1 metric computation.

    Applies a fast short-circuit: if no '<' is present the text is returned
    unchanged. Otherwise BeautifulSoup extracts plain text content, preventing
    HTML tag fragments (e.g. '<p>france</p>') from being treated as tokens.

    Args:
        text: Arbitrary string, possibly containing HTML markup.

    Returns:
        Plain text with all HTML tags removed and whitespace normalised.
    """
    if "<" not in text:
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


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
    anywhere in the prediction context. HTML tags are stripped before
    comparison so that HTML-structured contexts are handled correctly.

    Args:
        prediction: Retrieved context string (output of naive_rag or htmlrag_style).
        gold: Gold answer string from HotpotQA.

    Returns:
        1.0 if gold appears in prediction (case-insensitive), else 0.0.
    """
    prediction = _strip_html_for_metrics(prediction)
    return 1.0 if gold.lower() in prediction.lower() else 0.0


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score for retrieval quality.

    HTML tags are stripped from prediction before tokenising so that HTML
    tag fragments (e.g. '<p>france</p>') do not absorb text tokens and
    produce zero overlap with the gold answer.

    Tokenizes both strings by whitespace (after lowercasing) and computes
    precision, recall, and F1 based on token multiset overlap.

    Args:
        prediction: Retrieved context string (output of naive_rag or htmlrag_style).
        gold: Gold answer string from HotpotQA.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    prediction = _strip_html_for_metrics(prediction)
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
    """HyperRAG: dense retrieval + 1-hop graph expansion.

    1. Dense retrieval: top-k pages from FAISS (always included in output)
    2. Graph expansion: collect 1-hop neighbors via BOTH out-edges AND in-edges
    3. Keep only neighbor pages that are not already in the initial set
    4. Re-rank neighbor candidates by cosine similarity; take top expand_k
    5. Return context = initial_pages + top_expand_k_neighbors

    The initial k pages are always preserved in the output — graph neighbors are
    ADDED on top, not used as replacements. This ensures HyperRAG always has at
    least as much coverage as naive_rag while also including bridge pages for
    multi-hop questions.

    Args:
        query: User query string.
        index: FAISS index (full corpus).
        corpus: Full corpus.
        graph: NetworkX DiGraph of hyperlinks.
        k: Number of initial pages to retrieve via FAISS.
        expand_k: Maximum number of additional graph-neighbor pages to append.

    Returns:
        Context string: initial pages followed by up to expand_k neighbor pages,
        each separated by a double newline.
    """
    # Step 1: Dense retrieval — top-k pages from FAISS
    page_ids = [p["title"] for p in corpus]
    initial_pages = search(query, index, page_ids, corpus, k=k)

    # Step 2: 1-hop expansion via both out-edges (successors) AND in-edges (predecessors)
    title_to_page = {p["title"]: p for p in corpus}
    initial_titles: set[str] = {p["title"] for p in initial_pages}
    neighbor_titles: set[str] = set()

    for page in initial_pages:
        if graph.has_node(page["title"]):
            for nbr in graph.successors(page["title"]):
                if nbr in title_to_page and nbr not in initial_titles:
                    neighbor_titles.add(nbr)
            for nbr in graph.predecessors(page["title"]):
                if nbr in title_to_page and nbr not in initial_titles:
                    neighbor_titles.add(nbr)

    # Step 3: Collect neighbor page objects (new pages only)
    neighbor_pages = [title_to_page[t] for t in neighbor_titles]

    # Step 4: Re-rank neighbors by cosine similarity; take top expand_k
    if neighbor_pages and expand_k > 0:
        actual_expand = min(expand_k, len(neighbor_pages))
        temp_index, temp_page_ids = build_faiss_index(neighbor_pages)
        top_neighbors = search(query, temp_index, temp_page_ids, neighbor_pages, k=actual_expand)
    else:
        top_neighbors = []

    # Step 5: Assemble context — initial pages always first, then graph-neighbor pages
    all_pages = initial_pages + top_neighbors
    return "\n\n".join(p["text"] for p in all_pages)


def hyperrag_2hop(
    query: str,
    index: Any,
    corpus: list[dict[str, Any]],
    graph: nx.DiGraph,
    k: int = 5,
    expand_k: int = 3,
) -> str:
    """HyperRAG with 2-hop graph expansion.

    Same as hyperrag() but traverses 2 hops in the hyperlink graph instead of 1,
    squaring the reachable neighborhood and increasing the chance of finding
    multi-hop bridge pages.

    Args:
        query: User query string.
        index: FAISS index (full corpus).
        corpus: Full corpus.
        graph: NetworkX DiGraph of hyperlinks.
        k: Number of initial pages to retrieve via FAISS.
        expand_k: Maximum number of additional graph-neighbor pages to append.

    Returns:
        Context string: initial pages followed by up to expand_k neighbor pages.
    """
    page_ids = [p["title"] for p in corpus]
    initial_pages = search(query, index, page_ids, corpus, k=k)

    title_to_page = {p["title"]: p for p in corpus}
    initial_titles: set[str] = {p["title"] for p in initial_pages}

    # Hop 1: direct neighbors
    hop1_titles: set[str] = set()
    for page in initial_pages:
        if graph.has_node(page["title"]):
            for nbr in graph.successors(page["title"]):
                if nbr in title_to_page:
                    hop1_titles.add(nbr)
            for nbr in graph.predecessors(page["title"]):
                if nbr in title_to_page:
                    hop1_titles.add(nbr)

    # Hop 2: neighbors of neighbors
    hop2_titles: set[str] = set()
    for title in hop1_titles:
        if graph.has_node(title):
            for nbr in graph.successors(title):
                if nbr in title_to_page:
                    hop2_titles.add(nbr)
            for nbr in graph.predecessors(title):
                if nbr in title_to_page:
                    hop2_titles.add(nbr)

    # Combine all neighbor titles, exclude initial pages
    all_neighbor_titles = (hop1_titles | hop2_titles) - initial_titles
    neighbor_pages = [title_to_page[t] for t in all_neighbor_titles]

    if neighbor_pages and expand_k > 0:
        actual_expand = min(expand_k, len(neighbor_pages))
        temp_index, temp_page_ids = build_faiss_index(neighbor_pages)
        top_neighbors = search(query, temp_index, temp_page_ids, neighbor_pages, k=actual_expand)
    else:
        top_neighbors = []

    all_pages = initial_pages + top_neighbors
    return "\n\n".join(p["text"] for p in all_pages)


def compute_supporting_recall(
    retrieved_titles: list[str],
    gold_titles: list[str],
) -> float:
    """Compute recall of gold supporting-fact pages among retrieved pages.

    Measures what fraction of the gold supporting pages were retrieved,
    which is the right metric for evaluating multi-hop retrieval quality.

    Args:
        retrieved_titles: Titles of pages returned by the retrieval system.
        gold_titles: Titles of gold supporting-fact pages from HotpotQA.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not gold_titles:
        return 0.0
    retrieved_set = set(retrieved_titles)
    hits = sum(1 for t in gold_titles if t in retrieved_set)
    return hits / len(gold_titles)


def get_retrieved_titles(
    query: str,
    index: Any,
    corpus: list[dict[str, Any]],
    graph: nx.DiGraph | None = None,
    k: int = 5,
    expand_k: int = 3,
    method: str = "naive",
) -> list[str]:
    """Return titles of pages retrieved by a given method.

    Utility for computing supporting-fact recall without re-implementing
    each retrieval pipeline.

    Args:
        query: User query string.
        index: FAISS index.
        corpus: Full corpus.
        graph: NetworkX DiGraph (required for hyperrag/hyperrag_2hop methods).
        k: Number of initial FAISS results.
        expand_k: Max graph-expansion pages (for hyperrag methods).
        method: One of 'naive', 'hyperrag', 'hyperrag_2hop'.

    Returns:
        List of retrieved page titles.
    """
    from src.embeddings import search as faiss_search

    page_ids = [p["title"] for p in corpus]
    initial_pages = faiss_search(query, index, page_ids, corpus, k=k)
    initial_titles = [p["title"] for p in initial_pages]

    if method == "naive":
        return initial_titles

    if graph is None:
        return initial_titles

    title_to_page = {p["title"]: p for p in corpus}
    initial_set = set(initial_titles)

    # 1-hop neighbors
    hop1: set[str] = set()
    for page in initial_pages:
        if graph.has_node(page["title"]):
            for nbr in graph.successors(page["title"]):
                if nbr in title_to_page:
                    hop1.add(nbr)
            for nbr in graph.predecessors(page["title"]):
                if nbr in title_to_page:
                    hop1.add(nbr)

    if method == "hyperrag":
        neighbor_titles = hop1 - initial_set
    elif method == "hyperrag_2hop":
        hop2: set[str] = set()
        for title in hop1:
            if graph.has_node(title):
                for nbr in graph.successors(title):
                    if nbr in title_to_page:
                        hop2.add(nbr)
                for nbr in graph.predecessors(title):
                    if nbr in title_to_page:
                        hop2.add(nbr)
        neighbor_titles = (hop1 | hop2) - initial_set
    else:
        neighbor_titles = set()

    # Re-rank neighbors by cosine similarity
    neighbor_pages = [title_to_page[t] for t in neighbor_titles]
    if neighbor_pages and expand_k > 0:
        actual_expand = min(expand_k, len(neighbor_pages))
        temp_index, temp_page_ids = build_faiss_index(neighbor_pages)
        top_neighbors = faiss_search(query, temp_index, temp_page_ids, neighbor_pages, k=actual_expand)
        return initial_titles + [p["title"] for p in top_neighbors]

    return initial_titles
