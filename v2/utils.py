"""
HyperRAG V2 — Shared Utilities
Embed texts, build FAISS index, retrieve pages, compute EM/F1.
Imported by notebooks 02, 03, and 04.
"""

import html as html_lib
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None  # lazy singleton


def get_model() -> SentenceTransformer:
    """Load (or return cached) SentenceTransformer model."""
    global _model
    if _model is None:
        print(f"Loading model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")
    return _model


# ── Embeddings & FAISS ────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings → L2-normalised float32 vectors.
    Shape returned: (len(texts), 384)
    L2 normalisation makes inner product == cosine similarity.
    """
    model = get_model()
    vecs = model.encode(texts, show_progress_bar=len(texts) > 30, convert_to_numpy=True)
    vecs = vecs.astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def build_index(corpus: list[dict]) -> faiss.IndexFlatIP:
    """
    Build a FAISS cosine-similarity index from corpus page texts.
    Returns a faiss.IndexFlatIP ready for .search().
    """
    texts = [p["text"] for p in corpus]
    embeddings = embed_texts(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    corpus: list[dict],
    k: int = 5,
) -> list[dict]:
    """
    Retrieve top-k pages from corpus for a query.
    Returns list of page dicts, each with an added '_score' key.
    """
    q_vec = embed_texts([query])
    scores, indices = index.search(q_vec, k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < len(corpus):
            page = dict(corpus[idx])
            page["_score"] = float(score)
            results.append(page)
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Remove HTML tags — used before computing metrics."""
    if "<" not in text:
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


def compute_em(context: str, gold_answer: str) -> float:
    """
    Exact Match: 1.0 if the gold answer appears (case-insensitive)
    anywhere in the retrieved context, else 0.0.
    """
    return 1.0 if gold_answer.lower() in _strip_html(context).lower() else 0.0


def compute_f1(context: str, gold_answer: str) -> float:
    """
    Token-level F1: measures word overlap between context and gold answer.
    Higher = more overlap. Scores between 0.0 and 1.0.
    """
    context = _strip_html(context)
    pred_tokens = context.lower().split()
    gold_tokens = gold_answer.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_c = Counter(pred_tokens)
    gold_c = Counter(gold_tokens)
    overlap = sum((pred_c & gold_c).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_supporting_recall(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """
    What fraction of gold supporting pages were retrieved?
    This is the KEY metric for multi-hop retrieval.
    Example: gold=[A,B], retrieved=[A,C,D] → recall = 0.5
    """
    if not gold_titles:
        return 0.0
    hits = sum(1 for t in gold_titles if t in set(retrieved_titles))
    return hits / len(gold_titles)


# ── HTML Utilities ────────────────────────────────────────────────────────────

def clean_html(html: str) -> str:
    """
    Keep structural tags (h1, h2, p, li, table) but remove
    script, style, nav, sup elements.  Used by HtmlRAG.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "sup"]):
        tag.decompose()
    return str(soup)


def extract_links_from_html(html: str) -> list[str]:
    """
    Extract Wikipedia article titles from HTML <a href> links.
    Keeps only /wiki/ links with no namespace colon.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    titles = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/") and ":" not in href[6:]:
            title = href[6:].replace("_", " ")
            title = html_lib.unescape(title)
            if title not in seen:
                seen.add(title)
                titles.append(title)
    return titles
