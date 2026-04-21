"""Corpus module: HotpotQA dataset loading and Wikipedia corpus building."""

import html
import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


DEFAULT_CORPUS = Path("data/corpus.json")

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_USER_AGENT = "HyperRAG-M2/1.0 (academic research)"


def load_hotpotqa(split: str = "train", n_samples: int = 500) -> list[dict[str, Any]]:
    """Load HotpotQA dataset from HuggingFace.

    Args:
        split: Dataset split to load (default: 'train').
        n_samples: Number of QA items to load (default: 500).

    Returns:
        List of QA item dicts with keys: id, question, answer, supporting_facts, context.
    """
    from datasets import load_dataset  # imported here — heavy; not needed for cache-only runs

    ds = load_dataset(
        "hotpot_qa",
        "fullwiki",
        split=f"{split}[:{n_samples}]",
    )
    return list(ds)


def extract_supporting_pages(qa_items: list[dict[str, Any]]) -> set[str]:
    """Extract unique Wikipedia page titles from QA supporting facts.

    Args:
        qa_items: List of QA items from load_hotpotqa.

    Returns:
        Set of Wikipedia page titles as plain human-readable strings.
    """
    titles: set[str] = set()
    for item in qa_items:
        for title in item["supporting_facts"]["title"]:
            titles.add(title)
    return titles


def _fetch_one_page(session: requests.Session, title: str) -> dict[str, str] | None:
    """Fetch a single Wikipedia page via the MediaWiki action=parse API.

    Args:
        session: requests.Session with User-Agent header set.
        title: Plain Wikipedia page title (will be HTML-unescaped internally).

    Returns:
        Page dict with keys title, text, html, links — or None if page is missing.
    """
    clean_title = html.unescape(title)
    params = {
        "action": "parse",
        "page": clean_title,
        "prop": "text",
        "format": "json",
        "redirects": "",
    }
    try:
        resp = session.get(_WIKI_API, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    if "error" in data:
        # missingtitle or other API error — skip silently
        return None

    raw_html = data["parse"]["text"]["*"]
    canonical_title = data["parse"]["title"]

    soup = BeautifulSoup(raw_html, "html.parser")
    plain_text = soup.get_text(separator=" ", strip=True)
    links = _extract_wiki_links(soup)

    return {
        "title": canonical_title,
        "text": plain_text,
        "html": raw_html,
        "links": links,
    }


def _extract_wiki_links(soup: BeautifulSoup) -> list[str]:
    """Extract outbound Wikipedia article links from a parsed page.

    Filters out File:, Category:, Help:, Talk:, and other namespace links.
    Decodes URL encoding and converts underscores to spaces.

    Args:
        soup: BeautifulSoup parse tree of the Wikipedia page HTML.

    Returns:
        Deduplicated list of plain Wikipedia article titles.
    """
    seen: dict[str, None] = {}
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if href.startswith("/wiki/") and ":" not in href[6:]:
            title = urllib.parse.unquote(href[6:]).replace("_", " ")
            seen[title] = None  # dict preserves insertion order, deduplicates
    return list(seen.keys())


def fetch_wikipedia_pages(
    titles: set[str],
    output_path: Path = DEFAULT_CORPUS,
    delay: float = 0.5,
    max_retries: int = 3,
    force: bool = False,
) -> list[dict[str, str]]:
    """Fetch Wikipedia pages by title using the MediaWiki action=parse API.

    Implements incremental saving every 50 pages so a partial run is resumable.
    Skips already-fetched titles when resuming from a partial corpus file.

    Args:
        titles: Set of Wikipedia page titles to fetch.
        output_path: Path to save the corpus JSON (default: data/corpus.json).
        delay: Seconds to sleep between requests (default: 0.5).
        max_retries: Max retry attempts on rate-limit errors (default: 3).
        force: If True, ignore existing cache and re-fetch everything.

    Returns:
        List of page dicts with keys: title, text, html, links.
    """
    output_path = Path(output_path)

    # DATA-04: cache check — return immediately if corpus already exists
    if output_path.exists() and not force:
        print(f"Loading cached corpus from {output_path}")
        return load_corpus(output_path)

    # Resume support: load partial corpus to skip already-fetched titles
    pages: list[dict[str, str]] = []
    fetched_titles: set[str] = set()
    if output_path.exists():
        pages = load_corpus(output_path)
        fetched_titles = {p["title"] for p in pages}

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    remaining = [t for t in titles if t not in fetched_titles]

    for title in tqdm(remaining, desc="Fetching Wikipedia pages"):
        page = None
        for attempt in range(max_retries):
            page = _fetch_one_page(session, title)
            if page is not None:
                break
            # On None from a non-missing-title cause, brief backoff before retry
            time.sleep(delay * (attempt + 1))

        if page is not None:
            pages.append(page)

        time.sleep(delay)

        # Incremental save every 50 pages so crashes don't lose progress
        if len(pages) % 50 == 0 and len(pages) > 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_corpus(pages, output_path)

    # Final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_corpus(pages, output_path)
    return pages


def save_corpus(pages: list[dict[str, str]], path: str | Path = DEFAULT_CORPUS) -> None:
    """Save corpus pages to a JSON file.

    Args:
        pages: List of page dicts to save.
        path: Output file path (created with parents if needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict[str, str]]:
    """Load corpus pages from a JSON file.

    Args:
        path: Path to corpus JSON file.

    Returns:
        List of page dicts with keys: title, text, html, links.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus(
    output_path: Path = DEFAULT_CORPUS,
    n_samples: int = 500,
    force: bool = False,
) -> list[dict[str, str]]:
    """Orchestrate the full corpus build pipeline.

    Pipeline:
        1. Load n_samples QA items from HotpotQA fullwiki train split.
        2. Extract unique supporting page titles from all QA items.
        3. Fetch each page from Wikipedia API (with incremental caching).
        4. Save corpus to output_path.

    Cache pattern: if output_path exists and force=False, loads and returns
    the cached corpus immediately without re-fetching anything.

    Args:
        output_path: Destination path for corpus.json (default: data/corpus.json).
        n_samples: Number of HotpotQA items to source titles from (default: 500).
        force: If True, ignore cache and rebuild from scratch.

    Returns:
        List of page dicts with keys: title, text, html, links.
    """
    output_path = Path(output_path)

    # DATA-04: fast path — return cached corpus without touching the network
    if output_path.exists() and not force:
        print(f"Corpus already exists at {output_path} — loading from cache.")
        return load_corpus(output_path)

    print(f"Loading {n_samples} HotpotQA items...")
    qa_items = load_hotpotqa(split="train", n_samples=n_samples)

    titles = extract_supporting_pages(qa_items)
    print(f"Extracted {len(titles)} unique supporting page titles.")

    print(f"Fetching {len(titles)} Wikipedia pages (this may take 20-40 minutes)...")
    pages = fetch_wikipedia_pages(titles=titles, output_path=output_path, force=force)

    print(f"Corpus complete: {len(pages)} pages saved to {output_path}")
    return pages
