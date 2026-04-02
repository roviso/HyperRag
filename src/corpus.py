"""Corpus module: HotpotQA dataset loading and Wikipedia corpus building."""

import json
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = Path("data/corpus.json")


def load_hotpotqa(split: str = "train", n_samples: int = 500) -> list[dict[str, Any]]:
    """Load HotpotQA dataset from HuggingFace.

    Args:
        split: Dataset split to load (default: 'train').
        n_samples: Number of QA items to load (default: 500).

    Returns:
        List of QA item dicts with keys: question, answer, supporting_facts, context.
    """
    raise NotImplementedError("Implemented in Phase 2")


def extract_supporting_pages(qa_items: list[dict[str, Any]]) -> set[str]:
    """Extract unique Wikipedia page titles from QA supporting facts.

    Args:
        qa_items: List of QA items from load_hotpotqa.

    Returns:
        Set of Wikipedia page titles.
    """
    raise NotImplementedError("Implemented in Phase 2")


def fetch_wikipedia_pages(titles: set[str]) -> list[dict[str, str]]:
    """Fetch Wikipedia pages by title.

    Args:
        titles: Set of Wikipedia page titles to fetch.

    Returns:
        List of page dicts with keys: title, text, html, links.
    """
    raise NotImplementedError("Implemented in Phase 2")


def save_corpus(pages: list[dict[str, str]], path: str | Path = DEFAULT_CORPUS) -> None:
    """Save corpus pages to JSON file.

    Args:
        pages: List of page dicts to save.
        path: Output file path.
    """
    raise NotImplementedError("Implemented in Phase 2")


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict[str, str]]:
    """Load corpus pages from JSON file.

    Args:
        path: Path to corpus JSON file.

    Returns:
        List of page dicts.
    """
    raise NotImplementedError("Implemented in Phase 2")
