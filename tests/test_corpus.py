"""Unit tests for src/corpus.py — covers DATA-01 through DATA-04."""

from unittest.mock import patch, MagicMock
import pytest


def test_load_hotpotqa():
    """DATA-01: load_hotpotqa returns list of QA dicts from HotpotQA fullwiki."""
    from src.corpus import load_hotpotqa

    items = load_hotpotqa(split="train", n_samples=3)

    assert isinstance(items, list)
    assert len(items) == 3
    required_keys = {"id", "question", "answer", "supporting_facts"}
    assert required_keys.issubset(items[0].keys()), (
        f"Missing keys: {required_keys - set(items[0].keys())}"
    )


def test_extract_supporting_pages(sample_qa_items):
    """DATA-02: extract_supporting_pages returns set of plain title strings."""
    from src.corpus import extract_supporting_pages

    titles = extract_supporting_pages(sample_qa_items)

    assert isinstance(titles, set)
    assert len(titles) > 0
    assert all(isinstance(t, str) for t in titles)
    # Titles must be plain human-readable (no underscores — those are URL artifacts)
    assert all("_" not in t for t in titles), (
        f"Found underscore in titles: {[t for t in titles if '_' in t]}"
    )
    # Verify known titles from the fixture are present
    assert "Albert Einstein" in titles
    assert "Normandy" in titles


def test_page_schema(sample_page):
    """DATA-03: page dicts have non-empty title, text, html, links fields."""
    required_keys = {"title", "text", "html", "links"}
    assert required_keys.issubset(sample_page.keys())
    assert sample_page["title"] != ""
    assert sample_page["text"] != ""
    assert sample_page["html"] != ""
    assert isinstance(sample_page["links"], list)
    assert len(sample_page["links"]) > 0


def test_corpus_cache(corpus_file, sample_page):
    """DATA-04: fetch_wikipedia_pages loads from cache when file exists — no HTTP."""
    from src.corpus import fetch_wikipedia_pages

    with patch("src.corpus.requests") as mock_requests:
        result = fetch_wikipedia_pages(
            titles={"Albert Einstein"},
            output_path=corpus_file,
            force=False,
        )

    # Session.get must never be called — cache was hit
    mock_requests.Session.assert_not_called()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["title"] == sample_page["title"]
