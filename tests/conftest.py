"""Shared pytest fixtures for HyperRAG-M2 tests."""

import json
import pytest
from pathlib import Path

import networkx as nx


@pytest.fixture
def sample_qa_items():
    """Three minimal QA items matching HotpotQA fullwiki schema."""
    return [
        {
            "id": "abc1",
            "question": "Which magazine was started first?",
            "answer": "Arthur's Magazine",
            "type": "comparison",
            "level": "medium",
            "supporting_facts": {
                "title": ["Arthur's Magazine", "First for Women"],
                "sent_id": [0, 0],
            },
            "context": {"title": [], "sentences": []},
        },
        {
            "id": "abc2",
            "question": "Who was born first, Einstein or Bohr?",
            "answer": "Albert Einstein",
            "type": "comparison",
            "level": "hard",
            "supporting_facts": {
                "title": ["Albert Einstein", "Niels Bohr"],
                "sent_id": [0, 0],
            },
            "context": {"title": [], "sentences": []},
        },
        {
            "id": "abc3",
            "question": "In what country is Normandy located?",
            "answer": "France",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {
                "title": ["Normandy"],
                "sent_id": [0],
            },
            "context": {"title": [], "sentences": []},
        },
    ]


@pytest.fixture
def sample_page():
    """One minimal page object matching corpus schema."""
    return {
        "title": "Albert Einstein",
        "text": "Albert Einstein was a theoretical physicist.",
        "html": "<p>Albert Einstein was a theoretical physicist.</p>",
        "links": ["Theoretical physics", "Nobel Prize in Physics"],
    }


@pytest.fixture
def corpus_file(tmp_path, sample_page):
    """A pre-existing corpus JSON file with one page (for cache tests)."""
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps([sample_page]), encoding="utf-8")
    return p


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    """Small DiGraph with 3 nodes and 2 edges for graph.py tests."""
    G = nx.DiGraph()
    pages = [
        {"title": "Albert Einstein", "url": "https://en.wikipedia.org/wiki/Albert_Einstein"},
        {"title": "Niels Bohr", "url": "https://en.wikipedia.org/wiki/Niels_Bohr"},
        {"title": "Quantum mechanics", "url": "https://en.wikipedia.org/wiki/Quantum_mechanics"},
    ]
    for p in pages:
        G.add_node(p["title"], title=p["title"], url=p["url"])
    G.add_edge("Albert Einstein", "Niels Bohr")
    G.add_edge("Niels Bohr", "Quantum mechanics")
    return G


@pytest.fixture
def sample_corpus_for_graph() -> list[dict]:
    """3-page mini corpus with cross-links for build_graph tests."""
    return [
        {"title": "Albert Einstein", "text": "physicist", "html": "<p>physicist</p>",
         "links": ["Niels Bohr", "Quantum mechanics", "Nonexistent Page"]},
        {"title": "Niels Bohr", "text": "physicist", "html": "<p>physicist</p>",
         "links": ["Quantum mechanics", "Albert Einstein"]},
        {"title": "Quantum mechanics", "text": "physics", "html": "<p>physics</p>",
         "links": ["Albert Einstein"]},
    ]
