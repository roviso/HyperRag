"""Shared pytest fixtures for HyperRAG-M2 tests."""

import json
import pytest
from pathlib import Path


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
