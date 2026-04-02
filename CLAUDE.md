# CLAUDE.md — HyperRAG-M2

> Project instructions for Claude Code. Read this at the start of every session.

## Project Context

**Project:** HyperRAG — Link-Graph Augmented RAG for Multi-Hop Web QA
**Milestone:** 2 — Progress Submission (~9 April 2026, MST portal)
**Core value:** Demonstrate that 1-hop hyperlink graph expansion measurably improves EM/F1 over single-page RAG baselines on HotpotQA multi-hop questions.

**GSD planning:** `.planning/` directory — read `.planning/STATE.md` at the start of every session.

---

## Architecture Decisions (DO NOT change without discussion)

| Decision | Value | Reason |
|----------|-------|--------|
| ML framework | PyTorch (CPU fallback, GPU auto-detected) | No code changes needed between environments |
| Vector store | FAISS (faiss-cpu) | Lighter than ChromaDB, sufficient for prototype |
| Graph library | NetworkX DiGraph | Native, educational, well-documented |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Fast, CPU-friendly, adequate quality |
| Dataset | HotpotQA `fullwiki` split (HuggingFace datasets) | Required by assignment |
| Eval scope (M2) | 50 questions, retrieval EM/F1, no LLM inference | Partial results acceptable per brief |
| Pipeline | Direct Python (no LangChain, no Django) | Simpler to explain and grade |

---

## Project Structure

```
HyperRAG-M2/
├── data/               # gitignored — corpus, graph, FAISS index
├── src/
│   ├── __init__.py
│   ├── corpus.py       # Phase 2: HotpotQA loading + Wikipedia corpus
│   ├── graph.py        # Phase 3: NetworkX hyperlink graph
│   ├── embeddings.py   # Phase 4: sentence-transformers + FAISS
│   ├── retrieval.py    # Phase 5-6: B1, B2, HyperRAG retrieval
│   └── evaluate.py     # Phase 5-6: EM + F1 metrics
├── notebooks/
│   └── demo.ipynb      # Colab-ready end-to-end demo
├── writing/            # Phase 7: academic writing
├── .planning/          # GSD planning artifacts
├── requirements.txt
├── README.md
└── CLAUDE.md           # this file
```

---

## Phase Responsibilities

Each `src/` file is owned by a specific phase. Implement functions in the correct phase — do not skip ahead:

| File | Phase | Status |
|------|-------|--------|
| `src/corpus.py` | Phase 2 | Stub only in Phase 1 |
| `src/graph.py` | Phase 3 | Stub only in Phase 1 |
| `src/embeddings.py` | Phase 4 | Stub only in Phase 1 |
| `src/retrieval.py` | Phase 5-6 | Stub only in Phase 1 |
| `src/evaluate.py` | Phase 5-6 | Stub only in Phase 1 |

---

## Coding Standards

### Python Style
- Python 3.10+ syntax (use `list[...]`, `dict[...]`, `tuple[...]` type hints — not `List`, `Dict`)
- Every function must have: docstring, typed parameters, typed return value
- Every module must have a module-level docstring on line 1
- No hardcoded paths — use `pathlib.Path` for file I/O
- GPU/CPU agnostic: use `torch.cuda.is_available()` to set device; never hardcode `"cuda"` or `"cpu"`

### File I/O Patterns
```python
# CORRECT — pathlib, no hardcoded paths
from pathlib import Path
DEFAULT_CORPUS = Path("data/corpus.json")

def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict]:
    path = Path(path)
    ...

# WRONG — hardcoded string paths
def load_corpus(path="data/corpus.json"):
    with open(path) as f:  # breaks on Windows if called from different cwd
    ...
```

### Caching Pattern (required for all data-fetch functions)
All expensive operations (fetch, embed, build) must check for cached output before running:
```python
def build_something(output_path: Path = DEFAULT_PATH, force: bool = False):
    if output_path.exists() and not force:
        print(f"Loading cached {output_path}")
        return load_something(output_path)
    # ... build from scratch
```

### Imports
- Standard library first, then third-party, then local (PEP 8)
- No wildcard imports (`from x import *`)
- Import `networkx as nx`, `numpy as np`, `faiss` directly (no aliasing)

---

## Requirements to Cover (Phase 1)

- **ENV-01**: All deps installable via `pip install -r requirements.txt`
- **ENV-02**: Clear folder structure + README explaining how to run
- **ENV-03**: Code runs on CPU fallback; GPU preferred but no code changes needed

---

## Data Directory

`data/` is gitignored. The following files live there at runtime:

| File | Created in | Size (approx) |
|------|-----------|---------------|
| `data/corpus.json` | Phase 2 | ~50MB (500 Wikipedia pages) |
| `data/hyperlink_graph.graphml` | Phase 3 | ~5MB |
| `data/faiss_index.bin` | Phase 4 | ~10MB |
| `data/faiss_ids.json` | Phase 4 | ~50KB |
| `data/results.csv` | Phase 6 | ~10KB |

Never commit data files. Never hardcode data file paths in src/ — always accept `path` as a parameter with a sensible default.

---

## Evaluation Notes (M2 Scope)

- EM and F1 are computed as **retrieval quality** metrics (does the gold answer appear in the retrieved context?), NOT generation quality — no LLM inference needed for M2.
- Eval subset: 50 HotpotQA questions from `train` split, indices 0-49.
- Results table columns: `question_id, system, em, f1, context_length`

---

## Academic Integrity

This is a coursework submission. All code must be original. Do not copy implementations verbatim from external sources. Citations for algorithms and papers go in `writing/related_work_draft.md`, not in code comments.

---

*Last updated: 2 April 2026 — Phase 1 planning*
