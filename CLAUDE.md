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
├── data/               # gitignored — corpus, graph, FAISS index, results
├── src/
│   ├── __init__.py
│   ├── corpus.py       # HotpotQA loading + Wikipedia corpus (8 functions)
│   ├── graph.py        # NetworkX hyperlink graph (5 functions)
│   ├── embeddings.py   # sentence-transformers + FAISS (5 functions)
│   ├── retrieval.py    # B1, B2, HyperRAG retrieval + EM/F1 metrics
│   └── evaluate.py     # Evaluation loop (delegates to retrieval metrics)
├── notebooks/
│   ├── demo.ipynb              # Colab-ready end-to-end demo (14 cells)
│   ├── tutorial.ipynb          # Educational walkthrough with visualizations
│   ├── llm_eval.ipynb          # LLM generation-based evaluation (TinyLlama)
│   ├── naive_rag_explained.ipynb   # Diagnostic: Naive RAG step-by-step
│   ├── htmlrag_explained.ipynb     # Diagnostic: HtmlRAG metrics analysis
│   └── hyperrag_explained.ipynb    # Diagnostic: HyperRAG graph expansion
├── tests/
│   ├── conftest.py     # Shared fixtures
│   ├── test_corpus.py  # 4 tests (DATA-01..04)
│   ├── test_graph.py   # 3 tests (GRAPH-01..03)
│   ├── test_embeddings.py # 3 tests (EMBED-01..03)
│   ├── test_retrieval.py  # 10+ tests (B1, B2, HRAG)
│   └── test_evaluate.py
├── writing/
│   ├── related_work_draft.md   # ~730 words, 7 citations
│   └── dataset_description.md  # ~246-word academic description
├── run_all_systems.py  # Evaluates B1/B2/HyperRAG → data/results.csv
├── .planning/          # GSD planning artifacts
├── requirements.txt
├── README.md
└── CLAUDE.md           # this file
```

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Environment Setup | ✅ Complete |
| Phase 2 | Dataset & Corpus (`src/corpus.py`) | ✅ Complete |
| Phase 3 | Graph Construction (`src/graph.py`) | ✅ Complete |
| Phase 4 | Embeddings & Index (`src/embeddings.py`) | ✅ Complete |
| Phase 5 | Baseline Systems (`src/retrieval.py`, `src/evaluate.py`) | ✅ Complete |
| Phase 6 | HyperRAG + Evaluation (`run_all_systems.py`) | ✅ Complete |
| Phase 7 | Related Work Draft (`writing/`) | ✅ Complete |
| Phase 8 | Submission Package (README, demo notebook) | ✅ Complete |
| Phase 9 | Tutorial notebook with visualizations | ✅ Complete |
| Phase 10 | Diagnostic notebooks (identical EM/F1 investigation) | ✅ Complete |
| Phase 11 | LLM generation-based evaluation notebook | ✅ Complete |

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

## Requirements Covered

- **ENV-01**: All deps installable via `pip install -r requirements.txt` ✅
- **ENV-02**: Clear folder structure + README explaining how to run ✅
- **ENV-03**: Code runs on CPU fallback; GPU preferred but no code changes needed ✅
- **DATA-01..04**: HotpotQA loading, page extraction, Wikipedia fetch, corpus persistence ✅
- **GRAPH-01..03**: Graph construction, persistence, stats ✅
- **EMBED-01..03**: FAISS index build, persistence, search ✅
- **EVAL**: B1/B2/HyperRAG retrieval + EM/F1 metrics on 50 questions ✅

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
| `data/llm_eval_results.csv` | Phase 11 | ~5KB |

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

## Key Design Decisions (accumulated)

| Decision | Date | Rationale |
|----------|------|-----------|
| Incremental save every 50 pages | 3 Apr | Protects corpus fetch during 20-40 min Wikipedia API calls |
| `html.unescape()` on all titles | 3 Apr | Required for HotpotQA HTML entity titles |
| Link filter `/wiki/` + no colon | 3 Apr | Prevents namespace pollution in graph |
| L2 norm + IndexFlatIP for cosine | 3 Apr | Inner product equals cosine after L2 normalise |
| Monkeypatch SentenceTransformer | 3 Apr | Deterministic tests without model download |
| Separate tutorial.ipynb from demo.ipynb | 3 Apr | Educational verbose vs operational concise |

## Known Issues

- All three retrieval systems (B1/B2/HyperRAG) produce identical or near-identical EM/F1 scores. Diagnostic notebooks in `notebooks/*_explained.ipynb` investigate root causes.

*Last updated: 9 April 2026 — All 11 phases complete*
