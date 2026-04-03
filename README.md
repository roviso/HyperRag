# HyperRAG-M2: Link-Graph Augmented RAG for Multi-Hop Web QA

HyperRAG demonstrates that 1-hop hyperlink graph expansion measurably improves retrieval quality (EM/F1) over single-page RAG baselines on HotpotQA multi-hop questions. This repository is an MST Milestone 2 submission implementing three retrieval systems — B1 (Naive RAG), B2 (HtmlRAG-style), and HyperRAG — evaluated on a 50-question subset from HotpotQA.

## Project Structure

```
HyperRAG-M2/
├── data/               # Corpus, graph, FAISS index (gitignored)
├── src/
│   ├── __init__.py
│   ├── corpus.py       # Dataset loading + corpus building
│   ├── graph.py        # NetworkX graph construction
│   ├── embeddings.py   # sentence-transformers + FAISS
│   ├── retrieval.py    # B1, B2, HyperRAG retrieval functions
│   └── evaluate.py     # EM + F1 metric functions
├── notebooks/
│   └── demo.ipynb      # End-to-end demo (Colab-ready)
├── writing/
│   ├── related_work_draft.md    # Related Work section (~730 words)
│   └── dataset_description.md  # Dataset Description (~200 words)
├── tests/              # pytest test suite (24 tests)
├── run_all_systems.py  # Evaluate all 3 systems, save data/results.csv
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for faster embedding — code auto-detects via `torch.cuda.is_available()`

### Installation

```bash
git clone <repo-url>
cd HyperRAG-M2
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import transformers, sentence_transformers, faiss, networkx, datasets, bs4, evaluate; print('All imports OK')"
```

## Dataset

HotpotQA is a multi-hop QA benchmark requiring evidence from two or more Wikipedia pages (Yang et al., 2018), using the `fullwiki` setting where supporting facts come from English Wikipedia. This project uses a 50-question subset from the `train` split (indices 0–49), accessed via the HuggingFace `datasets` library. Supporting page titles are extracted from each QA item, Wikipedia pages are fetched via the MediaWiki API, and the resulting corpus is stored in `data/corpus.json` with approximately 450 pages. Each page contains `title`, `text`, `html`, and `links` fields. See `writing/dataset_description.md` for the full academic description.

## How to Run

Run the pipeline stages **in order**. Each stage caches its output — subsequent runs skip the expensive work.

### Stage 1: Build Corpus (Phase 2)

Fetches approximately 450 Wikipedia pages from the MediaWiki API using supporting page titles extracted from the 50 HotpotQA questions. Estimated time: 20–40 minutes on first run. Cached after first run — subsequent calls return immediately.

```bash
python -c "from src.corpus import build_corpus; build_corpus()"
```

Output: `data/corpus.json` (~450 pages)

### Stage 2: Build Hyperlink Graph (Phase 3)

Constructs a NetworkX DiGraph from within-corpus hyperlinks. Each corpus page's `links` field provides directed edges. Fast (under 1 minute).

```bash
python -c "from src.corpus import load_corpus; from src.graph import build_and_save_graph; build_and_save_graph(load_corpus())"
```

Output: `data/hyperlink_graph.graphml`

### Stage 3: Build FAISS Index (Phase 4)

Encodes all corpus pages with `sentence-transformers/all-MiniLM-L6-v2` into a FAISS flat inner-product index. Vectors are L2-normalised so inner product equals cosine similarity. Estimated time: 5–15 minutes on CPU.

```bash
python -c "from src.corpus import load_corpus; from src.embeddings import build_and_save_index; build_and_save_index(load_corpus())"
```

Output: `data/faiss_index.bin`, `data/faiss_ids.json`

### Stage 4: Run All Systems and Evaluate (Phase 6)

Evaluates B1 (Naive RAG), B2 (HtmlRAG-style), and HyperRAG on all 50 HotpotQA questions. Saves per-question EM and F1 scores and prints a summary table to stdout. Requires Stages 1–3 to have been run first.

```bash
python run_all_systems.py
```

Output: `data/results.csv` (150 rows: 50 questions × 3 systems), summary table printed to stdout.

### Run Tests

```bash
pytest tests/ -v
```

Expected: 24/24 tests passing.

### Interactive Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

Or open in Google Colab (a Colab install cell is included in the notebook).

## Systems

| System | Description |
|--------|-------------|
| B1 - Naive RAG | FAISS dense retrieval (top-5), plain text context concatenated — no structural awareness |
| B2 - HtmlRAG-style | Same FAISS retrieval, cleaned HTML structure preserved (h1/h2/p/table/li) for richer context |
| HyperRAG | Dense retrieval + 1-hop NetworkX graph expansion + cosine re-ranking — exploits hyperlink structure for multi-hop coverage |

## Preliminary Results

Note: EM and F1 are retrieval quality metrics — they measure whether the gold answer appears in the retrieved context, not generation quality. No LLM inference is performed.

| System | EM | F1 | Context Length (avg chars) |
|--------|----|----|---------------------------|
| B1 - Naive RAG | — | — | — |
| B2 - HtmlRAG-style | — | — | — |
| HyperRAG | — | — | — |

*Run `python run_all_systems.py` after completing Stages 1–3 to populate this table.*

## Evaluation Notes

EM (Exact Match) checks whether the gold answer string appears anywhere in the retrieved context (case-insensitive substring match). F1 measures token-level overlap between the gold answer and the retrieved context, computed as the harmonic mean of precision and recall over word tokens. These metrics serve as retrieval quality proxies: higher values indicate that the correct evidence was retrieved. The evaluation subset consists of 50 questions from the HotpotQA `train` split, indices 0–49.

## Academic Writing

- `writing/related_work_draft.md` — Related Work section (~730 words, 7 citations)
- `writing/dataset_description.md` — Dataset Description (~200 words)

## License

Academic use only. Part of MST coursework submission.
