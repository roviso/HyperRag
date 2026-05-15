# HyperRAG: Link-Graph Augmented RAG for Multi-Hop Web QA

HyperRAG demonstrates that 1-hop hyperlink graph expansion measurably improves retrieval quality and LLM generation over single-page RAG baselines on HotpotQA multi-hop questions. Three retrieval systems — Simple RAG (B1), HtmlRAG (B2), and HyperRAG — are evaluated across six chunk sizes on 6,625 HotpotQA questions using Qwen3.5-9B for answer generation.

## Project Structure

```
HyperRAG/
├── notebooks/
│   ├── 01_data_exploration_and_scrapper.ipynb  # HotpotQA loading + Wikipedia corpus fetch
│   ├── 02_data_preparation.ipynb               # Corpus cleaning, chunking, graph construction
│   ├── 03_simple_rag.ipynb                     # B1 — FAISS dense retrieval baseline
│   ├── 04_htmlrag.ipynb                        # B2 — HTML-structure-aware retrieval
│   ├── 05_hyperrag.ipynb                       # HyperRAG — dense + 1-hop graph expansion
│   ├── 05_llm_eval.ipynb                       # LLM-based evaluation (Qwen3.5-9B)
│   ├── 06_LLM_inference.ipynb                  # Full-scale LLM inference across chunk sizes
│   ├── utils.py                                # Shared utility functions
│   ├── make_combined_figures.py                # Combined figure export
│   ├── visualize_results.py                    # Results visualisation helpers
│   ├── data/                                   # Corpus and question snapshots 
│   ├── results/                                # CSV/JSON results + figures
│   └── figures/                                # figures from the result
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA — embedding and LLM inference auto-detect via `torch.cuda.is_available()`

### Installation

```bash
git clone <repo-url>
cd HyperRAG
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

HotpotQA is a multi-hop QA benchmark requiring evidence from two or more Wikipedia pages (Yang et al., 2018), using the `fullwiki` setting. This project uses 7,926 questions drawn from the `train` split against a corpus of 24,473 Wikipedia distractor pages, accessed via the HuggingFace `datasets` library. See `writing/dataset_description.md` for the full academic description.

## How to Run

Run the notebooks **in order**. Each notebook saves its outputs so subsequent runs load from cache.

| Step | Artifact | Description |
|------|----------|-------------|
| 1 | `01_data_exploration_and_scrapper.ipynb` | Load HotpotQA, fetch Wikipedia pages via MediaWiki API |
| 2 | `02_data_preparation.ipynb` | Clean corpus, build chunks, construct hyperlink graph |
| 3 | `03_simple_rag.ipynb` | Build FAISS index, run Simple RAG retrieval |
| 4 | `04_htmlrag.ipynb` | Run HtmlRAG-style retrieval with HTML-aware chunking |
| 5 | `05_hyperrag.ipynb` | Run HyperRAG with 1-hop graph expansion + cosine re-ranking |
| 6 | `05_llm_eval.ipynb` | Evaluate retrieval with Qwen3.5-9B generation |
| 7 | `06_LLM_inference.ipynb` | Full-scale inference across all chunk sizes |
| 8 | `visualize_results.py` | Build publication figures from step 7 results (see below) |

After step 7 finishes, it writes combined JSON files under `notebooks/results/llm_results/`. Generate the summary plots from those files:

```bash
cd notebooks
python visualize_results.py
```

Outputs land in `notebooks/figures/` (`fig1_retrieval_vs_generation.png` through `fig5_generation_quality.png`, plus `RESULTS_SUMMARY.txt`).

## Systems

| System | Description |
|--------|-------------|
| Simple RAG | FAISS dense retrieval (top-5), plain text context — no structural awareness |
| HtmlRAG | Same FAISS retrieval, cleaned HTML structure preserved (h1/h2/p/table/li) for richer context |
| HyperRAG | Dense retrieval + 1-hop NetworkX graph expansion + cosine re-ranking — exploits hyperlink structure for multi-hop coverage |

## Results

**Experiment config:** Qwen3.5-9B · `all-MiniLM-L6-v2` embeddings · K=5 retrieval · expand\_K=5 · 64-char overlap · 7,926 questions · 24,473-page corpus

*Source: `notebooks/figures/RESULTS_SUMMARY.txt` (from `06_LLM_inference.ipynb` + `visualize_results.py`).*

### Retrieval EM

| Chunk | Simple RAG | HtmlRAG | HyperRAG |
|-------|-----------|---------|----------|
| 512   | 0.347 | 0.166 | **0.462** |
| 1 K   | 0.398 | 0.230 | **0.511** |
| 2 K   | 0.438 | 0.311 | **0.554** |
| 4 K   | 0.479 | 0.389 | **0.609** |
| 8 K   | 0.512 | 0.471 | **0.658** |
| Full page | 0.596 | 0.647 | **0.885** |

### Supporting Recall

| Chunk | Simple RAG | HtmlRAG | HyperRAG |
|-------|-----------|---------|----------|
| 512   | 0.602 | 0.602 | **0.781** |
| 1 K   | 0.590 | 0.590 | **0.754** |
| 2 K   | 0.590 | 0.590 | **0.732** |
| 4 K   | 0.590 | 0.590 | **0.724** |
| 8 K   | 0.589 | 0.589 | **0.721** |
| Full page | 0.614 | 0.614 | **0.799** |

### Generation (Qwen3.5-9B) — EM / F1

| Chunk | Simple RAG EM/F1 | HtmlRAG EM/F1 | HyperRAG EM/F1 |
|-------|-----------------|--------------|----------------|
| 512   | 0.299 / 0.393 | 0.218 / 0.292 | **0.316 / 0.413** |
| 1 K   | 0.318 / 0.418 | 0.244 / 0.325 | **0.333 / 0.436** |
| 2 K   | 0.335 / 0.437 | 0.277 / 0.360 | **0.356 / 0.459** |
| 4 K   | 0.351 / 0.452 | 0.315 / 0.400 | **0.383 / 0.485** |
| 8 K   | 0.359 / 0.459 | 0.338 / 0.429 | **0.389 / 0.491** |
| Full page | 0.360 / 0.459 | 0.326 / 0.410 | **0.371 / 0.470** |

### Key Findings

- **Graph expansion consistently wins.** HyperRAG leads on Retrieval EM and Supporting Recall at every chunk size, with Supporting Recall ~14–19 pp above both baselines (e.g. 0.781 vs 0.602 at 512 chars).
- **Chunk size is the dominant retrieval factor.** Retrieval EM rises monotonically for all systems; HyperRAG reaches **0.885** at full-page granularity (Simple RAG 0.596, HtmlRAG 0.647).
- **2–4 K is the practical sweet-spot.** Marginal Gen EM gains shrink above 2 K characters (full-page uplift vs 8 K is only ~1–2 pp), balancing context length, inference cost, and answer quality.
- **Extraction gap persists.** At full-page, HyperRAG Retrieval EM is 0.885 but Gen EM is 0.371 — a **51.4 pp** gap showing the reader model often fails to extract the answer even when it is present in context.
- **HtmlRAG underperforms on generation.** Lowest Gen EM at every chunk size (e.g. 0.218 at 512 vs 0.299 Simple RAG, 0.316 HyperRAG); cleaned-HTML formatting noise hurts fragile HotpotQA answers despite higher Retrieval F1 at small chunks.

## Evaluation Notes

- **Retrieval EM** — binary: does the gold answer string appear anywhere in the retrieved context (case-insensitive)?
- **Supporting Recall** — fraction of gold supporting passages recovered in the retrieved context.
- **Gen EM / F1** — Qwen3.5-9B generates an answer from the retrieved context; scored against gold answer using relaxed string match / token overlap F1.
- **Retrieval F1** (token overlap between full retrieved context and short gold answer) is near-zero across all configurations due to context length dilution; it is not a useful metric in this setup.


## License

Academic use only. Part of MST coursework submission.
