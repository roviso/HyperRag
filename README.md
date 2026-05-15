# HyperRAG: Link-Graph Augmented RAG for Multi-Hop Web QA

HyperRAG demonstrates that 1-hop hyperlink graph expansion measurably improves retrieval quality and LLM generation over single-page RAG baselines on HotpotQA multi-hop questions. Three retrieval systems — Simple RAG (B1), HtmlRAG (B2), and HyperRAG — are evaluated across six chunk sizes on 4,028 HotpotQA questions using Qwen3.5-9B for answer generation.

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

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_data_exploration_and_scrapper.ipynb` | Load HotpotQA, fetch Wikipedia pages via MediaWiki API |
| 2 | `02_data_preparation.ipynb` | Clean corpus, build chunks, construct hyperlink graph |
| 3 | `03_simple_rag.ipynb` | Build FAISS index, run Simple RAG retrieval |
| 4 | `04_htmlrag.ipynb` | Run HtmlRAG-style retrieval with HTML-aware chunking |
| 5 | `05_hyperrag.ipynb` | Run HyperRAG with 1-hop graph expansion + cosine re-ranking |
| 6 | `05_llm_eval.ipynb` | Evaluate retrieval with Qwen3.5-9B generation |
| 7 | `06_LLM_inference.ipynb` | Full-scale inference across all chunk sizes |
| 8 | `07_llm_result_analysis.ipynb` | Aggregate results, generate figures |

## Systems

| System | Description |
|--------|-------------|
| Simple RAG | FAISS dense retrieval (top-5), plain text context — no structural awareness |
| HtmlRAG | Same FAISS retrieval, cleaned HTML structure preserved (h1/h2/p/table/li) for richer context |
| HyperRAG | Dense retrieval + 1-hop NetworkX graph expansion + cosine re-ranking — exploits hyperlink structure for multi-hop coverage |

## Results

**Experiment config:** Qwen3.5-9B · `all-MiniLM-L6-v2` embeddings · K=5 retrieval · expand\_K=5 · 64-char overlap · 7,926 questions · 24,473-page corpus

### Retrieval EM

| Chunk | Simple RAG | HtmlRAG | HyperRAG |
|-------|-----------|---------|----------|
| 512   | 0.539 | 0.256 | **0.654** |
| 1 K   | 0.615 | 0.350 | **0.729** |
| 2 K   | 0.676 | 0.472 | **0.779** |
| 4 K   | 0.733 | 0.591 | **0.832** |
| 8 K   | 0.782 | 0.718 | **0.878** |
| Full page | 0.902 | 0.915 | **1.000** |

### Supporting Recall

| Chunk | Simple RAG | HtmlRAG | HyperRAG |
|-------|-----------|---------|----------|
| 512   | 0.663 | 0.663 | **0.842** |
| 1 K   | 0.655 | 0.655 | **0.826** |
| 2 K   | 0.666 | 0.666 | **0.810** |
| 4 K   | 0.677 | 0.677 | **0.811** |
| 8 K   | 0.688 | 0.688 | **0.814** |
| Full page | 0.739 | 0.739 | **0.878** |

### Generation (Qwen3.5-9B) — EM / F1

| Chunk | Simple RAG EM/F1 | HtmlRAG EM/F1 | HyperRAG EM/F1 |
|-------|-----------------|--------------|----------------|
| 512   | 0.525 / 0.513 | 0.383 / 0.381 | **0.545 / 0.535** |
| 1 K   | 0.558 / 0.545 | 0.425 / 0.426 | **0.581 / 0.566** |
| 2 K   | 0.589 / 0.574 | 0.478 / 0.473 | **0.616 / 0.600** |
| 4 K   | 0.621 / 0.605 | 0.541 / 0.533 | **0.662 / 0.641** |
| 8 K   | 0.642 / 0.623 | 0.592 / 0.580 | **0.670 / 0.650** |
| Full page | 0.649 / 0.631 | 0.561 / 0.549 | **0.653 / 0.634** |

### Key Findings

- **Graph expansion consistently wins.** HyperRAG leads on Retrieval EM and Supporting Recall at every chunk size, with Supporting Recall 13–14 pp above both baselines.
- **Chunk size is the dominant retrieval factor.** Retrieval EM rises monotonically for all systems; HyperRAG reaches 1.000 at full-page granularity.
- **2–4 K is the practical sweet-spot.** Marginal gains shrink above 2 K characters, balancing context length, inference cost, and answer quality.
- **Extraction gap persists.** Even at Retrieval EM = 1.000, Gen EM is 0.653 — a 34.7 pp gap attributable to the LLM failing to locate the correct span in long, noisy context.
- **HtmlRAG underperforms.** Formatting noise from cleaned HTML degrades generation (lowest Gen EM at every chunk size) despite sometimes higher token-overlap F1.

## Evaluation Notes

- **Retrieval EM** — binary: does the gold answer string appear anywhere in the retrieved context (case-insensitive)?
- **Supporting Recall** — fraction of gold supporting passages recovered in the retrieved context.
- **Gen EM / F1** — Qwen3.5-9B generates an answer from the retrieved context; scored against gold answer using relaxed string match / token overlap F1.
- **Retrieval F1** (token overlap between full retrieved context and short gold answer) is near-zero across all configurations due to context length dilution; it is not a useful metric in this setup.


## License

Academic use only. Part of MST coursework submission.
