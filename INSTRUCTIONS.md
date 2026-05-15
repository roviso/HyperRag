# HyperRAG — Step-by-Step Run Guide

Follow these instructions in order. Each step must complete successfully before moving to the next.

---

## Prerequisites

- Python 3.10 or higher
- pip (bundled with Python)
- Git
- (Optional) NVIDIA GPU with CUDA — the pipeline auto-detects it; CPU works too

---

## Step 1 — Clone the Repository

```bash
git clone <repo-url>
cd HyperRAG
```

---

## Step 2 — Create a Virtual Environment

**Linux / macOS**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt)**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You should see `(venv)` prefixed to your terminal prompt once the environment is active.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, Transformers, sentence-transformers, FAISS, NetworkX, HuggingFace Datasets, BeautifulSoup4, and Jupyter. It may take 3–5 minutes.

### Verify the installation

```bash
python -c "import transformers, sentence_transformers, faiss, networkx, datasets, bs4, evaluate; print('All imports OK')"
```

Expected output: `All imports OK`

---

## Step 4 — Launch Jupyter

```bash
jupyter notebook
```

This opens the Jupyter interface in your browser. Navigate to the `notebooks/` folder and run the notebooks **in the numbered order below**.

> Each notebook saves its outputs to `notebooks/data/` and `notebooks/results/`. If you stop and restart, re-running a completed notebook will load from cache — you do not need to re-fetch or re-compute.

---

## Notebook Execution Order

### Notebook 1 — `01_data_exploration_and_scrapper.ipynb`

**What it does:**
- Downloads 7,926 multi-hop questions from the HotpotQA `fullwiki` split via HuggingFace Datasets.
- Fetches the 24,473 associated Wikipedia distractor pages using the MediaWiki API.
- Saves raw HTML and text to `notebooks/data/`.

**What you get:**
- `notebooks/data/corpus_raw.json` — raw Wikipedia pages
- Console output showing fetch progress and page count

**Expected runtime:** 20–40 minutes (Wikipedia API, network-dependent). Pages are saved incrementally every 50 fetches, so an interrupted run can be resumed.

---

### Notebook 2 — `02_data_preparation.ipynb`

**What it does:**
- Cleans the raw HTML corpus (strips boilerplate, normalises whitespace).
- Chunks each page at six sizes: 512, 1K, 2K, 4K, 8K characters, and full page.
- Builds a directed hyperlink graph (NetworkX DiGraph) from Wikipedia inter-page links.
- Saves the cleaned corpus, chunk index, and graph.

**What you get:**
- `notebooks/data/corpus_clean.json` — cleaned page text
- `notebooks/data/chunks_<size>.json` — chunked text for each chunk size
- `notebooks/data/hyperlink_graph.graphml` — the link graph (~5 MB)

**Expected runtime:** 5–10 minutes.

---

### Notebook 3 — `03_simple_rag.ipynb` — Simple RAG Baseline (B1)

**What it does:**
- Embeds all chunks using `sentence-transformers/all-MiniLM-L6-v2`.
- Builds a FAISS IndexFlatIP index (cosine similarity via L2-normalised vectors).
- Runs retrieval for all 7,926 questions at each chunk size (top-5 results).
- Computes **Retrieval EM** (does the gold answer appear in the retrieved context?) and **Supporting Recall** (fraction of gold supporting passages recovered).

**What you get:**
- `notebooks/results/simple_rag_results.csv` — per-question EM and recall at each chunk size
- Console table of aggregate EM / Supporting Recall per chunk size

**Expected runtime:** 30–60 minutes (embedding 24K pages × 6 chunk sizes; GPU reduces this significantly).

---

### Notebook 4 — `04_htmlrag.ipynb` — HtmlRAG Baseline (B2)

**What it does:**
- Re-uses the FAISS index from Notebook 3.
- Builds context by preserving HTML structure tags (h1, h2, p, table, li) rather than stripping them, giving the LLM structural cues.
- Computes the same retrieval metrics as B1.

**What you get:**
- `notebooks/results/htmlrag_results.csv` — per-question EM and recall
- Comparison printout: Simple RAG vs HtmlRAG side by side

**Expected runtime:** 10–20 minutes (retrieval only, no re-embedding).

---

### Notebook 5 — `05_hyperrag.ipynb` — HyperRAG System

**What it does:**
- Starts from the same FAISS top-5 results as B1.
- Expands each retrieved chunk by following 1-hop outgoing hyperlinks in the NetworkX graph, fetching up to 5 additional linked pages per result.
- Re-ranks the expanded candidate set by cosine similarity to the query.
- Computes Retrieval EM and Supporting Recall for the expanded context.

**What you get:**
- `notebooks/results/hyperrag_results.csv` — per-question EM and recall
- Summary table showing HyperRAG vs B1 vs B2 across all chunk sizes

**Expected runtime:** 20–40 minutes.

---

### Notebook 6 — `06_LLM_inference.ipynb` — Full LLM Inference

**What it does:**
- Loads Qwen3.5-9B from HuggingFace (downloads ~18 GB on first run; cached locally afterwards).
- For each system (Simple RAG, HtmlRAG, HyperRAG) and each chunk size (512 → full page), generates answers for all 7,926 questions.
- Scores generated answers using **Gen EM** (relaxed string match) and **Gen F1** (token overlap).

**What you get:**
- `notebooks/results/llm_results_<system>_<chunk>.csv` — one file per system/chunk combination
- Aggregate Gen EM / Gen F1 table printed to console

**Expected runtime:** Several hours on CPU. GPU (A100/RTX 3090+) recommended — reduces to 1–2 hours. Results are cached per system/chunk so interrupted runs resume.

> If you only want retrieval metrics (no LLM), you can skip this notebook. Notebook 7 will still produce retrieval-only figures.

---

## Output Summary

After running all notebooks, these are the key artefacts:

| Artefact | Location | Description |
|----------|----------|-------------|
| Cleaned corpus | `notebooks/data/corpus_clean.json` | 24,473 Wikipedia pages |
| Hyperlink graph | `notebooks/data/hyperlink_graph.graphml` | Directed link graph |
| FAISS index | `notebooks/data/faiss_index.bin` | Dense vector index |
| Retrieval results | `notebooks/results/*_results.csv` | EM / Recall per system |
| LLM results | `notebooks/results/llm_results_*.csv` | Gen EM / F1 per system |
| Figures | `notebooks/figures/*.png` | Comparison plots |

---

## Key Results (what to expect)

HyperRAG should outperform both baselines on Retrieval EM and Supporting Recall at every chunk size:

| Chunk | Simple RAG EM | HtmlRAG EM | HyperRAG EM |
|-------|--------------|------------|-------------|
| 512   | 0.539 | 0.256 | **0.654** |
| 2 K   | 0.676 | 0.472 | **0.779** |
| Full  | 0.902 | 0.915 | **1.000** |

Graph expansion provides a consistent +13–14 pp boost on Supporting Recall, confirming that hyperlink-guided expansion recovers the second supporting page missed by dense-only retrieval.

---

## Troubleshooting

**Import errors after installation**
Make sure the virtual environment is activated (`(venv)` in prompt) before launching Jupyter.

**Wikipedia fetch stalls or times out**
Notebook 1 saves progress every 50 pages. Simply re-run the cell — it will skip already-fetched pages.

**CUDA out-of-memory during LLM inference**
Reduce batch size in Notebook 6 or switch to CPU inference (slower but always works).

**Jupyter kernel crashes during embedding**
Increase system RAM or reduce the chunk size list in Notebook 3 to process one size at a time.
