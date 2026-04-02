# HyperRAG-M2: Link-Graph Augmented RAG for Multi-Hop Web QA

Milestone 2 progress submission for the HyperRAG project. Demonstrates that
1-hop hyperlink graph expansion improves retrieval quality (EM/F1) over
single-page RAG baselines on HotpotQA multi-hop questions.

## Project Structure

```
HyperRAG-M2/
├── data/               # Corpus, graph, FAISS index (gitignored)
├── src/
│   ├── corpus.py       # Dataset loading + corpus building
│   ├── graph.py        # NetworkX graph construction
│   ├── embeddings.py   # sentence-transformers + FAISS
│   ├── retrieval.py    # B1, B2, HyperRAG retrieval functions
│   └── evaluate.py     # EM + F1 metric functions
├── notebooks/
│   └── demo.ipynb      # End-to-end demo (Colab-ready)
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for faster embedding

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd HyperRAG-M2

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import transformers, sentence_transformers, faiss, networkx, datasets, bs4, evaluate; print('All imports OK')"
```

## How to Run

### Quick Start (Notebook)

Open `notebooks/demo.ipynb` in Jupyter or Google Colab for an interactive walkthrough.

### Command Line

```bash
# Phase 2: Build corpus
python -c "from src.corpus import load_hotpotqa; print(len(load_hotpotqa()))"

# Phase 5-6: Run all systems and evaluate
python run_all_systems.py
```

## Systems

| System | Description |
|--------|-------------|
| B1 - Naive RAG | FAISS dense retrieval, plain text context |
| B2 - HtmlRAG-style | Same retrieval, HTML structure preserved |
| HyperRAG | Dense retrieval + 1-hop graph expansion + re-ranking |

## Results

*To be completed after Phase 6 evaluation.*

| System | EM | F1 |
|--------|----|----|
| B1 - Naive RAG | -- | -- |
| B2 - HtmlRAG-style | -- | -- |
| HyperRAG | -- | -- |

## License

Academic use only. Part of MST coursework submission.
