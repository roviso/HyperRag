# ROADMAP — HyperRAG Milestone 2

**Project:** HyperRAG — Link-Graph Augmented RAG for Multi-Hop Web QA
**Milestone:** 2 — Progress Submission
**Deadline:** ~9 April 2026 (MST portal)
**Granularity:** Standard (8 phases)

---

## Phase Overview

| Phase | Name | Goal | Requirements | Est. Days |
|-------|------|------|-------------|-----------|
| 1 | Environment Setup | Reproducible Python project structure | ENV-01–03 | 0.5 |
| 2 | Dataset & Corpus | 1/1 | Complete    | 2026-04-02 |
| 3 | Graph Construction | 1/1 | Complete   | 2026-04-03 |
| 4 | Embeddings & Index | FAISS vector index built & persisted | EMBED-01–03 | 0.5 |
| 5 | Baseline Systems B1 & B2 | Naive RAG + HtmlRAG-style retrieval working | B1-01–03, B2-01–03 | 1 |
| 6 | HyperRAG + Evaluation | 1-hop expansion + results table (50 questions) | HRAG-01–05, EVAL-01–03 | 1.5 |
| 7 | Related Work Draft | 600–800 word academic section | RW-01–04 | 1 |
| 8 | Submission Package | README + dataset desc + packaged deliverables | PKG-01–04 | 0.5 |

**Total estimated: ~7 days** (starts after Milestone 1 submission on 3 April)

---

## Phase 1: Environment Setup

**Goal:** A clean, reproducible Python project that anyone can clone and run.

**Deliverables:**
- `requirements.txt` with pinned versions
- `README.md` skeleton (setup section + how to run)
- Project folder structure:
  ```
  HyperRAG-M2/
  ├── data/           # corpus, graph, FAISS index (gitignored)
  ├── src/
  │   ├── corpus.py       # dataset loading + corpus building
  │   ├── graph.py        # NetworkX graph construction
  │   ├── embeddings.py   # sentence-transformers + FAISS
  │   ├── retrieval.py    # B1, B2, HyperRAG retrieval functions
  │   └── evaluate.py     # EM + F1 metric functions
  ├── notebooks/
  │   └── demo.ipynb      # end-to-end demo (Colab-ready)
  ├── requirements.txt
  └── README.md
  ```
- Verified: `pip install -r requirements.txt` works cleanly
- Verified: imports work on CPU (no GPU required for setup)

**Success criteria:**
- Fresh virtual environment can install all deps without errors
- `python -c "import transformers, sentence_transformers, faiss, networkx, datasets, bs4, evaluate"` passes

**Plans:** 1/1 plans complete

Plans:
- [x] 01-01-PLAN.md — Project skeleton, requirements.txt, source stubs, README, and install verification

---

## Phase 2: Dataset & Corpus Building

**Goal:** HotpotQA loaded and a ~500-page Wikipedia corpus extracted and saved locally.

**Deliverables:**
- `src/corpus.py` — functions:
  - `load_hotpotqa(split, n_samples)` → returns list of QA items
  - `extract_supporting_pages(qa_items)` → returns set of Wikipedia page titles
  - `fetch_wikipedia_pages(titles)` → returns list of `{title, text, html, links}`
  - `save_corpus(pages, path)` / `load_corpus(path)` → JSON persistence
- `data/corpus.json` (gitignored, ~500 pages)
- Each page object: `{"title": str, "text": str, "html": str, "links": [str]}`

**Step-by-step (documented in README):**
1. Load HotpotQA: `ds = load_dataset("hotpot_qa", "fullwiki")`
2. Extract unique supporting page titles from first 500 QA items
3. Fetch each page from Wikipedia API (or use pre-downloaded dump)
4. Parse HTML with BeautifulSoup — extract text, preserve HTML, extract `<a href>` links
5. Save to `data/corpus.json`

**Success criteria:**
- `data/corpus.json` exists with ≥400 pages
- Each page has non-empty `text`, `html`, and `links` fields
- Script can be re-run without re-fetching (load from cache if exists)

**Plans:** 1/1 plans complete

Plans:
- [x] 02-01-PLAN.md — Implement corpus.py functions (replace stubs) + build data/corpus.json via Wikipedia API fetch

---

## Phase 3: Graph Construction

**Goal:** A NetworkX directed graph where nodes = Wikipedia pages, edges = hyperlinks between corpus pages.

**Deliverables:**
- `src/graph.py` — functions:
  - `build_graph(corpus)` → returns `nx.DiGraph`
  - `save_graph(G, path)` / `load_graph(path)` → GraphML persistence
  - `print_graph_stats(G)` → prints: nodes, edges, avg degree, top-5 connected pages
- `data/hyperlink_graph.graphml` (gitignored)
- Documented explanation of graph structure in code comments

**Step-by-step (documented in README):**
1. For each page in corpus: add node `page.title`
2. For each link in `page.links`: if link target is in corpus → `G.add_edge(source, target)`
3. Print stats: how many cross-corpus links found
4. Save graph to GraphML

**NetworkX concepts explained in code comments:**
- What a DiGraph is and why directed (hyperlinks are one-way)
- `G.add_node()`, `G.add_edge()` usage
- `G.successors(node)` for 1-hop expansion
- `nx.pagerank(G)` for future use (M3)

**Success criteria:**
- Graph has ≥200 nodes and ≥100 edges
- `G.successors("Some_Page")` returns non-empty list for at least some pages
- GraphML file saves and loads correctly

**Plans:** 1/1 plans complete

Plans:
- [x] 03-01-PLAN.md — Implement graph.py (5 functions) + tests/test_graph.py (3 tests) + conftest fixtures

---

## Phase 4: Embeddings & Vector Index

**Goal:** All corpus pages embedded and stored in a FAISS index for fast similarity search.

**Deliverables:**
- `src/embeddings.py` — functions:
  - `build_faiss_index(corpus, model_name)` → builds and returns `(index, page_ids)`
  - `save_index(index, page_ids, path)` / `load_index(path)` → disk persistence
  - `search(query, index, page_ids, corpus, k)` → returns top-k pages
- `data/faiss_index.bin` + `data/faiss_ids.json` (gitignored)
- Model: `sentence-transformers/all-MiniLM-L6-v2` (fast, CPU-friendly)

**Step-by-step (documented in README):**
1. Load corpus from `data/corpus.json`
2. Encode each page's text: `model.encode([page["text"] for page in corpus])`
3. Build FAISS flat index: `faiss.IndexFlatIP(dim)` (inner product = cosine similarity)
4. Save index + page ID mapping to disk

**Success criteria:**
- Index builds without memory errors
- `search("Who wrote Harry Potter?", ...)` returns plausible pages
- Re-running skips rebuild if files exist

**Plans:** 1/1 plans complete

Plans:
- [x] 04-01-PLAN.md — Implement embeddings.py (5 functions) + tests/test_embeddings.py (3 tests) + conftest fixtures

---

## Phase 5: Baseline Systems B1 & B2

**Goal:** Two retrieval baselines implemented, tested, and producing EM/F1 scores.

**Deliverables:**
- `src/retrieval.py` — functions:
  - `naive_rag(query, index, corpus, k)` → returns plain text context string (B1)
  - `htmlrag_style(query, index, corpus, k)` → returns HTML-preserved context string (B2)
  - `compute_em(prediction, gold)` → exact match score
  - `compute_f1(prediction, gold)` → token F1 score
- `src/evaluate.py` — `run_evaluation(system_fn, qa_items, k)` → results dict
- Baseline results on 10-question smoke test (not full 50 yet)

**B1 — Naive RAG:**
1. Encode query → search FAISS → get top-K pages
2. Concatenate plain text of top-K pages → context string
3. (For M2 eval: check if gold answer string appears in context → proxy EM/F1)

**B2 — HtmlRAG-style:**
1. Same retrieval as B1
2. Instead of plain text, use BeautifulSoup to clean HTML: keep `<h1>`, `<h2>`, `<p>`, `<table>`, `<li>` — remove `<script>`, `<style>`, `<nav>`
3. Context string = cleaned HTML with structural tags preserved

**Success criteria:**
- Both systems produce context strings for any input query
- EM and F1 functions pass unit tests (hardcoded examples)
- 10-question smoke test completes without errors

**Plans:** 1 plan

Plans:
- [ ] 05-01-PLAN.md — Implement retrieval.py (naive_rag, htmlrag_style, compute_em, compute_f1) + evaluate.py (run_evaluation) + 10 tests

---

## Phase 6: HyperRAG + Full Evaluation

**Goal:** HyperRAG implemented and all 3 systems evaluated on 50 HotpotQA questions.

**Deliverables:**
- `src/retrieval.py` additions:
  - `hyperrag(query, index, corpus, graph, k, expand_k)` → context string with graph expansion
- `run_all_systems.py` — evaluates B1, B2, HyperRAG on 50 questions → saves results CSV
- `data/results.csv` — columns: `question_id, system, em, f1, context_length`
- Preliminary results table (printed + in README)

**HyperRAG step-by-step:**
1. Dense retrieval: top-K pages (same as B1)
2. Graph expansion: for each retrieved page → get `G.successors(page)` → add to candidate set
3. Deduplicate candidates
4. Re-rank by cosine similarity to query (re-encode query, sort by similarity)
5. Take top-N from re-ranked list as final context
6. Assemble context string (plain text for now)

**Evaluation loop:**
```python
for qa in qa_items[:50]:
    for system in [naive_rag, htmlrag_style, hyperrag]:
        context = system(qa["question"], ...)
        em = compute_em(qa["answer"], context)
        f1 = compute_f1(qa["answer"], context)
        # save to results
```

**Success criteria:**
- All 3 systems produce results for all 50 questions
- Results CSV exists with no NaN values
- HyperRAG retrieves more relevant context than B1 on at least some questions (qualitative check)

---

## Phase 7: Related Work Draft

**Goal:** A 600–800 word Related Work section suitable for the conference paper.

**Deliverables:**
- `writing/related_work_draft.md` — formatted academic prose
- Structure:
  1. **RAG Foundations** (~150 words): Lewis et al. 2020, DPR (Karpukhin 2020)
  2. **HTML-Aware RAG** (~150 words): HtmlRAG (Tan 2024), AXE (2026)
  3. **Graph-Augmented RAG** (~200 words): GraphRAG (Edge 2024) — key differentiator explained
  4. **Multi-Hop QA** (~150 words): HotpotQA (Yang 2018), IRCoT (Trivedi 2023)
  5. **The Gap** (~100 words): No prior work exploits native HTML hyperlink graphs for web QA

**Success criteria:**
- 600–800 words (use word count tool)
- At minimum 4 citations: Lewis2020, Tan2024, Edge2024, Yang2018
- Gap paragraph clearly states HyperRAG's differentiator from GraphRAG and HtmlRAG
- Academic tone — no first person ("we show" is fine, "I think" is not)

---

## Phase 8: Submission Package

**Goal:** Everything packaged cleanly for MST portal submission.

**Deliverables:**
- `README.md` — complete with: project description, setup, how to run B1/B2/HyperRAG, expected outputs, sample results table
- `writing/dataset_description.md` — 200-word description of HotpotQA + Wikipedia corpus
- Updated `requirements.txt` with all final dependencies
- All code clean: docstrings on every function, no dead code, no hardcoded paths
- `notebooks/demo.ipynb` — end-to-end walkthrough (Colab-compatible)
- Submission checklist verified

**Submission checklist:**
- [ ] `pip install -r requirements.txt` works from clean environment
- [ ] `python run_all_systems.py` runs without errors and prints results table
- [ ] Related Work draft is 600–800 words
- [ ] Dataset description is ~200 words
- [ ] All Python files have module-level docstrings
- [ ] README has run instructions
- [ ] Submitted to MST portal before 9 April 2026

**Plans:** 1 plan

Plans:
- [ ] 08-01-PLAN.md — Complete README, dataset description, src/ docstring audit, demo notebook, and submission verification

---

## Dependency Graph

```
Phase 1 (Env)
    └── Phase 2 (Dataset)
            ├── Phase 3 (Graph)   ──┐
            └── Phase 4 (FAISS)  ──┤
                                    └── Phase 5 (Baselines)
                                              └── Phase 6 (HyperRAG + Eval)
                                                        └── Phase 8 (Package)
Phase 7 (Related Work) ──────────────────────────────────── Phase 8 (Package)
```

Phases 3 and 4 can run in parallel after Phase 2.
Phase 7 can run in parallel with Phases 3–6 (it's writing, not coding).

---

## Sprint Calendar (7 days)

| Day | Date | Phases | Output |
|-----|------|--------|--------|
| 1 | 4 Apr | Phase 1 | Project skeleton + requirements.txt |
| 2 | 5 Apr | Phase 2 | Corpus JSON built |
| 3 | 6 Apr | Phase 3 + 4 (parallel) | Graph + FAISS index |
| 3–4 | 6–7 Apr | Phase 7 (writing in parallel) | Related Work draft |
| 4 | 7 Apr | Phase 5 | B1 + B2 working |
| 5 | 8 Apr | Phase 6 | HyperRAG + results table |
| 6 | 9 Apr | Phase 8 | Final packaging + submission |

### Phase 9: understaing everything step by step execution with explanation and visualization

**Goal:** Create a detailed educational tutorial notebook (notebooks/tutorial.ipynb) covering all 4 pipeline stages with step-by-step explanations, live execution on 2-3 HotpotQA questions, and 4 visualizations (graph plot, similarity chart, pipeline diagram, system comparison chart).
**Requirements**: D-01 through D-08 (captured in 09-CONTEXT.md)
**Depends on:** Phase 8
**Plans:** 2 plans

Plans:
- [x] 09-01-PLAN.md — Create tutorial.ipynb with introduction, Stage 1 (Corpus), Stage 2 (Graph), graph visualization, and pipeline flow diagram
- [x] 09-02-PLAN.md — Add Stage 3 (Embeddings & FAISS) with similarity chart, Stage 4 (Retrieval + Eval) with comparison chart, and conclusion

### Phase 10: recheck if naive_rag, htmlrag and hyperrag are actually working as expected — find and fix flaws causing identical EM/F1 scores

**Goal:** [To be planned]
**Requirements**: TBD
**Depends on:** Phase 9
**Plans:** 0 plans

Plans:
- [ ] TBD (run /gsd:plan-phase 10 to break down)

### Phase 11: Local LLM inference with Llama for generation-based EM/F1 evaluation notebook

**Goal:** Create a standalone notebook (notebooks/llm_eval.ipynb) that loads a local LLM via HuggingFace transformers, generates short answers from retrieved context for all 3 systems (B1, B2, HyperRAG) on 10 HotpotQA questions, and compares generation-based EM/F1 against retrieval-based EM/F1 in a summary table.
**Requirements**: D-01 through D-15 (captured in 11-CONTEXT.md)
**Depends on:** Phase 10
**Plans:** 1 plan

Plans:
- [ ] 11-01-PLAN.md — Create llm_eval.ipynb with LLM pipeline, evaluation loop, and retrieval vs generation EM/F1 comparison table

---
*Roadmap created: 2 April 2026*
*Last updated: 8 April 2026 — Phase 11 planned: 11-01-PLAN.md created*
