---
phase: quick-260408-ukk
plan: 01
subsystem: diagnostics
tags: [notebooks, diagnosis, retrieval, visualization]
dependency_graph:
  requires: [src/retrieval.py, src/embeddings.py, src/corpus.py, src/graph.py]
  provides: [notebooks/naive_rag_explained.ipynb, notebooks/htmlrag_explained.ipynb, notebooks/hyperrag_explained.ipynb]
  affects: []
tech_stack:
  added: []
  patterns: [nbformat-v4-notebook-creation, matplotlib-visualization, networkx-subgraph-visualization]
key_files:
  created:
    - notebooks/naive_rag_explained.ipynb
    - notebooks/htmlrag_explained.ipynb
    - notebooks/hyperrag_explained.ipynb
  modified: []
decisions:
  - Used 10 questions for naive/hyperrag notebooks, 5 for htmlrag (sufficient to prove identity)
  - Added fallback context nodes in hyperrag subgraph visualization when few graph nodes match
metrics:
  duration: "~5 minutes"
  completed: "2026-04-08"
  tasks_completed: 3
  tasks_total: 3
---

# Quick Task 260408-ukk: Diagnostic Notebooks for Identical Results

3 diagnostic Jupyter notebooks explaining why naive_rag, htmlrag_style, and hyperrag produce identical EM/F1 scores -- each with step-by-step code, visualizations, and root cause diagnosis.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Naive RAG Deep Dive | ef60a52 | notebooks/naive_rag_explained.ipynb |
| 2 | HtmlRAG Metrics Collapse | 9880280 | notebooks/htmlrag_explained.ipynb |
| 3 | HyperRAG Graph Expansion Failure | 6841391 | notebooks/hyperrag_explained.ipynb |

## What Was Built

### Notebook 1: naive_rag_explained.ipynb (11 cells)
- FAISS search walkthrough with similarity score horizontal bar chart
- Supporting_facts comparison showing retrieval recall
- Manual step-by-step EM and F1 computation with token breakdown
- Batch evaluation on 10 questions with grouped bar chart
- Diagnosis: shared retrieval backbone forms the baseline for all 3 systems

### Notebook 2: htmlrag_explained.ipynb (11 cells)
- Proves retrieval is identical (same search() call, same pages, same order)
- Side-by-side: plain text vs raw HTML vs cleaned HTML for same page
- Demonstrates _strip_html_for_metrics() collapse: HTML context becomes plain text before scoring
- Token set analysis proving functional identity after stripping
- Scatter plot: naive_f1 vs html_f1, all points on y=x line
- Diagnosis: shared retrieval + metrics collapse = mathematically guaranteed identical scores

### Notebook 3: hyperrag_explained.ipynb (13 cells)
- Graph structure overview with degree distribution histogram
- Node ID matching diagnostic: graph_nodes vs corpus_titles intersection/difference
- Per-page neighbor expansion trace: has_node, successors, predecessors, within-corpus filtering
- NetworkX subgraph visualization (blue=initial, green=neighbors, gray=context)
- Full pipeline comparison: Naive vs HyperRAG on 10 questions with grouped bar chart
- Diagnosis: sparse graph (~450 nodes), possible encoding mismatches, empty neighbor sets

## Root Causes Documented

1. **Shared FAISS Retrieval:** All 3 systems call the same search() with the same index
2. **HtmlRAG Metrics Collapse:** _strip_html_for_metrics() strips HTML before EM/F1 computation
3. **HyperRAG Graph Sparsity:** ~450-node corpus produces a very sparse within-corpus graph
4. **Node ID Encoding Mismatches:** GraphML save/load may alter title encoding

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

None -- all notebooks contain complete code cells ready to execute.

## Self-Check: PASSED
