"""Script to append Stage 3 and Stage 4 cells to notebooks/tutorial.ipynb."""

import json
from pathlib import Path


def make_markdown(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


def make_code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ---------------------------------------------------------------------------
# Stage 3 cells
# ---------------------------------------------------------------------------

stage3_intro = make_markdown("tutorial-cell-20", """\
## Stage 3: Embeddings & FAISS Vector Index

To search efficiently across hundreds of Wikipedia pages, we need a way to measure \
*semantic similarity* between a question and a page — even when they don't share the \
same keywords. We solve this with dense vector embeddings and the FAISS similarity \
search library.

**What are embeddings?** An embedding is a numerical vector representation of text. \
Using the `sentence-transformers/all-MiniLM-L6-v2` model, each Wikipedia page's text \
is converted into a fixed-length 384-dimensional vector. Pages about similar topics \
(e.g., two pages about the same film festival) produce vectors that point in similar \
directions in this high-dimensional space — even if they use different words. The model \
was trained specifically to capture semantic meaning, not just keyword overlap.

**What is FAISS?** Facebook AI Similarity Search (FAISS) is a library optimized for \
fast nearest-neighbor lookups in vector spaces. Instead of comparing a query vector \
against every document vector individually (which would take O(n) time and become \
expensive at scale), FAISS uses optimized internal data structures to find the most \
similar vectors in near-constant time. We use `IndexFlatIP` — the exact flat inner-product \
index — which is guaranteed accurate (no approximation) and fast enough for a corpus of \
450 pages.

**How does cosine similarity work?** Cosine similarity measures the *angle* between two \
vectors rather than their absolute distance. Two vectors pointing in the same direction \
have cosine similarity = 1.0 (maximum similarity); orthogonal vectors score 0.0; \
opposite vectors score −1.0. We achieve this via *L2 normalisation*: before building \
the index, every embedding vector is divided by its own length (L2 norm), making all \
vectors unit-length. After normalisation, the inner product between two vectors equals \
their cosine similarity — and FAISS's `IndexFlatIP` computes inner products natively, \
so no extra computation is needed.

**Why this matters for HyperRAG:** Dense retrieval is the *entry point* of the \
HyperRAG pipeline. Stage 4 (graph expansion) starts from the k pages that FAISS \
returns here. If the initial dense retrieval misses important pages entirely, graph \
expansion cannot recover them — the two stages are complementary, not interchangeable.\
""")


stage3_build = make_code("tutorial-cell-21", """\
# Build FAISS index: encode all corpus pages into 384-dim vectors
# model: all-MiniLM-L6-v2 (fast on CPU, good semantic quality)
# Internally: L2-normalises each vector so inner product = cosine similarity
index, page_ids = build_faiss_index(corpus)

print(f"FAISS index built:")
print(f"  Vectors:    {index.ntotal}")          # number of indexed page vectors
print(f"  Dimensions: {index.d}")               # embedding dimensionality (384)
print(f"  Page IDs:   {len(page_ids)} titles")  # maps each vector slot to a page title
""")


stage3_search_intro = make_markdown("tutorial-cell-22", """\
### Step 3.1: Searching the Index

Given a natural-language question, we encode it into the same 384-dimensional vector \
space and ask FAISS for the k nearest page vectors. The distance metric — cosine \
similarity — captures *meaning*, not just word overlap. A question about "a director's \
most famous film" can match a page titled "Cannes Film Festival winners" without sharing \
any keywords, because both talk about films and awards.

The `search()` function handles encoding, normalisation, and the FAISS query in one \
call. It returns a list of page dicts sorted by cosine similarity, each augmented with \
a `score` key. We use this ranked list as the starting point for both B1 baseline \
retrieval (Stage 4) and HyperRAG's graph expansion step.\
""")


stage3_search_code = make_code("tutorial-cell-23", """\
# Use the first question from our 3-question HotpotQA tutorial subset
example_question = qa_items[0]["question"]
example_answer   = qa_items[0]["answer"]
print(f"Query:       {example_question}")
print(f"Gold answer: {example_answer}\\n")

# Search the FAISS index for the top-5 most semantically similar pages
top_pages = search(example_question, index, page_ids, corpus, k=5)

# Display each result; flag pages that contain the gold answer
for i, page in enumerate(top_pages):
    # Did retrieval succeed for this page? (EM check)
    contains_answer = example_answer.lower() in page["text"].lower()
    marker = "  [CONTAINS ANSWER]" if contains_answer else ""
    print(f"  {i+1}. {page['title']}{marker}")
    print(f"     Score: {page.get('score', 0.0):.4f}  |  "
          f"Preview: {page['text'][:80]}...")
    print()
""")


stage3_viz_intro = make_markdown("tutorial-cell-24", """\
### Step 3.2: Visualizing Similarity Scores

The bar chart below shows the cosine similarity score between our query and each of \
the top-10 retrieved pages. A few things to notice:

- **Scores drop off:** The gap between rank 1 and rank 5 can be substantial. If this \
  gap is large, higher k values add progressively less relevant pages.
- **No threshold:** FAISS always returns k results, even if some scores are very low. \
  In production systems, you might filter by a minimum similarity threshold.
- **Why [CONTAINS ANSWER] matters:** If the page containing the gold answer is not in \
  the top-k, no retrieval system can score EM=1 for this question — the graph expansion \
  in Stage 4 offers a second chance by following links FROM the top-k results.\
""")


stage3_viz_code = make_code("tutorial-cell-25", """\
# --- Similarity Score Bar Chart (D-05) ---
# We re-encode the query and call index.search() directly to get raw scores
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode and L2-normalise the query (same steps as build_faiss_index)
query_vec  = model.encode([example_question]).astype(np.float32)
import faiss as _faiss_local
_faiss_local.normalize_L2(query_vec)  # in-place; makes IP == cosine similarity

# Ask FAISS for the top-10 results (more than top-5 for a richer visualization)
distances, indices_arr = index.search(query_vec, 10)
scores = distances[0]         # cosine similarity scores
titles = [page_ids[idx] for idx in indices_arr[0]]  # map back to page titles

# Truncate titles that are too long to fit on the chart axis
short_titles = [t[:38] + "..." if len(t) > 38 else t for t in titles]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(range(len(scores)), scores, color="#3498db", edgecolor="white")

# Y-axis: page titles; invert so highest-scoring page is at the top
ax.set_yticks(range(len(scores)))
ax.set_yticklabels(short_titles, fontsize=9)
ax.invert_yaxis()

ax.set_xlabel("Cosine Similarity Score", fontsize=11)
ax.set_title(
    f'Top-10 Pages by Similarity to:\\n"{example_question[:80]}..."',
    fontsize=11,
)

# Add score values at the end of each bar for readability
for bar, score in zip(bars, scores):
    ax.text(
        bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
        f"{score:.3f}", va="center", fontsize=9,
    )

plt.tight_layout()
plt.show()  # inline display only — no file saving (D-06)
""")


stage3_summary = make_markdown("tutorial-cell-26", """\
### Stage 3 Summary

We now have a FAISS vector index that lets us find the most semantically similar corpus \
pages to any natural-language query in milliseconds. The key results of this stage:

- **Encoding:** Every Wikipedia page is stored as a 384-dimensional L2-normalised vector.
- **Search:** A query is encoded with the same model and compared against all page \
  vectors using cosine similarity (inner product after normalisation).
- **Output:** A ranked list of the top-k most similar pages, with their cosine scores.

However, FAISS alone can only find pages that are *directly* similar to the question \
text. Multi-hop questions often require information from a page that is topically \
adjacent but not directly matched. That is exactly the gap HyperRAG fills in Stage 4 \
by following one hop of hyperlinks from the top-k results.\
""")


# ---------------------------------------------------------------------------
# Stage 4 cells
# ---------------------------------------------------------------------------

stage4_intro = make_markdown("tutorial-cell-27", """\
## Stage 4: Retrieval Systems & Evaluation

This stage compares three retrieval systems on our HotpotQA questions. Each system \
takes a question, a FAISS index, and a corpus, and returns a *context string* — the \
text that would be given to a question-answering model. We then measure how well that \
context string covers the gold answer using two metrics: EM and F1.

**B1 — Naive RAG:** Encode the question, search FAISS for the top-K pages, concatenate \
their plain text. This is the simplest possible dense-retrieval system: fast, \
interpretable, and entirely driven by vector similarity. Its weakness is that it can \
only find pages whose text is semantically close to the question — if the answer \
requires information from a page that is topically distant but hyperlinked, B1 misses it.

**B2 — HtmlRAG-style:** Identical retrieval to B1 (same FAISS search, same top-K), but \
instead of returning plain text, it preserves structural HTML tags — headings (`<h1>`, \
`<h2>`), paragraphs (`<p>`), tables (`<table>`), and list items (`<li>`) — while \
removing scripts, styles, and navigation elements. The hypothesis behind HtmlRAG \
(Tan et al., 2024) is that structural markup helps downstream readers (human or LLM) \
parse dense Wikipedia content more accurately. Note: for our *retrieval* evaluation \
(does the gold answer appear in the context?), B1 and B2 often score similarly.

**HyperRAG — our approach:** Retrieves top-K pages via FAISS (same as B1), then \
**expands** the candidate set by following outgoing hyperlinks in the NetworkX graph \
(one hop: `G.successors(page_title)`). After deduplication, it re-ranks *all* \
candidates — both the initially retrieved pages and their link-neighbors — by cosine \
similarity to the query, and takes the top `expand_k` results. The insight: a page \
about "2003 Cannes Film Festival" might link to the actor page that contains the \
answer to "Who won Best Actor at Cannes 2003?" — HyperRAG follows that link. The cost \
is minimal because the hyperlink graph already exists inside the HTML `<a href>` tags; \
no LLM graph construction is needed.\
""")


stage4_b1_intro = make_markdown("tutorial-cell-28", """\
### Step 4.1: B1 — Naive RAG

Baseline 1 is a standard dense retrieval pipeline: encode → FAISS search → plain text. \
No graph, no HTML structure awareness. It represents the performance you would get from \
a minimal off-the-shelf RAG implementation.\
""")


stage4_b1_code = make_code("tutorial-cell-29", """\
# B1: Naive RAG — top-k pages by FAISS cosine similarity, plain text output
context_b1 = naive_rag(example_question, index, corpus, k=5)

# Measure retrieval quality against the gold answer
em_b1 = compute_em(context_b1, example_answer)   # 1.0 if answer found anywhere, else 0.0
f1_b1 = compute_f1(context_b1, example_answer)   # token-level overlap in [0, 1]

print("B1 — Naive RAG:")
print(f"  Context length : {len(context_b1):,} chars")
print(f"  EM             : {em_b1:.1f}   (1.0 = answer found in context)")
print(f"  F1             : {f1_b1:.4f}")
print(f"  Answer found   : {'YES' if em_b1 else 'NO'}  (gold: '{example_answer}')")
print(f"\\n  Context preview (first 300 chars):\\n  {context_b1[:300]}...")
""")


stage4_b2_intro = make_markdown("tutorial-cell-30", """\
### Step 4.2: B2 — HtmlRAG-style

Baseline 2 uses the same FAISS retrieval as B1, but returns HTML-structured context \
rather than plain text. The structural tags (headings, tables, lists) remain, giving \
downstream readers richer formatting cues. This mirrors the approach of Tan et al. \
(2024) HtmlRAG paper.\
""")


stage4_b2_code = make_code("tutorial-cell-31", """\
# B2: HtmlRAG-style — same retrieval as B1, but preserves structural HTML tags
context_b2 = htmlrag_style(example_question, index, corpus, k=5)

em_b2 = compute_em(context_b2, example_answer)
f1_b2 = compute_f1(context_b2, example_answer)

print("B2 — HtmlRAG-style:")
print(f"  Context length : {len(context_b2):,} chars")
print(f"  EM             : {em_b2:.1f}   (1.0 = answer found in context)")
print(f"  F1             : {f1_b2:.4f}")
print(f"  Answer found   : {'YES' if em_b2 else 'NO'}  (gold: '{example_answer}')")
print(f"\\n  Context preview (first 300 chars):\\n  {context_b2[:300]}...")
""")


stage4_hr_intro = make_markdown("tutorial-cell-32", """\
### Step 4.3: HyperRAG — Graph-Augmented Retrieval

HyperRAG adds a graph expansion step between dense retrieval and context assembly. \
The algorithm proceeds in five steps:

1. **Dense retrieval:** Encode the query and retrieve the top-K most similar pages from \
   FAISS — identical to B1.
2. **Graph expansion:** For each of the K retrieved pages, call `G.successors(page_title)` \
   in the NetworkX DiGraph to find every page it hyperlinks to. These are pages a human \
   reader would naturally *click* to while researching the question.
3. **Deduplicate:** Combine the K initially retrieved pages with all their link-neighbors \
   into a single candidate set (a Python `set` removes duplicates automatically).
4. **Re-rank:** Build a temporary FAISS index from just the candidate pages, then search \
   it with the original query. This gives each candidate a fresh cosine similarity score \
   against the query.
5. **Assemble context:** Return the plain text of the top `expand_k` re-ranked candidates.

The critical advantage: if the answer lives in a page that was NOT directly similar to \
the question (and therefore missed by B1), but IS linked to by one of the top-K pages, \
HyperRAG discovers it via graph expansion and promotes it via re-ranking. This is the \
core innovation — "one hop reasoning" at retrieval time.\
""")


stage4_hr_code = make_code("tutorial-cell-33", """\
# HyperRAG: FAISS retrieval + 1-hop graph expansion + cosine re-ranking
# k=5: initial FAISS pages; expand_k=5: final pages after re-ranking the expanded set
context_hr = hyperrag(example_question, index, corpus, graph, k=5, expand_k=5)

em_hr = compute_em(context_hr, example_answer)
f1_hr = compute_f1(context_hr, example_answer)

print("HyperRAG:")
print(f"  Context length : {len(context_hr):,} chars")
print(f"  EM             : {em_hr:.1f}   (1.0 = answer found in context)")
print(f"  F1             : {f1_hr:.4f}")
print(f"  Answer found   : {'YES' if em_hr else 'NO'}  (gold: '{example_answer}')")
print(f"\\n  Context preview (first 300 chars):\\n  {context_hr[:300]}...")
""")


stage4_eval_intro = make_markdown("tutorial-cell-34", """\
### Step 4.4: Evaluating on Multiple Questions

A single question is not enough to judge a system reliably — the result for one \
question could be an outlier in either direction. We now run all three systems on our \
complete tutorial set of 3 HotpotQA questions and compute average scores.

**Exact Match (EM)** checks whether the gold answer string appears *anywhere* in the \
retrieved context, case-insensitively. EM = 1.0 means the retrieval succeeded for that \
question; EM = 0.0 means the answer was not found. For retrieval-only evaluation (no \
LLM generation), EM is the primary metric.

**F1** measures token-level overlap between the retrieved context and the gold answer. \
Unlike EM, F1 gives partial credit: if the context contains some but not all tokens of \
a multi-word answer, F1 will be between 0 and 1. F1 is a softer and often more \
informative metric for multi-word answers.

Both metrics are computed over retrieval quality (does the context *contain* the \
answer?) — no language model inference is needed.\
""")


stage4_eval_code = make_code("tutorial-cell-35", """\
# Evaluate all 3 systems on our 3 tutorial questions (live execution — not pre-computed)
results = {}

for name, system_fn, extra_kwargs in [
    ("B1: Naive RAG",    naive_rag,      {}),
    ("B2: HtmlRAG",      htmlrag_style,  {}),
    ("HyperRAG",         hyperrag,       {"graph": graph, "expand_k": 5}),
]:
    em_scores  = []
    f1_scores  = []
    for qa in qa_items:
        # Run retrieval system on this question
        ctx = system_fn(qa["question"], index, corpus, k=5, **extra_kwargs)
        em_scores.append(compute_em(ctx, qa["answer"]))   # 1.0 or 0.0
        f1_scores.append(compute_f1(ctx, qa["answer"]))   # 0.0 – 1.0

    results[name] = {
        "avg_em": sum(em_scores) / len(em_scores),
        "avg_f1": sum(f1_scores) / len(f1_scores),
    }

# Print results table
print(f"{'System':<20}  {'Avg EM':>8}  {'Avg F1':>8}")
print("-" * 42)
for name, scores in results.items():
    print(f"{name:<20}  {scores['avg_em']:>8.3f}  {scores['avg_f1']:>8.3f}")

print(f"\\nEvaluated on {len(qa_items)} multi-hop HotpotQA questions")
print("EM  = does the gold answer appear anywhere in retrieved context? (1.0 / 0.0)")
print("F1  = token-level overlap between retrieved context and gold answer (0.0 – 1.0)")
""")


stage4_chart_intro = make_markdown("tutorial-cell-36", """\
### Step 4.5: Comparison Visualization

The grouped bar chart below plots the average EM and F1 scores for all three systems \
side by side. Each system has two bars: one for EM (blue) and one for F1 (red). \
Value labels above each bar make the exact scores readable without squinting at the \
y-axis.

What to look for: if HyperRAG's bars are taller than B1 and B2, the 1-hop graph \
expansion successfully retrieved pages that dense search alone would miss. On a 3-question \
sample the gap may be small or inconsistent; the full 50-question evaluation in \
`data/results.csv` gives a more reliable estimate.\
""")


stage4_chart_code = make_code("tutorial-cell-37", """\
# --- B1 vs B2 vs HyperRAG Comparison Chart (D-05) ---

systems = list(results.keys())
em_vals  = [results[s]["avg_em"] for s in systems]
f1_vals  = [results[s]["avg_f1"] for s in systems]

x     = np.arange(len(systems))  # group positions along the x-axis
width = 0.35                      # width of each individual bar

fig, ax = plt.subplots(figsize=(9, 5))

# Two sets of bars: EM (blue, shifted left) and F1 (red, shifted right)
bars_em = ax.bar(x - width / 2, em_vals, width, label="Exact Match (EM)",
                 color="#3498db", edgecolor="white")
bars_f1 = ax.bar(x + width / 2, f1_vals, width, label="F1 Score",
                 color="#e74c3c", edgecolor="white")

# Add value labels above each bar
for bar in bars_em:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars_f1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Score", fontsize=11)
ax.set_title("Retrieval Quality: B1 vs B2 vs HyperRAG", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=10)
ax.legend(fontsize=10)
# Add headroom above the tallest bar so labels don't get clipped
max_val = max(max(em_vals), max(f1_vals)) if (em_vals or f1_vals) else 1.0
ax.set_ylim(0, max_val * 1.25 + 0.05)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()  # inline display only — no file saving (D-06)
""")


stage4_conclusion = make_markdown("tutorial-cell-38", """\
## Conclusion

This tutorial has walked through the complete HyperRAG pipeline from raw data to \
evaluated retrieval results:

- **Stage 1 (Corpus Loading):** We loaded 3 HotpotQA multi-hop questions and the \
  pre-built Wikipedia corpus of ~450 pages. Each page carries plain text (for B1), \
  raw HTML (for B2), and outgoing hyperlinks (for the graph in Stage 2).
- **Stage 2 (Graph Construction):** We built a directed NetworkX DiGraph where nodes \
  are Wikipedia pages and edges are `<a href>` hyperlinks. The graph encodes the \
  same navigational paths a human researcher would follow — and HyperRAG exploits \
  them via `G.successors()`.
- **Stage 3 (Embeddings & FAISS):** We encoded every page as a 384-dimensional \
  L2-normalised vector using `all-MiniLM-L6-v2` and stored them in a FAISS \
  `IndexFlatIP` for fast cosine similarity search.
- **Stage 4 (Retrieval + Evaluation):** We compared B1 (Naive RAG), B2 \
  (HtmlRAG-style), and HyperRAG on our 3-question sample. HyperRAG's 1-hop graph \
  expansion adds a reasoning step that baselines lack — by following hyperlinks from \
  retrieved pages, it can discover relevant pages that vector similarity alone would miss.

**Key insight:** The hyperlink graph already *exists* inside the HTML `<a href>` tags — \
no LLM-extracted knowledge graph construction is required. HyperRAG's graph expansion \
costs almost nothing extra and offers a principled way to address multi-hop questions \
that single-vector-search systems struggle with.

**Running the full evaluation:** For results on 50 HotpotQA questions, run \
`python run_all_systems.py` from the project root after building the data files \
(Stages 1–3). Results are saved to `data/results.csv` and summarised in the README \
results table.

**Further reading:** See `writing/related_work_draft.md` for academic context on RAG, \
HtmlRAG, GraphRAG, and multi-hop QA — including citations for Lewis et al. (2020), \
Tan et al. (2024), and Edge et al. (2024).\
""")


# ---------------------------------------------------------------------------
# Append all new cells to the notebook
# ---------------------------------------------------------------------------

NB_PATH = Path("notebooks/tutorial.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

new_cells = [
    stage3_intro,
    stage3_build,
    stage3_search_intro,
    stage3_search_code,
    stage3_viz_intro,
    stage3_viz_code,
    stage3_summary,
    stage4_intro,
    stage4_b1_intro,
    stage4_b1_code,
    stage4_b2_intro,
    stage4_b2_code,
    stage4_hr_intro,
    stage4_hr_code,
    stage4_eval_intro,
    stage4_eval_code,
    stage4_chart_intro,
    stage4_chart_code,
    stage4_conclusion,
]

nb["cells"].extend(new_cells)

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

print(f"Done. Total cells: {len(nb['cells'])}")
print("New cells added:", len(new_cells))
