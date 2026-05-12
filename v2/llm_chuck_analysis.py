# CONFIGURATION
N_FAIL = 'all'   # int or 'all'
N_BOTH = 'all'   # int or 'all'
CHUNK_SIZES   = [512,1024,2048,4000,8000,'full_page']
CHUNK_OVERLAP = 64
K        = 5
EXPAND_K = 5
LOCAL_MODEL       = 'Qwen/Qwen3.5-9B'
MAX_CTX_CHARS_CAP = 32000
SAVE_RESULTS = True

GPU_ID = 0   # <-- single source of truth for GPU selection

# ── Fix: set CUDA_VISIBLE_DEVICES BEFORE importing torch ──────────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)   # must happen before torch import

subset_desc = f'{N_FAIL} graph-wins + {N_BOTH} both-pass'
if isinstance(N_FAIL, int) and isinstance(N_BOTH, int):
    subset_desc += f' = up to {N_FAIL + N_BOTH} questions'
else:
    subset_desc += ' = all available in each bucket'

print('=' * 62)
print('CONFIGURATION SUMMARY')
print('=' * 62)
print(f'  Subset     : {subset_desc}')
print(f'  Chunks     : {CHUNK_SIZES}')
print(f'  K={K}  expand_k={EXPAND_K}')
print(f'  LLM        : {LOCAL_MODEL}')
print(f'  Max ctx cap: {MAX_CTX_CHARS_CAP:,} chars')
print('=' * 62)


import json, sys, re, string, csv, time
from pathlib import Path
from collections import Counter, defaultdict

import faiss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline

if Path('utils.py').exists():
    BASE_DIR = Path('.')
elif Path('v2/utils.py').exists():
    BASE_DIR = Path('v2')
else:
    raise FileNotFoundError('utils.py not found. Run from v2/ or HyperRAG-M2/')

DATA_DIR   = BASE_DIR / 'data'
OUTPUT_DIR = DATA_DIR / 'chunk_size_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("results_by_chunk")
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from utils import (
    embed_texts, build_index, retrieve,
    compute_em, compute_f1, compute_supporting_recall, clean_html
)

print(f'BASE_DIR   : {BASE_DIR.resolve()}')
print(f'OUTPUT_DIR : {OUTPUT_DIR.resolve()}')
print('All imports OK.')

# # ── Single GPU setup ───────────────────────────────────────────────────────
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {DEVICE}")
# if torch.cuda.is_available():
#     free, total = torch.cuda.mem_get_info(0)
#     print(f"  GPU 0: {free/(1024**3):.1f} / {total/(1024**3):.1f} GB free")

# # ── Load LLM once on the single GPU ───────────────────────────────────────
# print(f"\nLoading model {LOCAL_MODEL} on {DEVICE}...")
device_str  = 'GPU (cuda:0)' if torch.cuda.is_available() else 'CPU'
torch_dtype = torch.float16  if torch.cuda.is_available() else torch.float32

print(f'Loading: {LOCAL_MODEL}')
print(f'Actual GPU: {GPU_ID}  mapped as cuda:0  |  dtype={torch_dtype}')
print('First run downloads model weights -- may take a few minutes...')

local_llm = pipeline(
    "text-generation",
    model=LOCAL_MODEL,
    torch_dtype=torch_dtype,
    device=0,                  # correct: CUDA_VISIBLE_DEVICES already remapped GPU_ID -> 0
    trust_remote_code=True,
)

print('\nQwen LLM ready.')


# ── Data loading ───────────────────────────────────────────────────────────
print('Loading corpus and questions...')
with open(DATA_DIR / 'corpus_001.json')    as f: corpus    = json.load(f)
with open(DATA_DIR / 'questions_001.json') as f: questions = json.load(f)

for fname in ['results_simple_rag.json', 'results_htmlrag.json', 'results_hyperrag.json']:
    if not (DATA_DIR / fname).exists():
        raise FileNotFoundError(f'{fname} missing - run notebooks 02, 03, 04 first.')

with open(DATA_DIR / 'results_simple_rag.json') as f: simple_results = json.load(f)
with open(DATA_DIR / 'results_htmlrag.json')    as f: html_results   = json.load(f)
with open(DATA_DIR / 'results_hyperrag.json')   as f: hyper_results  = json.load(f)

simple_per_q = {r['id']: r for r in simple_results['per_question']}
html_per_q   = {r['id']: r for r in html_results['per_question']}
hyper_per_q  = {r['id']: r for r in hyper_results['per_question']}

print(f'Corpus    : {len(corpus):,} pages')
print(f'Questions : {len(questions):,}')
print()
print(f'  Simple RAG -- mean EM = {simple_results["mean_em"]:.3f}')
print(f'  HtmlRAG    -- mean EM = {html_results["mean_em"]:.3f}')
print(f'  HyperRAG   -- mean EM = {hyper_results["mean_em"]:.3f}')


# ── Subset building ────────────────────────────────────────────────────────
fail_then_graph = []
both_pass       = []
both_fail       = []
graph_fails     = []

for q in questions:
    sq = simple_per_q.get(q['id'])
    hq = hyper_per_q.get(q['id'])
    if not sq or not hq:
        continue
    s, h = int(sq['em']), int(hq['em'])
    if   s == 0 and h == 1: fail_then_graph.append(q)
    elif s == 1 and h == 1: both_pass.append(q)
    elif s == 0 and h == 0: both_fail.append(q)
    else:                   graph_fails.append(q)

n = len(questions)
print('EM Pattern Breakdown')
print(f'  graph-wins (S=0,H=1): {len(fail_then_graph):5d} ({100*len(fail_then_graph)/n:.1f}%)')
print(f'  both-pass  (S=1,H=1): {len(both_pass):5d} ({100*len(both_pass)/n:.1f}%)')
print(f'  both-fail  (S=0,H=0): {len(both_fail):5d} ({100*len(both_fail)/n:.1f}%)')
print()

if N_FAIL in ('all', None):
    n_fail = len(fail_then_graph)
else:
    n_fail = min(int(N_FAIL), len(fail_then_graph))

if N_BOTH in ('all', None):
    n_both = len(both_pass)
else:
    n_both = min(int(N_BOTH), len(both_pass))

subset_fail = fail_then_graph[:n_fail]
subset_both = both_pass[:n_both]
subset      = subset_fail + subset_both

print(f'Mixed subset: {n_fail} graph-wins + {n_both} both-pass = {len(subset)} questions')
print()
print(f'  {"Type":<12}  {"Gold Answer":<28}  Question')
print('-' * 85)
for q in subset[:4]:
    kind = 'graph-wins' if simple_per_q[q['id']]['em'] == 0.0 else 'both-pass'
    print(f'  {kind:<12}  {q["answer"][:26]:<28}  {q["question"][:48]}')


# ── Chunking utilities ─────────────────────────────────────────────────────
def split_into_chunks(text: str, chunk_size: int, overlap: int = 64) -> list[str]:
    'Split text into overlapping chunks, breaking at word boundaries.'
    if len(text) <= chunk_size:
        return [text.strip()]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        nxt = end - overlap
        start = nxt if nxt > start else end
    return chunks


def build_chunk_corpus(corpus: list[dict], chunk_size: int, overlap: int = 64) -> list[dict]:
    'Convert page corpus into overlapping fixed-size chunk corpus.'
    docs = []
    for page in corpus:
        text_chunks = split_into_chunks(page['text'], chunk_size, overlap)
        raw_html    = page.get('html', '') or ''
        html_clean  = clean_html(raw_html) if raw_html else page['text']
        html_chunks = split_into_chunks(html_clean, chunk_size, overlap)
        n = len(text_chunks)
        for i, tc in enumerate(text_chunks):
            hc = html_chunks[i] if i < len(html_chunks) else tc
            docs.append({
                'title'    : page['title'],
                'chunk_idx': i,
                'n_chunks' : n,
                'text'     : tc,
                'html_text': hc,
                'links'    : page.get('links', []),
            })
    return docs


def build_full_page_corpus(corpus: list[dict]) -> list[dict]:
    'One chunk per page -- entire page text, no splitting.'
    docs = []
    for page in corpus:
        raw_html   = page.get('html', '') or ''
        html_clean = clean_html(raw_html) if raw_html else page['text']
        docs.append({
            'title'    : page['title'],
            'chunk_idx': 0,
            'n_chunks' : 1,
            'text'     : page['text'],
            'html_text': html_clean,
            'links'    : page.get('links', []),
        })
    return docs


def build_chunk_index(chunk_docs: list[dict]) -> faiss.IndexFlatIP:
    'Build FAISS cosine-similarity index over chunk plain texts.'
    vecs = embed_texts([d['text'] for d in chunk_docs])
    idx  = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return idx


def retrieve_chunks(query: str, chunk_index: faiss.IndexFlatIP,
                    chunk_docs: list[dict], k: int = 5) -> list[dict]:
    'Retrieve top-k chunks by cosine similarity.'
    q_vec    = embed_texts([query])
    k_actual = min(k, chunk_index.ntotal)
    scores, indices = chunk_index.search(q_vec, k_actual)
    results = []
    for ix, sc in zip(indices[0], scores[0]):
        if 0 <= ix < len(chunk_docs):
            c = dict(chunk_docs[ix])
            c['_score'] = float(sc)
            results.append(c)
    return results


def simple_rag_chunked(question: str, chunk_index, chunk_docs: list[dict], k: int = 5):
    'Simple RAG: retrieve top-k chunks, context = plain text.'
    chunks  = retrieve_chunks(question, chunk_index, chunk_docs, k=k)
    context = '\n\n'.join(c['text'] for c in chunks)
    titles  = list(dict.fromkeys(c['title'] for c in chunks))
    return context, titles, chunks


def html_rag_chunked(question: str, chunk_index, chunk_docs: list[dict], k: int = 5):
    'HtmlRAG: same retrieval, context uses cleaned HTML text.'
    chunks  = retrieve_chunks(question, chunk_index, chunk_docs, k=k)
    context = '\n\n'.join(c['html_text'] for c in chunks)
    titles  = list(dict.fromkeys(c['title'] for c in chunks))
    return context, titles, chunks


def hyper_rag_chunked(question: str, chunk_index, chunk_docs: list[dict],
                      corpus: list[dict], graph: nx.DiGraph,
                      k: int = 5, expand_k: int = 3):
    'HyperRAG: FAISS top-k + 1-hop graph neighbor expansion.'
    title_to_page   = {p['title']: p for p in corpus}
    initial         = retrieve_chunks(question, chunk_index, chunk_docs, k=k)
    init_ttls       = set(c['title'] for c in initial)
    neighbor_titles = set()
    for t in init_ttls:
        if graph.has_node(t):
            for nbr in list(graph.successors(t)) + list(graph.predecessors(t)):
                if nbr in title_to_page and nbr not in init_ttls:
                    neighbor_titles.add(nbr)
    expanded = []
    if neighbor_titles and expand_k > 0:
        nbr_chunks = [c for c in chunk_docs if c['title'] in neighbor_titles]
        if nbr_chunks:
            nv  = embed_texts([c['text'] for c in nbr_chunks])
            qv  = embed_texts([question])
            sc  = (qv @ nv.T)[0]
            top = sc.argsort()[::-1][:expand_k]
            for ix in top:
                ec = dict(nbr_chunks[ix])
                ec['_score']    = float(sc[ix])
                ec['_expanded'] = True
                expanded.append(ec)
    all_chunks = initial + expanded
    context    = '\n\n'.join(c['text'] for c in all_chunks)
    titles     = list(dict.fromkeys(c['title'] for c in all_chunks))
    return context, titles, all_chunks


print('Chunk + retrieval functions defined.')


# ── Retrieval quality ──────────────────────────────────────────────────────
def compute_retrieval_quality(context: str, chunks: list[dict],
                               gold_answer: str, gold_titles: list[str]) -> dict:
    'Compute retrieval EM, F1, supporting recall for retrieved chunks.'
    retrieved_titles = list(dict.fromkeys(c['title'] for c in chunks))
    ret_em  = float(compute_em(context, gold_answer))
    ret_f1  = round(float(compute_f1(context, gold_answer)), 4)
    sup_rec = round(float(compute_supporting_recall(retrieved_titles, gold_titles)), 4)
    chunk_details = []
    for rank, c in enumerate(chunks, 1):
        chunk_details.append({
            'rank'         : rank,
            'page_title'   : c['title'],
            'chunk_idx'    : c.get('chunk_idx', 0),
            'n_chunks'     : c.get('n_chunks', 1),
            'score'        : round(c.get('_score', 0.0), 4),
            'is_gold_page' : c['title'] in set(gold_titles),
            'has_answer'   : gold_answer.lower() in c['text'].lower(),
            'is_expanded'  : c.get('_expanded', False),
            'text_preview' : c['text'][:120].replace('\n', ' '),
        })
    return {
        'retrieval_em'      : ret_em,
        'retrieval_f1'      : ret_f1,
        'supporting_recall' : sup_rec,
        'retrieved_titles'  : retrieved_titles,
        'chunk_details'     : chunk_details,
    }


# ── Generation utilities ───────────────────────────────────────────────────
def normalize_answer(text: str) -> str:
    'HotpotQA normalization: lowercase, strip articles and punctuation.'
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def compute_gen_em(generated: str, gold: str) -> float:
    """
    Generation EM:
      1 if normalized generated == normalized gold          (exact match)
      1 if normalized generated is contained in normalized gold  (LLM gave partial answer)
      1 if normalized gold is contained in normalized generated  (LLM gave verbose answer)
      0 otherwise
    """
    norm_gen  = normalize_answer(generated)
    norm_gold = normalize_answer(gold)

    if not norm_gen or not norm_gold:
        return 0.0

    # 1. Exact match
    if norm_gen == norm_gold:
        return 1.0

    gen_tokens  = set(norm_gen.split())
    gold_tokens = set(norm_gold.split())

    # 2. Generated tokens are a subset of gold tokens
    #    e.g. "victor mature" ⊆ "victor john mature"
    if gen_tokens.issubset(gold_tokens):
        return 1.0

    # 3. Gold tokens are a subset of generated tokens
    #    e.g. "victor john mature" ⊆ "victor john mature sr"
    if gold_tokens.issubset(gen_tokens):
        return 1.0

    # 4. Keep contiguous substring checks too (for phrase-level matching)
    if norm_gen in norm_gold:
        return 1.0
    if norm_gold in norm_gen:
        return 1.0

    return 0.0


def compute_gen_f1(generated: str, gold: str) -> float:
    'Generation F1: token-overlap F1 between generated and gold.'
    gen_t  = normalize_answer(generated).split()
    gold_t = normalize_answer(gold).split()
    if not gen_t or not gold_t:
        return 0.0
    gen_c  = Counter(gen_t)
    gold_c = Counter(gold_t)
    overlap = sum((gen_c & gold_c).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(gen_t)
    rec  = overlap / len(gold_t)
    return round(2 * prec * rec / (prec + rec), 4)


def generate_qwen(question: str, context: str,
                  max_ctx_chars: int = MAX_CTX_CHARS_CAP) -> str:
    'Generate a short answer with Qwen using ChatML prompt format.'
    ctx = context[:max_ctx_chars]
    
    # Explicitly disable thinking in the prompt
    prompt = (
        '<|im_start|>system\n'
        'You are a precise QA assistant. '
        'Answer the question with a short phrase (1-5 words). '
        'Extract the answer directly from the provided context. '
        'Do not think out loud. Do not explain. Just output the answer.<|im_end|>\n'
        f'<|im_start|>user\nContext:\n{ctx}\n\nQuestion: {question}\n\n'
        'Answer (short phrase only, no explanation):<|im_end|>\n'
        '<|im_start|>assistant\n'
        # Prefill with </think> to force model to skip thinking block
        '<think>\n</think>\n'
    )
    try:
        raw = local_llm(
            prompt,
            max_new_tokens=128,   # enough for answer + any stray tokens
            do_sample=False,
            return_full_text=False,
        )[0]['generated_text']

        # ── Strip thinking block (complete or truncated) ───────────────────
        # Remove complete <think>...</think> blocks
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        # Remove any leftover opening <think> tag and everything after it
        raw = re.sub(r'<think>.*', '', raw, flags=re.DOTALL)
        # Remove stray </think> tags
        raw = raw.replace('</think>', '')

        # ── Strip ChatML stop tokens ───────────────────────────────────────
        raw = raw.split('<|im_end|>')[0]
        raw = raw.split('<|endoftext|>')[0]
        raw = raw.split('<|im_start|>')[0]

        # ── Clean up whitespace ────────────────────────────────────────────
        raw = raw.strip()

        # ── Fallback: take only the first non-empty line ───────────────────
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        raw   = lines[0] if lines else ''

        # ── Last resort: if still empty return placeholder ─────────────────
        return raw if raw else '[no answer]'

    except Exception as e:
        return f'[LLM error: {e}]'


print('Generation functions defined.')
print(f'  Max ctx cap: {MAX_CTX_CHARS_CAP:,} chars')

import pickle

def get_or_build_chunk_data(
    corpus       : list[dict],
    cs,
    chunk_overlap: int = CHUNK_OVERLAP,
    cache_dir    : Path = RESULTS_DIR / "index_cache",
) -> tuple[list[dict], faiss.IndexFlatIP]:
    """
    Load chunk_docs + FAISS index from disk if cached, otherwise build and save.
    Cache is keyed by chunk_size and corpus length so stale caches are detected.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cs_label   = "full_page" if cs == "full_page" else str(cs)
    # Include corpus length in filename so cache is invalidated if corpus changes
    corpus_tag = f"n{len(corpus)}"
    docs_path  = cache_dir / f"chunk_docs_{cs_label}_{corpus_tag}.pkl"
    idx_path   = cache_dir / f"faiss_index_{cs_label}_{corpus_tag}.bin"

    # ── Load from cache ────────────────────────────────────────────────────
    if docs_path.exists() and idx_path.exists():
        print(f"  [CACHE HIT] Loading chunk_docs  ← {docs_path.name}")
        with open(docs_path, "rb") as f:
            chunk_docs = pickle.load(f)

        print(f"  [CACHE HIT] Loading FAISS index ← {idx_path.name}")
        chunk_index = faiss.read_index(str(idx_path))

        print(f"  Loaded {len(chunk_docs):,} docs  |  {chunk_index.ntotal:,} vectors")
        return chunk_docs, chunk_index

    # ── Build from scratch ─────────────────────────────────────────────────
    print(f"  [CACHE MISS] Building chunk corpus for cs={cs_label} ...")
    if cs == "full_page":
        chunk_docs = build_full_page_corpus(corpus)
    else:
        chunk_docs = build_chunk_corpus(corpus, cs, chunk_overlap)
    print(f"  Chunks: {len(chunk_docs):,}")

    print(f"  Building FAISS index ...")
    chunk_index = build_chunk_index(chunk_docs)
    print(f"  Index : {chunk_index.ntotal:,} vectors")

    # ── Persist to disk ────────────────────────────────────────────────────
    print(f"  Saving chunk_docs  → {docs_path.name}")
    with open(docs_path, "wb") as f:
        pickle.dump(chunk_docs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Saving FAISS index → {idx_path.name}")
    faiss.write_index(chunk_index, str(idx_path))

    print(f"  Cache saved to {cache_dir}")
    return chunk_docs, chunk_index


# ── Main experiment loop (single GPU, sequential) ──────────────────────────
def run_experiment(
    corpus, subset, graph, chunk_sizes,
    k=K, expand_k=EXPAND_K,
    max_ctx_cap=MAX_CTX_CHARS_CAP,
    chunk_overlap=CHUNK_OVERLAP,
):
    all_results = []

    for cs in chunk_sizes:
        cs_label  = "full_page" if cs == "full_page" else str(cs)
        csv_path  = RESULTS_DIR / f"results_chunk_{cs_label}.csv"
        json_path = RESULTS_DIR / f"results_chunk_{cs_label}.json"

        # Resume: skip already-finished chunk sizes
        if csv_path.exists() and json_path.exists():
            print(f"[SKIP] chunk={cs_label} already saved → {csv_path}")
            existing = pd.read_csv(csv_path).to_dict("records")
            all_results.extend(existing)
            continue

        print(f'\n{"=" * 65}')
        print(f'CHUNK SIZE : {cs_label}')
        print(f'{"=" * 65}')

        # ── Load or build index (cached) ───────────────────────────────────
        chunk_docs, chunk_index = get_or_build_chunk_data(corpus, cs, chunk_overlap)

        max_ctx = max_ctx_cap if cs == "full_page" else min(cs * k, max_ctx_cap)
        print(f"  Max ctx → Qwen : {max_ctx:,} chars")

        cs_results  = []
        total_items = len(subset) * 3

        pbar = tqdm(total=total_items, desc=f"chunk={cs_label}", unit="items", ncols=80)

        t0 = time.time()
        for q in subset:
            gold   = q["answer"]
            golds  = q["supporting_titles"]
            q_type = "graph-wins" if simple_per_q[q["id"]]["em"] == 0.0 else "both-pass"

            systems = {
                "Simple RAG": lambda _q=q: simple_rag_chunked(
                    _q["question"], chunk_index, chunk_docs, k=k),
                "HtmlRAG":    lambda _q=q: html_rag_chunked(
                    _q["question"], chunk_index, chunk_docs, k=k),
                "HyperRAG":   lambda _q=q: hyper_rag_chunked(
                    _q["question"], chunk_index, chunk_docs,
                    corpus, graph, k=k, expand_k=expand_k),
            }

            for sys_name, fn in systems.items():
                ctx, titles, chunks = fn()
                quality = compute_retrieval_quality(ctx, chunks, gold, golds)

                answer = generate_qwen(q["question"], ctx, max_ctx_chars=max_ctx)
                gen_em = compute_gen_em(answer, gold)
                gen_f1 = compute_gen_f1(answer, gold)

                row = {
                    "chunk_size"        : cs_label,
                    "question_id"       : q["id"],
                    "question"          : q["question"],
                    "gold_answer"       : gold,
                    "supporting_titles" : list(golds),
                    "subset_type"       : q_type,
                    "system"            : sys_name,
                    "k"                 : k,
                    "expand_k"          : expand_k if sys_name == "HyperRAG" else 0,
                    "retrieval_em"      : quality["retrieval_em"],
                    "retrieval_f1"      : quality["retrieval_f1"],
                    "supporting_recall" : quality["supporting_recall"],
                    "retrieved_titles"  : quality["retrieved_titles"],
                    "qwen_answer"       : answer,
                    "qwen_gen_em"       : gen_em,
                    "qwen_gen_f1"       : gen_f1,
                }
                cs_results.append(row)
                pbar.update(1)

                if len(cs_results) % 30 == 0:
                    recent  = cs_results[-30:]
                    avg_em  = np.mean([r["qwen_gen_em"] for r in recent])
                    avg_f1  = np.mean([r["qwen_gen_f1"] for r in recent])
                    tqdm.write(
                        f"  [{cs_label}] {len(cs_results):>4}/{total_items} | "
                        f"last-30 GenEM={avg_em:.3f}  GenF1={avg_f1:.3f} | "
                        f'sample: "{cs_results[-1]["qwen_answer"][:60]}"'
                    )

        pbar.close()
        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.1f}s  |  {len(cs_results):,} items")

        # ── Save CSV ───────────────────────────────────────────────────────
        df_cs     = pd.DataFrame(cs_results)
        df_cs_csv = df_cs.copy()
        df_cs_csv["supporting_titles"] = df_cs_csv["supporting_titles"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else x
        )
        df_cs_csv["retrieved_titles"] = df_cs_csv["retrieved_titles"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else x
        )
        df_cs_csv.to_csv(csv_path, index=False)
        print(f"  CSV  saved → {csv_path}")

        # ── Save JSON ──────────────────────────────────────────────────────
        json_out = {
            "chunk_size" : cs_label,
            "n_items"    : len(cs_results),
            "config"     : {"k": k, "expand_k": expand_k, "max_ctx": max_ctx},
            "mean_by_system" : {},
            "per_item"   : cs_results,
        }
        for sys_name in ["Simple RAG", "HtmlRAG", "HyperRAG"]:
            sys_rows = [r for r in cs_results if r["system"] == sys_name]
            if sys_rows:
                json_out["mean_by_system"][sys_name] = {
                    "retrieval_em"      : round(np.mean([r["retrieval_em"]      for r in sys_rows]), 4),
                    "retrieval_f1"      : round(np.mean([r["retrieval_f1"]      for r in sys_rows]), 4),
                    "supporting_recall" : round(np.mean([r["supporting_recall"] for r in sys_rows]), 4),
                    "qwen_gen_em"       : round(np.mean([r["qwen_gen_em"]       for r in sys_rows]), 4),
                    "qwen_gen_f1"       : round(np.mean([r["qwen_gen_f1"]       for r in sys_rows]), 4),
                }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_out, f, indent=2, ensure_ascii=False)
        print(f"  JSON saved → {json_path}")

        summary = (
            df_cs.groupby("system")[["retrieval_f1", "qwen_gen_em", "qwen_gen_f1"]]
                 .mean().round(4)
        )
        print(f"\n  Summary (chunk={cs_label}):")
        print(summary.to_string())

        all_results.extend(cs_results)

    return pd.DataFrame(all_results)


# ── Build hyperlink graph ──────────────────────────────────────────────────
print("Building hyperlink graph...")
all_titles = {p["title"] for p in corpus}
graph      = nx.DiGraph()
graph.add_nodes_from(all_titles)
for page in corpus:
    for link in page.get("links", []):
        if link in all_titles and link != page["title"]:
            graph.add_edge(page["title"], link)
print(f"  Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges\n")


# ── Run experiment ─────────────────────────────────────────────────────────
df_all = run_experiment(
    corpus      = corpus,
    subset      = subset,
    graph       = graph,
    chunk_sizes = CHUNK_SIZES,
)

df_all.to_csv(RESULTS_DIR / "results_ALL.csv", index=False)
print(f"\nDone. {len(df_all)} rows → {RESULTS_DIR}/results_ALL.csv")

all_results = df_all.to_dict('records')
sys_names   = ['Simple RAG', 'HtmlRAG', 'HyperRAG']
cs_labels   = [str(cs) for cs in CHUNK_SIZES]


def mean_safe(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else 0.0


# ── Save results ───────────────────────────────────────────────────────────
if SAVE_RESULTS:
    json_path = OUTPUT_DIR / 'chunk_comparison_results.json'
    out = {
        'config': {
            'chunk_sizes'      : cs_labels,
            'chunk_overlap'    : CHUNK_OVERLAP,
            'k'                : K,
            'expand_k'         : EXPAND_K,
            'local_model'      : LOCAL_MODEL,
            'max_ctx_chars_cap': MAX_CTX_CHARS_CAP,
            'n_questions'      : len(subset),
        },
        'results': all_results,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'JSON saved  ->  {json_path}')

    csv_path    = OUTPUT_DIR / 'chunk_comparison_per_question.csv'
    flat_fields = [
        'chunk_size', 'question_id', 'subset_type', 'system',
        'question', 'gold_answer',
        'retrieval_em', 'retrieval_f1', 'supporting_recall',
        'retrieved_titles',
        'qwen_answer', 'qwen_gen_em', 'qwen_gen_f1',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            row = dict(r)
            row['retrieved_titles'] = ' | '.join(r.get('retrieved_titles', []))
            writer.writerow(row)
    print(f'CSV saved   ->  {csv_path}')

    agg_path = OUTPUT_DIR / 'chunk_comparison_summary.csv'
    agg_rows = []
    for csl in cs_labels:
        for sys_name in sys_names:
            rows = [r for r in all_results
                    if r['chunk_size'] == csl and r['system'] == sys_name]
            if not rows:
                continue
            agg_rows.append({
                'chunk_size'             : csl,
                'system'                 : sys_name,
                'n_questions'            : len(rows),
                'mean_retrieval_em'      : round(mean_safe(rows, 'retrieval_em'),      4),
                'mean_retrieval_f1'      : round(mean_safe(rows, 'retrieval_f1'),      4),
                'mean_supporting_recall' : round(mean_safe(rows, 'supporting_recall'), 4),
                'mean_qwen_gen_em'       : round(mean_safe(rows, 'qwen_gen_em'),       4),
                'mean_qwen_gen_f1'       : round(mean_safe(rows, 'qwen_gen_f1'),       4),
            })
    with open(agg_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f'CSV 2 saved ->  {agg_path}')


# ── Plotting ───────────────────────────────────────────────────────────────
def get_means(metric, sys_name):
    return [
        mean_safe(
            [r for r in all_results
             if r['chunk_size'] == csl and r['system'] == sys_name],
            metric
        )
        for csl in cs_labels
    ]

sys_colors  = {'Simple RAG': 'steelblue', 'HtmlRAG': 'teal', 'HyperRAG': 'coral'}
sys_markers = {'Simple RAG': 'o', 'HtmlRAG': 's', 'HyperRAG': '^'}
metrics_info = [
    ('retrieval_em',  'Retrieval EM\n(gold in context?)'),
    ('retrieval_f1',  'Retrieval F1\n(token overlap)'),
    ('qwen_gen_em',   'Qwen Gen EM\n(exact match)'),
    ('qwen_gen_f1',   'Qwen Gen F1\n(token F1)'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
fig.suptitle(
    f'HyperRAG -- Chunk Size Impact\n(K={K}, expand_k={EXPAND_K} | {len(subset)} questions)',
    fontsize=13, fontweight='bold',
)
for ax, (metric, title) in zip(axes, metrics_info):
    for sys_name in sys_names:
        vals = get_means(metric, sys_name)
        ax.plot(cs_labels, vals, marker=sys_markers[sys_name],
                color=sys_colors[sys_name], label=sys_name, linewidth=2.2, markersize=8)
        for xi, v in enumerate(vals):
            ax.annotate(f'{v:.3f}', (xi, v), textcoords='offset points',
                        xytext=(0, 9), ha='center', fontsize=7,
                        color=sys_colors[sys_name], fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Chunk Size (chars)', fontsize=9)
    ax.set_xticks(range(len(cs_labels)))
    ax.set_xticklabels(cs_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart1_path = OUTPUT_DIR / 'chunk_size_line_plots.png'
plt.savefig(str(chart1_path), dpi=130, bbox_inches='tight')
plt.show()
print(f'Chart 1 saved -> {chart1_path}')


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f'Qwen3.5-9B Generation Quality by Chunk Size & System\n'
    f'(K={K}, expand_k={EXPAND_K}  |  {len(subset)} questions)',
    fontsize=12, fontweight='bold',
)
x       = np.arange(len(cs_labels))
width   = 0.25
offsets = [-width, 0, width]

for ax_idx, (metric, ylabel) in enumerate([
    ('qwen_gen_em', 'Generation EM (Exact Match)'),
    ('qwen_gen_f1', 'Generation F1 (Token F1)'),
]):
    ax = axes[ax_idx]
    for i, sys_name in enumerate(sys_names):
        vals = get_means(metric, sys_name)
        bars = ax.bar(x + offsets[i], vals, width,
                      label=sys_name, color=sys_colors[sys_name], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.3f}', ha='center', va='bottom',
                        fontsize=6.5, fontweight='bold')
    ax.set_title(ylabel, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cs_labels, rotation=30, ha='right', fontsize=8)
    ax.set_xlabel('Chunk Size (chars)', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart2_path = OUTPUT_DIR / 'chunk_size_grouped_bars.png'
plt.savefig(str(chart2_path), dpi=130, bbox_inches='tight')
plt.show()
print(f'Chart 2 saved -> {chart2_path}')


fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle(
    'Retrieval EM vs Qwen Generation EM -- Extraction Gap by System\n'
    f'(K={K}, expand_k={EXPAND_K}  |  {len(subset)} questions)',
    fontsize=12, fontweight='bold',
)
x     = np.arange(len(cs_labels))
width = 0.35

for ax, sys_name in zip(axes, sys_names):
    ret_vals = get_means('retrieval_em', sys_name)
    gen_vals = get_means('qwen_gen_em',  sys_name)
    ax.bar(x - width/2, ret_vals, width,
           label='Retrieval EM', color='#aec6cf', alpha=0.9)
    ax.bar(x + width/2, gen_vals, width,
           label='Qwen Gen EM',  color=sys_colors[sys_name], alpha=0.85)
    for xi, (rv, gv) in enumerate(zip(ret_vals, gen_vals)):
        ax.text(xi - width/2, rv + 0.012, f'{rv:.2f}',
                ha='center', va='bottom', fontsize=6.5)
        ax.text(xi + width/2, gv + 0.012, f'{gv:.2f}',
                ha='center', va='bottom', fontsize=6.5)
    ax.set_title(sys_name, fontsize=11, fontweight='bold',
                 color=sys_colors[sys_name])
    ax.set_xticks(x)
    ax.set_xticklabels(cs_labels, rotation=30, ha='right', fontsize=7)
    ax.set_xlabel('Chunk Size', fontsize=9)
    ax.set_ylabel('EM' if sys_name == sys_names[0] else '')
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart3_path = OUTPUT_DIR / 'retrieval_vs_generation_gap.png'
plt.savefig(str(chart3_path), dpi=130, bbox_inches='tight')
plt.show()
print(f'Chart 3 saved -> {chart3_path}')


# ── Final aggregate table ──────────────────────────────────────────────────
def fmt(v): return f'{v:.3f}'

print('AGGREGATE RESULTS -- Chunk Size x System')
print('=' * 100)
hdr = (f'  {"Chunk":<12} {"System":<13} {"Ret.EM":>7} '
       f'{"Ret.F1":>7} {"SuppRec":>8} {"GenEM":>7} {"GenF1":>7}')
print(hdr)
print(f'  {"-" * 63}')

for csl in cs_labels:
    for sys_name in sys_names:
        rows = [r for r in all_results
                if r['chunk_size'] == csl and r['system'] == sys_name]
        if not rows:
            continue
        print(f'  {csl:<12} {sys_name:<13} '
              f'{fmt(mean_safe(rows, "retrieval_em")):>7} '
              f'{fmt(mean_safe(rows, "retrieval_f1")):>7} '
              f'{fmt(mean_safe(rows, "supporting_recall")):>8} '
              f'{fmt(mean_safe(rows, "qwen_gen_em")):>7} '
              f'{fmt(mean_safe(rows, "qwen_gen_f1")):>7}')
    print()