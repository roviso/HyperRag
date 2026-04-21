"""run_all_systems_llm.py — Evaluate B1, B2, and HyperRAG with LLM-based answer generation.

Extends run_all_systems.py by loading a local LLM (TinyLlama by default) and computing
both retrieval-based EM/F1 (does the gold answer appear in retrieved context?) and
generation-based EM/F1 (does the LLM's generated answer match the gold answer?).

Usage:
    python run_all_systems_llm.py                  # default: 10 questions, TinyLlama
    python run_all_systems_llm.py --n 20           # evaluate 20 questions
    python run_all_systems_llm.py --force          # re-run even if CSV exists
    python run_all_systems_llm.py --model "meta-llama/Llama-3.2-3B-Instruct"
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

from src.corpus import load_corpus, load_hotpotqa
from src.embeddings import load_index
from src.graph import load_graph
from src.retrieval import naive_rag, htmlrag_style, hyperrag, compute_em, compute_f1

DATA_DIR = Path("data")
DEFAULT_RESULTS_CSV = DATA_DIR / "llm_eval_results.csv"
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Retrieval constants (mirror run_all_systems.py)
K = 5        # pages retrieved by FAISS
EXPAND_K = 3 # additional graph-neighbor pages HyperRAG appends

# LLM constants
MAX_CONTEXT_TOKENS = 1500  # truncate context to this many tokens before passing to LLM
MAX_NEW_TOKENS = 20        # max tokens for generated answer


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def truncate_to_tokens(text: str, tokenizer: Any, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    """Truncate text to at most max_tokens using the model's tokenizer.

    Truncates from the END — keeps the beginning of the context, which contains
    the most relevant retrieved page (FAISS top-1 is always first).

    Args:
        text: Raw context string from retrieval.
        tokenizer: Model tokenizer (pipe.tokenizer).
        max_tokens: Maximum number of tokens to keep.

    Returns:
        Truncated plain text string.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)


def prepare_context_for_llm(context: str, system_name: str) -> str:
    """Strip HTML from B2 context before LLM inference.

    B2 (htmlrag_style) returns cleaned HTML — suitable for retrieval-based EM/F1
    (tags are stripped by _strip_html_for_metrics) but harmful for LLM input:
    the model wastes context tokens on angle brackets and tag names.
    Plain-text contexts (B1, HyperRAG) pass through unchanged.

    Args:
        context: Retrieved context string.
        system_name: One of 'B1_naive_rag', 'B2_htmlrag', 'HyperRAG'.

    Returns:
        Plain text context string.
    """
    if system_name == "B2_htmlrag" and "<" in context:
        return BeautifulSoup(context, "html.parser").get_text(separator=" ", strip=True)
    return context


def generate_answer(
    question: str,
    context: str,
    pipe: Any,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Generate a short answer from retrieved context using an LLM pipeline.

    Truncates context, constructs a short-answer chat prompt, and returns the
    generated answer stripped and lowercased.

    Args:
        question: HotpotQA question string.
        context: Retrieved plain-text context (HTML stripped before calling).
        pipe: HuggingFace text-generation pipeline (loaded once, reused).
        max_context_tokens: Maximum tokens to use for context.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Generated answer string, lowercased and stripped.
    """
    truncated = truncate_to_tokens(context, pipe.tokenizer, max_context_tokens)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a question-answering assistant. "
                "Answer using ONLY the provided context. "
                "Give a SHORT answer (1-5 words). Do not explain."
            ),
        },
        {
            "role": "user",
            "content": f"Context: {truncated}\n\nQuestion: {question}\n\nAnswer:",
        },
    ]

    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return outputs[0]["generated_text"].strip().lower()


def load_llm_pipeline(model_name: str) -> Any:
    """Load a HuggingFace text-generation pipeline with GPU/CPU auto-detection.

    Uses 4-bit quantization (bitsandbytes) when available on GPU.
    Falls back to float32 on GPU or CPU otherwise.

    Args:
        model_name: HuggingFace model ID.

    Returns:
        Loaded text-generation pipeline.
    """
    _bnb_available = False
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            _bnb_available = True
        except ImportError:
            print("bitsandbytes not available — skipping 4-bit quantization")

    if _bnb_available:
        pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            model_kwargs={"load_in_4bit": True},
        )
        print(f"Loaded {model_name} with 4-bit quantization (GPU)")
    else:
        device = 0 if torch.cuda.is_available() else -1
        pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float32,
            device=device,
        )
        env = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"Loaded {model_name} in float32 ({env})")

    return pipe


def print_summary_table(rows: list[dict[str, Any]], n_questions: int) -> None:
    """Print a formatted summary table comparing retrieval vs generation EM/F1.

    Args:
        rows: List of result row dicts (as written to CSV).
        n_questions: Number of questions evaluated per system.
    """
    system_names = ["B1_naive_rag", "B2_htmlrag", "HyperRAG"]
    summary: dict[str, dict[str, float]] = {}

    for sname in system_names:
        srows = [r for r in rows if r["system"] == sname]
        if not srows:
            continue
        summary[sname] = {
            "retrieval_em":  sum(float(r["retrieval_em"])  for r in srows) / len(srows),
            "retrieval_f1":  sum(float(r["retrieval_f1"])  for r in srows) / len(srows),
            "generation_em": sum(float(r["generation_em"]) for r in srows) / len(srows),
            "generation_f1": sum(float(r["generation_f1"]) for r in srows) / len(srows),
        }

    header = (
        f"{'System':<20} | {'Retr. EM':>8} | {'Retr. F1':>8} | "
        f"{'Gen. EM':>8} | {'Gen. F1':>8}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for sname, m in summary.items():
        print(
            f"{sname:<20} | "
            f"{m['retrieval_em']:>8.3f} | "
            f"{m['retrieval_f1']:>8.4f} | "
            f"{m['generation_em']:>8.3f} | "
            f"{m['generation_f1']:>8.4f}"
        )
    print(sep)
    print(
        f"\nTotal rows: {len(rows)} "
        f"({n_questions} questions × {len(summary)} systems)\n"
    )
    print(
        "Note: Retrieval F1 is near-zero because it computes token overlap between\n"
        "the full retrieved context (~5000 tokens) and the gold answer (1-3 words).\n"
        "Retrieval EM (does the gold string appear anywhere in context?) is the\n"
        "meaningful retrieval quality metric. Generation EM/F1 measure the LLM output."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all 3 RAG systems with LLM generation, compute retrieval + generation EM/F1."""
    parser = argparse.ArgumentParser(
        description="Evaluate B1/B2/HyperRAG with LLM-based generation (TinyLlama default)"
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of HotpotQA questions to evaluate (default: 10)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=(
            "HuggingFace model ID. Default: TinyLlama (ungated, ~2.2GB). "
            "For Llama 3.x: requires `huggingface-cli login` + Meta approval."
        )
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_RESULTS_CSV),
        help="Path for output CSV (default: data/llm_eval_results.csv)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if output CSV already exists (skip cache)"
    )
    args = parser.parse_args()

    n_questions: int = args.n
    model_name: str = args.model
    results_csv = Path(args.output)

    # --- Load cached results if available ---
    if results_csv.exists() and not args.force:
        print(f"Loading cached results from {results_csv}")
        with open(results_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Infer n_questions from cached rows (may differ from --n arg)
        n_questions = len(rows) // 3 if rows else n_questions
        print(f"Loaded {len(rows)} cached rows ({n_questions} questions × 3 systems)")
        print_summary_table(rows, n_questions)
        return

    # --- Load data ---
    print("Loading corpus, FAISS index, and hyperlink graph ...")
    corpus = load_corpus(DATA_DIR / "corpus.json")
    index, _page_ids = load_index(DATA_DIR)
    graph = load_graph(DATA_DIR / "hyperlink_graph.graphml")

    print(f"Loading {n_questions} HotpotQA questions ...")
    qa_items = load_hotpotqa(split="train", n_samples=n_questions)
    print(
        f"Corpus: {len(corpus)} pages | "
        f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )

    # --- Load LLM ---
    print(f"\nLoading LLM: {model_name}")
    print("  TinyLlama is ungated and downloads to ~/.cache/huggingface (~2.2GB on first run)")
    print("  On CPU (float32): ~5-15 min total for inference. Use --force=False to cache.\n")
    pipe = load_llm_pipeline(model_name)

    # --- System definitions ---
    systems: dict[str, Any] = {
        "B1_naive_rag": lambda q, k: naive_rag(q, index, corpus, k),
        "B2_htmlrag":   lambda q, k: htmlrag_style(q, index, corpus, k),
        "HyperRAG":     lambda q, k: hyperrag(q, index, corpus, graph, k=k, expand_k=EXPAND_K),
    }

    # --- Evaluation loop ---
    rows: list[dict[str, Any]] = []
    total = n_questions * len(systems)

    with tqdm(total=total, desc="LLM Evaluation") as pbar:
        for i, qa in enumerate(qa_items):
            question_id = str(qa.get("id", i))
            question = qa["question"]
            gold_answer = qa["answer"]

            for system_name, system_fn in systems.items():
                # Retrieve context
                context = system_fn(question, K)

                # Retrieval-based metrics (existing method: gold answer in context?)
                retrieval_em = compute_em(context, gold_answer)
                retrieval_f1 = round(compute_f1(context, gold_answer), 4)

                # Generation-based: strip HTML → truncate → generate → score
                llm_context = prepare_context_for_llm(context, system_name)
                generated_answer = generate_answer(
                    question, llm_context, pipe,
                    MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS,
                )
                generation_em = compute_em(generated_answer, gold_answer)
                generation_f1 = round(compute_f1(generated_answer, gold_answer), 4)

                rows.append({
                    "question_id":    question_id,
                    "question":       question,
                    "gold_answer":    gold_answer,
                    "system":         system_name,
                    "retrieval_em":   retrieval_em,
                    "retrieval_f1":   retrieval_f1,
                    "generated_answer": generated_answer,
                    "generation_em":  generation_em,
                    "generation_f1":  generation_f1,
                })
                pbar.update(1)

    # --- Save CSV ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id", "question", "gold_answer", "system",
        "retrieval_em", "retrieval_f1",
        "generated_answer", "generation_em", "generation_f1",
    ]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {results_csv} ({len(rows)} rows)")

    # --- Print summary ---
    print_summary_table(rows, n_questions)


if __name__ == "__main__":
    main()
