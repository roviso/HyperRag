"""run_all_systems.py — Evaluate B1, B2, and HyperRAG on 50 HotpotQA questions."""

import csv
from pathlib import Path

from src.corpus import load_corpus, load_hotpotqa
from src.embeddings import load_index
from src.graph import load_graph
from src.retrieval import naive_rag, htmlrag_style, hyperrag, compute_em, compute_f1

DATA_DIR = Path("data")
RESULTS_CSV = DATA_DIR / "results.csv"
N_QUESTIONS = 50
K = 5
EXPAND_K = 5


def main() -> None:
    """Load data, run all 3 retrieval systems on 50 HotpotQA questions, save CSV."""
    print("Loading corpus, index, and graph ...")
    corpus = load_corpus(DATA_DIR / "corpus.json")
    index, _page_ids = load_index(DATA_DIR)
    graph = load_graph(DATA_DIR / "hyperlink_graph.graphml")

    print(f"Loading {N_QUESTIONS} HotpotQA questions ...")
    qa_items = load_hotpotqa(split="train", n_samples=N_QUESTIONS)

    systems = {
        "B1_naive_rag": lambda q, k: naive_rag(q, index, corpus, k),
        "B2_htmlrag": lambda q, k: htmlrag_style(q, index, corpus, k),
        "HyperRAG": lambda q, k: hyperrag(q, index, corpus, graph, k=k, expand_k=EXPAND_K),
    }

    rows: list[dict] = []
    for i, qa in enumerate(qa_items):
        question_id = str(qa.get("id", i))
        question = qa["question"]
        answer = qa["answer"]
        print(f"  [{i + 1}/{N_QUESTIONS}] {question[:60]}...")

        for system_name, system_fn in systems.items():
            context = system_fn(question, K)
            em = compute_em(context, answer)
            f1 = compute_f1(context, answer)
            context_length = len(context)
            rows.append({
                "question_id": question_id,
                "system": system_name,
                "em": em,
                "f1": f1,
                "context_length": context_length,
            })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question_id", "system", "em", "f1", "context_length"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {RESULTS_CSV}")
    print(f"Total rows: {len(rows)}")

    # Print summary table
    print("\nSystem Summary:")
    print(f"  {'System':<20}  {'EM':>6}  {'F1':>6}")
    print("  " + "-" * 36)
    for system_name in systems:
        system_rows = [r for r in rows if r["system"] == system_name]
        avg_em = sum(r["em"] for r in system_rows) / len(system_rows)
        avg_f1 = sum(r["f1"] for r in system_rows) / len(system_rows)
        print(f"  {system_name:<20}  EM={avg_em:.3f}  F1={avg_f1:.3f}")


if __name__ == "__main__":
    main()
