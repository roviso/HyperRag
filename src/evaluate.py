"""Evaluation module: EM and F1 metrics for retrieval quality assessment."""

from typing import Any

from src.retrieval import compute_em, compute_f1


def run_evaluation(
    system_fn: Any,
    qa_items: list[dict[str, Any]],
    k: int = 5,
    **system_kwargs: Any,
) -> dict[str, Any]:
    """Run evaluation of a retrieval system on QA items.

    Calls system_fn(question, k=k, **system_kwargs) for each QA item, then
    computes EM and F1 against the gold answer using retrieval-quality metrics
    (no LLM inference needed for M2).

    Args:
        system_fn: Retrieval function (naive_rag or htmlrag_style). Must accept
            (query: str, k: int, **kwargs) and return a context string.
        qa_items: List of QA dicts, each with 'question' and 'answer' keys.
            Optional 'id' key used as question_id; falls back to loop index.
        k: Number of pages to retrieve per query.
        **system_kwargs: Additional keyword arguments forwarded to system_fn
            (e.g. index=..., corpus=...).

    Returns:
        Dict with keys:
            avg_em (float): Mean EM across all questions.
            avg_f1 (float): Mean F1 across all questions.
            per_question (list[dict]): One entry per QA item with keys
                question_id (str), em (float), f1 (float).
    """
    per_question: list[dict[str, Any]] = []

    for i, qa in enumerate(qa_items):
        question_id = str(qa.get("id", i))
        context = system_fn(qa["question"], k=k, **system_kwargs)
        em = compute_em(context, qa["answer"])
        f1 = compute_f1(context, qa["answer"])
        per_question.append({"question_id": question_id, "em": em, "f1": f1})

    if not per_question:
        return {"avg_em": 0.0, "avg_f1": 0.0, "per_question": []}

    avg_em = sum(r["em"] for r in per_question) / len(per_question)
    avg_f1 = sum(r["f1"] for r in per_question) / len(per_question)

    return {"avg_em": avg_em, "avg_f1": avg_f1, "per_question": per_question}
