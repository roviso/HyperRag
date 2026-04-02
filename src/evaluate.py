"""Evaluation module: EM and F1 metrics for retrieval quality assessment."""

from typing import Any


def compute_em(prediction: str, gold: str) -> float:
    """Compute Exact Match score.

    Checks if the gold answer string appears exactly in the prediction context.

    Args:
        prediction: Retrieved context string.
        gold: Gold answer string.

    Returns:
        1.0 if exact match found, 0.0 otherwise.
    """
    raise NotImplementedError("Implemented in Phase 5")


def compute_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 score.

    Tokenizes both strings and computes precision, recall, and F1
    based on token overlap.

    Args:
        prediction: Retrieved context string.
        gold: Gold answer string.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    raise NotImplementedError("Implemented in Phase 5")


def run_evaluation(
    system_fn: Any,
    qa_items: list[dict[str, Any]],
    k: int = 5,
    **system_kwargs: Any,
) -> dict[str, float]:
    """Run evaluation of a retrieval system on QA items.

    Args:
        system_fn: Retrieval function (naive_rag, htmlrag_style, or hyperrag).
        qa_items: List of QA items with 'question' and 'answer' keys.
        k: Number of pages to retrieve per query.
        **system_kwargs: Additional kwargs passed to system_fn.

    Returns:
        Dict with keys: avg_em, avg_f1, per_question (list of individual scores).
    """
    raise NotImplementedError("Implemented in Phase 5/6")
