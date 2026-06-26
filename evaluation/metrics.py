"""
metrics.py
==========
Retrieval-quality metrics. These answer the question a healthcare team will
always ask: "before the LLM even speaks, are we finding the RIGHT evidence?"

Every function takes:
  retrieved : list of doc ids, in ranked order (best first)
  relevant  : set/list of doc ids that are actually relevant (the "gold" answer)

These are pure Python with no dependencies, so they're easy to trust and test.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def _as_set(relevant: Iterable[str]) -> set[str]:
    return set(relevant)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Of all the relevant docs, what fraction did we retrieve in the top k?"""
    relevant = _as_set(relevant)
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Of the top k we retrieved, what fraction were actually relevant?"""
    relevant = _as_set(relevant)
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / k


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant doc is in the top k, else 0.0."""
    relevant = _as_set(relevant)
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """
    1 / (rank of the first relevant doc). Rewards putting a correct doc HIGH.
    Returns 0.0 if no relevant doc was retrieved at all.
    """
    relevant = _as_set(relevant)
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(results: list[tuple[Sequence[str], Iterable[str]]]) -> float:
    """MRR across many queries. `results` is a list of (retrieved, relevant)."""
    if not results:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in results) / len(results)


def aggregate(results: list[tuple[Sequence[str], Iterable[str]]], k: int = 5) -> dict:
    """Convenience: compute the headline metrics over a whole eval set."""
    n = len(results) or 1
    return {
        f"recall@{k}": sum(recall_at_k(r, rel, k) for r, rel in results) / n,
        f"precision@{k}": sum(precision_at_k(r, rel, k) for r, rel in results) / n,
        f"hit@{k}": sum(hit_at_k(r, rel, k) for r, rel in results) / n,
        "mrr": mean_reciprocal_rank(results),
    }
