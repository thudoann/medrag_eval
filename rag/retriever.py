"""
retriever.py
============
The retriever decides which chunks the LLM gets to see. We support three modes
so the evaluation harness can prove which one actually works best on YOUR data:

  - "dense"  : semantic similarity only (embeddings)
  - "bm25"   : keyword matching only
  - "hybrid" : combine both scores (usually the winner)

Hybrid scoring uses a simple, robust technique: normalise each score list to
[0, 1] and take a weighted sum. (Reciprocal-rank fusion is a common
alternative -- noted as a follow-up in the README.)
"""

from __future__ import annotations

import numpy as np

from rag.bm25 import BM25
from rag.chunking import Chunk
from rag.embeddings import Embedder
from rag.vector_store import VectorStore


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None,
                 mode: str = "hybrid", alpha: float = 0.5):
        """
        chunks   : the corpus, already split into Chunk objects.
        mode     : "dense" | "bm25" | "hybrid".
        alpha    : hybrid weight on dense vs bm25 (1.0 = pure dense).
        """
        self.chunks = chunks
        self.mode = mode
        self.alpha = alpha
        self.embedder = embedder or Embedder()
        self.id_to_text = {c.chunk_id: c.text for c in chunks}

        # Build BM25 index over chunk texts.
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._bm25 = BM25([c.text for c in chunks])

        # Build the dense index (skipped for pure-bm25 mode to avoid the model load).
        self._store = None
        if mode in ("dense", "hybrid"):
            vecs = self.embedder.encode([c.text for c in chunks])
            self._store = VectorStore()
            self._store.add(self._chunk_ids, vecs)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        """Return the top_k chunks as (chunk_id, text), best first."""
        dense_scores: dict[str, float] = {}
        if self._store is not None:
            qv = self.embedder.encode([query])[0]
            dense_scores = dict(self._store.search(qv, top_k=top_k * 4))

        bm25_scores: dict[str, float] = {}
        if self.mode in ("bm25", "hybrid"):
            for idx, score in self._bm25.rank(query, top_k=top_k * 4):
                bm25_scores[self._chunk_ids[idx]] = score

        if self.mode == "dense":
            ranked = sorted(dense_scores.items(), key=lambda x: -x[1])
        elif self.mode == "bm25":
            ranked = sorted(bm25_scores.items(), key=lambda x: -x[1])
        else:  # hybrid: normalise then weighted-sum the two score lists
            d, b = _minmax(dense_scores), _minmax(bm25_scores)
            keys = set(d) | set(b)
            combined = {k: self.alpha * d.get(k, 0.0) + (1 - self.alpha) * b.get(k, 0.0)
                        for k in keys}
            ranked = sorted(combined.items(), key=lambda x: -x[1])

        return [(cid, self.id_to_text[cid]) for cid, _ in ranked[:top_k]]
