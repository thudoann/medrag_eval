"""
vector_store.py
===============
A tiny in-memory vector store using cosine similarity. It's deliberately simple
and dependency-light (just NumPy) so the project runs anywhere and the maths is
visible. For a real corpus of millions of chunks you'd swap this for FAISS or a
managed vector DB -- the interface below is designed so that swap is a one-file
change. (That separation is itself a good engineering talking point.)
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self):
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Store chunk ids alongside their (L2-normalised) embedding vectors."""
        vectors = np.asarray(vectors, dtype=np.float32)
        # Normalise so that a dot product equals cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        self.ids.extend(ids)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Return (chunk_id, similarity) for the top_k closest chunks."""
        if self.vectors is None:
            return []
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        q = q / max(np.linalg.norm(q), 1e-12)
        sims = self.vectors @ q                     # cosine similarity to every chunk
        order = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in order]
