"""
embeddings.py
=============
Turns text into vectors. Default is a small, free, LOCAL model
(sentence-transformers) so the project costs nothing to run and needs no key.
You can switch to a hosted embedding API by setting EMBED_PROVIDER=openai.

The first call downloads the local model (~90 MB) once, then caches it.
"""

from __future__ import annotations

import os
import numpy as np


class Embedder:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.environ.get("EMBED_PROVIDER", "local")).lower()
        self.model_name = model or (
            "sentence-transformers/all-MiniLM-L6-v2" if self.provider == "local"
            else "text-embedding-3-small"
        )
        self._model = None  # lazy-loaded on first use

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings -> array of shape (len(texts), dim)."""
        if self.provider == "local":
            return self._encode_local(texts)
        if self.provider == "openai":
            return self._encode_openai(texts)
        raise ValueError(f"Unknown embedding provider: {self.provider}")

    def _encode_local(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return np.asarray(self._model.encode(texts, show_progress_bar=False))

    def _encode_openai(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model=self.model_name, input=texts)
        return np.asarray([d.embedding for d in resp.data])
