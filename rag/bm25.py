"""
bm25.py
=======
BM25 keyword search, implemented from scratch (~40 lines of the classic
Okapi BM25 formula). Why bother when we also have semantic embeddings?

Dense embeddings are great at meaning but can miss exact terms -- drug names,
gene symbols, dosages ("BRCA1", "50mg"). BM25 nails those. Combining the two
("hybrid search") reliably beats either alone, and being able to say why is a
strong signal in an interview.

BM25 score of a document d for a query q:
    sum over query terms t of:
        IDF(t) * ( f(t,d) * (k1 + 1) ) / ( f(t,d) + k1 * (1 - b + b * |d|/avgdl) )
where f(t,d) is term frequency, |d| is doc length, avgdl the average length.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_tokenize(d) for d in corpus]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.docs)) if self.docs else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        n_docs = len(self.docs)
        df: Counter = Counter()
        for doc in self.docs:
            for term in set(doc):
                df[term] += 1
        # Smoothed IDF (the standard BM25 variant, always non-negative).
        return {
            term: math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def score(self, query: str, index: int) -> float:
        score = 0.0
        freqs = self.freqs[index]
        dl = self.doc_len[index]
        for term in _tokenize(query):
            if term not in freqs:
                continue
            idf = self.idf.get(term, 0.0)
            tf = freqs[term]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def rank(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return (doc_index, score) for the top_k documents, best first."""
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
