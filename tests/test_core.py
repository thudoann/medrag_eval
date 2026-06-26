"""Tests for the pure-Python core: metrics, BM25, vector store, chunking."""
import numpy as np
from evaluation.metrics import (recall_at_k, precision_at_k, reciprocal_rank,
                                 mean_reciprocal_rank, aggregate)
from rag.bm25 import BM25
from rag.vector_store import VectorStore
from rag.chunking import fixed_token_chunks, sentence_chunks


# ---- metrics ----
def test_recall_and_precision():
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = ["d2", "d4", "d9"]
    assert recall_at_k(retrieved, relevant, 4) == 2 / 3      # found 2 of 3
    assert precision_at_k(retrieved, relevant, 4) == 2 / 4   # 2 of top-4 right

def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == 1 / 3
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0

def test_mrr_and_aggregate():
    results = [(["a", "b"], ["a"]), (["x", "y"], ["y"])]
    assert mean_reciprocal_rank(results) == (1.0 + 0.5) / 2
    agg = aggregate(results, k=2)
    assert set(agg) == {"recall@2", "precision@2", "hit@2", "mrr"}


# ---- BM25: exact-term retrieval ----
def test_bm25_finds_keyword_doc():
    corpus = [
        "metformin is a first line treatment for type 2 diabetes",
        "aspirin reduces the risk of heart attack",
        "the patient was prescribed insulin for glucose control",
    ]
    bm25 = BM25(corpus)
    top = bm25.rank("metformin diabetes", top_k=1)
    assert top[0][0] == 0          # the metformin doc ranks first
    assert top[0][1] > 0


# ---- vector store: cosine similarity ----
def test_vector_store_finds_nearest():
    store = VectorStore()
    store.add(["a", "b", "c"], np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
    hits = store.search(np.array([1.0, 0.0]), top_k=1)
    assert hits[0][0] == "a"       # identical direction -> top hit
    assert abs(hits[0][1] - 1.0) < 1e-6


# ---- chunking ----
def test_fixed_chunks_overlap():
    text = " ".join(str(i) for i in range(300))
    chunks = fixed_token_chunks("doc1", text, size=120, overlap=20)
    assert all(c.doc_id == "doc1" for c in chunks)
    assert len(chunks) >= 2

def test_sentence_chunks():
    text = "First fact here. Second fact here. Third one. Fourth. Fifth."
    chunks = sentence_chunks("doc1", text, max_sentences=2)
    assert len(chunks) == 3        # 5 sentences in groups of 2 -> 3 chunks
