# Architecture & design decisions

## The flow

```
                ┌─────────────┐
   PubMed  ───► │  ingestion  │  fetch_pubmed.py  → data/corpus.json
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │  chunking   │  sentence / fixed-window chunks
                └──────┬──────┘
            ┌──────────┴──────────┐
            ▼                     ▼
      ┌───────────┐        ┌───────────┐
      │  BM25     │        │ embeddings│  (local, free)
      │ (keyword) │        │  + vector │
      └─────┬─────┘        │   store   │
            │              └─────┬─────┘
            └────────┬───────────┘
                     ▼
              ┌─────────────┐
              │  retriever  │  dense | bm25 | HYBRID
              └──────┬──────┘
                     ▼
        ┌───────────────────────┐
        │  guards: abstain?     │  weak retrieval → refuse
        └──────────┬────────────┘
                   ▼
              ┌─────────────┐
              │  LLM (RAG)  │  grounded prompt, forced [n] citations
              └──────┬──────┘
                     ▼
        ┌───────────────────────┐
        │ guards: citations ok? │  evaluation: faithfulness + retrieval metrics
        │ PHI flags?            │
        └───────────────────────┘
```

## Why these choices

**ELT-style "retrieve raw, judge later."** The retriever returns evidence; the
guards and evaluation decide whether to trust it. Keeping retrieval, generation,
and judging as separate stages makes each one independently testable and
swappable.

**Hybrid retrieval (dense + BM25).** Dense embeddings capture meaning but miss
exact tokens (drug names, gene symbols, dosages); BM25 nails those. Combining
them is consistently stronger, and the eval harness *proves* it on your data
instead of asserting it.

**Provider-agnostic LLM + embeddings.** One env var switches vendors. No part of
the pipeline is married to a single API, so adapting to an employer's stack is a
config change, not a rewrite.

**Guardrails before and after generation.** Abstain when evidence is weak;
require valid citations after. In healthcare, a calibrated "I don't know" beats a
confident hallucination every time.

**A from-scratch vector store + BM25.** Kept simple (NumPy-only) so the maths is
visible and the project runs anywhere. The `VectorStore` interface is designed so
swapping in FAISS or a managed vector DB touches exactly one file.

## What I'd change for production
- Surface real similarity scores end-to-end (the abstain guard currently
  approximates confidence from rank).
- Swap the in-memory store for FAISS / a managed vector DB; add incremental
  indexing instead of rebuilding.
- Add a reranker (cross-encoder) between retrieval and generation.
- Expand the gold set and add reciprocal-rank fusion as a hybrid variant.
