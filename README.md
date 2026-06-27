# MedRAG-Eval

A retrieval-augmented generation (RAG) system over biomedical literature, built
with a focus on evaluation and guardrails rather than just a chat demo.

Most "chat with your medical PDFs" projects stop at the demo. This project
measures retrieval quality, answer faithfulness, and safe refusal instead of
assuming they work.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your LLM key

pytest -q                     # run the tests (no key needed)

# compare retrieval modes on the bundled gold set
python -m evaluation.run_eval --gold data/gold_qa.example.json

# add answer faithfulness (needs an LLM key)
python -m evaluation.run_eval --gold data/gold_qa.example.json --judge

# end-to-end demo
python -m examples.demo

# optional: build a real corpus from PubMed
python -m ingestion.fetch_pubmed --query "metformin type 2 diabetes" --max 50
```

## Features

- **Hybrid retrieval, measured.** Dense embeddings + from-scratch BM25,
  combined. The eval harness compares all three modes to show which wins —
  exact drug names and dosages are where keyword search beats semantic search.
- **A real evaluation harness.** Retrieval metrics (recall@k, precision@k,
  MRR) plus answer-level faithfulness: deterministic citation coverage and an
  LLM-as-judge groundedness score that catches "cited but wrong".
- **Guardrails.** The system abstains when retrieval is weak (calibrated "I
  don't know"), forces inline `[n]` citations, validates them against the
  real sources, and flags obvious PHI.
- **Provider-agnostic.** One env var switches between Anthropic / OpenAI /
  local Ollama, and local-or-hosted embeddings. Nothing is locked to a vendor.

## Result

Output of `python -m evaluation.run_eval --gold data/gold_qa.example.json` on
the bundled sample (4 questions, 5 chunks):

```
mode       recall@5   precision@5   hit@5    mrr
bm25         1.000        0.250      1.000   1.000
dense        1.000        0.250      1.000   1.000
hybrid       1.000        0.250      1.000   1.000
```

The bundled gold set is too small to separate the three modes — it exists to
prove the harness runs end-to-end. A larger, harder gold set (see Roadmap) is
needed before the comparison says anything real about which mode wins.

## Structure

```
ingestion/fetch_pubmed.py   # pull abstracts from PubMed (free NCBI API)
rag/
  chunking.py               # sentence vs fixed-window chunking (so you can compare)
  bm25.py                   # Okapi BM25, from scratch
  embeddings.py             # local (free) or hosted embeddings
  vector_store.py           # numpy cosine store (FAISS-swappable)
  retriever.py              # dense | bm25 | hybrid
  llm.py                    # provider-agnostic LLM wrapper
  pipeline.py               # retrieve → guard → generate w/ citations → guard
guardrails/guards.py        # abstain, citation validation, PHI flagging
evaluation/
  metrics.py                # recall@k, precision@k, MRR  (unit-tested)
  faithfulness.py           # citation coverage + LLM-as-judge groundedness
  run_eval.py               # the before/after comparison harness
docs/architecture.md        # diagram and design rationale
tests/                      # pure-Python pieces are tested
```

The deterministic parts (metrics, BM25, vector store, guards) have unit
tests — `pytest -q` runs them with no API key.

## Lessons learned

- Hybrid retrieval's edge comes almost entirely from exact-token queries; on
  paraphrased questions, dense alone is already strong.
- Citation coverage is cheap and catches a lot, but "cited but wrong" needs
  the LLM judge — the two metrics are complementary, not redundant.
- A good abstain threshold is a real tuning problem: too eager and it's
  useless, too lax and it hallucinates.

## Roadmap

- [ ] Add a cross-encoder reranker between retrieval and generation
- [ ] Reciprocal-rank fusion as a second hybrid strategy
- [ ] Swap the numpy store for FAISS; incremental indexing
- [ ] Bigger, expert-reviewed gold set
- [ ] Proper de-identification (Presidio / scispaCy) instead of the regex stub

## Safety note

This is a research/portfolio project over public literature. It is **not** a
medical device and must not be used for clinical decisions.

## License

MIT — see [LICENSE](LICENSE).
