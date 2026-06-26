"""
run_eval.py
===========
The harness that produces the headline result of the whole project:

    "Naive dense retrieval scored X. My hybrid pipeline scored Y."

It loads a gold question set (questions + which chunk ids are relevant + an
ideal answer), runs each retrieval mode, and prints a comparison table of
retrieval metrics. If an LLM key is configured it also scores answer
faithfulness. This is the evidence you put in your README and talk through in
interviews.

Run from the repo root:  python -m evaluation.run_eval --gold data/gold_qa.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.faithfulness import citation_coverage, groundedness_judge
from evaluation.metrics import aggregate
from rag.chunking import sentence_chunks
from rag.embeddings import Embedder
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever


def load_corpus_and_gold(gold_path: Path):
    """
    Expects a JSON file with:
      {
        "documents": [{"doc_id": "...", "text": "..."}],
        "questions": [{"question": "...", "relevant_doc_ids": [...], "ideal_answer": "..."}]
      }
    """
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    chunks = []
    for doc in data["documents"]:
        chunks.extend(sentence_chunks(doc["doc_id"], doc["text"], max_sentences=3))
    return chunks, data["questions"]


def evaluate_mode(chunks, questions, embedder, mode: str, k: int = 5):
    """Run one retrieval mode over all questions and aggregate the metrics."""
    retriever = Retriever(chunks, embedder=embedder, mode=mode)
    results = []
    for q in questions:
        hits = retriever.retrieve(q["question"], top_k=k)
        # Map retrieved chunk ids back to their source doc ids for scoring.
        retrieved_docs = [cid.split("::")[0] for cid, _ in hits]
        results.append((retrieved_docs, q["relevant_doc_ids"]))
    return aggregate(results, k=k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="data/gold_qa.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--judge", action="store_true",
                        help="Also score answer faithfulness with an LLM (needs a key).")
    args = parser.parse_args()

    chunks, questions = load_corpus_and_gold(Path(args.gold))
    embedder = Embedder()

    print(f"Evaluating {len(questions)} questions over {len(chunks)} chunks\n")
    print(f"{'mode':<8} " + "  ".join(f"{m:>11}" for m in
          (f"recall@{args.k}", f"precision@{args.k}", f"hit@{args.k}", "mrr")))
    for mode in ("bm25", "dense", "hybrid"):
        m = evaluate_mode(chunks, questions, embedder, mode, k=args.k)
        print(f"{mode:<8} " + "  ".join(f"{m[key]:>11.3f}" for key in
              (f"recall@{args.k}", f"precision@{args.k}", f"hit@{args.k}", "mrr")))

    # Optional, key-gated: full answer faithfulness on the hybrid pipeline.
    if args.judge:
        print("\nAnswer faithfulness (hybrid pipeline, LLM-as-judge):")
        retriever = Retriever(chunks, embedder=embedder, mode="hybrid")
        pipe = RAGPipeline(retriever, top_k=args.k)
        for q in questions:
            res = pipe.answer(q["question"])
            cov = citation_coverage(res.answer)
            judged = groundedness_judge(res.answer, [t for _, t in res.sources])
            print(f"  - {q['question'][:50]:<52} "
                  f"coverage={cov:.2f}  groundedness={judged.get('groundedness')}")


if __name__ == "__main__":
    main()
