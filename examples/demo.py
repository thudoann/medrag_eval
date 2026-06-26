"""
demo.py -- a minimal end-to-end run on the bundled sample corpus.
Needs: local embedding model (auto-downloads) + an LLM key in .env.

    python -m examples.demo
"""
import json
from pathlib import Path

from rag.chunking import sentence_chunks
from rag.retriever import Retriever
from rag.pipeline import RAGPipeline


def main():
    data = json.loads(Path("data/gold_qa.example.json").read_text())
    chunks = []
    for doc in data["documents"]:
        chunks.extend(sentence_chunks(doc["doc_id"], doc["text"], max_sentences=3))

    retriever = Retriever(chunks, mode="hybrid")
    pipe = RAGPipeline(retriever, top_k=3)

    res = pipe.answer("What is the first-line treatment for type 2 diabetes?")
    print("Q:", res.question)
    print("A:", res.answer)
    print("abstained:", res.abstained, "| citations valid:", res.citations_valid)
    print("sources used:", [cid for cid, _ in res.sources])


if __name__ == "__main__":
    main()
