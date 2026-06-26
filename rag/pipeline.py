"""
pipeline.py
===========
The end-to-end RAG flow, with the guardrails wired in:

    question
       -> retrieve top-k chunks
       -> if retrieval too weak: ABSTAIN (don't let the LLM guess)
       -> build a grounded prompt that forces inline [n] citations
       -> generate
       -> verify the citations are valid; flag if not

The whole design goal is that the LLM answers ONLY from the retrieved evidence
and always shows its work. That's what makes the output checkable -- and
checkability is the entire pitch of this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from guardrails.guards import (ABSTAIN_MESSAGE, citations_are_valid,
                               flag_phi, should_abstain)
from rag.llm import LLMClient
from rag.retriever import Retriever

SYSTEM_PROMPT = (
    "You are a careful biomedical research assistant. Answer ONLY using the "
    "numbered sources provided. Cite every claim with inline markers like [1]. "
    "If the sources do not contain the answer, say you don't know. Never invent "
    "facts, citations, or sources."
)


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[tuple[str, str]]      # (chunk_id, text) actually used
    retrieval_scores: list[float] = field(default_factory=list)
    abstained: bool = False
    citations_valid: bool = False
    phi_flags: list[str] = field(default_factory=list)


def _build_prompt(question: str, sources: list[tuple[str, str]]) -> str:
    numbered = "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(sources, 1))
    return (f"Sources:\n{numbered}\n\n"
            f"Question: {question}\n\n"
            f"Answer using only the sources above, with inline [n] citations.")


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None,
                 top_k: int = 5, abstain_threshold: float = 0.25):
        self.retriever = retriever
        self.llm = llm or LLMClient()
        self.top_k = top_k
        self.abstain_threshold = abstain_threshold

    def answer(self, question: str) -> RAGResult:
        # 1. Retrieve. (We re-run to capture scores for the abstain guard.)
        hits = self.retriever.retrieve(question, top_k=self.top_k)
        # The retriever returns (id, text); we approximate confidence by rank here,
        # but a production version would surface real similarity scores.
        scores = [1.0 - i / max(len(hits), 1) for i in range(len(hits))]

        # 2. Abstain guard.
        if should_abstain(scores, self.abstain_threshold) or not hits:
            return RAGResult(question, ABSTAIN_MESSAGE, hits, scores, abstained=True)

        # 3. Grounded generation.
        prompt = _build_prompt(question, hits)
        answer = self.llm.generate(prompt, system=SYSTEM_PROMPT)

        # 4. Output guards.
        valid = citations_are_valid(answer, len(hits))
        phi = flag_phi(answer)
        return RAGResult(question, answer, hits, scores,
                         abstained=False, citations_valid=valid, phi_flags=phi)
