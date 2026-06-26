"""
faithfulness.py
===============
Answer-level evaluation: is the generated answer actually SUPPORTED by the
retrieved evidence, or is it hallucinating? This is the crown jewel of the
project -- "how do you know it's not making things up?" is the first question
any healthcare team will ask.

Two complementary signals:
  1. citation_coverage  -- cheap, deterministic: what fraction of the answer's
     sentences carry a [n] citation? No model needed.
  2. groundedness (LLM-as-judge) -- an LLM rates whether each claim is entailed
     by the cited sources. More expensive but catches "cited but wrong".
"""

from __future__ import annotations

import json
import re

from rag.llm import LLMClient


def citation_coverage(answer: str) -> float:
    """Fraction of sentences that include at least one [n] citation marker."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
    return cited / len(sentences)


JUDGE_SYSTEM = (
    "You are a strict fact-checker. You will be given SOURCES and an ANSWER. "
    "Decide what fraction of the answer's factual claims are directly supported "
    "by the sources. Respond with ONLY a JSON object: "
    '{"groundedness": <float 0..1>, "unsupported_claims": [<strings>]}.'
)


def groundedness_judge(answer: str, sources: list[str],
                       llm: LLMClient | None = None) -> dict:
    """
    Use an LLM as a judge to score how well the answer is grounded in sources.
    Returns {"groundedness": float, "unsupported_claims": [...]}.
    Falls back gracefully if the judge doesn't return clean JSON.
    """
    llm = llm or LLMClient()
    numbered = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1))
    prompt = f"SOURCES:\n{numbered}\n\nANSWER:\n{answer}"
    raw = llm.generate(prompt, system=JUDGE_SYSTEM)

    # Be defensive: pull the first JSON object out of the response.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"groundedness": None, "unsupported_claims": [], "raw": raw}
    try:
        parsed = json.loads(match.group(0))
        return {
            "groundedness": float(parsed.get("groundedness", 0.0)),
            "unsupported_claims": parsed.get("unsupported_claims", []),
        }
    except (json.JSONDecodeError, ValueError):
        return {"groundedness": None, "unsupported_claims": [], "raw": raw}
