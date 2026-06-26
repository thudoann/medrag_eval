"""
guards.py
=========
Safety guardrails. In healthcare these aren't optional polish -- they're the
difference between a demo and something you'd let near a clinician. Three guards:

  1. ABSTAIN: if retrieval is weak, the system should say "I don't have enough
     evidence" instead of guessing. Calibrated refusal > confident hallucination.
  2. CITATIONS: every answer must cite its sources; we verify the citations
     actually point at retrieved chunks (no invented [4] when only 3 were given).
  3. PHI AWARENESS: a hook to flag/redact obvious personal identifiers, a nod to
     the de-identification work real clinical NLP requires.
"""

from __future__ import annotations

import re

ABSTAIN_MESSAGE = ("I don't have enough reliable evidence in the retrieved "
                   "sources to answer this safely.")


def should_abstain(retrieval_scores: list[float], threshold: float = 0.25) -> bool:
    """True if even the best retrieved chunk is too weak to trust."""
    if not retrieval_scores:
        return True
    return max(retrieval_scores) < threshold


def extract_citations(answer: str) -> list[int]:
    """Pull out citation markers like [1], [2] from an answer."""
    return [int(n) for n in re.findall(r"\[(\d+)\]", answer)]


def citations_are_valid(answer: str, n_sources: int) -> bool:
    """Every [n] in the answer must reference a real retrieved source (1..n)."""
    cited = extract_citations(answer)
    if not cited:
        return False  # we REQUIRE citations; an uncited answer fails the guard
    return all(1 <= c <= n_sources for c in cited)


# A deliberately small PHI flagger -- real systems use Presidio / scispaCy.
_PHI_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\d{3}[.\-]\d{3}[.\-]\d{4}\b"),
    "mrn": re.compile(r"\bMRN[:#]?\s*\d+\b", re.IGNORECASE),
}


def flag_phi(text: str) -> list[str]:
    """Return the kinds of likely PHI found in the text (empty list = clean)."""
    return [kind for kind, pat in _PHI_PATTERNS.items() if pat.search(text)]
