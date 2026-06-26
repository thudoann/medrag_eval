"""Tests for guardrails + citation coverage (pure-Python, no key needed)."""
from guardrails.guards import (should_abstain, citations_are_valid,
                               extract_citations, flag_phi)
from evaluation.faithfulness import citation_coverage


def test_abstain_on_weak_retrieval():
    assert should_abstain([], 0.25) is True
    assert should_abstain([0.1, 0.05], 0.25) is True
    assert should_abstain([0.9, 0.2], 0.25) is False

def test_citation_validity():
    assert citations_are_valid("Metformin is first-line [1].", n_sources=3) is True
    assert citations_are_valid("No citation here.", n_sources=3) is False
    assert citations_are_valid("Invented source [9].", n_sources=3) is False

def test_extract_citations():
    assert extract_citations("a [1] b [3] c [1]") == [1, 3, 1]

def test_phi_flagging():
    assert "ssn" in flag_phi("patient SSN 123-45-6789")
    assert flag_phi("no identifiers here") == []

def test_citation_coverage():
    ans = "Metformin is first-line [1]. It is weight-neutral [1]. It is common."
    cov = citation_coverage(ans)          # 2 of 3 sentences cited
    assert abs(cov - 2/3) < 1e-9
