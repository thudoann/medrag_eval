"""
chunking.py
===========
Splitting documents into retrievable chunks. This sounds trivial but it's one
of the highest-leverage knobs in a RAG system: chunks that are too big dilute
the signal, too small lose context. We implement two strategies so the
evaluation harness can MEASURE which one retrieves better -- that comparison is
exactly the kind of evidence that separates an engineer from a tutorial.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str          # which source document this came from
    chunk_id: str        # unique id for this chunk
    text: str


def _split_sentences(text: str) -> list[str]:
    # A light sentence splitter. (Medical text is abbreviation-heavy; a real
    # system might use scispaCy, but this keeps the project dependency-light.)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def fixed_token_chunks(doc_id: str, text: str,
                       size: int = 120, overlap: int = 20) -> list[Chunk]:
    """
    Split into ~`size`-word windows with `overlap` words shared between
    neighbours (overlap stops a fact being cut in half at a boundary).
    """
    words = text.split()
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        window = words[i:i + size]
        if not window:
            continue
        chunks.append(Chunk(doc_id, f"{doc_id}::w{i}", " ".join(window)))
        if i + size >= len(words):
            break
    return chunks


def sentence_chunks(doc_id: str, text: str, max_sentences: int = 4) -> list[Chunk]:
    """Group whole sentences together (keeps ideas intact)."""
    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    for i in range(0, len(sentences), max_sentences):
        group = sentences[i:i + max_sentences]
        chunks.append(Chunk(doc_id, f"{doc_id}::s{i}", " ".join(group)))
    return chunks
