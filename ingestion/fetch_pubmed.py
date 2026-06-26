"""
fetch_pubmed.py
===============
Pull abstracts from PubMed using NCBI's free E-utilities API. No key needed for
light use (an optional NCBI_API_KEY raises your rate limit).

Two-step dance:
  1. esearch -> get a list of PubMed IDs (PMIDs) matching a query
  2. efetch  -> get the abstract text for those PMIDs

Run from the repo root:
    python -m ingestion.fetch_pubmed --query "metformin type 2 diabetes" --max 50
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OUT_DIR = Path("data")


def _params(extra: dict) -> dict:
    base = {"db": "pubmed", "retmode": "xml"}
    if os.environ.get("NCBI_API_KEY"):
        base["api_key"] = os.environ["NCBI_API_KEY"]
    return {**base, **extra}


def search_pmids(query: str, max_results: int) -> list[str]:
    r = requests.get(ESEARCH, params=_params(
        {"term": query, "retmax": max_results, "retmode": "json"}), timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    r = requests.get(EFETCH, params=_params(
        {"id": ",".join(pmids), "rettype": "abstract"}), timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    docs = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = article.findtext(".//ArticleTitle") or ""
        # Abstracts can have several labelled sections; join them.
        parts = [el.text or "" for el in article.findall(".//AbstractText")]
        abstract = " ".join(p for p in parts if p).strip()
        if abstract:
            docs.append({"doc_id": f"PMID{pmid}", "title": title,
                         "text": f"{title}. {abstract}"})
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--max", type=int, default=50)
    args = parser.parse_args()

    print(f"Searching PubMed for: {args.query!r}")
    pmids = search_pmids(args.query, args.max)
    print(f"  found {len(pmids)} PMIDs; fetching abstracts...")
    time.sleep(0.4)  # be polite to the free API
    docs = fetch_abstracts(pmids)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "corpus.json"
    out.write_text(json.dumps({"documents": docs}, indent=2), encoding="utf-8")
    print(f"Saved {len(docs)} abstracts to {out}")


if __name__ == "__main__":
    main()
