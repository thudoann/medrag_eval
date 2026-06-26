"""Central configuration. Override anything via environment variables."""
import os

# Which providers to use (see rag/llm.py and rag/embeddings.py).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "local")

# Retrieval defaults.
TOP_K = int(os.environ.get("TOP_K", "5"))
HYBRID_ALPHA = float(os.environ.get("HYBRID_ALPHA", "0.5"))   # dense vs bm25 weight
ABSTAIN_THRESHOLD = float(os.environ.get("ABSTAIN_THRESHOLD", "0.25"))
