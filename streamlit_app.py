import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="MedRAG-Eval",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── data / index ──────────────────────────────────────────────────────────────

@st.cache_data
def load_corpus():
    return json.loads(Path("data/gold_qa.example.json").read_text())

@st.cache_resource(show_spinner="Building retrieval index…")
def build_retriever(mode: str):
    from rag.chunking import sentence_chunks
    from rag.retriever import Retriever
    data = load_corpus()
    chunks = []
    for doc in data["documents"]:
        chunks.extend(sentence_chunks(doc["doc_id"], doc["text"], max_sentences=3))
    return Retriever(chunks, mode=mode)

def resolve_api_key(manual: str) -> str | None:
    if manual:
        return manual
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("MedRAG-Eval")
    st.caption("Clinical RAG · hybrid retrieval · guardrails · evaluation")
    st.divider()

    mode = st.selectbox(
        "Retrieval mode",
        ["hybrid", "dense", "bm25"],
        help="hybrid = weighted BM25 + dense embeddings; strong on exact drug/dose queries",
    )
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=5, value=3)

    st.divider()
    st.subheader("LLM key")
    manual_key = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-… (optional)",
        label_visibility="visible",
    )
    api_key = resolve_api_key(manual_key)
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        st.success("Generation enabled", icon="✓")
    else:
        st.info("No key — retrieval only", icon="ℹ")

    st.divider()
    st.caption(
        "Corpus: 4 PubMed-style abstracts (bundled sample). "
        "Fetch a real corpus with `python -m ingestion.fetch_pubmed`."
    )

# ── header ────────────────────────────────────────────────────────────────────

st.title("MedRAG-Eval")
st.markdown(
    "Ask a biomedical question. The system retrieves relevant passages, "
    "guards against weak retrieval, then generates a citation-grounded answer — "
    "or abstains when it can't."
)
st.divider()

# ── question input ────────────────────────────────────────────────────────────

corpus = load_corpus()
example_qs = [q["question"] for q in corpus["questions"]]

col_pick, col_ask = st.columns([2, 3], gap="medium")
with col_pick:
    picked = st.selectbox(
        "Example questions",
        options=["— pick one —"] + example_qs,
        label_visibility="collapsed",
    )
with col_ask:
    question = st.text_input(
        "Your question",
        value="" if picked == "— pick one —" else picked,
        placeholder="e.g. What is the first-line treatment for type 2 diabetes?",
        label_visibility="collapsed",
    )

run = st.button("Ask", type="primary", disabled=not question.strip())

# ── results ───────────────────────────────────────────────────────────────────

if run and question.strip():
    retriever = build_retriever(mode)

    with st.spinner("Retrieving…"):
        hits = retriever.retrieve(question.strip(), top_k=top_k)

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader(f"Retrieved chunks · {mode} · k={top_k}")
        for i, (cid, text) in enumerate(hits, 1):
            with st.container(border=True):
                st.caption(f"[{i}] {cid}")
                st.write(text)

    with right:
        st.subheader("Answer")
        if api_key:
            from rag.pipeline import RAGPipeline

            pipe = RAGPipeline(retriever, top_k=top_k)
            with st.spinner("Generating…"):
                result = pipe.answer(question.strip())

            if result.abstained:
                st.warning(
                    "**Abstained** — retrieval confidence too low to answer safely.",
                    icon="⚠️",
                )
            else:
                st.markdown(result.answer)
                st.divider()
                m1, m2 = st.columns(2)
                m1.metric("Citations valid", "✓" if result.citations_valid else "✗")
                m2.metric("PHI flags", len(result.phi_flags) or "none")
                if result.phi_flags:
                    st.error(f"PHI detected: {', '.join(result.phi_flags)}")
        else:
            st.info(
                "Add an Anthropic API key in the sidebar to generate an answer. "
                "Retrieval results are on the left.",
                icon="ℹ️",
            )

# ── footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Research / portfolio project · public biomedical literature · "
    "**not** a medical device · "
    "[GitHub](https://github.com/thudoann/medrag_eval)"
)
