"""Streamlit UI: upload PDFs, ask questions, inspect answer, citations and evidence."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    import rag_assistant  # noqa: F401
except ImportError:  # running without `pip install -e .` (e.g. Streamlit Cloud)
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from rag_assistant.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, get_chunk_settings
from rag_assistant.embeddings import FastEmbedEmbedder
from rag_assistant.llm import get_provider
from rag_assistant.pdf_loader import PdfError
from rag_assistant.pipeline import RagPipeline

SAMPLE_DIR = ROOT / "data" / "sample_pdfs"

st.set_page_config(page_title="Industrial RAG Assistant", page_icon="🏭", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~220 MB)...")
def get_embedder() -> FastEmbedEmbedder:
    embedder = FastEmbedEmbedder()
    embedder.embed(["warm up"])  # load the model once, here, with a spinner
    return embedder


def new_pipeline(provider_name: str, api_key: str | None) -> RagPipeline:
    size, overlap = get_chunk_settings()
    return RagPipeline(
        get_embedder(),
        get_provider(provider_name, api_key),
        chunk_size=st.session_state.get("chunk_size", size),
        chunk_overlap=st.session_state.get("chunk_overlap", overlap),
        top_k=st.session_state.get("top_k", DEFAULT_TOP_K),
        min_score=st.session_state.get("min_score", DEFAULT_MIN_SCORE),
    )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    provider_name = st.radio(
        "Answer engine",
        ["extractive", "anthropic"],
        format_func=lambda x: (
            "Offline (quotes the documents, no API key)"
            if x == "extractive"
            else "Claude (API key required)"
        ),
    )
    api_key = None
    if provider_name == "anthropic":
        api_key = st.text_input("Anthropic API key", type="password", help="Kept in memory only.")

    with st.expander("Retrieval parameters"):
        size, overlap = get_chunk_settings()
        st.number_input("Chunk size (chars)", 100, 4000, size, 50, key="chunk_size")
        st.number_input("Chunk overlap (chars)", 0, 1000, overlap, 10, key="chunk_overlap")
        st.number_input("Passages retrieved (top-k)", 1, 20, DEFAULT_TOP_K, key="top_k")
        st.slider("Minimum similarity", 0.0, 0.9, DEFAULT_MIN_SCORE, 0.05, key="min_score")
        st.caption("Changing chunk size/overlap applies to documents added afterwards.")

    st.divider()
    st.subheader("Documents")
    uploads = st.file_uploader("Upload PDF manuals", type="pdf", accept_multiple_files=True)
    use_samples = st.button("Load synthetic sample manuals")
    st.caption(
        "Files are processed in memory and are not stored. "
        "Do not upload confidential documents to a public demo."
    )

# ---------------------------------------------------------------- state
config_key = (provider_name, api_key)
if "pipeline" not in st.session_state or st.session_state.get("config_key") != config_key:
    previous = st.session_state.get("files", {})
    try:
        st.session_state.pipeline = new_pipeline(provider_name, api_key or None)
        st.session_state.config_key = config_key
        st.session_state.files = {}
        for name, data in previous.items():  # re-index when only the engine changed
            st.session_state.pipeline.add_pdf_bytes(data, name)
            st.session_state.files[name] = data
    except ValueError as exc:  # e.g. missing API key
        st.session_state.pipeline = None
        st.session_state.config_key = None
        st.sidebar.warning(str(exc))

pipeline: RagPipeline | None = st.session_state.get("pipeline")
files: dict[str, bytes] = st.session_state.setdefault("files", {})


def add_file(name: str, data: bytes) -> None:
    try:
        with st.spinner(f"Indexing {name}..."):
            n = pipeline.add_pdf_bytes(data, name)
        files[name] = data
        st.sidebar.success(f"{name}: {n} passages indexed")
    except PdfError as exc:
        st.sidebar.error(str(exc))


if pipeline:
    for upload in uploads or []:
        data = upload.getvalue()
        if files.get(upload.name) != data:
            add_file(upload.name, data)
    if use_samples:
        for pdf in sorted(SAMPLE_DIR.glob("*.pdf")):
            if pdf.name not in files:
                add_file(pdf.name, pdf.read_bytes())

# ---------------------------------------------------------------- main
st.title("🏭 Industrial RAG Assistant")
st.caption(
    "Ask questions about your technical manuals. Every answer cites document and page; "
    "if the documents do not contain the answer, the assistant says so."
)

if pipeline is None:
    st.info("Enter an API key in the sidebar, or switch to the offline answer engine.")
    st.stop()

if not files:
    st.info("Upload one or more PDFs in the sidebar, or load the sample manuals, to start.")
    st.stop()

with st.expander(f"Indexed documents ({len(files)}) - {pipeline.chunk_count} passages"):
    for name in list(files):
        col1, col2 = st.columns([6, 1])
        col1.write(f"📄 {name}")
        if col2.button("Remove", key=f"rm_{name}"):
            pipeline.remove(name)
            del files[name]
            st.rerun()

question = st.text_input(
    "Your question",
    placeholder="e.g. What is the tightening torque of the impeller nut?",
)

if question:
    pipeline.top_k = st.session_state.get("top_k", DEFAULT_TOP_K)
    pipeline.min_score = st.session_state.get("min_score", DEFAULT_MIN_SCORE)
    try:
        with st.spinner("Searching the documents..."):
            answer = pipeline.ask(question)
    except Exception as exc:  # network/API errors must not crash the page
        st.error(f"Could not get an answer: {exc}")
        st.stop()

    if answer.status == "answered":
        st.success(answer.text)
    elif answer.status == "insufficient":
        st.warning(answer.text)
    else:
        st.warning(
            "The model answered without valid citations, so the answer cannot be verified:\n\n"
            + answer.text
        )

    if answer.citations:
        st.subheader("Sources")
        for c in answer.citations:
            st.markdown(f"- **{c.doc_name}**, page {c.page_number}")
    if answer.invalid_citations:
        st.error(f"The model cited non-existent sources: {', '.join(answer.invalid_citations)}")
    if answer.flagged:
        st.warning(
            f"{len(answer.flagged)} passage(s) looked like instructions aimed at the AI "
            "(possible prompt injection) and were excluded from the answer."
        )

    st.subheader("Retrieved passages")
    st.caption("Check the evidence yourself. Score = cosine similarity (higher is closer).")

    def key(r):
        return (r.chunk.doc_name, r.chunk.page_number, r.chunk.chunk_index)

    used = {key(r) for r in answer.evidence}
    for r in pipeline.search(question):
        tag = "sent to answer engine" if key(r) in used else "not sent (low score/flagged)"
        where = f"{r.chunk.doc_name}, p. {r.chunk.page_number}"
        with st.expander(f"#{r.rank} · {where} · score {r.score:.2f} · {tag}"):
            st.write(r.chunk.text)
