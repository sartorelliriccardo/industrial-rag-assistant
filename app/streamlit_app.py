"""Streamlit UI (Italian / English): upload PDFs, ask questions, inspect the evidence."""

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
from rag_assistant.i18n import DEFAULT_LANGUAGE, LANGUAGES, t
from rag_assistant.llm import get_provider
from rag_assistant.pdf_loader import (
    PdfCorruptedError,
    PdfError,
    PdfNoTextError,
    PdfNotFoundError,
)
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


def pdf_error_message(exc: PdfError, name: str) -> str:
    if isinstance(exc, PdfNoTextError):
        key = "err_no_text"
    elif isinstance(exc, PdfCorruptedError):
        key = "err_corrupted"
    elif isinstance(exc, PdfNotFoundError):
        key = "err_not_found"
    else:
        key = "err_pdf"
    return t(lang, key, name=name)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    lang = st.radio(
        "Lingua / Language",
        list(LANGUAGES),
        format_func=LANGUAGES.get,
        index=list(LANGUAGES).index(DEFAULT_LANGUAGE),
        horizontal=True,
        key="lang",
    )
    st.header(t(lang, "settings"))
    provider_name = st.radio(
        t(lang, "engine"),
        ["extractive", "anthropic"],
        format_func=lambda x: t(lang, f"engine_{x}"),
    )
    api_key = None
    if provider_name == "anthropic":
        api_key = st.text_input(t(lang, "api_key"), type="password", help=t(lang, "api_key_help"))

    with st.expander(t(lang, "params")):
        size, overlap = get_chunk_settings()
        st.number_input(t(lang, "chunk_size"), 100, 4000, size, 50, key="chunk_size")
        st.number_input(t(lang, "chunk_overlap"), 0, 1000, overlap, 10, key="chunk_overlap")
        st.number_input(t(lang, "top_k"), 1, 20, DEFAULT_TOP_K, key="top_k")
        st.slider(t(lang, "min_score"), 0.0, 0.9, DEFAULT_MIN_SCORE, 0.05, key="min_score")
        st.caption(t(lang, "params_note"))

    st.divider()
    st.subheader(t(lang, "documents"))
    uploads = st.file_uploader(
        t(lang, "upload"), type="pdf", accept_multiple_files=True, key="uploader"
    )
    use_samples = st.button(t(lang, "load_samples"))
    st.caption(t(lang, "privacy"))

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
    except ValueError:  # missing API key
        st.session_state.pipeline = None
        st.session_state.config_key = None

pipeline: RagPipeline | None = st.session_state.get("pipeline")
files: dict[str, bytes] = st.session_state.setdefault("files", {})


def add_file(name: str, data: bytes) -> None:
    try:
        with st.spinner(t(lang, "indexing", name=name)):
            n = pipeline.add_pdf_bytes(data, name)
        files[name] = data
        st.sidebar.success(t(lang, "indexed", name=name, n=n))
    except PdfError as exc:
        st.sidebar.error(pdf_error_message(exc, name))


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
st.title(f"🏭 {t(lang, 'title')}")
st.caption(t(lang, "subtitle"))

if pipeline is None:
    st.info(t(lang, "need_key"))
    st.stop()

if not files:
    st.info(t(lang, "start_hint"))
    st.stop()

with st.expander(t(lang, "indexed_docs", n=len(files), chunks=pipeline.chunk_count)):
    for name in list(files):
        col1, col2 = st.columns([6, 1])
        col1.write(f"📄 {name}")
        if col2.button(t(lang, "remove"), key=f"rm_{name}"):
            pipeline.remove(name)
            del files[name]
            st.rerun()

question = st.text_input(t(lang, "question"), placeholder=t(lang, "placeholder"), key="question")

if question:
    pipeline.top_k = st.session_state.get("top_k", DEFAULT_TOP_K)
    pipeline.min_score = st.session_state.get("min_score", DEFAULT_MIN_SCORE)
    try:
        with st.spinner(t(lang, "searching")):
            answer = pipeline.ask(question)
    except Exception as exc:  # network/API errors must not crash the page
        st.error(t(lang, "answer_error", error=exc))
        st.stop()

    if answer.status == "answered":
        st.success(answer.text)
    elif answer.status == "insufficient":
        st.warning(t(lang, "insufficient"))
    else:
        st.warning(f"{t(lang, 'uncited')}\n\n{answer.text}")

    if answer.citations:
        st.subheader(t(lang, "sources"))
        for c in answer.citations:
            st.markdown("- " + t(lang, "source_line", doc=c.doc_name, page=c.page_number))
    if answer.invalid_citations:
        st.error(t(lang, "invalid_citations", ids=", ".join(answer.invalid_citations)))
    if answer.flagged:
        st.warning(t(lang, "flagged", n=len(answer.flagged)))

    st.subheader(t(lang, "passages"))
    st.caption(t(lang, "passages_help"))

    def key(r):
        return (r.chunk.doc_name, r.chunk.page_number, r.chunk.chunk_index)

    used = {key(r) for r in answer.evidence}
    for r in pipeline.search(question):
        tag = t(lang, "tag_used" if key(r) in used else "tag_unused")
        title = t(
            lang,
            "passage_title",
            rank=r.rank,
            doc=r.chunk.doc_name,
            page=r.chunk.page_number,
            score=r.score,
            tag=tag,
        )
        with st.expander(title):
            st.write(r.chunk.text)
