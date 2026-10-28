"""Streamlit UI (Italian / English): upload PDFs, ask questions, inspect the evidence."""

import json
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
from rag_assistant.generation import has_critical_values
from rag_assistant.i18n import (
    DEFAULT_LANGUAGE,
    EXAMPLE_QUESTIONS,
    LANGUAGES,
    PROBLEM_KEYS,
    t,
)
from rag_assistant.llm import get_provider
from rag_assistant.pdf_loader import (
    PdfCorruptedError,
    PdfError,
    PdfNoTextError,
    PdfNotFoundError,
)
from rag_assistant.pipeline import RagPipeline
from rag_assistant.preview import render_cited_page, select_sentences

SAMPLE_DIR = ROOT / "data" / "sample_pdfs"
RESULTS_PATH = ROOT / "eval" / "results.json"
MAX_PREVIEWS = 3

st.set_page_config(page_title="Industrial RAG Assistant", page_icon="🏭", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~120 MB)...")
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


def pdf_error_message(lang: str, exc: PdfError, name: str) -> str:
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
        st.sidebar.error(pdf_error_message(lang, exc, name))


if pipeline:
    for upload in uploads or []:
        data = upload.getvalue()
        if files.get(upload.name) != data:
            add_file(upload.name, data)
    if use_samples:
        for pdf in sorted(SAMPLE_DIR.glob("*.pdf")):
            if pdf.name not in files:
                add_file(pdf.name, pdf.read_bytes())


# ---------------------------------------------------------------- tabs
def set_question(text: str) -> None:
    st.session_state.question = text


def render_assistant() -> None:
    if pipeline is None:
        st.info(t(lang, "need_key"))
        return
    if not files:
        st.info(t(lang, "start_hint"))
        return

    with st.expander(t(lang, "indexed_docs", n=len(files), chunks=pipeline.chunk_count)):
        for name in list(files):
            col1, col2 = st.columns([6, 1])
            col1.write(f"📄 {name}")
            if col2.button(t(lang, "remove"), key=f"rm_{name}"):
                pipeline.remove(name)
                del files[name]
                st.rerun()

    st.caption(t(lang, "examples_title"))
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, (key, example) in zip(columns, EXAMPLE_QUESTIONS.items(), strict=True):
        column.button(
            t(lang, key),
            key=f"btn_{key}",
            on_click=set_question,
            args=(example,),
            use_container_width=True,
        )

    question = st.text_input(
        t(lang, "question"), placeholder=t(lang, "placeholder"), key="question"
    )
    if not question:
        return

    pipeline.top_k = st.session_state.get("top_k", DEFAULT_TOP_K)
    pipeline.min_score = st.session_state.get("min_score", DEFAULT_MIN_SCORE)
    try:
        with st.spinner(t(lang, "searching")):
            answer = pipeline.ask(question)
    except Exception as exc:  # network/API errors must not crash the page
        st.error(t(lang, "answer_error", error=exc))
        return

    if answer.status == "answered":
        st.success(answer.text)
        if has_critical_values(answer.text):
            st.warning(t(lang, "verify_values"), icon="⚠️")
    elif answer.status == "insufficient":
        st.warning(t(lang, "insufficient"))
    else:
        st.warning(f"{t(lang, 'uncited')}\n\n{answer.text}")

    if answer.citations:
        st.subheader(t(lang, "sources"))
        for c in answer.citations:
            st.markdown("- " + t(lang, "source_line", doc=c.doc_name, page=c.page_number))
        render_previews(answer)
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


def render_previews(answer) -> None:
    """Show the cited PDF pages with the used passage highlighted."""
    st.subheader(t(lang, "preview_title"))
    for position, c in enumerate(answer.citations[:MAX_PREVIEWS]):
        evidence = answer.evidence[int(c.source_id[1:]) - 1]
        pdf_bytes = files.get(c.doc_name)
        if pdf_bytes is None:
            continue
        try:
            used_text = select_sentences(evidence.chunk.text, answer.text)
            png, highlighted = render_cited_page(pdf_bytes, c.page_number, used_text)
        except Exception as exc:
            st.error(t(lang, "preview_error", error=exc))
            continue
        caption_key = "preview_caption" if highlighted else "preview_no_highlight"
        with st.expander(c.label, expanded=position == 0):
            st.image(png, caption=t(lang, caption_key, doc=c.doc_name, page=c.page_number))


def architecture_diagram() -> str:
    """Two-flow diagram: each node shows the step and the module that implements it."""

    def node(name: str, key: str, module: str = "", fill: str = "") -> str:
        text = t(lang, key).replace('"', "'")
        if module:
            text += "\\n" + module
        style = f', fillcolor="{fill}"' if fill else ""
        return f'{name} [label="{text}"{style}];'

    def cluster(name: str, title_key: str, body: list[str]) -> list[str]:
        head = [
            f"subgraph cluster_{name} {{",
            f'label="{t(lang, title_key)}"; fontcolor="#9ca3af"; color="#4b5563";',
            'fontname="Helvetica";',
        ]
        return head + body + ["}"]

    ingest = [
        node("upload", "flow_upload", "app"),
        node("pages", "flow_pages", "pdf_loader.py"),
        node("chunks", "flow_chunks", "chunking.py"),
        node("vectors", "flow_vectors", "embeddings.py"),
        node("index", "flow_index", "index.py", "#1e3a8a"),
        "upload -> pages -> chunks -> vectors -> index;",
    ]
    query = [
        node("question", "flow_question", "app"),
        node("embed_q", "flow_embed_q", "embeddings.py"),
        node("search", "flow_search", "retrieval.py"),
        node("filter", "flow_filter", "generation.py"),
        node("prompt", "flow_prompt", "prompts.py"),
        node("engine", "flow_engine", "llm.py"),
        node("verify", "flow_verify", "generation.py"),
        node("answer", "flow_answer", "app", "#166534"),
        node("no", "flow_no", "", "#92400e"),
        "question -> embed_q -> search -> filter;",
        'filter -> prompt [label="✓"]; filter -> no [label="✗"];',
        "prompt -> engine -> verify -> answer;",
    ]
    lines = [
        "digraph {",
        'rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;',
        'node [shape=box, style="rounded,filled", fillcolor="#374151", fontcolor="white",',
        '      color="#6b7280", fontname="Helvetica", fontsize=11];',
        'edge [color="#9ca3af", fontcolor="#9ca3af", fontname="Helvetica"];',
        *cluster("ingest", "cluster_ingest", ingest),
        *cluster("query", "cluster_query", query),
        'index -> search [style=dashed, label="search"];',
        "}",
    ]
    return "\n".join(lines)


def render_about() -> None:
    st.markdown(t(lang, "about_md"))


def render_architecture() -> None:
    st.subheader(t(lang, "arch_title"))
    st.caption(t(lang, "arch_intro"))
    st.graphviz_chart(architecture_diagram(), use_container_width=True)
    st.markdown(t(lang, "arch_md"))


def render_evaluation() -> None:
    st.subheader(t(lang, "eval_title"))
    if not RESULTS_PATH.exists():
        st.info(t(lang, "eval_missing"))
        return
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    summary, results = data["summaries"][-1], data["results"]
    cfg = summary["config"]
    st.caption(
        t(
            lang,
            "eval_intro",
            n=summary["n_questions"],
            n_ans=summary["n_answerable"],
            n_unans=summary["n_unanswerable"],
            size=cfg["chunk_size"],
            overlap=cfg["chunk_overlap"],
            k=cfg["top_k"],
            min_score=cfg["min_score"],
        )
    )
    metrics = [
        ("m_hit1", "retrieval_hit_at_1"),
        ("m_hit3", "retrieval_hit_at_3"),
        ("m_mrr", "retrieval_mrr"),
        ("m_correct", "answer_correct"),
        ("m_cit_hit", "citation_hit"),
        ("m_cit_prec", "citation_precision"),
        ("m_abst", "abstention_correct"),
        ("m_false_abst", "false_abstention"),
    ]
    columns = st.columns(4)
    for i, (label_key, field) in enumerate(metrics):
        value = summary.get(field)
        columns[i % 4].metric(t(lang, label_key), "-" if value is None else f"{value:.0%}")
    st.warning(t(lang, "eval_limits_md"))

    st.subheader(t(lang, "eval_problems"))
    rows = [
        {
            t(lang, "col_id"): r["id"],
            t(lang, "col_category"): r["category"],
            t(lang, "col_question"): r["question"],
            t(lang, "col_problem"): (t(lang, PROBLEM_KEYS[p]) if p in PROBLEM_KEYS else p),
        }
        for r in results
        for p in r["problems"]
    ]
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.success(t(lang, "eval_no_problems"))


# ---------------------------------------------------------------- main
st.title(f"🏭 {t(lang, 'title')}")
st.caption(t(lang, "subtitle"))

tab_assistant, tab_about, tab_arch, tab_eval = st.tabs(
    [t(lang, "tab_assistant"), t(lang, "tab_about"), t(lang, "tab_arch"), t(lang, "tab_eval")]
)
with tab_assistant:
    render_assistant()
with tab_about:
    render_about()
with tab_arch:
    render_architecture()
with tab_eval:
    render_evaluation()
