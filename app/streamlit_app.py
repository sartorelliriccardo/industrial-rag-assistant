"""Streamlit UI (Italian / English): upload PDFs, ask questions, inspect the evidence."""

import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
try:
    import rag_assistant  # noqa: F401
except ImportError:  # running without `pip install -e .` (e.g. Streamlit Cloud)
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from streamlit import runtime as st_runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

from rag_assistant.config import (
    DEFAULT_TOP_K,
    MIN_SCORE_BY_SEARCH,
    get_chunk_settings,
)
from rag_assistant.embeddings import FastEmbedEmbedder, LexicalEmbedder
from rag_assistant.generation import has_critical_values
from rag_assistant.i18n import (
    DEFAULT_LANGUAGE,
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
from rag_assistant.session import check_interval_seconds, idle_expired, passwords_match
from rag_assistant.workspace import Workspace, get_store

EVAL_DIR = ROOT / "eval"
# uploaded documents are dropped after this many idle minutes (0 disables it)
IDLE_MINUTES = float(os.environ.get("DOC_IDLE_MINUTES", "15"))
MAX_PREVIEWS = 3

st.set_page_config(page_title="Industrial RAG Assistant", page_icon="🏭", layout="wide")


def configured_password() -> str:
    """Optional access password: APP_PASSWORD in the environment or in Streamlit secrets."""
    value = os.environ.get("APP_PASSWORD", "")
    if not value:
        try:
            value = str(st.secrets.get("APP_PASSWORD", ""))
        except Exception:  # no secrets file configured
            value = ""
    return value


def require_password() -> None:
    """Stop the script until the right password is typed. Does nothing if none is set."""
    expected = configured_password()
    if not expected or st.session_state.get("authenticated"):
        return
    typed = st.text_input("Password", type="password", key="app_password")
    if typed and passwords_match(expected, typed):
        st.session_state.authenticated = True
        st.rerun()
    if typed:
        st.error("Password errata / Wrong password")
    st.stop()


require_password()
st.session_state.last_active = time.time()  # every full run is a user action


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~120 MB)...")
def get_embedder() -> FastEmbedEmbedder:
    embedder = FastEmbedEmbedder()
    embedder.embed(["warm up"])  # load the model once, here, with a spinner
    return embedder


def min_score_for(search: str) -> float:
    return st.session_state.get(f"min_score_{search}", MIN_SCORE_BY_SEARCH[search])


def new_pipeline(search: str, provider_name: str, api_key: str | None) -> RagPipeline:
    size, overlap = get_chunk_settings()
    # keyword search needs no model (a few MB); semantic search loads ~400 MB of weights
    embedder = LexicalEmbedder() if search == "keyword" else get_embedder()
    return RagPipeline(
        embedder,
        get_provider(provider_name, api_key),
        chunk_size=st.session_state.get("chunk_size", size),
        chunk_overlap=st.session_state.get("chunk_overlap", overlap),
        top_k=st.session_state.get("top_k", DEFAULT_TOP_K),
        min_score=min_score_for(search),
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


def session_is_active(session_id: str) -> bool:
    return st_runtime.get_instance().is_active_session(session_id)


def current_session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is not None:
        return ctx.session_id
    return st.session_state.setdefault("fallback_session_id", uuid4().hex)


# One store for the whole process: documents live here, not in Streamlit's session state,
# so they are discarded a few seconds after the browser tab goes away (see workspace.py).
store = get_store(session_is_active)


def workspace() -> Workspace:
    return store.get_or_create(current_session_id())


def discard_upload_buffers() -> None:
    """Delete Streamlit's own server-side copy of the files uploaded in this session.

    Streamlit keeps every uploaded file in memory until the session is cleaned up, even
    after the widget is reset. We already hold the bytes we need in our workspace.
    """
    try:
        st_runtime.get_instance().uploaded_file_mgr.remove_session_files(current_session_id())
    except Exception:  # best effort: the session cleanup will still remove them eventually
        pass


def flash(kind: str, text: str) -> None:
    """Queue a message that survives the st.rerun() that follows an upload."""
    st.session_state.setdefault("flash", []).append((kind, text))


def clear_documents(language: str, message_key: str = "cleared") -> None:
    """Drop every uploaded document from memory and reset the upload widget."""
    ws = workspace()
    if ws.pipeline is not None:
        ws.pipeline.clear()
    ws.files.clear()
    discard_upload_buffers()
    st.session_state.question = ""
    st.session_state.upload_gen = st.session_state.get("upload_gen", 0) + 1
    st.toast(t(language, message_key, minutes=f"{IDLE_MINUTES:g}"))


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
    search = st.radio(
        t(lang, "search"),
        ["keyword", "semantic"],
        format_func=lambda x: t(lang, f"search_{x}"),
        key="search",
    )
    if search == "semantic":
        st.caption(t(lang, "search_semantic_note"))
    provider_name = st.radio(
        t(lang, "engine"),
        ["extractive", "anthropic"],
        format_func=lambda x: t(lang, f"engine_{x}"),
    )
    api_key = None
    if provider_name == "anthropic":
        api_key = st.text_input(t(lang, "api_key"), type="password", help=t(lang, "api_key_help"))
        st.warning(t(lang, "claude_privacy"))

    with st.expander(t(lang, "params")):
        size, overlap = get_chunk_settings()
        st.number_input(t(lang, "chunk_size"), 100, 4000, size, 50, key="chunk_size")
        st.number_input(t(lang, "chunk_overlap"), 0, 1000, overlap, 10, key="chunk_overlap")
        st.number_input(t(lang, "top_k"), 1, 20, DEFAULT_TOP_K, key="top_k")
        st.slider(
            t(lang, "min_score"),
            0.0,
            0.9,
            MIN_SCORE_BY_SEARCH[search],
            0.05,
            key=f"min_score_{search}",
        )
        st.caption(t(lang, "params_note"))

    st.divider()
    st.subheader(t(lang, "documents"))
    confirmed = st.checkbox(t(lang, "confirm_rights"), key="confirm_rights")
    uploads = st.file_uploader(
        t(lang, "upload"),
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.get('upload_gen', 0)}",
        disabled=not confirmed,
    )
    for kind, text in st.session_state.pop("flash", []):
        (st.success if kind == "success" else st.error)(text)
    clear_slot = st.empty()  # filled below, once uploads have been processed
    st.caption(t(lang, "privacy"))
    if IDLE_MINUTES > 0:
        st.caption(t(lang, "idle_note", minutes=f"{IDLE_MINUTES:g}"))

# ---------------------------------------------------------------- state
config_key = (search, provider_name, api_key)
ws = workspace()
if ws.pipeline is None or ws.config_key != config_key:
    previous = dict(ws.files)
    try:
        ws.pipeline = new_pipeline(search, provider_name, api_key or None)
        ws.config_key = config_key
        ws.files = {}
        for name, data in previous.items():  # re-index when only the engine changed
            ws.pipeline.add_pdf_bytes(data, name)
            ws.files[name] = data
    except ValueError:  # missing API key
        ws.pipeline = None
        ws.config_key = None

pipeline: RagPipeline | None = ws.pipeline
files: dict[str, bytes] = ws.files


def add_file(name: str, data: bytes) -> None:
    try:
        with st.spinner(t(lang, "indexing", name=name)):
            n = pipeline.add_pdf_bytes(data, name)
        files[name] = data
        flash("success", t(lang, "indexed", name=name, n=n))
    except PdfError as exc:
        flash("error", pdf_error_message(lang, exc, name))


if pipeline:
    consumed = False
    for upload in uploads or []:
        data = upload.getvalue()
        consumed = True
        if files.get(upload.name) != data:
            add_file(upload.name, data)
    if consumed:
        # we hold our own copy now: drop Streamlit's, reset the widget, show the result
        discard_upload_buffers()
        st.session_state.upload_gen = st.session_state.get("upload_gen", 0) + 1
        st.rerun()

clear_slot.button(t(lang, "clear_all"), on_click=clear_documents, args=(lang,), disabled=not files)


# ---------------------------------------------------------------- tabs
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

    question = st.text_input(
        t(lang, "question"), placeholder=t(lang, "placeholder"), key="question"
    )
    if not question:
        return

    pipeline.top_k = st.session_state.get("top_k", DEFAULT_TOP_K)
    pipeline.min_score = min_score_for(search)
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


def architecture_diagram(flow: str) -> str:
    """One flow as a top-to-bottom diagram: each box shows the step and the module behind it.

    Vertical layout keeps the drawing narrow, so it stays at a readable size instead of being
    shrunk to fit the page width.
    """

    def node(name: str, key: str, module: str = "", fill: str = "") -> str:
        text = t(lang, key).replace('"', "'")
        if module:
            text += "\\n" + module
        style = f', fillcolor="{fill}"' if fill else ""
        return f'{name} [label="{text}"{style}];'

    if flow == "ingest":
        body = [
            node("upload", "flow_upload", "app"),
            node("pages", "flow_pages", "pdf_loader.py"),
            node("chunks", "flow_chunks", "chunking.py"),
            node("vectors", "flow_vectors", "embeddings.py"),
            node("index", "flow_index", "index.py", "#1e3a8a"),
            "upload -> pages -> chunks -> vectors -> index;",
        ]
    else:
        body = [
            node("question", "flow_question", "app"),
            node("embed_q", "flow_embed_q", "embeddings.py"),
            node("index", "flow_index", "index.py", "#1e3a8a"),
            node("search", "flow_search", "retrieval.py"),
            node("filter", "flow_filter", "generation.py"),
            node("prompt", "flow_prompt", "prompts.py"),
            node("engine", "flow_engine", "llm.py"),
            node("verify", "flow_verify", "generation.py"),
            node("answer", "flow_answer", "app", "#166534"),
            node("no", "flow_no", "", "#92400e"),
            "{rank=same; embed_q; index}",
            "question -> embed_q -> search; index -> search;",
            'search -> filter; filter -> prompt [label=" ✓"]; filter -> no [label=" ✗"];',
            "prompt -> engine -> verify -> answer;",
        ]
    lines = [
        "digraph {",
        'rankdir=TB; bgcolor="transparent"; nodesep=0.4; ranksep=0.45;',
        'node [shape=box, style="rounded,filled", fillcolor="#374151", fontcolor="white",',
        '      color="#6b7280", fontname="Helvetica", fontsize=13, margin="0.2,0.12"];',
        'edge [color="#6b7280", fontcolor="#6b7280", fontname="Helvetica", fontsize=13];',
        *body,
        "}",
    ]
    return "\n".join(lines)


def render_about() -> None:
    st.markdown(t(lang, "about_md"))


def render_architecture() -> None:
    st.subheader(t(lang, "arch_title"))
    st.caption(t(lang, "arch_intro"))
    # column widths follow the natural widths of the two drawings (about 1 : 2.2), so both
    # are shown at the same scale and the text has the same size in each
    left, right = st.columns([1, 2.2])
    with left:
        st.markdown(f"**{t(lang, 'cluster_ingest')}**")
        st.graphviz_chart(architecture_diagram("ingest"), use_container_width=True)
    with right:
        st.markdown(f"**{t(lang, 'cluster_query')}**")
        st.graphviz_chart(architecture_diagram("query"), use_container_width=True)
    st.markdown(t(lang, "arch_md"))


def render_evaluation() -> None:
    st.subheader(t(lang, "eval_title"))
    results_path = EVAL_DIR / f"results_{search}.json"  # the file matching the chosen search
    if not results_path.exists():
        st.info(t(lang, "eval_missing"))
        return
    data = json.loads(results_path.read_text(encoding="utf-8"))
    summary, results = data["summaries"][-1], data["results"]
    cfg = summary["config"]
    st.caption(
        t(
            lang,
            "eval_intro",
            mode=t(lang, f"search_{search}"),
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


@st.fragment(run_every=check_interval_seconds(IDLE_MINUTES))
def idle_guard() -> None:
    """Drop the documents if nobody touched the app for IDLE_MINUTES.

    Runs by itself every few seconds while the page is open (a fragment re-run does not
    count as user activity). Once the page is closed nothing runs here: the session is then
    discarded by Streamlit.
    """
    if workspace().files and idle_expired(
        st.session_state.get("last_active", time.time()), time.time(), IDLE_MINUTES
    ):
        clear_documents(lang, "idle_cleared")
        st.rerun()


if IDLE_MINUTES > 0:
    idle_guard()
