"""The left sidebar: language, search method, answer engine, parameters and the upload."""

import os
from dataclasses import dataclass
from typing import Any

import streamlit as st

from rag_assistant.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_TOP_K,
    MIN_SCORE_BY_SEARCH,
    get_chunk_settings,
)
from rag_assistant.i18n import DEFAULT_LANGUAGE, LANGUAGES, t
from rag_assistant.pdf_loader import (
    PdfCorruptedError,
    PdfError,
    PdfNoTextError,
    PdfNotFoundError,
)
from ui_privacy import IDLE_MINUTES, discard_upload_buffers
from ui_state import Ctx


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


def flash(kind: str, text: str) -> None:
    """Queue a message that survives the st.rerun() that follows an upload."""
    st.session_state.setdefault("flash", []).append((kind, text))


@dataclass
class Sidebar:
    """What the user chose in the sidebar."""

    lang: str
    search: str
    provider_name: str
    api_key: str | None
    model: str | None  # only chosen for the Groq engine; None means "the engine's default"
    uploads: list | None
    clear_slot: Any  # placeholder for the "delete all" button, filled once uploads are handled


def render_sidebar() -> Sidebar:
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
            ["extractive", "groq", "anthropic"],
            format_func=lambda x: t(lang, f"engine_{x}"),
        )
        api_key = None
        model = None
        if provider_name == "groq":
            api_key = st.text_input(
                t(lang, "api_key_groq"), type="password", help=t(lang, "api_key_groq_help")
            )
            model = st.text_input(
                t(lang, "model_groq"),
                value=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
                help=t(lang, "model_groq_help"),
            ).strip()
            st.warning(t(lang, "groq_privacy"))
        elif provider_name == "anthropic":
            api_key = st.text_input(
                t(lang, "api_key"), type="password", help=t(lang, "api_key_help")
            )
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
    return Sidebar(lang, search, provider_name, api_key, model, uploads, clear_slot)


def add_file(ctx: Ctx, name: str, data: bytes) -> None:
    lang, pipeline, files = ctx.lang, ctx.pipeline, ctx.files
    try:
        with st.spinner(t(lang, "indexing", name=name)):
            n = pipeline.add_pdf_bytes(data, name)
        files[name] = data
        skipped = pipeline.skipped_pages.get(name, [])
        if skipped:
            pages = ", ".join(str(p) for p in skipped)
            flash("success", t(lang, "indexed_skipped", name=name, n=n, pages=pages))
        else:
            flash("success", t(lang, "indexed", name=name, n=n))
    except PdfError as exc:
        flash("error", pdf_error_message(lang, exc, name))


def process_uploads(ctx: Ctx, uploads: list | None) -> None:
    """Index the newly uploaded PDFs, then drop Streamlit's own copy and reset the widget."""
    if not ctx.pipeline:
        return
    consumed = False
    for upload in uploads or []:
        data = upload.getvalue()
        consumed = True
        if ctx.files.get(upload.name) != data:
            add_file(ctx, upload.name, data)
    if consumed:
        # we hold our own copy now: drop Streamlit's, reset the widget, show the result
        discard_upload_buffers()
        st.session_state.upload_gen = st.session_state.get("upload_gen", 0) + 1
        st.rerun()
