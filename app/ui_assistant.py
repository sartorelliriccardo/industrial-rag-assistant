"""The Assistant tab: ask a question, see the answer, its sources and the retrieved passages."""

import streamlit as st

from rag_assistant.generation import has_critical_values
from rag_assistant.i18n import t
from rag_assistant.preview import render_cited_page, select_sentences
from ui_state import Ctx

MAX_PREVIEWS = 3


def render_assistant(ctx: Ctx) -> None:
    lang, pipeline, files = ctx.lang, ctx.pipeline, ctx.files
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

    ctx.apply_search_settings()
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
        render_previews(ctx, answer)
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


def render_previews(ctx: Ctx, answer) -> None:
    """Show the cited PDF pages with the used passage highlighted."""
    lang, files = ctx.lang, ctx.files
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
