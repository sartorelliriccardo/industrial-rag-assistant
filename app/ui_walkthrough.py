"""The Step by step tab: one question followed through every stage of the pipeline."""

import streamlit as st

from rag_assistant.i18n import t
from rag_assistant.llm import ExtractiveProvider
from rag_assistant.walkthrough import explain
from ui_state import Ctx

MAX_CHUNK_ROWS = 60


def render_walkthrough(ctx: Ctx) -> None:
    """Follow one question through every stage, with the real data of each stage."""
    lang, pipeline, files, provider_name = ctx.lang, ctx.pipeline, ctx.files, ctx.provider_name
    st.subheader(t(lang, "walk_title"))
    st.caption(t(lang, "walk_intro"))
    if pipeline is None or not files:
        st.info(t(lang, "walk_need_docs"))
        return
    if provider_name != "extractive":
        st.info(t(lang, "walk_engine_note"))

    st.session_state.setdefault("walk_question", st.session_state.get("question", ""))
    question = st.text_input(t(lang, "walk_question"), key="walk_question")
    if not question.strip():
        st.info(t(lang, "walk_empty_q"))
        return
    ctx.apply_search_settings()
    try:
        w = explain(pipeline, question, ExtractiveProvider())
    except ValueError as exc:
        st.error(t(lang, "walk_error", error=exc))
        return

    # 1. the question
    with st.container(border=True):
        st.markdown(f"#### {t(lang, 's1_title')}")
        st.write(t(lang, "s1_text"))
        if w.terms:
            st.markdown(f"**{t(lang, 's1_terms')}:** " + ", ".join(f"`{x}`" for x in w.terms))
        else:
            st.warning(t(lang, "s1_none"))

    # 2. chunks
    with st.container(border=True):
        st.markdown(f"#### {t(lang, 's2_title')}")
        st.write(
            t(
                lang,
                "s2_text",
                size=w.chunk_size,
                overlap=w.chunk_overlap,
                n_chunks=w.n_chunks,
                n_docs=w.n_documents,
            )
        )
        shown = sorted(w.chunks, key=lambda c: not c.retrieved)[:MAX_CHUNK_ROWS]
        st.dataframe(
            [
                {
                    t(lang, "col_doc"): c.doc,
                    t(lang, "col_page"): c.page,
                    t(lang, "col_chunk"): c.index + 1,
                    t(lang, "col_chars"): c.chars,
                    t(lang, "col_start"): c.preview,
                    t(lang, "col_found"): "✓" if c.retrieved else "",
                }
                for c in shown
            ],
            hide_index=True,
            use_container_width=True,
        )
        if len(w.chunks) > len(shown):
            st.caption(t(lang, "s2_more", shown=len(shown), total=len(w.chunks)))

    # 3. vectors
    with st.container(border=True):
        st.markdown(f"#### {t(lang, 's3_title')}")
        if w.term_positions:
            st.write(t(lang, "s3_keyword", dim=w.vector_dim, nonzero=w.vector_nonzero))
            st.dataframe(
                [
                    {
                        t(lang, "col_word"): tp.term,
                        t(lang, "col_position"): tp.position,
                        t(lang, "col_weight"): round(tp.weight, 2),
                    }
                    for tp in w.term_positions
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write(t(lang, "s3_other", dim=w.vector_dim, nonzero=w.vector_nonzero))

    # 4. ranking
    with st.container(border=True):
        st.markdown(f"#### {t(lang, 's4_title')}")
        st.write(t(lang, "s4_text", k=len(w.ranking), min_score=f"{w.min_score:.2f}"))
        if w.ranking:
            score_label = t(lang, "col_score")
            st.dataframe(
                [
                    {
                        t(lang, "col_rank"): p.rank,
                        t(lang, "col_doc"): p.doc,
                        t(lang, "col_page"): p.page,
                        score_label: max(p.score, 0.0),
                        t(lang, "col_outcome"): t(lang, f"out_{p.outcome}"),
                    }
                    for p in w.ranking
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    score_label: st.column_config.ProgressColumn(
                        score_label, min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )
        else:
            st.info(t(lang, "s4_none"))

    # 5. injection filter
    with st.container(border=True):
        st.markdown(f"#### {t(lang, 's5_title')}")
        st.write(t(lang, "s5_text"))
        if w.removed_sentences:
            st.markdown(f"**{t(lang, 's5_removed')}**")
            for doc, page, sentence in w.removed_sentences:
                st.code(f"{doc}, p. {page}: {sentence}", language=None, wrap_lines=True)
        else:
            st.success(t(lang, "s5_nothing"))

    # 6. prompt
    with st.container(border=True):
        st.markdown(f"#### {t(lang, 's6_title')}")
        st.write(t(lang, "s6_text"))
        if w.answer.engine_called:
            st.markdown(f"**{t(lang, 's6_rules')}**")
            st.code(w.system_prompt, language=None, wrap_lines=True)
            st.markdown(f"**{t(lang, 's6_input', chars=len(w.user_prompt))}**")
            st.code(w.user_prompt, language="xml", wrap_lines=True)
        else:
            st.warning(t(lang, "s6_not_called"))

    # 7. engine
    with st.container(border=True):
        st.markdown(f"#### {t(lang, 's7_title')}")
        st.write(t(lang, "s7_text"))
        if not w.answer.engine_called:
            st.info(t(lang, "s7_skipped"))
        elif "INSUFFICIENT_EVIDENCE" in w.answer.raw:
            st.warning(t(lang, "s7_refused"))
        else:
            st.markdown(f"**{t(lang, 's7_raw')}**")
            st.code(w.answer.raw, language=None, wrap_lines=True)

    # 8. citation check
    with st.container(border=True):
        st.markdown(f"#### {t(lang, 's8_title')}")
        st.write(t(lang, "s8_text"))
        if w.answer.citations:
            st.dataframe(
                [
                    {
                        t(lang, "col_tag"): c.source_id,
                        t(lang, "col_doc"): c.doc_name,
                        t(lang, "col_page"): c.page_number,
                    }
                    for c in w.answer.citations
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info(t(lang, "s8_none"))
        if w.answer.invalid_citations:
            st.error(t(lang, "s8_invalid", ids=", ".join(w.answer.invalid_citations)))

    # 9. result
    with st.container(border=True):
        st.markdown(f"#### {t(lang, 's9_title')}")
        st.caption(t(lang, "s9_text"))
        if w.answer.status == "answered":
            st.success(w.answer.text)
        elif w.answer.status == "insufficient":
            st.warning(t(lang, "insufficient"))
        else:
            st.warning(f"{t(lang, 'uncited')}\n\n{w.answer.text}")
