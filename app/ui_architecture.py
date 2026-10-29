"""The Architecture tab: the two flows drawn as diagrams, and the component table."""

import streamlit as st

from rag_assistant.i18n import t


def architecture_diagram(lang: str, flow: str) -> str:
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


def render_architecture(lang: str) -> None:
    st.subheader(t(lang, "arch_title"))
    st.caption(t(lang, "arch_intro"))
    # column widths follow the natural widths of the two drawings (about 1 : 2.2), so both
    # are shown at the same scale and the text has the same size in each
    left, right = st.columns([1, 2.2])
    with left:
        st.markdown(f"**{t(lang, 'cluster_ingest')}**")
        st.graphviz_chart(architecture_diagram(lang, "ingest"), use_container_width=True)
    with right:
        st.markdown(f"**{t(lang, 'cluster_query')}**")
        st.graphviz_chart(architecture_diagram(lang, "query"), use_container_width=True)
    st.markdown(t(lang, "arch_md"))
