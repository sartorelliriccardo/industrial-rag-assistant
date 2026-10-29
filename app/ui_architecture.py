"""The Architecture tab: keyword search, neural-network search, and both with Groq.

Each variant has its own diagram and its own explanation, followed by the component table.
"""

import textwrap

import streamlit as st

from rag_assistant.i18n import t

INDEX = "#1e3a8a"  # the search index
OK = "#166534"  # the answer
DECLINE = "#92400e"  # "not enough information"
OUTSIDE = "#6b21a8"  # what leaves the server

# A flow is (nodes, edges, same_rank). A node is (id, text key, module shown under it, fill colour
# where "" means the default grey). An edge is (from, to, dot attributes).
INGEST_START = [
    ("upload", "flow_upload", "app", ""),
    ("pages", "flow_pages", "pdf_loader.py", ""),
    ("toc", "flow_toc", "pipeline.py", ""),
    ("chunks", "flow_chunks", "chunking.py", ""),
]


def _chain(*names: str) -> list[tuple[str, str, str]]:
    return [(a, b, "") for a, b in zip(names, names[1:], strict=False)]


FLOWS = {
    "kw_ingest": (
        INGEST_START
        + [
            ("words", "flow_kw_words", "textutil.py", ""),
            ("idf", "flow_kw_idf", "embeddings.py", ""),
            ("vec", "flow_kw_vector", "embeddings.py", ""),
            ("index", "flow_index", "index.py", INDEX),
        ],
        _chain("upload", "pages", "toc", "chunks", "words", "idf", "vec", "index"),
        [],
    ),
    "kw_query": (
        [
            ("question", "flow_question", "app", ""),
            ("same", "flow_kw_same", "embeddings.py", ""),
            ("index", "flow_index", "index.py", INDEX),
            ("search", "flow_search", "retrieval.py", ""),
            ("thr", "flow_thr_kw", "generation.py", ""),
        ],
        _chain("question", "same", "search", "thr") + [("index", "search", "style=dashed")],
        [("same", "index")],
    ),
    "nn_ingest": (
        INGEST_START
        + [
            ("pieces", "flow_nn_pieces", "embeddings.py", ""),
            ("layers", "flow_nn_layers", "embeddings.py", ""),
            ("index", "flow_index", "index.py", INDEX),
        ],
        _chain("upload", "pages", "toc", "chunks", "pieces", "layers", "index"),
        [],
    ),
    "nn_query": (
        [
            ("question", "flow_question", "app", ""),
            ("enc", "flow_nn_q", "embeddings.py", ""),
            ("index", "flow_index", "index.py", INDEX),
            ("search", "flow_search", "retrieval.py", ""),
            ("thr", "flow_thr_nn", "generation.py", ""),
        ],
        _chain("question", "enc", "search", "thr") + [("index", "search", "style=dashed")],
        [("enc", "index")],
    ),
    "groq": (
        [
            ("found", "flow_g_found", "retrieval.py", ""),
            ("filter", "flow_filter", "generation.py", ""),
            ("prompt", "flow_prompt", "prompts.py", ""),
            ("call", "flow_g_call", "llm.py", OUTSIDE),
            ("norm", "flow_g_norm", "prompts.py", ""),
            ("verify", "flow_verify", "generation.py", ""),
            ("answer", "flow_answer", "app", OK),
            ("no", "flow_no", "", DECLINE),
        ],
        _chain("found", "filter")
        + [
            ("filter", "prompt", 'label=" ✓"'),
            ("filter", "no", 'label=" ✗"'),
            ("prompt", "call", ""),
            ("call", "norm", ""),
            ("call", "no", 'label=" ✗"'),
            ("norm", "verify", ""),
            ("verify", "answer", ""),
        ],
        [],
    ),
}


def architecture_diagram(lang: str, flows: list[tuple[str, str]]) -> str:
    """Flows as top-to-bottom diagrams, each box showing the step and the module behind it.

    Several flows go into ONE drawing (each in a titled frame), so they share a single scale and
    the text has the same size in all of them.
    """
    lines = [
        "digraph {",
        'rankdir=TB; bgcolor="transparent"; nodesep=0.4; ranksep=0.45; compound=true;',
        'node [shape=box, style="rounded,filled", fillcolor="#374151", fontcolor="white",',
        '      color="#6b7280", fontname="Helvetica", fontsize=13, margin="0.2,0.12"];',
        'edge [color="#6b7280", fontcolor="#6b7280", fontname="Helvetica", fontsize=13];',
    ]
    for prefix, (flow, title) in enumerate(flows):
        nodes, edges, same_rank = FLOWS[flow]
        framed = bool(title)
        if framed:
            lines += [
                f"subgraph cluster_{prefix} {{",
                f'label="{title}"; labeljust=l; fontname="Helvetica-Bold"; fontsize=14;',
                'fontcolor="#9ca3af"; color="#4b5563"; style="rounded"; margin=16;',
            ]
        for name, key, module, fill in nodes:
            # short lines keep the boxes narrow, so the drawing is shown at a readable size
            text = "\\n".join(textwrap.wrap(t(lang, key).replace('"', "'"), 24))
            if module:
                text += "\\n" + module
            style = f', fillcolor="{fill}"' if fill else ""
            lines.append(f'n{prefix}_{name} [label="{text}"{style}];')
        for a, b in same_rank:
            lines.append(f"{{rank=same; n{prefix}_{a}; n{prefix}_{b};}}")
        for a, b, attrs in edges:
            suffix = f" [{attrs}]" if attrs else ""
            lines.append(f"n{prefix}_{a} -> n{prefix}_{b}{suffix};")
        if framed:
            lines.append("}")
    lines.append("}")
    return "\n".join(lines)


def _two_flows(lang: str, ingest: str, query: str) -> None:
    dot = architecture_diagram(
        lang, [(ingest, t(lang, "cluster_ingest")), (query, t(lang, "cluster_query"))]
    )
    st.graphviz_chart(dot, use_container_width=True)


def render_architecture(lang: str) -> None:
    st.subheader(t(lang, "arch_title"))
    st.caption(t(lang, "arch_intro"))
    tab_kw, tab_nn, tab_groq = st.tabs(
        [t(lang, "arch_sub_kw"), t(lang, "arch_sub_nn"), t(lang, "arch_sub_groq")]
    )
    with tab_kw:
        _two_flows(lang, "kw_ingest", "kw_query")
        st.markdown(t(lang, "arch_kw_md"))
    with tab_nn:
        _two_flows(lang, "nn_ingest", "nn_query")
        st.markdown(t(lang, "arch_nn_md"))
    with tab_groq:
        left, right = st.columns([1, 1.6])
        with left:
            st.graphviz_chart(architecture_diagram(lang, [("groq", "")]), use_container_width=True)
        with right:
            st.markdown(t(lang, "arch_groq_md"))
        st.caption(t(lang, "arch_legend"))
    st.markdown(t(lang, "arch_md"))
