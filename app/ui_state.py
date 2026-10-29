"""What every tab needs: the choices made in the sidebar and this session's documents."""

from dataclasses import dataclass

import streamlit as st

from rag_assistant.config import DEFAULT_TOP_K, MIN_SCORE_BY_SEARCH, get_chunk_settings
from rag_assistant.embeddings import FastEmbedEmbedder, LexicalEmbedder
from rag_assistant.llm import get_provider
from rag_assistant.pipeline import RagPipeline
from ui_privacy import workspace


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~120 MB)...")
def get_embedder() -> FastEmbedEmbedder:
    embedder = FastEmbedEmbedder()
    embedder.embed(["warm up"])  # load the model once, here, with a spinner
    return embedder


def min_score_for(search: str) -> float:
    return st.session_state.get(f"min_score_{search}", MIN_SCORE_BY_SEARCH[search])


def new_pipeline(
    search: str, provider_name: str, api_key: str | None, model: str | None = None
) -> RagPipeline:
    size, overlap = get_chunk_settings()
    # keyword search needs no model (a few MB); semantic search loads ~400 MB of weights
    embedder = LexicalEmbedder() if search == "keyword" else get_embedder()
    return RagPipeline(
        embedder,
        get_provider(provider_name, api_key, model),
        chunk_size=st.session_state.get("chunk_size", size),
        chunk_overlap=st.session_state.get("chunk_overlap", overlap),
        top_k=st.session_state.get("top_k", DEFAULT_TOP_K),
        min_score=min_score_for(search),
    )


@dataclass
class Ctx:
    """The state one script run hands to every tab."""

    lang: str
    search: str  # "keyword" or "semantic"
    provider_name: str  # "extractive" or "anthropic"
    pipeline: RagPipeline | None  # None when the Claude engine is chosen without an API key
    files: dict[str, bytes]  # name -> PDF bytes of the uploaded documents

    def apply_search_settings(self) -> None:
        """Use the top-k and minimum similarity currently set in the sidebar."""
        self.pipeline.top_k = st.session_state.get("top_k", DEFAULT_TOP_K)
        self.pipeline.min_score = min_score_for(self.search)


def open_workspace(
    lang: str, search: str, provider_name: str, api_key: str | None, model: str | None = None
) -> Ctx:
    """This session's pipeline and files; the pipeline is rebuilt only if the settings changed."""
    config_key = (search, provider_name, api_key, model)
    ws = workspace()
    if ws.pipeline is None or ws.config_key != config_key:
        previous = dict(ws.files)
        try:
            ws.pipeline = new_pipeline(search, provider_name, api_key or None, model or None)
            ws.config_key = config_key
            ws.files = {}
            for name, data in previous.items():  # re-index when only the engine changed
                ws.pipeline.add_pdf_bytes(data, name)
                ws.files[name] = data
        except ValueError:  # missing API key
            ws.pipeline = None
            ws.config_key = None
    return Ctx(lang, search, provider_name, ws.pipeline, ws.files)
