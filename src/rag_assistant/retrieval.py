"""Question -> top-k most similar chunks, with scores."""

from rag_assistant.embeddings import Embedder
from rag_assistant.index import VectorIndex
from rag_assistant.models import RetrievedChunk


def retrieve(
    question: str, index: VectorIndex, embedder: Embedder, k: int = 5
) -> list[RetrievedChunk]:
    """Embed the question and return the k nearest chunks, best first.

    This is the baseline: pure vector search, no re-ranking and no score threshold
    (deciding what counts as 'enough evidence' is the generation step's job).
    """
    if not question.strip():
        raise ValueError("question is empty")
    query_vector = embedder.embed([question])[0]
    hits = index.search(query_vector, k)
    return [RetrievedChunk(chunk=c, score=s, rank=r) for r, (c, s) in enumerate(hits, start=1)]
