import numpy as np
import pytest

from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.index import VectorIndex
from rag_assistant.models import Chunk
from rag_assistant.retrieval import retrieve

CHUNKS = [
    Chunk("pump.pdf", 1, 0, "impeller nut tightening torque 120 N.m"),
    Chunk("pump.pdf", 2, 0, "bearing lubrication grease every 2000 hours"),
    Chunk("safety.pdf", 1, 0, "fire assembly point is Gate B"),
]


@pytest.fixture
def index_and_embedder():
    embedder = HashingEmbedder()
    index = VectorIndex(embedder.dim)
    index.add(CHUNKS, embedder.embed([c.text for c in CHUNKS]))
    return index, embedder


def test_embeddings_are_unit_length_and_deterministic():
    e = HashingEmbedder()
    a, b = e.embed(["torque of the nut"]), e.embed(["torque of the nut"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, b)


def test_most_relevant_chunk_ranks_first_with_metadata(index_and_embedder):
    index, embedder = index_and_embedder
    results = retrieve("what is the impeller nut torque", index, embedder, k=3)
    assert results[0].chunk.page_number == 1 and results[0].chunk.doc_name == "pump.pdf"
    assert results[0].rank == 1
    assert [r.score for r in results] == sorted([r.score for r in results], reverse=True)


def test_k_larger_than_index_returns_everything(index_and_embedder):
    index, embedder = index_and_embedder
    assert len(retrieve("anything", index, embedder, k=50)) == 3


def test_empty_question_is_rejected(index_and_embedder):
    index, embedder = index_and_embedder
    with pytest.raises(ValueError):
        retrieve("   ", index, embedder)


def test_remove_document(index_and_embedder):
    index, embedder = index_and_embedder
    index.remove_document("pump.pdf")
    assert index.doc_names == ["safety.pdf"]
    assert [r.chunk.doc_name for r in retrieve("torque", index, embedder, k=5)] == ["safety.pdf"]


def test_empty_index_returns_no_results():
    assert VectorIndex(8).search(np.ones(8, dtype=np.float32), 3) == []


def test_wrong_dimension_is_rejected():
    index = VectorIndex(8)
    with pytest.raises(ValueError):
        index.add(CHUNKS[:1], np.ones((1, 4), dtype=np.float32))
