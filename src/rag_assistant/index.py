"""In-memory FAISS index that keeps chunks and their vectors together."""

import faiss
import numpy as np

from rag_assistant.models import Chunk


class VectorIndex:
    """Exact cosine search. With unit-length vectors, inner product == cosine similarity."""

    def __init__(self, dim: int):
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._faiss = faiss.IndexFlatIP(dim)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def doc_names(self) -> list[str]:
        return sorted({c.doc_name for c in self._chunks})

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks) == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of dim {self.dim}, got {vectors.shape[1]}")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._chunks.extend(chunks)
        self._vectors = np.vstack([self._vectors, vectors])
        self._faiss.add(vectors)

    def remove_document(self, doc_name: str) -> None:
        """FAISS flat indexes cannot delete rows, so we rebuild from the kept vectors."""
        keep = [i for i, c in enumerate(self._chunks) if c.doc_name != doc_name]
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep]
        self._faiss = faiss.IndexFlatIP(self.dim)
        if len(self._chunks):
            self._faiss.add(self._vectors)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if len(self._chunks) == 0:
            return []
        k = min(k, len(self._chunks))
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, ids = self._faiss.search(query, k)
        return [
            (self._chunks[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0
        ]
