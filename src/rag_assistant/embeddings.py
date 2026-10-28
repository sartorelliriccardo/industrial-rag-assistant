"""Text -> L2-normalised vectors. Two interchangeable embedders behind one protocol."""

import re
import zlib
from typing import Protocol

import numpy as np

from rag_assistant.config import EMBEDDING_MODEL, enable_system_certificates


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 matrix (len(texts), dim) with unit-length rows."""
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # keep all-zero rows as zeros instead of dividing by 0
    return (vectors / norms).astype(np.float32)


class FastEmbedEmbedder:
    """Multilingual sentence embeddings (ONNX, no PyTorch). Model is downloaded on first use."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            enable_system_certificates()
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = np.array(list(self._load().embed(texts)), dtype=np.float32)
        return _normalize(vectors)


class HashingEmbedder:
    """Deterministic bag-of-words embedder: no model, no network. Used by the unit tests.

    Each word is hashed into one of `dim` buckets. Texts sharing words get a high
    cosine similarity; it is lexical only, it does NOT understand synonyms.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                vectors[row, zlib.crc32(word.encode("utf-8")) % self.dim] += 1.0
        return _normalize(vectors)
