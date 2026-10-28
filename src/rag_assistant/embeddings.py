"""Text -> L2-normalised vectors. Two interchangeable embedders behind one protocol."""

import math
import re
import zlib
from typing import Protocol

import numpy as np

from rag_assistant.config import EMBEDDING_MODEL, enable_system_certificates
from rag_assistant.textutil import content_tokens

# 8-bit quantised ONNX export of the same sentence-transformers model (Apache-2.0).
QUANTIZED_MODEL_REPO = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
QUANTIZED_MODEL_FILE = "onnx/model_quantized.onnx"
QUANTIZED_MODEL_NAME = "custom/paraphrase-multilingual-MiniLM-L12-v2-int8"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 matrix (len(texts), dim) with unit-length rows."""
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # keep all-zero rows as zeros instead of dividing by 0
    return (vectors / norms).astype(np.float32)


class FastEmbedEmbedder:
    """Multilingual sentence embeddings (ONNX, no PyTorch). Model is downloaded on first use.

    By default it uses an 8-bit quantised copy of the model: about 140 MB less memory
    than the full-precision one (measured: +424 MB vs +560 MB) with embeddings that
    match the original (mean cosine 0.994 on a small sample). That is what lets the app
    fit in the ~1 GB of a free Streamlit Cloud instance. `quantized=False` gives the
    original model.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, quantized: bool = True):
        self.model_name = model_name
        self.quantized = quantized
        self._model = None

    def _load(self):
        if self._model is None:
            enable_system_certificates()
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType

            name = self.model_name
            if self.quantized:
                name = QUANTIZED_MODEL_NAME
                try:
                    TextEmbedding.add_custom_model(
                        model=name,
                        pooling=PoolingType.MEAN,
                        normalization=True,
                        sources=ModelSource(hf=QUANTIZED_MODEL_REPO),
                        dim=384,
                        model_file=QUANTIZED_MODEL_FILE,
                    )
                except ValueError:  # already registered in this process
                    pass
            # no memory arena: the arena keeps freed buffers around, costing extra RAM
            self._model = TextEmbedding(name, enable_cpu_mem_arena=False)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        # small batches keep the peak memory low (fastembed's default is 256 texts at a time)
        vectors = np.array(list(self._load().embed(texts, batch_size=16)), dtype=np.float32)
        return _normalize(vectors)


class LexicalEmbedder:
    """Keyword search as a vector: stemmed content words hashed into `dim` buckets.

    No model and no download (a few MB of memory instead of ~400 MB). Texts that share
    words score high; it does NOT understand synonyms or paraphrases, and it cannot match
    a question in one language with a document in another. Term frequency is damped with
    1 + log(tf) so a repeated word does not dominate.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in content_tokens(text):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                vectors[row, zlib.crc32(token.encode("utf-8")) % self.dim] += 1 + math.log(count)
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
