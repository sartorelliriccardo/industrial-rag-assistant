"""High-level facade: add PDFs, ask questions."""

from pathlib import Path

from rag_assistant.chunking import split_pages
from rag_assistant.config import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    get_chunk_settings,
)
from rag_assistant.embeddings import Embedder
from rag_assistant.generation import generate_answer
from rag_assistant.index import VectorIndex
from rag_assistant.llm import LLMProvider
from rag_assistant.models import Answer, Chunk, RetrievedChunk
from rag_assistant.pdf_loader import load_pdf_bytes
from rag_assistant.retrieval import retrieve
from rag_assistant.textutil import is_table_of_contents


class RagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        provider: LLMProvider,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        env_size, env_overlap = get_chunk_settings()
        self.chunk_size = chunk_size or env_size
        self.chunk_overlap = env_overlap if chunk_overlap is None else chunk_overlap
        self.embedder = embedder
        self.provider = provider
        self.top_k = top_k
        self.min_score = min_score
        self._index: VectorIndex | None = None
        self.skipped_pages: dict[str, list[int]] = {}  # index pages left out, per document

    @property
    def _learns_weights(self) -> bool:
        """Keyword search must see all chunks to know which words are rare."""
        return hasattr(self.embedder, "fit")

    def _rebuild(self, chunks: list[Chunk]) -> None:
        """(Re)build the whole index from chunks, refitting the word weights first."""
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)  # type: ignore[attr-defined]
        vectors = self.embedder.embed(texts)
        self._index = VectorIndex(dim=vectors.shape[1])
        self._index.add(chunks, vectors)

    @property
    def documents(self) -> list[str]:
        return self._index.doc_names if self._index else []

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._index.chunks if self._index else ()

    @property
    def chunk_count(self) -> int:
        return len(self._index) if self._index else 0

    def add_pdf_bytes(self, data: bytes, doc_name: str) -> int:
        """Index a PDF (replacing any previous one with the same name). Returns #chunks."""
        pages = load_pdf_bytes(data, doc_name)  # raises PdfError before touching the index
        kept = [p for p in pages if not is_table_of_contents(p.text)]
        if not kept:  # never drop a whole document just because it looks like an index
            kept = pages
        self.skipped_pages[doc_name] = [p.page_number for p in pages if p not in kept]
        chunks = split_pages(kept, self.chunk_size, self.chunk_overlap)

        if self._learns_weights:
            others = [c for c in self.chunks if c.doc_name != doc_name]
            self._rebuild(others + chunks)
            return len(chunks)

        vectors = self.embedder.embed([c.text for c in chunks])
        if self._index is None:
            self._index = VectorIndex(dim=vectors.shape[1])
        self._index.remove_document(doc_name)
        self._index.add(chunks, vectors)
        return len(chunks)

    def add_pdf_path(self, path: str | Path) -> int:
        path = Path(path)
        return self.add_pdf_bytes(path.read_bytes(), path.name)

    def clear(self) -> None:
        """Forget every document (the index and the extracted text go with it)."""
        self._index = None
        self.skipped_pages.clear()

    def remove(self, doc_name: str) -> None:
        self.skipped_pages.pop(doc_name, None)
        if not self._index:
            return
        if self._learns_weights:
            remaining = [c for c in self.chunks if c.doc_name != doc_name]
            if remaining:
                self._rebuild(remaining)
            else:
                self._index = None
        else:
            self._index.remove_document(doc_name)

    def search(self, question: str) -> list[RetrievedChunk]:
        if not self._index or len(self._index) == 0:
            raise ValueError("no documents indexed yet")
        return retrieve(question, self._index, self.embedder, self.top_k)

    def ask(self, question: str) -> Answer:
        retrieved = self.search(question)
        return generate_answer(question, retrieved, self.provider, self.min_score)
