"""Run one question through the real pipeline and keep what each stage produced.

Powers the "step by step" view. It reuses the same functions as `RagPipeline.ask`
(`pipeline.search`, `generate_answer`, `build_user_prompt`), so what it shows is what
happened, not a separate re-implementation.
"""

from dataclasses import dataclass, field

from rag_assistant.generation import build_user_prompt, generate_answer
from rag_assistant.llm import LLMProvider
from rag_assistant.models import Answer
from rag_assistant.pipeline import RagPipeline
from rag_assistant.prompts import (
    SENTENCE_SPLIT,
    SYSTEM_PROMPT,
    looks_like_injection,
    strip_injected_sentences,
)
from rag_assistant.textutil import content_tokens

PREVIEW_CHARS = 140


@dataclass(frozen=True)
class ChunkRow:
    doc: str
    page: int
    index: int  # position inside its page
    chars: int
    preview: str
    retrieved: bool  # among the passages returned for this question


@dataclass(frozen=True)
class RankedPassage:
    rank: int
    doc: str
    page: int
    score: float
    # used | below_threshold | removed (every sentence looked like an injected instruction)
    outcome: str
    sentences_removed: int  # injected-looking sentences dropped from this passage


@dataclass(frozen=True)
class TermPosition:
    term: str  # the word as the search sees it (lower-case, stemmed)
    position: int  # index in the vector this word writes to
    weight: float


@dataclass
class Walkthrough:
    question: str
    embedder: str  # class name, e.g. LexicalEmbedder
    terms: list[str]  # content words of the question (stopwords out, stemmed)
    term_positions: list[TermPosition]  # only for the keyword search; empty otherwise
    vector_dim: int
    vector_nonzero: int
    n_documents: int
    n_chunks: int
    chunk_size: int
    chunk_overlap: int
    chunks: list[ChunkRow]
    ranking: list[RankedPassage]
    min_score: float
    removed_sentences: list[tuple[str, int, str]]  # (doc, page, sentence)
    system_prompt: str
    user_prompt: str  # empty when the engine was not called
    answer: Answer
    notes: list[str] = field(default_factory=list)

    @property
    def used(self) -> int:
        return sum(p.outcome == "used" for p in self.ranking)


def _preview(text: str) -> str:
    return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1].rstrip() + "…"


def explain(
    pipeline: RagPipeline, question: str, provider: LLMProvider | None = None
) -> Walkthrough:
    """Ask `question` and record every stage. `provider` overrides the pipeline's engine
    (the view uses the offline engine so that exploring never calls a paid API)."""
    retrieved = pipeline.search(question)  # raises ValueError if empty question / no documents
    answer = generate_answer(question, retrieved, provider or pipeline.provider, pipeline.min_score)

    query_vector = pipeline.embedder.embed([question])[0]
    describe = getattr(pipeline.embedder, "describe", None)
    term_positions = [TermPosition(t, p, w) for t, p, w in describe(question)] if describe else []

    ranking: list[RankedPassage] = []
    removed: list[tuple[str, int, str]] = []
    for r in retrieved:
        cleaned, modified = strip_injected_sentences(r.chunk.text)
        dropped = [s for s in SENTENCE_SPLIT.split(r.chunk.text) if looks_like_injection(s)]
        removed.extend((r.chunk.doc_name, r.chunk.page_number, s.strip()) for s in dropped)
        if not cleaned:
            outcome = "removed"
        elif r.score < pipeline.min_score:
            outcome = "below_threshold"
        else:
            outcome = "used"
        ranking.append(
            RankedPassage(
                rank=r.rank,
                doc=r.chunk.doc_name,
                page=r.chunk.page_number,
                score=r.score,
                outcome=outcome,
                sentences_removed=len(dropped) if modified else 0,
            )
        )

    found = {(r.chunk.doc_name, r.chunk.page_number, r.chunk.chunk_index) for r in retrieved}
    chunks = [
        ChunkRow(
            doc=c.doc_name,
            page=c.page_number,
            index=c.chunk_index,
            chars=len(c.text),
            preview=_preview(c.text),
            retrieved=(c.doc_name, c.page_number, c.chunk_index) in found,
        )
        for c in pipeline.chunks
    ]

    return Walkthrough(
        question=question,
        embedder=type(pipeline.embedder).__name__,
        terms=content_tokens(question),
        term_positions=term_positions,
        vector_dim=int(query_vector.shape[0]),
        vector_nonzero=int((query_vector != 0).sum()),
        n_documents=len(pipeline.documents),
        n_chunks=pipeline.chunk_count,
        chunk_size=pipeline.chunk_size,
        chunk_overlap=pipeline.chunk_overlap,
        chunks=chunks,
        ranking=ranking,
        min_score=pipeline.min_score,
        removed_sentences=removed,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(question, answer.evidence) if answer.engine_called else "",
        answer=answer,
    )
