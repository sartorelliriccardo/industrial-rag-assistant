"""Plain data types shared by all modules."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    """Text of one PDF page. `page_number` is 1-based, as printed in a PDF viewer."""

    doc_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A piece of a single page. Never crosses a page boundary, so citations stay exact."""

    doc_name: str
    page_number: int
    chunk_index: int  # position of the chunk inside its page
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]
    rank: int  # 1 = best


@dataclass(frozen=True)
class Citation:
    source_id: str  # "S1": the id the LLM saw in the prompt
    doc_name: str
    page_number: int

    @property
    def label(self) -> str:
        return f"{self.doc_name}, p. {self.page_number}"


@dataclass
class Answer:
    # answered | insufficient (no usable evidence) | uncited (answer without valid citations)
    status: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    evidence: list[RetrievedChunk] = field(default_factory=list)  # passages given to the LLM
    flagged: list[RetrievedChunk] = field(default_factory=list)  # excluded: look like injection
    invalid_citations: list[str] = field(default_factory=list)  # ids cited but never provided
    raw: str = ""  # what the answer engine returned, before citations were rendered
    engine_called: bool = False  # False when no passage was good enough to ask the engine
