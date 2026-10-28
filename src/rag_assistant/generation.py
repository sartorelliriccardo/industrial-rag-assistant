"""Evidence -> grounded prompt -> LLM -> answer with verified citations."""

import re
from dataclasses import replace

from rag_assistant.llm import LLMProvider
from rag_assistant.models import Answer, Citation, RetrievedChunk
from rag_assistant.prompts import (
    CITATION_RE,
    INSUFFICIENT_TOKEN,
    SYSTEM_PROMPT,
    escape_source_text,
    strip_injected_sentences,
)

CRITICAL_VALUE_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:N\.?m|bar|psi|kPa|MPa|°\s?C|degC|kW|rpm|mm/s|dB|V|A|kg)(?![A-Za-z])"
)


def has_critical_values(text: str) -> bool:
    """True if the text contains a number with a technical unit (torque, pressure, ...).

    Used to remind the reader to verify such values on the cited page.
    """
    return bool(CRITICAL_VALUE_RE.search(text))


INSUFFICIENT_MESSAGE = (
    "The uploaded documents do not contain enough information to answer this question."
)


def build_user_prompt(question: str, evidence: list[RetrievedChunk]) -> str:
    """Sources are numbered S1..Sn in the order given; the LLM may cite only those ids."""
    blocks = [
        f'<source id="S{i}" document="{escape_source_text(r.chunk.doc_name)}" '
        f'page="{r.chunk.page_number}">\n{escape_source_text(r.chunk.text)}\n</source>'
        for i, r in enumerate(evidence, start=1)
    ]
    return (
        "<sources>\n" + "\n".join(blocks) + f"\n</sources>\n\n<question>\n{question}\n</question>"
    )


def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    provider: LLMProvider,
    min_score: float,
) -> Answer:
    """Decide whether the evidence is enough, ask the LLM, and verify its citations."""
    flagged: list[RetrievedChunk] = []
    evidence: list[RetrievedChunk] = []
    for r in retrieved:
        cleaned, modified = strip_injected_sentences(r.chunk.text)
        if modified:
            flagged.append(r)  # shown as a warning even if the rest of the passage is kept
        if cleaned and r.score >= min_score:
            chunk = replace(r.chunk, text=cleaned) if modified else r.chunk
            evidence.append(replace(r, chunk=chunk))

    if not evidence:
        return Answer(status="insufficient", text=INSUFFICIENT_MESSAGE, flagged=flagged)

    raw = provider.complete(SYSTEM_PROMPT, build_user_prompt(question, evidence)).strip()
    if INSUFFICIENT_TOKEN in raw:
        return Answer(
            status="insufficient", text=INSUFFICIENT_MESSAGE, evidence=evidence, flagged=flagged
        )

    citations: list[Citation] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for source_id in CITATION_RE.findall(raw):
        if source_id in seen:
            continue
        seen.add(source_id)
        position = int(source_id[1:]) - 1
        if 0 <= position < len(evidence):
            chunk = evidence[position].chunk
            citations.append(Citation(source_id, chunk.doc_name, chunk.page_number))
        else:
            invalid.append(source_id)

    return Answer(
        status="answered" if citations else "uncited",
        text=_render_citations(raw, citations),
        citations=citations,
        evidence=evidence,
        flagged=flagged,
        invalid_citations=invalid,
    )


def _render_citations(text: str, citations: list[Citation]) -> str:
    """Replace [S1] with [document.pdf, p. 3] so the reader sees real sources."""
    for c in citations:
        text = text.replace(f"[{c.source_id}]", f"[{c.label}]")
    return text
