"""Render a PDF page as an image with the cited passage highlighted."""

import re

import pymupdf

from rag_assistant.prompts import SENTENCE_SPLIT


def _find(page: pymupdf.Page, text: str) -> list:
    """Locate `text` on the page; fall back to short word windows if the phrase is split
    in a way the PDF text layer does not reproduce (line breaks, ligatures)."""
    hits = page.search_for(text, quads=True)
    if hits:
        return hits
    words = text.split()
    found = []
    for start in range(0, len(words), 6):
        window = " ".join(words[start : start + 6])
        if len(window) >= 8:
            found.extend(page.search_for(window, quads=True))
    return found


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


def select_sentences(passage: str, answer_text: str, max_sentences: int = 2) -> str:
    """Keep the sentences of `passage` that share the most words with the answer.

    Highlighting the whole passage is noisy; this narrows it to what the answer used.
    Falls back to the whole passage when nothing overlaps.
    """
    answer_words = _words(answer_text)
    scored = [
        (len(_words(s) & answer_words), s) for s in SENTENCE_SPLIT.split(passage) if s.strip()
    ]
    best = sorted((x for x in scored if x[0] > 0), key=lambda x: -x[0])[:max_sentences]
    if not best:
        return passage
    chosen = {s for _, s in best}
    return " ".join(s for _, s in scored if s in chosen)  # keep original order


def render_cited_page(
    pdf_bytes: bytes, page_number: int, passage: str, zoom: float = 1.6
) -> tuple[bytes, int]:
    """Return (PNG bytes, number of highlighted areas) for a 1-based page number.

    The PDF is opened from memory and never modified on disk.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if not 1 <= page_number <= len(doc):
            raise ValueError(f"page {page_number} is out of range (1-{len(doc)})")
        page = doc[page_number - 1]
        highlighted = 0
        for sentence in SENTENCE_SPLIT.split(passage):
            sentence = sentence.strip()
            if not sentence:
                continue
            for quad in _find(page, sentence):
                page.add_highlight_annot(quad)
                highlighted += 1
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pixmap.tobytes("png"), highlighted
