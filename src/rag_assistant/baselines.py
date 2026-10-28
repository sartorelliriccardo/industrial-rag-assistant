"""A 'Ctrl+F' baseline: what a person can find by searching the PDFs for exact words.

Used to check that the assistant adds something over plain text search. It only models
*finding pages*: Ctrl+F gives no answer, so the comparison is about retrieval.
"""

import re
from pathlib import Path

from rag_assistant.models import Page
from rag_assistant.pdf_loader import load_pdf
from rag_assistant.textutil import STOPWORDS

Mode = str  # "all" | "rarest"


def search_words(question: str) -> list[str]:
    """The question's content words, exactly as typed (no stemming, no synonyms)."""
    seen: dict[str, None] = {}
    for word in re.findall(r"\w+", question.lower()):
        if (len(word) > 2 or word.isdigit()) and word not in STOPWORDS:
            seen.setdefault(word)
    return list(seen)


def ctrl_f_pages(question: str, pages: list[Page], mode: Mode) -> list[tuple[str, int]]:
    """Pages (document, number) a person would get with exact-word search.

    - ``all``: pages containing every content word of the question (searching the whole
      sentence). Strict: any word the manual words differently gives zero results.
    - ``rarest``: a person who types only the single most distinctive word that exists in
      the documents, i.e. the word appearing on the fewest pages. Generous: it assumes the
      person already knows which word will work.
    """
    words = search_words(question)
    texts = {(p.doc_name, p.page_number): p.text.lower() for p in pages}
    if mode == "all":
        return [key for key, text in texts.items() if words and all(w in text for w in words)]
    if mode == "rarest":
        counts = {w: sum(w in text for text in texts.values()) for w in words}
        present = {w: n for w, n in counts.items() if n > 0}
        if not present:
            return []
        best = min(present, key=lambda w: (present[w], words.index(w)))
        return [key for key, text in texts.items() if best in text]
    raise ValueError(f"unknown mode: {mode!r}")


def evaluate_ctrl_f(
    docs_dir: str | Path, questions: list[dict], mode: Mode, max_pages: int = 5
) -> dict:
    """Retrieval-only metrics for the baseline, comparable with the assistant's.

    ``all_gold_found``: every page the answer needs is among the results.
    ``usable``: ...and the person has to read at most `max_pages` pages (the assistant
    shows 5 passages).
    """
    pages = [p for pdf in sorted(Path(docs_dir).glob("*.pdf")) for p in load_pdf(pdf)]
    answerable = [q for q in questions if q["answerable"]]
    found = usable = 0
    sizes: list[int] = []
    for q in answerable:
        gold = {(g["document"], g["page"]) for g in q["gold"]}
        hits = ctrl_f_pages(q["question"], pages, mode)
        sizes.append(len(hits))
        if gold <= set(hits):
            found += 1
            usable += len(hits) <= max_pages
    n = len(answerable)
    return {
        "mode": mode,
        "n_answerable": n,
        "all_gold_found": round(found / n, 3) if n else None,
        "usable": round(usable / n, 3) if n else None,
        "avg_pages_to_read": round(sum(sizes) / n, 1) if n else None,
        "n_pages_in_corpus": len(pages),
    }
