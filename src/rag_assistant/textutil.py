"""Small text helpers shared by the keyword search and the offline answer engine."""

import re

STOPWORDS = frozenset(
    "the a an of to in on at for and or is are be what which how many much does do with by from "
    "where when who why must should can within long often "
    "il lo la i gli le un una di del della dei delle da in con su per tra fra e o è sono qual "
    "quale quali quanto quanta quanti come che cosa dove quando".split()
)

STEM_LENGTH = 6


def stem(word: str) -> str:
    """Crude prefix stemmer: 'lubricated' and 'lubricate' both become 'lubric'.

    It works for English and Italian without language-specific rules. Words with digits
    (part numbers, values) are kept whole. It will sometimes merge unrelated words.
    """
    return word if any(c.isdigit() for c in word) else word[:STEM_LENGTH]


TOC_ENTRY_RE = re.compile(r"\.{5,}\s*\d+")
TOC_MIN_ENTRIES = 8


def is_table_of_contents(text: str) -> bool:
    """True for an index page: many 'heading ........ page number' lines.

    Such pages repeat every section title of the manual, so they match almost any question
    and crowd the real answer out of the results (measured on a 110-page manual: 8 of the
    60 top-5 slots on 12 questions). A normal page has no, or very few, dot-leader entries.
    """
    return len(TOC_ENTRY_RE.findall(text)) >= TOC_MIN_ENTRIES


def content_tokens(text: str) -> list[str]:
    """Lower-case, drop stopwords and very short words, stem the rest."""
    return [
        stem(w)
        for w in re.findall(r"\w+", text.lower())
        if (len(w) > 2 or w.isdigit()) and w not in STOPWORDS
    ]
