"""LLM providers behind a one-method interface, so the vendor is easy to swap."""

import math
import os
from typing import Protocol

from rag_assistant.config import DEFAULT_ANTHROPIC_MODEL, enable_system_certificates
from rag_assistant.prompts import INSUFFICIENT_TOKEN, QUESTION_RE, SENTENCE_SPLIT, SOURCE_RE
from rag_assistant.textutil import content_tokens


class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicProvider:
    """Claude via the Anthropic API. The key is read from the environment, never hard-coded."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self._client = None

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            enable_system_certificates()
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=700,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if block.type == "text").strip()


def _tokens(text: str) -> set[str]:
    return set(content_tokens(text))


class ExtractiveProvider:
    """Offline 'LLM' with no API key: quotes the best-matching sentences from the sources.

    It cannot reason or translate, but it is deterministic and always grounded: every
    word it outputs is copied from a source. Used for tests, the keyless demo and as an
    evaluation baseline for the real LLM.

    A sentence is scored by the share of the question's content words it covers, where
    rare words weigh more (IDF over the provided sentences). A question word that appears
    in no source at all ("warranty", "weight") weighs the most, which makes the provider
    refuse questions about a topic the sources only mention in passing.
    """

    def __init__(
        self, min_overlap: float = 0.5, max_sentences: int = 3, context_sentences: int = 1
    ):
        self.min_overlap = min_overlap
        self.max_sentences = max_sentences
        # Procedures in manuals are lists of steps, and the best-matching sentence is often
        # only the first step or the heading: also quote the next `context_sentences`.
        self.context_sentences = context_sentences

    def complete(self, system: str, user: str) -> str:
        match = QUESTION_RE.search(user)
        question_tokens = _tokens(match.group(1)) if match else set()
        if not question_tokens:
            return INSUFFICIENT_TOKEN

        per_source: dict[str, list[str]] = {}
        for source_id, text in SOURCE_RE.findall(user):
            per_source[source_id] = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
        token_sets = {sid: [_tokens(s) for s in sents] for sid, sents in per_source.items()}
        n = sum(len(sents) for sents in per_source.values())
        weight = {
            t: math.log(
                1 + n / max(sum(t in ts for sets in token_sets.values() for ts in sets), 0.5)
            )
            for t in question_tokens
        }
        total = sum(weight.values())

        # best sentence of each source: (score, source id, position inside the source)
        best_per_source: list[tuple[float, str, int]] = []
        for sid, sets in token_sets.items():
            scores = [sum(weight[t] for t in question_tokens & ts) / total for ts in sets]
            if scores:
                position = max(range(len(scores)), key=scores.__getitem__)
                best_per_source.append((scores[position], sid, position))

        best_per_source.sort(key=lambda x: -x[0])
        if not best_per_source or best_per_source[0][0] < self.min_overlap:
            return INSUFFICIENT_TOKEN
        top = best_per_source[0][0]
        # keep other sources only if they match nearly as well (e.g. a conflicting document)
        chosen = [x for x in best_per_source if x[0] >= max(self.min_overlap, 0.8 * top)]
        quotes = []
        for _, sid, position in chosen[: self.max_sentences]:
            window = per_source[sid][position : position + 1 + self.context_sentences]
            quotes.append(f"{' '.join(window)} [{sid}]")
        return " ".join(quotes)


def get_provider(name: str, api_key: str | None = None) -> LLMProvider:
    if name == "extractive":
        return ExtractiveProvider()
    if name == "anthropic":
        return AnthropicProvider(api_key=api_key)
    raise ValueError(f"unknown LLM provider: {name!r} (use 'extractive' or 'anthropic')")
