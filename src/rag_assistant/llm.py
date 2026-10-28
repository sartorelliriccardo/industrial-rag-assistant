"""LLM providers behind a one-method interface, so the vendor is easy to swap."""

import math
import os
import re
from typing import Protocol

from rag_assistant.config import DEFAULT_ANTHROPIC_MODEL, enable_system_certificates
from rag_assistant.prompts import INSUFFICIENT_TOKEN, QUESTION_RE, SENTENCE_SPLIT, SOURCE_RE


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


_STOPWORDS = set(
    "the a an of to in on at for and or is are be what which how many much does do with by from "
    "il lo la i gli le un una di del della dei delle da in con su per tra fra e o è sono qual "
    "quale quali quanto quanta quanti come che cosa dove quando".split()
)


def _tokens(text: str) -> set[str]:
    words = (w for w in re.findall(r"\w+", text.lower()) if len(w) > 2 and w not in _STOPWORDS)
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words}  # crude plural strip


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

    def __init__(self, min_overlap: float = 0.4, max_sentences: int = 3):
        self.min_overlap = min_overlap
        self.max_sentences = max_sentences

    def complete(self, system: str, user: str) -> str:
        match = QUESTION_RE.search(user)
        question_tokens = _tokens(match.group(1)) if match else set()
        if not question_tokens:
            return INSUFFICIENT_TOKEN

        sentences = [
            (source_id, s.strip())
            for source_id, text in SOURCE_RE.findall(user)
            for s in SENTENCE_SPLIT.split(text)
        ]
        token_sets = [_tokens(s) for _, s in sentences]
        n = len(sentences)
        weight = {
            t: math.log(1 + n / max(sum(t in ts for ts in token_sets), 0.5))
            for t in question_tokens
        }
        total = sum(weight.values())

        best_per_source: dict[str, tuple[float, str]] = {}
        for (source_id, sentence), ts in zip(sentences, token_sets, strict=True):
            score = sum(weight[t] for t in question_tokens & ts) / total
            if score > best_per_source.get(source_id, (-1.0, ""))[0]:
                best_per_source[source_id] = (score, sentence)
        best_per_source = [(s, sid, sent) for sid, (s, sent) in best_per_source.items()]

        best_per_source.sort(key=lambda x: -x[0])
        if not best_per_source or best_per_source[0][0] < self.min_overlap:
            return INSUFFICIENT_TOKEN
        top = best_per_source[0][0]
        # keep other sources only if they match nearly as well (e.g. a conflicting document)
        chosen = [x for x in best_per_source if x[0] >= max(self.min_overlap, 0.8 * top)]
        return " ".join(f"{sentence} [{sid}]" for _, sid, sentence in chosen[: self.max_sentences])


def get_provider(name: str, api_key: str | None = None) -> LLMProvider:
    if name == "extractive":
        return ExtractiveProvider()
    if name == "anthropic":
        return AnthropicProvider(api_key=api_key)
    raise ValueError(f"unknown LLM provider: {name!r} (use 'extractive' or 'anthropic')")
