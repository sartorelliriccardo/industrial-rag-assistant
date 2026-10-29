"""LLM providers behind a one-method interface, so the vendor is easy to swap."""

import json
import math
import os
import urllib.error
import urllib.request
from typing import Protocol

from rag_assistant.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_GROQ_MODEL,
    GROQ_BASE_URL,
    enable_system_certificates,
)
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


class ProviderError(Exception):
    """A hosted model refused or failed; the message is meant to be shown to the user."""


class OpenAICompatibleProvider:
    """Any host that speaks the OpenAI chat-completions dialect (Groq, OpenRouter, a local server).

    Plain standard library on purpose: one POST, no extra dependency to install or audit.
    """

    HINTS = {
        401: "the API key was rejected: check that it is correct and still active",
        403: "the host refused the request (key without access to this model, or blocked)",
        404: "model or address not found: check the model name",
        429: "rate limit reached: free plans allow few requests per minute, wait and retry",
    }

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        name: str = "openai-compatible",
        max_tokens: int = 700,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError(f"{name}: an API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = name
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system: str, user: str) -> str:
        enable_system_certificates()
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_completion_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # some hosts block the default Python user agent
                "User-Agent": "industrial-rag-assistant",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = self.HINTS.get(exc.code, "the request failed")
            raise ProviderError(f"{self.name}: HTTP {exc.code}, {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"{self.name}: cannot reach the service ({exc})") from exc
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: unexpected response from the service") from exc
        if not text or not text.strip():
            raise ProviderError(
                f"{self.name}: empty answer (a reasoning model may have spent every token "
                "thinking: try another model)"
            )
        return text.strip()


class GroqProvider(OpenAICompatibleProvider):
    """Groq: free plan for tests (limits shown in the Groq console), key from console.groq.com."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            base_url=os.environ.get("GROQ_BASE_URL", GROQ_BASE_URL),
            model=model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            name="groq",
        )


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
    if name == "groq":
        return GroqProvider(api_key=api_key)
    raise ValueError(f"unknown LLM provider: {name!r} (use 'extractive', 'groq' or 'anthropic')")
