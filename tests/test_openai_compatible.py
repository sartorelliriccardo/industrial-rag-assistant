"""The hosted-model provider, tested against a fake local server: no network, no real key."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import make_pdf_bytes
from rag_assistant.embeddings import LexicalEmbedder
from rag_assistant.llm import (
    GroqProvider,
    OpenAICompatibleProvider,
    ProviderError,
    get_provider,
)
from rag_assistant.pipeline import RagPipeline


class FakeHost:
    """A tiny HTTP server that records requests and replies with a scripted status and body."""

    def __init__(self, status: int = 200, body: dict | None = None):
        self.status = status
        self.body = body if body is not None else completion("ok")
        self.requests: list[dict] = []
        host = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (name fixed by the http.server API)
                length = int(self.headers.get("Content-Length", 0))
                host.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "agent": self.headers.get("User-Agent"),
                        "json": json.loads(self.rfile.read(length)),
                    }
                )
                payload = json.dumps(host.body).encode("utf-8")
                self.send_response(host.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # keep the test output clean
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def completion(text) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def provider(host: FakeHost, **kw) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider("secret-key", host.url, "some-model", name="fake", **kw)


def test_request_has_the_expected_shape():
    with FakeHost(body=completion("  Answer [S1]  ")) as host:
        text = provider(host).complete("SYSTEM RULES", "USER TEXT")
    request = host.requests[0]
    assert text == "Answer [S1]"  # trimmed
    assert request["path"] == "/v1/chat/completions"
    assert request["auth"] == "Bearer secret-key"
    assert request["agent"] == "industrial-rag-assistant"
    body = request["json"]
    assert body["model"] == "some-model"
    assert body["temperature"] == 0
    assert body["max_completion_tokens"] == 700
    assert body["messages"] == [
        {"role": "system", "content": "SYSTEM RULES"},
        {"role": "user", "content": "USER TEXT"},
    ]


def test_trailing_slash_in_base_url_is_harmless():
    with FakeHost() as host:
        OpenAICompatibleProvider("k", host.url + "/", "m").complete("s", "u")
    assert host.requests[0]["path"] == "/v1/chat/completions"


@pytest.mark.parametrize(
    "status, expected",
    [(401, "key was rejected"), (429, "rate limit"), (404, "model or address"), (500, "failed")],
)
def test_http_errors_become_readable_messages(status, expected):
    with FakeHost(status=status, body={"error": {"message": "nope"}}) as host:
        with pytest.raises(ProviderError) as info:
            provider(host).complete("s", "u")
    assert f"HTTP {status}" in str(info.value)
    assert expected in str(info.value)
    assert "secret-key" not in str(info.value)  # the key must never leak into a message


def test_unreachable_host_is_reported():
    # nothing listens on this port
    p = OpenAICompatibleProvider("k", "http://127.0.0.1:9/v1", "m", name="fake", timeout=2)
    with pytest.raises(ProviderError, match="cannot reach"):
        p.complete("s", "u")


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": {}}]}])
def test_malformed_response_is_reported(body):
    with FakeHost(body=body) as host:
        with pytest.raises(ProviderError, match="unexpected response"):
            provider(host).complete("s", "u")


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_answer_is_reported(content):
    with FakeHost(body=completion(content)) as host:
        with pytest.raises(ProviderError, match="empty answer"):
            provider(host).complete("s", "u")


def test_a_key_is_required():
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleProvider(None, "http://x/v1", "m")
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleProvider("", "http://x/v1", "m")


def test_groq_defaults_and_environment(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        GroqProvider()
    p = GroqProvider(api_key="k")
    assert p.base_url == "https://api.groq.com/openai/v1"
    assert p.model == "llama-3.3-70b-versatile"
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    monkeypatch.setenv("GROQ_MODEL", "another-model")
    p = GroqProvider()
    assert (p.api_key, p.model) == ("from-env", "another-model")


def test_get_provider_knows_groq(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert isinstance(get_provider("groq", "k"), GroqProvider)
    with pytest.raises(ValueError):  # the sidebar relies on this to ask for a key
        get_provider("groq", None)
    with pytest.raises(ValueError, match="groq"):
        get_provider("nope")


def test_whole_pipeline_with_a_hosted_model_keeps_citations_verified():
    """Ask a question end to end: the answer comes from the host, the citation is checked."""
    pdf = make_pdf_bytes(["The impeller nut torque is 120 N.m.", "Unrelated page about paint."])
    with FakeHost(body=completion("The torque is 120 N.m. [S1] Also see [S9].")) as host:
        llm = OpenAICompatibleProvider("k", host.url, "m", name="fake")
        pipeline = RagPipeline(LexicalEmbedder(), llm)
        pipeline.add_pdf_bytes(pdf, "pump.pdf")
        answer = pipeline.ask("What is the impeller nut torque?")
    sent = host.requests[0]["json"]["messages"]
    assert "<source id=" in sent[1]["content"]  # evidence was handed over as numbered sources
    assert "impeller nut torque is 120" in sent[1]["content"]
    assert answer.status == "answered"
    assert [c.label for c in answer.citations] == ["pump.pdf, p. 1"]
    assert answer.invalid_citations == ["S9"]  # an invented source is reported, not trusted


def test_refusal_from_the_host_becomes_insufficient():
    pdf = make_pdf_bytes(["The impeller nut torque is 120 N.m."])
    with FakeHost(body=completion("INSUFFICIENT_EVIDENCE")) as host:
        pipeline = RagPipeline(LexicalEmbedder(), OpenAICompatibleProvider("k", host.url, "m"))
        pipeline.add_pdf_bytes(pdf, "pump.pdf")
        answer = pipeline.ask("What is the impeller nut torque?")
    assert answer.status == "insufficient"
