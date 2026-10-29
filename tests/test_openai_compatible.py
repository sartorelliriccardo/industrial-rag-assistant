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

    def __init__(
        self, status: int = 200, body: dict | None = None, models: list[str] | None = None
    ):
        self.status = status
        self.body = body if body is not None else completion("ok")
        self.models = models  # ids offered by GET /models; None = the host has no such page
        self.requests: list[dict] = []
        host = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (name fixed by the http.server API)
                host.requests.append({"path": self.path, "auth": self.headers.get("Authorization")})
                if host.models is None or not self.path.endswith("/models"):
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                payload = json.dumps({"data": [{"id": m} for m in host.models]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

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


def test_a_retired_model_lists_what_the_key_can_use():
    """The real failure seen with Groq: HTTP 404 because the default model was shut down."""
    body = {"error": {"message": "The model `old-model` does not exist or you do not have access"}}
    offered = ["openai/gpt-oss-120b", "whisper-large-v3", "llama-guard-x", "openai/gpt-oss-20b"]
    with FakeHost(status=404, body=body, models=offered) as host:
        with pytest.raises(ProviderError) as info:
            provider(host).complete("s", "u")
    message = str(info.value)
    assert 'The service says: "The model `old-model` does not exist' in message
    assert "Models offered to this key: openai/gpt-oss-120b, openai/gpt-oss-20b." in message
    assert "whisper" not in message and "guard" not in message  # not chat models
    assert host.requests[-1]["auth"] == "Bearer secret-key"  # the listing used the same key


def test_no_suggestions_when_the_host_cannot_list_models():
    with FakeHost(status=404, body={"error": {"message": "gone"}}, models=None) as host:
        with pytest.raises(ProviderError) as info:
            provider(host).complete("s", "u")
    assert "Models offered" not in str(info.value)
    assert 'The service says: "gone"' in str(info.value)


def test_server_messages_never_echo_the_key():
    body = {"error": {"message": "bad key secret-key supplied"}}
    with FakeHost(status=401, body=body) as host:
        with pytest.raises(ProviderError) as info:
            provider(host).complete("s", "u")
    assert "secret-key" not in str(info.value)
    assert "***" in str(info.value)


def test_extra_body_fields_are_sent():
    with FakeHost() as host:
        provider(host, extra_body={"reasoning_effort": "low"}).complete("s", "u")
    assert host.requests[0]["json"]["reasoning_effort"] == "low"


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
    assert p.model == "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was shut down by Groq
    assert p.model != "llama-3.3-70b-versatile"
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    monkeypatch.setenv("GROQ_MODEL", "another-model")
    p = GroqProvider()
    assert (p.api_key, p.model) == ("from-env", "another-model")
    assert GroqProvider(api_key="k", model="chosen/in-the-box").model == "chosen/in-the-box"


def test_groq_keeps_gpt_oss_reasoning_short_and_leaves_other_models_alone():
    thinking = GroqProvider(api_key="k", model="openai/gpt-oss-20b")
    assert thinking.extra_body == {"reasoning_effort": "low", "include_reasoning": False}
    assert thinking.max_tokens == 1500  # room for the answer after a little reasoning
    assert GroqProvider(api_key="k", model="some/other-model").extra_body == {}


def test_get_provider_knows_groq(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert isinstance(get_provider("groq", "k"), GroqProvider)
    assert get_provider("groq", "k", "my/model").model == "my/model"
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
