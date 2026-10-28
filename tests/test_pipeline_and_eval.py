import pytest

from conftest import make_pdf_bytes
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.evaluation import run_evaluation
from rag_assistant.llm import ExtractiveProvider, get_provider
from rag_assistant.pdf_loader import PdfCorruptedError
from rag_assistant.pipeline import RagPipeline


def make_pipeline(**kwargs):
    kwargs.setdefault("min_score", 0.15)
    return RagPipeline(HashingEmbedder(), ExtractiveProvider(), chunk_size=300, **kwargs)


def test_end_to_end_question_returns_cited_answer():
    pipeline = make_pipeline()
    pipeline.add_pdf_bytes(
        make_pdf_bytes(["Intro page.", "The impeller nut tightening torque is 120 N.m."]),
        "pump.pdf",
    )
    answer = pipeline.ask("What is the tightening torque of the impeller nut?")
    assert answer.status == "answered"
    assert "120 N.m" in answer.text
    assert [(c.doc_name, c.page_number) for c in answer.citations] == [("pump.pdf", 2)]


def test_unrelated_question_is_declined():
    pipeline = make_pipeline(min_score=0.3)
    pipeline.add_pdf_bytes(make_pdf_bytes(["The impeller nut torque is 120 N.m."]), "pump.pdf")
    assert pipeline.ask("What is the capital of France?").status == "insufficient"


def test_asking_with_no_documents_is_an_error():
    with pytest.raises(ValueError, match="no documents"):
        make_pipeline().ask("anything")


def test_bad_pdf_does_not_alter_the_index():
    pipeline = make_pipeline()
    pipeline.add_pdf_bytes(make_pdf_bytes(["good page"]), "good.pdf")
    with pytest.raises(PdfCorruptedError):
        pipeline.add_pdf_bytes(b"garbage", "bad.pdf")
    assert pipeline.documents == ["good.pdf"]


def test_reuploading_same_name_replaces_the_document():
    pipeline = make_pipeline()
    pipeline.add_pdf_bytes(make_pdf_bytes(["old content"]), "a.pdf")
    n_before = pipeline.chunk_count
    pipeline.add_pdf_bytes(make_pdf_bytes(["new content"]), "a.pdf")
    assert pipeline.chunk_count == n_before


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        get_provider("nope")


def test_anthropic_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_provider("anthropic")


def test_evaluation_runs_offline_and_is_reproducible(sample_dir):
    args = (sample_dir, sample_dir.parent.parent / "eval" / "questions.jsonl", HashingEmbedder())
    kwargs = {"min_score": 0.15}
    first, results = run_evaluation(*args, **kwargs)
    second, _ = run_evaluation(*args, **kwargs)
    assert first == second
    assert first["n_questions"] == len(results) >= 30
    assert first["n_unanswerable"] >= 5
    for key in ("retrieval_hit_at_k", "answer_correct", "citation_hit", "abstention_correct"):
        assert 0.0 <= first[key] <= 1.0
