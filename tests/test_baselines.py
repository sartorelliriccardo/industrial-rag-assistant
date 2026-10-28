import pytest

from conftest import make_pdf_bytes
from rag_assistant.baselines import ctrl_f_pages, evaluate_ctrl_f, search_words
from rag_assistant.embeddings import LexicalEmbedder
from rag_assistant.evaluation import evaluate_question
from rag_assistant.llm import ExtractiveProvider
from rag_assistant.models import Page
from rag_assistant.pdf_loader import load_pdf_bytes
from rag_assistant.pipeline import RagPipeline

PAGES = [
    Page("a.pdf", 1, "The impeller nut torque is 120 N.m on the pump."),
    Page("a.pdf", 2, "Lubricate the bearings of the pump every 2000 hours."),
    Page("b.pdf", 1, "Fire assembly point is Gate B."),
]


def test_search_words_are_exact_content_words_without_duplicates():
    assert search_words("What is the torque of the torque nut?") == ["torque", "nut"]


def test_all_mode_requires_every_word_on_the_page():
    assert ctrl_f_pages("impeller nut torque", PAGES, "all") == [("a.pdf", 1)]
    assert ctrl_f_pages("impeller nut warranty", PAGES, "all") == []  # one word is missing


def test_exact_words_do_not_match_other_word_forms():
    assert ctrl_f_pages("lubricated bearings", PAGES, "all") == []  # page says "Lubricate"


def test_rarest_mode_uses_the_most_distinctive_word_that_exists():
    # "pump" is on two pages, "gate" on one: the person types the rarer word
    assert ctrl_f_pages("pump gate", PAGES, "rarest") == [("b.pdf", 1)]


def test_rarest_mode_ignores_words_absent_from_the_documents():
    assert ctrl_f_pages("warranty pump", PAGES, "rarest") == [("a.pdf", 1), ("a.pdf", 2)]


def test_no_known_word_means_no_results():
    assert ctrl_f_pages("warranty", PAGES, "rarest") == []


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        ctrl_f_pages("pump", PAGES, "fuzzy")


def test_evaluate_ctrl_f_counts_questions_needing_several_pages(tmp_path):
    (tmp_path / "a.pdf").write_bytes(
        make_pdf_bytes(["Isolate energy and lock out.", "Wear safety shoes and gloves."])
    )
    questions = [
        {  # the answer needs both pages; "lock" only appears on the first
            "answerable": True,
            "question": "lock",
            "gold": [{"document": "a.pdf", "page": 1}, {"document": "a.pdf", "page": 2}],
        },
        {"answerable": True, "question": "gloves", "gold": [{"document": "a.pdf", "page": 2}]},
        {"answerable": False, "question": "anything", "gold": []},
    ]
    result = evaluate_ctrl_f(tmp_path, questions, "rarest")
    assert result["n_answerable"] == 2  # the unanswerable one is not counted
    assert result["all_gold_found"] == 0.5
    assert result["n_pages_in_corpus"] == 2


def _pipeline(pages: list[str]) -> RagPipeline:
    pipeline = RagPipeline(LexicalEmbedder(), ExtractiveProvider(), chunk_size=300, min_score=0.05)
    pipeline.add_pdf_bytes(make_pdf_bytes(pages), "m.pdf")
    assert len(load_pdf_bytes(make_pdf_bytes(pages), "m.pdf")) == len(pages)
    return pipeline


def test_any_of_groups_need_one_alternative_each():
    pipeline = _pipeline(["The impeller nut torque is 120 N.m.", "Unrelated text about fire."])
    question = {
        "id": "x",
        "category": "t",
        "question": "What is the impeller nut torque?",
        "answerable": True,
        "expected_any_of": [["120", "twelve"], ["N.m", "Nm"]],
        "gold": [{"document": "m.pdf", "page": 1}],
    }
    result = evaluate_question(pipeline, question)
    assert result.correct is True and result.gold_recall == 1.0

    question["expected_any_of"] = [["120"], ["kilogram", "kg"]]  # second group not satisfied
    assert evaluate_question(pipeline, question).correct is False


def test_gold_recall_is_the_share_of_needed_pages_that_were_retrieved():
    pipeline = _pipeline(["The impeller nut torque is 120 N.m.", "Fire assembly point is Gate B."])
    question = {
        "id": "x",
        "category": "t",
        "question": "impeller nut torque",
        "answerable": True,
        "expected_any_of": [["120"]],
        "gold": [{"document": "m.pdf", "page": 1}, {"document": "m.pdf", "page": 2}],
    }
    pipeline.top_k = 1  # only one passage is retrieved: page 1 of the two pages the answer needs
    assert evaluate_question(pipeline, question).gold_recall == 0.5
    pipeline.top_k = 2
    assert evaluate_question(pipeline, question).gold_recall == 1.0
