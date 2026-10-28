import numpy as np
import pytest

from conftest import make_pdf_bytes
from rag_assistant.config import MIN_SCORE_BY_SEARCH
from rag_assistant.embeddings import LexicalEmbedder
from rag_assistant.llm import ExtractiveProvider
from rag_assistant.pipeline import RagPipeline
from rag_assistant.textutil import content_tokens, stem


def test_stemmer_merges_word_forms_in_english_and_italian():
    assert stem("lubricated") == stem("lubricate")
    assert stem("inspected") == stem("inspect")
    assert stem("sostituito") == stem("sostituire")


def test_values_and_part_numbers_are_kept_whole():
    assert stem("120") == "120"
    assert stem("m12bolts") == "m12bolts"


def test_stopwords_and_short_words_are_dropped():
    assert content_tokens("What is the torque of the nut?") == ["torque", "nut"]


def test_keyword_vectors_are_unit_length_and_deterministic():
    e = LexicalEmbedder()
    a, b = e.embed(["tighten the impeller nut"]), e.embed(["tighten the impeller nut"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, b)


def test_text_without_content_words_gives_a_zero_vector():
    assert not LexicalEmbedder().embed(["is of the a"]).any()


def test_word_forms_match_each_other():
    e = LexicalEmbedder()
    question, passage, other = e.embed(
        [
            "how often is the bearing lubricated",
            "Lubricate the bearing every 2000 hours",
            "fire gate",
        ]
    )
    assert question @ passage > 0.4 > question @ other


def test_repeating_a_word_does_not_dominate():
    e = LexicalEmbedder()
    once, many = e.embed(["torque nut", "torque " * 50 + "nut"])
    assert once @ many > 0.7


@pytest.mark.parametrize("search", ["keyword", "semantic"])
def test_each_search_mode_has_a_threshold(search):
    assert 0 < MIN_SCORE_BY_SEARCH[search] < 1


def test_end_to_end_with_keyword_search_and_stemmed_question():
    pipeline = RagPipeline(
        LexicalEmbedder(),
        ExtractiveProvider(),
        chunk_size=300,
        min_score=MIN_SCORE_BY_SEARCH["keyword"],
    )
    pipeline.add_pdf_bytes(
        make_pdf_bytes(["Intro page.", "Lubricate the bearings every 2000 operating hours."]),
        "pump.pdf",
    )
    answer = pipeline.ask("How often should the bearings be lubricated?")
    assert answer.status == "answered"
    assert "2000" in answer.text
    assert [(c.doc_name, c.page_number) for c in answer.citations] == [("pump.pdf", 2)]


def test_keyword_search_declines_an_unrelated_question():
    pipeline = RagPipeline(
        LexicalEmbedder(), ExtractiveProvider(), min_score=MIN_SCORE_BY_SEARCH["keyword"]
    )
    pipeline.add_pdf_bytes(make_pdf_bytes(["The impeller nut torque is 120 N.m."]), "pump.pdf")
    assert pipeline.ask("What is the capital of France?").status == "insufficient"
