"""Fixes found by testing a real 110-page manual: index pages, word weights, refits."""

import pytest

from conftest import make_pdf_bytes
from rag_assistant.embeddings import HashingEmbedder, LexicalEmbedder
from rag_assistant.llm import ExtractiveProvider
from rag_assistant.pipeline import RagPipeline
from rag_assistant.textutil import is_table_of_contents, stem

TOC_PAGE = "Contents " + " ".join(f"SECTION NUMBER{i} .......... {i + 3}" for i in range(10))
BODY = "Drain the condensate every day before starting work."


def test_an_index_page_is_recognised():
    assert is_table_of_contents(TOC_PAGE)


@pytest.mark.parametrize(
    "text",
    [
        BODY,
        "Short list: one ....... 1 and two ....... 2.",  # a couple of leaders is not an index
        "A line of dots ................ only",
        "",
    ],
)
def test_ordinary_pages_are_not_index_pages(text):
    assert not is_table_of_contents(text)


def test_index_pages_are_left_out_of_the_search_but_reported():
    pipeline = RagPipeline(LexicalEmbedder(), ExtractiveProvider(), chunk_size=800)
    pipeline.add_pdf_bytes(make_pdf_bytes([TOC_PAGE, BODY]), "m.pdf")
    assert {c.page_number for c in pipeline.chunks} == {2}
    assert pipeline.skipped_pages["m.pdf"] == [1]


def test_a_document_made_only_of_index_like_pages_is_not_dropped():
    pipeline = RagPipeline(LexicalEmbedder(), ExtractiveProvider(), chunk_size=800)
    pipeline.add_pdf_bytes(make_pdf_bytes([TOC_PAGE]), "toc_only.pdf")
    assert pipeline.chunk_count >= 1
    assert pipeline.skipped_pages["toc_only.pdf"] == []


def test_rare_words_weigh_more_after_fitting():
    embedder = LexicalEmbedder()
    embedder.fit(["dental unit one", "dental unit two", "dental unit three", "condensate cock"])
    weights = {term: w for term, _, w in embedder.describe("dental condensate")}
    assert weights[stem("condensate")] > weights[stem("dental")]


def test_before_fitting_every_word_weighs_the_same():
    weights = {term: w for term, _, w in LexicalEmbedder().describe("dental condensate")}
    assert weights[stem("condensate")] == weights[stem("dental")]


def test_fitting_does_not_change_the_vector_size_or_unit_length():
    embedder = LexicalEmbedder()
    embedder.fit(["alpha beta", "beta gamma"])
    vectors = embedder.embed(["alpha beta gamma", "unknown words only"])
    assert vectors.shape == (2, embedder.dim)
    assert abs(float((vectors[0] ** 2).sum()) - 1.0) < 1e-5


def test_word_weights_follow_the_documents_that_are_indexed():
    embedder = LexicalEmbedder()
    pipeline = RagPipeline(embedder, ExtractiveProvider(), chunk_size=800)
    pipeline.add_pdf_bytes(make_pdf_bytes(["alpha apple cider"]), "a.pdf")
    pipeline.add_pdf_bytes(make_pdf_bytes(["bravo banana cider"]), "b.pdf")
    assert stem("apple") in embedder._idf and stem("banana") in embedder._idf
    pipeline.remove("a.pdf")
    assert stem("apple") not in embedder._idf and stem("banana") in embedder._idf
    assert pipeline.documents == ["b.pdf"]


def test_replacing_a_document_does_not_duplicate_its_chunks():
    pipeline = RagPipeline(LexicalEmbedder(), ExtractiveProvider(), chunk_size=800)
    pipeline.add_pdf_bytes(make_pdf_bytes(["first version"]), "a.pdf")
    pipeline.add_pdf_bytes(make_pdf_bytes(["second version"]), "a.pdf")
    assert pipeline.chunk_count == 1 and "second" in pipeline.chunks[0].text


def test_removing_the_last_document_empties_the_pipeline():
    pipeline = RagPipeline(LexicalEmbedder(), ExtractiveProvider(), chunk_size=800)
    pipeline.add_pdf_bytes(make_pdf_bytes(["only document"]), "a.pdf")
    pipeline.remove("a.pdf")
    assert pipeline.chunk_count == 0
    with pytest.raises(ValueError):
        pipeline.search("anything")


def test_the_right_section_wins_over_pages_full_of_common_words():
    pages = [
        "The dental unit chair. The dental unit chair moves. Dental unit chair controls.",
        "The dental unit lamp. The dental unit lamp lights. Dental unit lamp controls.",
        "Dental unit handpieces. The dental unit handpiece. Dental unit handpiece sprays.",
        "Draining condensate: open the condensate drain cock of the dental unit each morning.",
    ]
    pipeline = RagPipeline(LexicalEmbedder(), ExtractiveProvider(), chunk_size=800, min_score=0.0)
    pipeline.add_pdf_bytes(make_pdf_bytes(pages), "m.pdf")
    assert (
        pipeline.search("How do I drain the condensate of the dental unit?")[0].chunk.page_number
        == 4
    )


def test_other_embedders_are_still_indexed_incrementally():
    pipeline = RagPipeline(HashingEmbedder(), ExtractiveProvider(), chunk_size=800)
    pipeline.add_pdf_bytes(make_pdf_bytes(["alpha"]), "a.pdf")
    pipeline.add_pdf_bytes(make_pdf_bytes(["bravo"]), "b.pdf")
    pipeline.remove("a.pdf")
    assert pipeline.documents == ["b.pdf"]
