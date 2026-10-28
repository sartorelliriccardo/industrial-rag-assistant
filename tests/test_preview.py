import pymupdf
import pytest

from conftest import make_pdf_bytes
from rag_assistant.generation import has_critical_values
from rag_assistant.preview import render_cited_page, select_sentences

PDF = make_pdf_bytes(
    ["First page.", "The impeller nut must be tightened to 120 N.m. Use a calibrated wrench."]
)


def test_cited_page_is_rendered_as_png_with_highlights():
    png, highlighted = render_cited_page(PDF, 2, "The impeller nut must be tightened to 120 N.m.")
    assert png.startswith(b"\x89PNG")
    assert highlighted >= 1


def test_text_not_on_the_page_gives_no_highlight_but_still_an_image():
    png, highlighted = render_cited_page(PDF, 1, "completely unrelated sentence here")
    assert png.startswith(b"\x89PNG") and highlighted == 0


def test_select_sentences_keeps_only_those_the_answer_used():
    passage = "Casing bolts need 65 N.m. The impeller nut needs 120 N.m. Use a wrench."
    chosen = select_sentences(passage, "The impeller nut must be tightened to 120 N.m.", 1)
    assert chosen == "The impeller nut needs 120 N.m."


def test_select_sentences_falls_back_to_the_whole_passage():
    passage = "Alpha beta. Gamma delta."
    assert select_sentences(passage, "zzz qqq") == passage


@pytest.mark.parametrize("page", [0, 3, -1])
def test_page_out_of_range_is_rejected(page):
    with pytest.raises(ValueError):
        render_cited_page(PDF, page, "x")


def test_original_pdf_bytes_are_not_modified():
    before = bytes(PDF)
    render_cited_page(PDF, 2, "The impeller nut")
    assert PDF == before
    assert len(pymupdf.open(stream=PDF, filetype="pdf")) == 2


@pytest.mark.parametrize(
    "text",
    [
        "tighten to 120 N.m",
        "max 10 bar",
        "up to 90 °C",
        "motor 22 kW",
        "5,2 m3/min and 8 bar",
        "185 kg",
    ],
)
def test_values_with_units_are_detected(text):
    assert has_critical_values(text)


@pytest.mark.parametrize("text", ["see page 3", "every 2000 hours", "Gate B", "5555", "10 apples"])
def test_plain_text_is_not_flagged(text):
    assert not has_critical_values(text)
