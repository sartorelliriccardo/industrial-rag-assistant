import pytest

from conftest import make_pdf_bytes
from rag_assistant.pdf_loader import (
    PdfCorruptedError,
    PdfNoTextError,
    PdfNotFoundError,
    load_pdf,
    load_pdf_bytes,
)


def test_extracts_text_with_document_name_and_page_numbers(pdf_bytes):
    pages = load_pdf_bytes(pdf_bytes, "manual.pdf")
    assert [(p.doc_name, p.page_number) for p in pages] == [("manual.pdf", 1), ("manual.pdf", 2)]
    assert "pumps" in pages[0].text
    assert "65 N.m" in pages[1].text


def test_blank_page_is_skipped_but_numbering_is_preserved():
    data = make_pdf_bytes(["page one", "", "page three"])
    assert [p.page_number for p in load_pdf_bytes(data, "x.pdf")] == [1, 3]


def test_line_breaks_are_collapsed():
    text = load_pdf_bytes(make_pdf_bytes(["word " * 80]), "x.pdf")[0].text
    assert "\n" not in text


def test_pdf_without_text_raises_no_text_error():
    with pytest.raises(PdfNoTextError):
        load_pdf_bytes(make_pdf_bytes(["", ""]), "scan.pdf")


@pytest.mark.parametrize("data", [b"", b"this is not a pdf", b"%PDF-1.7 truncated garbage"])
def test_invalid_bytes_raise_corrupted_error(data):
    with pytest.raises(PdfCorruptedError):
        load_pdf_bytes(data, "broken.pdf")


def test_missing_file_raises_not_found(tmp_path):
    with pytest.raises(PdfNotFoundError):
        load_pdf(tmp_path / "nope.pdf")


def test_load_pdf_uses_file_name_as_document_name(tmp_path, pdf_bytes):
    path = tmp_path / "pump.pdf"
    path.write_bytes(pdf_bytes)
    assert load_pdf(path)[0].doc_name == "pump.pdf"
