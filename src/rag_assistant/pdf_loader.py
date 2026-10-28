"""PDF -> list of pages with document name and page number."""

from pathlib import Path

import pymupdf

from rag_assistant.models import Page


class PdfError(Exception):
    """Base class for PDF loading problems."""


class PdfNotFoundError(PdfError):
    pass


class PdfCorruptedError(PdfError):
    pass


class PdfNoTextError(PdfError):
    """The PDF opens fine but contains no extractable text (e.g. a scan: it needs OCR)."""


def _normalize(text: str) -> str:
    # PDF line breaks are layout, not meaning: collapse all whitespace runs.
    return " ".join(text.split())


def load_pdf_bytes(data: bytes, doc_name: str) -> list[Page]:
    """Extract the text of every non-empty page. Page numbers are 1-based."""
    if not data:
        raise PdfCorruptedError(f"{doc_name}: the file is empty")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types for bad files
        raise PdfCorruptedError(f"{doc_name}: cannot open as PDF ({exc})") from exc

    pages: list[Page] = []
    with doc:
        if doc.needs_pass:
            raise PdfCorruptedError(f"{doc_name}: the PDF is password protected")
        try:
            for index, page in enumerate(doc, start=1):
                text = _normalize(page.get_text())
                if text:
                    pages.append(Page(doc_name=doc_name, page_number=index, text=text))
        except Exception as exc:
            raise PdfCorruptedError(f"{doc_name}: failed while reading pages ({exc})") from exc

    if not pages:
        raise PdfNoTextError(
            f"{doc_name}: no extractable text found (scanned PDF? OCR is not supported)"
        )
    return pages


def load_pdf(path: str | Path) -> list[Page]:
    path = Path(path)
    if not path.is_file():
        raise PdfNotFoundError(f"{path}: file not found")
    return load_pdf_bytes(path.read_bytes(), doc_name=path.name)
