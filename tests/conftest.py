from pathlib import Path

import pymupdf
import pytest

from rag_assistant.models import Chunk, RetrievedChunk

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_pdfs"


def make_pdf_bytes(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_bytes():
    return make_pdf_bytes(["First page about pumps.", "Second page about torque 65 N.m."])


@pytest.fixture
def sample_dir():
    return SAMPLE_DIR


def retrieved(text: str, score: float = 0.9, doc: str = "d.pdf", page: int = 1, rank: int = 1):
    return RetrievedChunk(Chunk(doc, page, 0, text), score, rank)


class ScriptedProvider:
    """Fake LLM: returns a fixed reply and records what it was sent."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply
