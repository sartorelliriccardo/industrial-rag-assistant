import pytest

from rag_assistant.chunking import split_page, split_pages
from rag_assistant.models import Page

TEXT = " ".join(f"word{i}" for i in range(200))


def page(text=TEXT, number=3):
    return Page("m.pdf", number, text)


def test_short_page_is_a_single_chunk():
    chunks = split_page(page("short text"), 400, 80)
    assert [c.text for c in chunks] == ["short text"]


def test_chunks_respect_size_and_keep_metadata():
    chunks = split_page(page(), 100, 20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.doc_name == "m.pdf" and c.page_number == 3 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    chunks = split_page(page(), 100, 30)
    first_tail = set(chunks[0].text.split()[-3:])
    assert first_tail & set(chunks[1].text.split())


def test_no_word_is_cut_in_half_and_nothing_is_lost():
    chunks = split_page(page(), 100, 20)
    words = {w for c in chunks for w in c.text.split()}
    assert words == set(TEXT.split())


def test_always_terminates_even_with_one_long_word():
    chunks = split_page(page("x" * 1000), 100, 99)
    assert chunks and all(len(c.text) <= 100 for c in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_page(page(), size, overlap)


def test_chunks_never_cross_pages():
    chunks = split_pages([page("alpha beta", 1), page("gamma delta", 2)], 400, 80)
    assert [(c.page_number, c.text) for c in chunks] == [(1, "alpha beta"), (2, "gamma delta")]
