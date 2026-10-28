"""Split page text into overlapping chunks, keeping source metadata."""

from rag_assistant.models import Chunk, Page


def split_page(page: Page, chunk_size: int, overlap: int) -> list[Chunk]:
    """Sliding window over one page, cutting on whitespace instead of mid-word.

    Overlap repeats the tail of a chunk at the head of the next one, so a fact that
    straddles a boundary is still fully contained in at least one chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = page.text
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # move the cut back to the last space, but not too far back
            cut = text.rfind(" ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    chunk_index=len(chunks),
                    text=piece,
                )
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always move forward
        if text[start - 1] != " ":  # do not start in the middle of a word
            next_space = text.find(" ", start, end)
            if next_space != -1:
                start = next_space + 1
    return chunks


def split_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    return [chunk for page in pages for chunk in split_page(page, chunk_size, overlap)]
