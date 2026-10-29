"""Configuration read from environment variables, plus small runtime helpers."""

import os

# 800 characters (~130 words): a procedure of a few steps fits in one chunk. Measured on a
# real 110-page manual: 400 gave 5 useful answers out of 9, 800 gave 6 (see docs/DETAILS.md).
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 5
# Minimum cosine similarity for a passage to count as evidence (tuned in eval, see docs/DETAILS.md).
DEFAULT_MIN_SCORE = 0.25  # semantic (neural) search
# Cosine scores are not comparable across embedders: each has its own sensible threshold.
# Keyword scores are lower since the word weights (IDF) were added: re-tuned from 0.10.
MIN_SCORE_BY_SEARCH = {"semantic": 0.25, "keyword": 0.05}
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
# Groq speaks the OpenAI "chat completions" dialect. Its models change often: llama-3.3-70b-
# versatile, my first default, was shut down on 16 Aug 2026 (Groq's deprecations page, which I
# had not read) and answered HTTP 404. gpt-oss-120b is the replacement Groq recommends.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def get_chunk_settings() -> tuple[int, int]:
    """Return (chunk_size, chunk_overlap) read from CHUNK_SIZE / CHUNK_OVERLAP.

    The chunker advances by (size - overlap) characters per step, so an overlap
    >= size would never move forward: we reject it up front.
    """
    size = int(os.environ.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    overlap = int(os.environ.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    if overlap >= size:
        raise ValueError(f"CHUNK_OVERLAP ({overlap}) must be smaller than CHUNK_SIZE ({size})")
    return size, overlap


def enable_system_certificates() -> None:
    """Use the OS certificate store for HTTPS if `truststore` is installed.

    Corporate networks often re-sign TLS traffic with a company CA that Python's
    bundled certificates do not know; this makes model downloads work there.
    """
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()
