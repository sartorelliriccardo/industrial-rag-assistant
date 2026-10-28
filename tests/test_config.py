import pytest

from rag_assistant.config import get_chunk_settings


def test_default(monkeypatch):
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    assert get_chunk_settings() == (400, 80)


def test_values_from_environment(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    assert get_chunk_settings() == (500, 50)


def test_overlap_not_smaller_than_size_is_rejected(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    with pytest.raises(ValueError, match="smaller"):
        get_chunk_settings()
