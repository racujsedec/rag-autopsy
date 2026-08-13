import pytest
from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig

def test_chunker_creates_overlapping_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(25))
    chunker = FixedSizeChunker(ChunkingConfig(chunk_size=10, overlap=2))
    chunks = chunker.chunk("doc-1", text)
    assert len(chunks) == 3
    assert chunks[0].start_word == 0
    assert chunks[0].end_word == 10
    assert chunks[1].start_word == 8
    assert chunks[1].end_word == 18
    assert chunks[2].start_word == 16
    assert chunks[2].end_word == 25

def test_empty_document_returns_no_chunks() -> None:
    assert FixedSizeChunker().chunk("empty", "") == []

def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, overlap=10)
