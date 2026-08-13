import pytest

from rag_autopsy.chunking import ParagraphChunker


def test_preserves_paragraph_boundaries() -> None:
    text = (
        "First paragraph contains one complete idea.\n\n"
        "Second paragraph contains another complete idea."
    )

    chunker = ParagraphChunker(
        max_words=50
    )

    chunks = chunker.chunk(
        "doc",
        text,
    )

    assert len(chunks) == 2

    assert chunks[0].text == (
        "First paragraph contains one complete idea."
    )

    assert chunks[1].text == (
        "Second paragraph contains another complete idea."
    )


def test_large_paragraph_is_split() -> None:
    text = " ".join(
        f"word{i}"
        for i in range(25)
    )

    chunker = ParagraphChunker(
        max_words=10
    )

    chunks = chunker.chunk(
        "doc",
        text,
    )

    assert len(chunks) == 3


def test_empty_document_returns_no_chunks() -> None:
    chunker = ParagraphChunker()

    assert chunker.chunk(
        "empty",
        ""
    ) == []


def test_invalid_max_words_is_rejected() -> None:
    with pytest.raises(ValueError):
        ParagraphChunker(
            max_words=0
        )
