from rag_autopsy.chunking import (
    Chunk,
    PreviousChunkContextEnricher,
)


def make_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    start_word: int,
    end_word: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        start_word=start_word,
        end_word=end_word,
    )


def test_first_chunk_in_document_is_unchanged() -> None:
    chunk = make_chunk(
        "doc::paragraph-0000",
        "doc",
        "Vector Systems analyzed service-desk tickets.",
        0,
        5,
    )

    enricher = PreviousChunkContextEnricher()

    enriched = enricher.enrich([chunk])

    assert len(enriched) == 1
    assert enriched[0].text == chunk.text


def test_previous_chunk_text_is_added_to_next_chunk() -> None:
    first = make_chunk(
        "doc::paragraph-0000",
        "doc",
        "Vector Systems analyzed service-desk tickets.",
        0,
        5,
    )

    second = make_chunk(
        "doc::paragraph-0001",
        "doc",
        "The team changed resolution-code controls.",
        5,
        10,
    )

    enricher = PreviousChunkContextEnricher()

    enriched = enricher.enrich(
        [first, second]
    )

    assert enriched[1].text == (
        "Vector Systems analyzed service-desk tickets."
        "\n\n"
        "The team changed resolution-code controls."
    )


def test_chunk_identity_and_boundaries_are_preserved() -> None:
    first = make_chunk(
        "doc::paragraph-0000",
        "doc",
        "First paragraph.",
        0,
        2,
    )

    second = make_chunk(
        "doc::paragraph-0001",
        "doc",
        "Second paragraph.",
        2,
        4,
    )

    enricher = PreviousChunkContextEnricher()

    enriched = enricher.enrich(
        [first, second]
    )

    assert enriched[1].chunk_id == second.chunk_id
    assert enriched[1].document_id == second.document_id
    assert enriched[1].start_word == second.start_word
    assert enriched[1].end_word == second.end_word


def test_context_does_not_cross_document_boundaries() -> None:
    doc_a = make_chunk(
        "doc_a::paragraph-0000",
        "doc_a",
        "Document A context.",
        0,
        3,
    )

    doc_b = make_chunk(
        "doc_b::paragraph-0000",
        "doc_b",
        "Document B answer.",
        0,
        3,
    )

    enricher = PreviousChunkContextEnricher()

    enriched = enricher.enrich(
        [doc_a, doc_b]
    )

    assert enriched[1].text == "Document B answer."
