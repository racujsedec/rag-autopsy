import pytest

from rag_autopsy.chunking import Chunk
from rag_autopsy.retrieval import (
    SearchResult,
    reciprocal_rank_fusion,
)


def make_result(
    chunk_id: str,
    score: float,
) -> SearchResult:
    document_id = chunk_id.split("::")[0]

    return SearchResult(
        score=score,
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text="example",
            start_word=0,
            end_word=1,
        ),
    )


def test_rrf_rewards_results_found_by_both_retrievers() -> None:
    bm25 = [
        make_result("a::chunk-0000", 10.0),
        make_result("b::chunk-0000", 8.0),
    ]

    semantic = [
        make_result("b::chunk-0000", 0.9),
        make_result("c::chunk-0000", 0.8),
    ]

    results = reciprocal_rank_fusion(
        [bm25, semantic]
    )

    assert (
        results[0].chunk.chunk_id
        == "b::chunk-0000"
    )


def test_rrf_returns_unique_chunks() -> None:
    first = [
        make_result("a::chunk-0000", 5.0),
    ]

    second = [
        make_result("a::chunk-0000", 0.9),
    ]

    results = reciprocal_rank_fusion(
        [first, second]
    )

    assert len(results) == 1


def test_rrf_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [],
            rrf_k=0,
        )
