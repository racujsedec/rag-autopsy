import pytest

from rag_autopsy.chunking import Chunk
from rag_autopsy.evaluation import recall_at_k, reciprocal_rank
from rag_autopsy.retrieval import SearchResult


def make_result(
    chunk_id: str,
    score: float,
) -> SearchResult:
    document_id = chunk_id.split("::")[0]

    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text="example text",
        start_word=0,
        end_word=2,
    )

    return SearchResult(
        score=score,
        chunk=chunk,
    )


def test_recall_at_k_finds_relevant_chunk() -> None:
    results = [
        make_result("retail::chunk-0000", 3.0),
        make_result("finance::chunk-0000", 2.0),
        make_result("platform::chunk-0000", 1.0),
    ]

    assert recall_at_k(
        results,
        relevant_chunk_ids=["finance::chunk-0000"],
        k=2,
    ) == 1.0


def test_recall_at_k_misses_relevant_chunk() -> None:
    results = [
        make_result("retail::chunk-0000", 3.0),
        make_result("finance::chunk-0000", 2.0),
    ]

    assert recall_at_k(
        results,
        relevant_chunk_ids=["finance::chunk-0000"],
        k=1,
    ) == 0.0


def test_reciprocal_rank_for_first_result() -> None:
    results = [
        make_result("finance::chunk-0000", 3.0),
        make_result("retail::chunk-0000", 2.0),
    ]

    assert reciprocal_rank(
        results,
        ["finance::chunk-0000"],
    ) == 1.0


def test_reciprocal_rank_for_second_result() -> None:
    results = [
        make_result("retail::chunk-0000", 3.0),
        make_result("finance::chunk-0000", 2.0),
    ]

    assert reciprocal_rank(
        results,
        ["finance::chunk-0000"],
    ) == 0.5


def test_reciprocal_rank_returns_zero_when_missing() -> None:
    results = [
        make_result("retail::chunk-0000", 3.0),
    ]

    assert reciprocal_rank(
        results,
        ["finance::chunk-0000"],
    ) == 0.0


def test_recall_at_k_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k(
            [],
            relevant_chunk_ids=["finance::chunk-0000"],
            k=0,
        )
