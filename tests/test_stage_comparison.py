from rag_autopsy.chunking import Chunk
from rag_autopsy.diagnostics import (
    StageComparisonType,
    compare_reranking_stages,
)
from rag_autopsy.retrieval import SearchResult


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


def test_reranker_improvement() -> None:
    before = [
        make_result("wrong::chunk-0000", 3.0),
        make_result("correct::chunk-0000", 2.0),
    ]

    after = [
        make_result("correct::chunk-0000", 5.0),
        make_result("wrong::chunk-0000", 1.0),
    ]

    result = compare_reranking_stages(
        before,
        after,
        ["correct::chunk-0000"],
    )

    assert (
        result.diagnosis
        == StageComparisonType.RERANKER_IMPROVEMENT
    )

    assert result.before_rank == 2
    assert result.after_rank == 1


def test_reranker_regression() -> None:
    before = [
        make_result("correct::chunk-0000", 3.0),
        make_result("wrong::chunk-0000", 2.0),
    ]

    after = [
        make_result("wrong::chunk-0000", 5.0),
        make_result("correct::chunk-0000", 1.0),
    ]

    result = compare_reranking_stages(
        before,
        after,
        ["correct::chunk-0000"],
    )

    assert (
        result.diagnosis
        == StageComparisonType.RERANKER_REGRESSION
    )

    assert result.before_rank == 1
    assert result.after_rank == 2


def test_no_change() -> None:
    before = [
        make_result("correct::chunk-0000", 3.0),
    ]

    after = [
        make_result("correct::chunk-0000", 9.0),
    ]

    result = compare_reranking_stages(
        before,
        after,
        ["correct::chunk-0000"],
    )

    assert (
        result.diagnosis
        == StageComparisonType.NO_CHANGE
    )

    assert result.before_rank == 1
    assert result.after_rank == 1


def test_retrieval_miss_to_retrieval_hit_is_improvement() -> None:
    before = [
        make_result("wrong::chunk-0000", 3.0),
    ]

    after = [
        make_result("correct::chunk-0000", 5.0),
    ]

    result = compare_reranking_stages(
        before,
        after,
        ["correct::chunk-0000"],
    )

    assert (
        result.diagnosis
        == StageComparisonType.RERANKER_IMPROVEMENT
    )

    assert result.before_rank is None
    assert result.after_rank == 1
