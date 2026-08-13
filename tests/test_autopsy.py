from rag_autopsy.chunking import Chunk
from rag_autopsy.diagnostics import (
    DiagnosisType,
    diagnose_retrieval,
)
from rag_autopsy.retrieval import SearchResult


def make_result(
    chunk_id: str,
    score: float,
) -> SearchResult:
    document_id = chunk_id.split("::")[0]

    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text="example evidence",
        start_word=0,
        end_word=2,
    )

    return SearchResult(
        score=score,
        chunk=chunk,
    )


def test_success_when_relevant_chunk_is_ranked_first() -> None:
    results = [
        make_result("finance::chunk-0000", 3.0),
        make_result("retail::chunk-0000", 2.0),
    ]

    diagnosis = diagnose_retrieval(
        results,
        ["finance::chunk-0000"],
    )

    assert diagnosis.diagnosis == DiagnosisType.SUCCESS
    assert diagnosis.relevant_rank == 1


def test_ranking_failure_when_relevant_chunk_is_lower() -> None:
    results = [
        make_result("retail::chunk-0000", 3.0),
        make_result("finance::chunk-0000", 2.0),
    ]

    diagnosis = diagnose_retrieval(
        results,
        ["finance::chunk-0000"],
    )

    assert diagnosis.diagnosis == DiagnosisType.RANKING_FAILURE
    assert diagnosis.relevant_rank == 2


def test_retrieval_miss_when_evidence_is_missing() -> None:
    results = [
        make_result("retail::chunk-0000", 3.0),
        make_result("platform::chunk-0000", 2.0),
    ]

    diagnosis = diagnose_retrieval(
        results,
        ["finance::chunk-0000"],
    )

    assert diagnosis.diagnosis == DiagnosisType.RETRIEVAL_MISS
    assert diagnosis.relevant_rank is None
