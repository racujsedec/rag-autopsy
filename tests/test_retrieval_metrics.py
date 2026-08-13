from rag_autopsy.chunking import Chunk
from rag_autopsy.evaluation.retrieval_metrics import recall_at_k
from rag_autopsy.retrieval import SearchResult


def make_result(
    document_id: str,
    score: float,
) -> SearchResult:
    chunk = Chunk(
        chunk_id=f"{document_id}::chunk-0000",
        document_id=document_id,
        text="example text",
        start_word=0,
        end_word=2,
    )

    return SearchResult(
        score=score,
        chunk=chunk,
    )


def test_recall_at_k_finds_relevant_document() -> None:
    results = [
        make_result("retail", 3.0),
        make_result("finance", 2.0),
        make_result("platform", 1.0),
    ]

    assert recall_at_k(
        results,
        relevant_document="finance",
        k=2,
    ) == 1.0


def test_recall_at_k_misses_relevant_document() -> None:
    results = [
        make_result("retail", 3.0),
        make_result("finance", 2.0),
    ]

    assert recall_at_k(
        results,
        relevant_document="finance",
        k=1,
    ) == 0.0


def test_recall_at_k_rejects_invalid_k() -> None:
    results = []

    try:
        recall_at_k(
            results,
            relevant_document="finance",
            k=0,
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError")
