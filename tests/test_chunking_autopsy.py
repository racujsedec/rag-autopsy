from rag_autopsy.chunking import Chunk
from rag_autopsy.diagnostics import (
    ChunkingDiagnosisType,
    diagnose_chunking,
)
from rag_autopsy.evaluation import GroundTruthResult
from rag_autopsy.retrieval import SearchResult


def make_result(
    chunk_id: str,
) -> SearchResult:
    document_id = chunk_id.split("::")[0]

    return SearchResult(
        score=1.0,
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text="example evidence",
            start_word=0,
            end_word=2,
        ),
    )


def test_complete_evidence_is_detected() -> None:
    ground_truth = GroundTruthResult(
        relevant_chunk_ids=["doc::chunk-0000"],
        supporting_chunk_ids=["doc::chunk-0000"],
        max_coverage=1.0,
        complete_evidence_preserved=True,
    )

    result = diagnose_chunking(
        ground_truth,
        [make_result("doc::chunk-0000")],
    )

    assert (
        result.diagnosis
        == ChunkingDiagnosisType.COMPLETE_EVIDENCE
    )


def test_fragmented_evidence_with_retrieval_is_risk() -> None:
    ground_truth = GroundTruthResult(
        relevant_chunk_ids=["doc::chunk-0001"],
        supporting_chunk_ids=["doc::chunk-0001"],
        max_coverage=0.90,
        complete_evidence_preserved=False,
    )

    result = diagnose_chunking(
        ground_truth,
        [
            make_result("doc::chunk-0000"),
            make_result("doc::chunk-0001"),
        ],
    )

    assert (
        result.diagnosis
        == ChunkingDiagnosisType.CHUNK_BOUNDARY_RISK
    )


def test_fragmented_evidence_without_retrieval_is_failure() -> None:
    ground_truth = GroundTruthResult(
        relevant_chunk_ids=["doc::chunk-0001"],
        supporting_chunk_ids=["doc::chunk-0001"],
        max_coverage=0.90,
        complete_evidence_preserved=False,
    )

    result = diagnose_chunking(
        ground_truth,
        [
            make_result("other::chunk-0000"),
        ],
    )

    assert (
        result.diagnosis
        == ChunkingDiagnosisType.CHUNK_BOUNDARY_FAILURE
    )
