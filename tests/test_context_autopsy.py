from rag_autopsy.chunking import Chunk
from rag_autopsy.diagnostics import (
    ContextDiagnosisType,
    diagnose_context,
)
from rag_autopsy.evaluation import GroundTruthResult
from rag_autopsy.retrieval import SearchResult


def make_chunk(
    chunk_id: str,
    text: str,
    start_word: int,
    end_word: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split("::")[0],
        text=text,
        start_word=start_word,
        end_word=end_word,
    )


def make_result(
    chunk: Chunk,
) -> SearchResult:
    return SearchResult(
        score=1.0,
        chunk=chunk,
    )


def test_context_is_intact_when_relevant_chunk_is_retrieved() -> None:
    context_chunk = make_chunk(
        "doc::paragraph-0000",
        "Vector Systems reviewed service desk tickets.",
        0,
        7,
    )

    evidence_chunk = make_chunk(
        "doc::paragraph-0001",
        (
            "Vector Systems changed service desk controls. "
            "Reopen rates declined afterward."
        ),
        7,
        17,
    )

    ground_truth = GroundTruthResult(
        relevant_chunk_ids=["doc::paragraph-0001"],
        supporting_chunk_ids=["doc::paragraph-0001"],
        max_coverage=1.0,
        complete_evidence_preserved=True,
    )

    result = diagnose_context(
        question=(
            "What service desk changes did Vector Systems "
            "introduce, and what was the result?"
        ),
        chunks=[
            context_chunk,
            evidence_chunk,
        ],
        ground_truth=ground_truth,
        retrieval_results=[
            make_result(evidence_chunk),
        ],
    )

    assert (
        result.diagnosis
        == ContextDiagnosisType.CONTEXT_INTACT
    )


def test_context_loss_is_detected_when_anchors_are_in_neighbor() -> None:
    context_chunk = make_chunk(
        "doc::paragraph-0000",
        (
            "Vector Systems analyzed why service desk "
            "tickets were being reopened."
        ),
        0,
        10,
    )

    evidence_chunk = make_chunk(
        "doc::paragraph-0001",
        (
            "The team made resolution codes mandatory "
            "and delayed automatic closure. "
            "Reopen rates declined afterward."
        ),
        10,
        24,
    )

    ground_truth = GroundTruthResult(
        relevant_chunk_ids=["doc::paragraph-0001"],
        supporting_chunk_ids=["doc::paragraph-0001"],
        max_coverage=1.0,
        complete_evidence_preserved=True,
    )

    result = diagnose_context(
        question=(
            "What service desk changes did Vector Systems "
            "introduce, and what was the result?"
        ),
        chunks=[
            context_chunk,
            evidence_chunk,
        ],
        ground_truth=ground_truth,
        retrieval_results=[
            make_result(context_chunk),
        ],
    )

    assert (
        result.diagnosis
        == ContextDiagnosisType.CHUNK_CONTEXT_LOSS
    )


def test_context_loss_is_not_reported_without_neighbor_anchors() -> None:
    unrelated_chunk = make_chunk(
        "doc::paragraph-0000",
        "Quarterly reporting remained on schedule.",
        0,
        5,
    )

    evidence_chunk = make_chunk(
        "doc::paragraph-0001",
        (
            "The team made resolution codes mandatory "
            "and reopen rates declined."
        ),
        5,
        15,
    )

    ground_truth = GroundTruthResult(
        relevant_chunk_ids=["doc::paragraph-0001"],
        supporting_chunk_ids=["doc::paragraph-0001"],
        max_coverage=1.0,
        complete_evidence_preserved=True,
    )

    result = diagnose_context(
        question=(
            "What service desk changes did Vector Systems "
            "introduce, and what was the result?"
        ),
        chunks=[
            unrelated_chunk,
            evidence_chunk,
        ],
        ground_truth=ground_truth,
        retrieval_results=[
            make_result(unrelated_chunk),
        ],
    )

    assert (
        result.diagnosis
        == ContextDiagnosisType.CONTEXT_INTACT
    )
