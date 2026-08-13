import pytest

from rag_autopsy.chunking import Chunk
from rag_autopsy.evaluation import (
    evidence_coverage,
    resolve_ground_truth,
)


def make_chunk(
    chunk_id: str,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        text=text,
        start_word=0,
        end_word=len(text.split()),
    )


def test_full_evidence_has_complete_coverage() -> None:
    evidence = (
        "Engineers mitigated the incident "
        "by changing the partitioning strategy."
    )

    chunk = (
        "The cluster remained healthy. "
        "Engineers mitigated the incident "
        "by changing the partitioning strategy. "
        "Monitoring was added later."
    )

    assert evidence_coverage(
        chunk,
        evidence,
    ) == 1.0


def test_partial_evidence_has_partial_coverage() -> None:
    evidence = (
        "The company plans to focus on pricing "
        "discipline and procurement negotiations."
    )

    chunk = (
        "The company plans to focus on pricing "
        "discipline"
    )

    coverage = evidence_coverage(
        chunk,
        evidence,
    )

    assert 0.0 < coverage < 1.0


def test_resolve_ground_truth_selects_best_chunk() -> None:
    evidence = (
        "Engineers mitigated the incident "
        "by changing the partitioning strategy."
    )

    chunks = [
        make_chunk(
            "doc::chunk-0000",
            "The incident caused query latency.",
        ),
        make_chunk(
            "doc::chunk-0001",
            (
                "Engineers mitigated the incident "
                "by changing the partitioning strategy."
            ),
        ),
    ]

    result = resolve_ground_truth(
        chunks,
        evidence,
    )

    assert result.relevant_chunk_ids == [
        "doc::chunk-0001"
    ]

    assert result.max_coverage == 1.0
    assert result.complete_evidence_preserved is True


def test_split_evidence_is_not_marked_complete() -> None:
    evidence = (
        "The operations team increased safety stock "
        "and added an exception queue."
    )

    chunks = [
        make_chunk(
            "doc::chunk-0000",
            "The operations team increased safety stock",
        ),
        make_chunk(
            "doc::chunk-0001",
            "and added an exception queue",
        ),
    ]

    result = resolve_ground_truth(
        chunks,
        evidence,
    )

    assert result.max_coverage < 1.0
    assert result.complete_evidence_preserved is False


def test_invalid_support_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_ground_truth(
            [],
            "evidence",
            support_threshold=1.5,
        )
