from rag_autopsy.diagnostics import (
    CitationDiagnosisType,
    diagnose_citations,
)
from rag_autopsy.generation import GroundedGenerationResult


def make_result(
    cited_chunk_ids=(),
    invalid_citation_ids=(),
) -> GroundedGenerationResult:
    return GroundedGenerationResult(
        answer="Generated answer.",
        cited_chunk_ids=cited_chunk_ids,
        invalid_citation_ids=invalid_citation_ids,
    )


def test_valid_citation_is_classified() -> None:
    result = diagnose_citations(
        make_result(
            cited_chunk_ids=(
                "doc::paragraph-0001",
            )
        )
    )

    assert (
        result.diagnosis
        == CitationDiagnosisType.VALID_CITATION
    )

    assert result.cited_chunk_ids == (
        "doc::paragraph-0001",
    )


def test_missing_citation_is_classified() -> None:
    result = diagnose_citations(
        make_result()
    )

    assert (
        result.diagnosis
        == CitationDiagnosisType.NO_CITATION
    )


def test_invalid_citation_is_classified() -> None:
    result = diagnose_citations(
        make_result(
            invalid_citation_ids=(
                "doc::paragraph-9999",
            )
        )
    )

    assert (
        result.diagnosis
        == CitationDiagnosisType.INVALID_CITATION
    )

    assert result.invalid_citation_ids == (
        "doc::paragraph-9999",
    )


def test_invalid_citation_takes_priority_over_valid_citation() -> None:
    result = diagnose_citations(
        make_result(
            cited_chunk_ids=(
                "doc::paragraph-0001",
            ),
            invalid_citation_ids=(
                "doc::paragraph-9999",
            ),
        )
    )

    assert (
        result.diagnosis
        == CitationDiagnosisType.INVALID_CITATION
    )
