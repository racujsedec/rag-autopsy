from rag_autopsy.diagnostics.citation_coverage import (
    CitationCoverageDiagnosisType,
    diagnose_citation_coverage,
)
from rag_autopsy.generation import GroundedGenerationResult


def test_multi_claim_answer_with_one_citation_has_partial_coverage() -> None:
    generation = GroundedGenerationResult(
        answer=(
            "Vector Systems introduced three service-desk changes:\n\n"
            "- Made resolution codes mandatory.\n"
            "- Delayed automatic closure when customers had replied recently.\n"
            "- Added weekly supervisor audits of reopened tickets.\n\n"
            "Reopen rates declined over the next two reporting periods. "
            "[service_desk::paragraph-0001]"
        ),
        cited_chunk_ids=(
            "service_desk::paragraph-0001",
        ),
        invalid_citation_ids=(),
    )

    result = diagnose_citation_coverage(
        generation
    )

    assert (
        result.diagnosis
        == CitationCoverageDiagnosisType.PARTIAL_CITATION_COVERAGE
    )
    assert result.total_claims == 4
    assert result.cited_claims == 1
    assert result.coverage == 0.25


def test_all_claims_cited_has_complete_coverage() -> None:
    generation = GroundedGenerationResult(
        answer=(
            "Resolution codes became mandatory. "
            "[doc::paragraph-0001]\n"
            "Reopen rates declined "
            "[doc::paragraph-0001]."
        ),
        cited_chunk_ids=(
            "doc::paragraph-0001",
        ),
        invalid_citation_ids=(),
    )

    result = diagnose_citation_coverage(
        generation
    )

    assert (
        result.diagnosis
        == CitationCoverageDiagnosisType.COMPLETE_CITATION_COVERAGE
    )
    assert result.total_claims == 2
    assert result.cited_claims == 2
    assert result.coverage == 1.0
    assert result.uncited_claims == ()


def test_uncited_claims_have_zero_coverage() -> None:
    generation = GroundedGenerationResult(
        answer=(
            "Resolution codes became mandatory. "
            "Reopen rates declined."
        ),
        cited_chunk_ids=(),
        invalid_citation_ids=(),
    )

    result = diagnose_citation_coverage(
        generation
    )

    assert (
        result.diagnosis
        == CitationCoverageDiagnosisType.NO_CITATION_COVERAGE
    )
    assert result.total_claims == 2
    assert result.cited_claims == 0
    assert result.coverage == 0.0


def test_citation_on_next_line_attaches_to_previous_claim() -> None:
    generation = GroundedGenerationResult(
        answer=(
            "Reopen rates declined.\n"
            "[doc::paragraph-0001]"
        ),
        cited_chunk_ids=(
            "doc::paragraph-0001",
        ),
        invalid_citation_ids=(),
    )

    result = diagnose_citation_coverage(
        generation
    )

    assert (
        result.diagnosis
        == CitationCoverageDiagnosisType.COMPLETE_CITATION_COVERAGE
    )
    assert result.total_claims == 1
    assert result.cited_claims == 1
    assert result.coverage == 1.0
