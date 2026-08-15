from rag_autopsy.chunking import Chunk
from rag_autopsy.diagnostics import (
    CitationSupportDiagnosisType,
    diagnose_citation_support,
)
from rag_autopsy.generation import GroundedGenerationResult
from rag_autopsy.retrieval import SearchResult


def make_search_result(
    chunk_id: str,
    text: str,
) -> SearchResult:
    return SearchResult(
        score=0.9,
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc",
            text=text,
            start_word=0,
            end_word=20,
        ),
    )


def make_generation_result(
    answer: str,
    cited_chunk_ids=(),
) -> GroundedGenerationResult:
    return GroundedGenerationResult(
        answer=answer,
        cited_chunk_ids=cited_chunk_ids,
        invalid_citation_ids=(),
    )


def test_supported_claim_has_textual_support() -> None:
    result = diagnose_citation_support(
        generation_result=make_generation_result(
            answer=(
                "Reopen rates declined "
                "[doc::paragraph-0001]."
            ),
            cited_chunk_ids=(
                "doc::paragraph-0001",
            ),
        ),
        retrieval_results=[
            make_search_result(
                "doc::paragraph-0001",
                (
                    "Reopen rates declined over the "
                    "next two reporting periods."
                ),
            )
        ],
    )

    assert (
        result.diagnosis
        == CitationSupportDiagnosisType.SUPPORTED_BY_TEXT
    )

    assert result.unsupported_citation_ids == ()


def test_unsupported_claim_is_detected() -> None:
    result = diagnose_citation_support(
        generation_result=make_generation_result(
            answer=(
                "Revenue increased by forty percent "
                "[doc::paragraph-0001]."
            ),
            cited_chunk_ids=(
                "doc::paragraph-0001",
            ),
        ),
        retrieval_results=[
            make_search_result(
                "doc::paragraph-0001",
                (
                    "Reopen rates declined over the "
                    "next two reporting periods."
                ),
            )
        ],
    )

    assert (
        result.diagnosis
        == CitationSupportDiagnosisType.LOW_TEXTUAL_SUPPORT
    )

    assert result.unsupported_citation_ids == (
        "doc::paragraph-0001",
    )


def test_no_valid_citation_is_not_evaluated() -> None:
    result = diagnose_citation_support(
        generation_result=make_generation_result(
            answer="The result improved."
        ),
        retrieval_results=[],
    )

    assert (
        result.diagnosis
        == CitationSupportDiagnosisType.NO_VALID_CITATION
    )


def test_one_unsupported_citation_fails_multiple_claims() -> None:
    result = diagnose_citation_support(
        generation_result=make_generation_result(
            answer=(
                "Reopen rates declined "
                "[doc::paragraph-0001]. "
                "Revenue doubled "
                "[doc::paragraph-0002]."
            ),
            cited_chunk_ids=(
                "doc::paragraph-0001",
                "doc::paragraph-0002",
            ),
        ),
        retrieval_results=[
            make_search_result(
                "doc::paragraph-0001",
                "Reopen rates declined.",
            ),
            make_search_result(
                "doc::paragraph-0002",
                "Customer satisfaction remained stable.",
            ),
        ],
    )

    assert (
        result.diagnosis
        == CitationSupportDiagnosisType.LOW_TEXTUAL_SUPPORT
    )

    assert result.unsupported_citation_ids == (
        "doc::paragraph-0002",
    )
