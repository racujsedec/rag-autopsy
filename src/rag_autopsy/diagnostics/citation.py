from dataclasses import dataclass
from enum import Enum

from rag_autopsy.generation import GroundedGenerationResult


class CitationDiagnosisType(str, Enum):
    VALID_CITATION = "VALID_CITATION"
    NO_CITATION = "NO_CITATION"
    INVALID_CITATION = "INVALID_CITATION"


@dataclass(frozen=True)
class CitationAutopsyResult:
    diagnosis: CitationDiagnosisType
    cited_chunk_ids: tuple[str, ...]
    invalid_citation_ids: tuple[str, ...]
    explanation: str


def diagnose_citations(
    generation_result: GroundedGenerationResult,
) -> CitationAutopsyResult:
    if generation_result.invalid_citation_ids:
        return CitationAutopsyResult(
            diagnosis=CitationDiagnosisType.INVALID_CITATION,
            cited_chunk_ids=generation_result.cited_chunk_ids,
            invalid_citation_ids=generation_result.invalid_citation_ids,
            explanation=(
                "The generated answer contains one or more "
                "citations that were not present in the "
                "retrieved context."
            ),
        )

    if not generation_result.cited_chunk_ids:
        return CitationAutopsyResult(
            diagnosis=CitationDiagnosisType.NO_CITATION,
            cited_chunk_ids=(),
            invalid_citation_ids=(),
            explanation=(
                "The generated answer does not cite any "
                "retrieved chunks."
            ),
        )

    return CitationAutopsyResult(
        diagnosis=CitationDiagnosisType.VALID_CITATION,
        cited_chunk_ids=generation_result.cited_chunk_ids,
        invalid_citation_ids=(),
        explanation=(
            "All cited chunk IDs were present in the "
            "retrieved context."
        ),
    )
