from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .autopsy import DiagnosisType
from .citation import CitationDiagnosisType
from .citation_support import CitationSupportDiagnosisType

if TYPE_CHECKING:
    from .report import RAGAutopsyReport


class RAGVerdictType(str, Enum):
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    RANKING_FAILURE = "RANKING_FAILURE"
    INVALID_CITATION = "INVALID_CITATION"
    LOW_TEXTUAL_SUPPORT = "LOW_TEXTUAL_SUPPORT"
    NO_CITATION = "NO_CITATION"
    SUCCESS = "SUCCESS"


@dataclass(frozen=True)
class RAGVerdictResult:
    diagnosis: RAGVerdictType
    explanation: str


def diagnose_rag_stages(
    retrieval,
    citations,
    citation_support,
) -> RAGVerdictResult:
    if retrieval.diagnosis == DiagnosisType.RETRIEVAL_MISS:
        return RAGVerdictResult(
            diagnosis=RAGVerdictType.RETRIEVAL_MISS,
            explanation=(
                "The primary failure occurred during retrieval: "
                "relevant evidence was not retrieved."
            ),
        )

    if retrieval.diagnosis == DiagnosisType.RANKING_FAILURE:
        return RAGVerdictResult(
            diagnosis=RAGVerdictType.RANKING_FAILURE,
            explanation=(
                "Relevant evidence was retrieved but ranked "
                "below the first position."
            ),
        )

    if (
        citations.diagnosis
        == CitationDiagnosisType.INVALID_CITATION
    ):
        return RAGVerdictResult(
            diagnosis=RAGVerdictType.INVALID_CITATION,
            explanation=(
                "The generated answer cited at least one chunk "
                "that was not present in the retrieved context."
            ),
        )

    if (
        citation_support.diagnosis
        == CitationSupportDiagnosisType.LOW_TEXTUAL_SUPPORT
    ):
        return RAGVerdictResult(
            diagnosis=RAGVerdictType.LOW_TEXTUAL_SUPPORT,
            explanation=(
                "At least one valid citation has insufficient "
                "textual support for its attached claim."
            ),
        )

    if (
        citations.diagnosis
        == CitationDiagnosisType.NO_CITATION
    ):
        return RAGVerdictResult(
            diagnosis=RAGVerdictType.NO_CITATION,
            explanation=(
                "Retrieval succeeded, but the generated answer "
                "did not cite retrieved evidence."
            ),
        )

    return RAGVerdictResult(
        diagnosis=RAGVerdictType.SUCCESS,
        explanation=(
            "Retrieval succeeded and the generated answer used "
            "valid citations with sufficient textual support."
        ),
    )


def diagnose_rag_verdict(
    report: "RAGAutopsyReport",
) -> RAGVerdictResult:
    return diagnose_rag_stages(
        retrieval=report.retrieval,
        citations=report.citations,
        citation_support=report.citation_support,
    )
