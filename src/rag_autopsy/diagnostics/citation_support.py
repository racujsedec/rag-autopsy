import re
from dataclasses import dataclass
from enum import Enum

from rag_autopsy.generation import GroundedGenerationResult
from rag_autopsy.retrieval import SearchResult


class CitationSupportDiagnosisType(str, Enum):
    SUPPORTED_BY_TEXT = "SUPPORTED_BY_TEXT"
    LOW_TEXTUAL_SUPPORT = "LOW_TEXTUAL_SUPPORT"
    NO_VALID_CITATION = "NO_VALID_CITATION"


@dataclass(frozen=True)
class CitationSupportAutopsyResult:
    diagnosis: CitationSupportDiagnosisType
    supported_citation_ids: tuple[str, ...]
    unsupported_citation_ids: tuple[str, ...]
    support_scores: tuple[tuple[str, float], ...]
    explanation: str


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _tokens(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text,
        )
    }

    return tokens - _STOP_WORDS


def _claim_before_citation(
    answer: str,
    citation_id: str,
) -> str:
    marker = f"[{citation_id}]"
    position = answer.find(marker)

    if position == -1:
        return ""

    prefix = answer[:position]

    boundaries = [
        prefix.rfind("."),
        prefix.rfind("!"),
        prefix.rfind("?"),
    ]

    start = max(boundaries) + 1

    return prefix[start:].strip()


def _textual_support_score(
    claim: str,
    evidence: str,
) -> float:
    claim_tokens = _tokens(claim)

    if not claim_tokens:
        return 0.0

    evidence_tokens = _tokens(evidence)

    return (
        len(claim_tokens & evidence_tokens)
        / len(claim_tokens)
    )


def diagnose_citation_support(
    generation_result: GroundedGenerationResult,
    retrieval_results: list[SearchResult],
    support_threshold: float = 0.50,
) -> CitationSupportAutopsyResult:
    if not generation_result.cited_chunk_ids:
        return CitationSupportAutopsyResult(
            diagnosis=(
                CitationSupportDiagnosisType.NO_VALID_CITATION
            ),
            supported_citation_ids=(),
            unsupported_citation_ids=(),
            support_scores=(),
            explanation=(
                "No valid retrieved chunk citation was "
                "available for textual support evaluation."
            ),
        )

    retrieved_by_id = {
        result.chunk.chunk_id: result.chunk
        for result in retrieval_results
    }

    supported = []
    unsupported = []
    scores = []

    for citation_id in generation_result.cited_chunk_ids:
        chunk = retrieved_by_id.get(
            citation_id
        )

        if chunk is None:
            score = 0.0
        else:
            claim = _claim_before_citation(
                generation_result.answer,
                citation_id,
            )

            score = _textual_support_score(
                claim,
                chunk.text,
            )

        scores.append(
            (
                citation_id,
                score,
            )
        )

        if score >= support_threshold:
            supported.append(
                citation_id
            )
        else:
            unsupported.append(
                citation_id
            )

    if unsupported:
        return CitationSupportAutopsyResult(
            diagnosis=(
                CitationSupportDiagnosisType.LOW_TEXTUAL_SUPPORT
            ),
            supported_citation_ids=tuple(supported),
            unsupported_citation_ids=tuple(unsupported),
            support_scores=tuple(scores),
            explanation=(
                "One or more cited chunks have low lexical "
                "support for the claims attached to them."
            ),
        )

    return CitationSupportAutopsyResult(
        diagnosis=(
            CitationSupportDiagnosisType.SUPPORTED_BY_TEXT
        ),
        supported_citation_ids=tuple(supported),
        unsupported_citation_ids=(),
        support_scores=tuple(scores),
        explanation=(
            "All cited claims have sufficient textual "
            "overlap with their cited retrieved chunks."
        ),
    )
