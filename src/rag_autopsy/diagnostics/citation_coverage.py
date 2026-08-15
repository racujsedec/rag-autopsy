import re
from dataclasses import dataclass
from enum import Enum

from rag_autopsy.generation import GroundedGenerationResult


class CitationCoverageDiagnosisType(str, Enum):
    COMPLETE_CITATION_COVERAGE = "COMPLETE_CITATION_COVERAGE"
    PARTIAL_CITATION_COVERAGE = "PARTIAL_CITATION_COVERAGE"
    NO_CITATION_COVERAGE = "NO_CITATION_COVERAGE"
    NO_CLAIMS = "NO_CLAIMS"


@dataclass(frozen=True)
class CitationCoverageAutopsyResult:
    diagnosis: CitationCoverageDiagnosisType
    total_claims: int
    cited_claims: int
    coverage: float
    uncited_claims: tuple[str, ...]
    explanation: str


_CITATION_PATTERN = re.compile(
    r"\[[^\[\]]+\]"
)


def _extract_claims(
    answer: str,
) -> list[tuple[str, bool]]:
    claims = []

    for raw_line in answer.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if _CITATION_PATTERN.fullmatch(line):
            if claims:
                claim, _ = claims[-1]
                claims[-1] = (claim, True)
            continue

        is_bullet = bool(
            re.match(r"^[-*•]\s+", line)
        )

        if is_bullet:
            line = re.sub(
                r"^[-*•]\s+",
                "",
                line,
                count=1,
            )

        if not is_bullet and line.endswith(":"):
            continue

        if is_bullet:
            cited = bool(
                _CITATION_PATTERN.search(line)
            )
            text = _CITATION_PATTERN.sub(
                "",
                line,
            ).strip()

            if text:
                claims.append((text, cited))

            continue

        segments = re.findall(
            r".+?(?:[.!?](?:\s*\[[^\[\]]+\])?|$)(?=\s+|$)",
            line,
        )

        for segment in segments:
            cited = bool(
                _CITATION_PATTERN.search(segment)
            )
            text = _CITATION_PATTERN.sub(
                "",
                segment,
            ).strip()

            if text:
                claims.append((text, cited))

    return claims


def diagnose_citation_coverage(
    generation_result: GroundedGenerationResult,
) -> CitationCoverageAutopsyResult:
    claims = _extract_claims(
        generation_result.answer
    )

    total_claims = len(claims)

    if total_claims == 0:
        return CitationCoverageAutopsyResult(
            diagnosis=CitationCoverageDiagnosisType.NO_CLAIMS,
            total_claims=0,
            cited_claims=0,
            coverage=0.0,
            uncited_claims=(),
            explanation=(
                "No factual claims were detected for "
                "citation-coverage analysis."
            ),
        )

    cited_claims = sum(
        1 for _, cited in claims if cited
    )
    coverage = cited_claims / total_claims

    uncited_claims = tuple(
        claim
        for claim, cited in claims
        if not cited
    )

    if cited_claims == total_claims:
        diagnosis = (
            CitationCoverageDiagnosisType.COMPLETE_CITATION_COVERAGE
        )
        explanation = (
            "Every detected claim has an attached citation."
        )
    elif cited_claims == 0:
        diagnosis = (
            CitationCoverageDiagnosisType.NO_CITATION_COVERAGE
        )
        explanation = (
            "None of the detected claims has an attached citation."
        )
    else:
        diagnosis = (
            CitationCoverageDiagnosisType.PARTIAL_CITATION_COVERAGE
        )
        explanation = (
            "Only some detected claims have attached citations."
        )

    return CitationCoverageAutopsyResult(
        diagnosis=diagnosis,
        total_claims=total_claims,
        cited_claims=cited_claims,
        coverage=coverage,
        uncited_claims=uncited_claims,
        explanation=explanation,
    )
