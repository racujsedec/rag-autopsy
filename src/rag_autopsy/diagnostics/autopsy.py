from dataclasses import dataclass
from enum import Enum

from rag_autopsy.retrieval import SearchResult


class DiagnosisType(str, Enum):
    SUCCESS = "SUCCESS"
    RANKING_FAILURE = "RANKING_FAILURE"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"


@dataclass(frozen=True)
class AutopsyResult:
    diagnosis: DiagnosisType
    relevant_rank: int | None
    explanation: str


def diagnose_retrieval(
    results: list[SearchResult],
    relevant_chunk_ids: list[str],
) -> AutopsyResult:
    """
    Diagnose the retrieval stage for a single question.

    SUCCESS:
        Relevant evidence is ranked #1.

    RANKING_FAILURE:
        Relevant evidence was retrieved, but ranked below #1.

    RETRIEVAL_MISS:
        Relevant evidence was not retrieved at all.
    """

    relevant = set(relevant_chunk_ids)

    for rank, result in enumerate(results, start=1):
        if result.chunk.chunk_id in relevant:

            if rank == 1:
                return AutopsyResult(
                    diagnosis=DiagnosisType.SUCCESS,
                    relevant_rank=rank,
                    explanation=(
                        "Relevant evidence was retrieved "
                        "and ranked first."
                    ),
                )

            return AutopsyResult(
                diagnosis=DiagnosisType.RANKING_FAILURE,
                relevant_rank=rank,
                explanation=(
                    "Relevant evidence was retrieved, "
                    f"but it was ranked #{rank} instead of #1."
                ),
            )

    return AutopsyResult(
        diagnosis=DiagnosisType.RETRIEVAL_MISS,
        relevant_rank=None,
        explanation=(
            "Relevant evidence did not appear "
            "in the retrieved results."
        ),
    )
