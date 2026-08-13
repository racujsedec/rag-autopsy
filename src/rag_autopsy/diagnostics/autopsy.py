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

class StageComparisonType(str, Enum):
    RERANKER_IMPROVEMENT = "RERANKER_IMPROVEMENT"
    RERANKER_REGRESSION = "RERANKER_REGRESSION"
    NO_CHANGE = "NO_CHANGE"


@dataclass(frozen=True)
class StageComparisonResult:
    diagnosis: StageComparisonType
    before_rank: int | None
    after_rank: int | None
    explanation: str


def compare_reranking_stages(
    before_results: list[SearchResult],
    after_results: list[SearchResult],
    relevant_chunk_ids: list[str],
) -> StageComparisonResult:
    """
    Compare retrieval quality before and after reranking.

    Lower rank is better:
        rank 1 > rank 2 > rank 3 > missing
    """

    relevant = set(relevant_chunk_ids)

    def find_rank(
        results: list[SearchResult],
    ) -> int | None:
        for rank, result in enumerate(
            results,
            start=1,
        ):
            if result.chunk.chunk_id in relevant:
                return rank

        return None

    before_rank = find_rank(before_results)
    after_rank = find_rank(after_results)

    # Missing evidence is treated as worse
    # than any retrieved position.
    before_score = (
        before_rank
        if before_rank is not None
        else float("inf")
    )

    after_score = (
        after_rank
        if after_rank is not None
        else float("inf")
    )

    if after_score < before_score:
        return StageComparisonResult(
            diagnosis=(
                StageComparisonType.RERANKER_IMPROVEMENT
            ),
            before_rank=before_rank,
            after_rank=after_rank,
            explanation=(
                "The reranker improved the position of "
                "the relevant evidence "
                f"from {before_rank} to {after_rank}."
            ),
        )

    if after_score > before_score:
        return StageComparisonResult(
            diagnosis=(
                StageComparisonType.RERANKER_REGRESSION
            ),
            before_rank=before_rank,
            after_rank=after_rank,
            explanation=(
                "The reranker degraded the position of "
                "the relevant evidence "
                f"from {before_rank} to {after_rank}."
            ),
        )

    return StageComparisonResult(
        diagnosis=StageComparisonType.NO_CHANGE,
        before_rank=before_rank,
        after_rank=after_rank,
        explanation=(
            "The reranker did not change the position "
            "of the relevant evidence."
        ),
    )


from rag_autopsy.evaluation import GroundTruthResult


class ChunkingDiagnosisType(str, Enum):
    COMPLETE_EVIDENCE = "COMPLETE_EVIDENCE"
    CHUNK_BOUNDARY_RISK = "CHUNK_BOUNDARY_RISK"
    CHUNK_BOUNDARY_FAILURE = "CHUNK_BOUNDARY_FAILURE"


@dataclass(frozen=True)
class ChunkingAutopsyResult:
    diagnosis: ChunkingDiagnosisType
    evidence_coverage: float
    explanation: str


def diagnose_chunking(
    ground_truth: GroundTruthResult,
    retrieval_results: list[SearchResult],
) -> ChunkingAutopsyResult:
    """
    Diagnose whether chunk boundaries damaged
    the expected evidence passage.

    COMPLETE_EVIDENCE:
        At least one chunk preserves the full evidence.

    CHUNK_BOUNDARY_RISK:
        Evidence is fragmented, but retrieval still
        surfaces the strongest supporting chunk.

    CHUNK_BOUNDARY_FAILURE:
        Evidence is fragmented and retrieval does not
        surface the strongest supporting chunk.
    """

    if ground_truth.complete_evidence_preserved:
        return ChunkingAutopsyResult(
            diagnosis=ChunkingDiagnosisType.COMPLETE_EVIDENCE,
            evidence_coverage=ground_truth.max_coverage,
            explanation=(
                "The expected evidence is preserved "
                "completely within at least one chunk."
            ),
        )

    relevant = set(
        ground_truth.relevant_chunk_ids
    )

    retrieved_ids = {
        result.chunk.chunk_id
        for result in retrieval_results
    }

    if relevant & retrieved_ids:
        return ChunkingAutopsyResult(
            diagnosis=(
                ChunkingDiagnosisType.CHUNK_BOUNDARY_RISK
            ),
            evidence_coverage=ground_truth.max_coverage,
            explanation=(
                "The expected evidence was fragmented "
                "by chunk boundaries, but retrieval still "
                "surfaced the strongest supporting chunk."
            ),
        )

    return ChunkingAutopsyResult(
        diagnosis=(
            ChunkingDiagnosisType.CHUNK_BOUNDARY_FAILURE
        ),
        evidence_coverage=ground_truth.max_coverage,
        explanation=(
            "The expected evidence was fragmented by "
            "chunk boundaries and the strongest supporting "
            "chunk was not retrieved within the evaluation window."
        ),
    )
