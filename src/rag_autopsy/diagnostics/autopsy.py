import re
from dataclasses import dataclass
from enum import Enum

from rag_autopsy.chunking import Chunk
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


class ContextDiagnosisType(str, Enum):
    CONTEXT_INTACT = "CONTEXT_INTACT"
    CHUNK_CONTEXT_LOSS = "CHUNK_CONTEXT_LOSS"


@dataclass(frozen=True)
class ContextAutopsyResult:
    diagnosis: ContextDiagnosisType
    relevant_chunk_id: str | None
    strongest_neighbor_chunk_id: str | None
    relevant_anchor_overlap: int
    neighbor_anchor_overlap: int
    explanation: str


_CONTEXT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
}


def _context_tokens(
    text: str,
) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text,
        )
    }

    return tokens - _CONTEXT_STOP_WORDS


def _anchor_overlap(
    question_tokens: set[str],
    chunk: Chunk,
) -> int:
    return len(
        question_tokens
        & _context_tokens(chunk.text)
    )


def diagnose_context(
    question: str,
    chunks: list[Chunk],
    ground_truth: GroundTruthResult,
    retrieval_results: list[SearchResult],
) -> ContextAutopsyResult:
    """
    Diagnose whether an evidence chunk lost surrounding
    context that helps connect it to the question.

    CONTEXT_INTACT:
        The relevant chunk was retrieved, or neighboring
        chunks do not contain substantially stronger
        question anchors.

    CHUNK_CONTEXT_LOSS:
        The relevant evidence is preserved completely,
        but the relevant chunk was missed and an adjacent
        chunk from the same document contains substantially
        stronger question anchors.
    """

    relevant_ids = set(
        ground_truth.relevant_chunk_ids
    )

    retrieved_ids = {
        result.chunk.chunk_id
        for result in retrieval_results
    }

    relevant_chunks = [
        chunk
        for chunk in chunks
        if chunk.chunk_id in relevant_ids
    ]

    if not relevant_chunks:
        return ContextAutopsyResult(
            diagnosis=ContextDiagnosisType.CONTEXT_INTACT,
            relevant_chunk_id=None,
            strongest_neighbor_chunk_id=None,
            relevant_anchor_overlap=0,
            neighbor_anchor_overlap=0,
            explanation=(
                "No relevant evidence chunk was available "
                "for context-loss analysis."
            ),
        )

    if relevant_ids & retrieved_ids:
        relevant_chunk = relevant_chunks[0]

        return ContextAutopsyResult(
            diagnosis=ContextDiagnosisType.CONTEXT_INTACT,
            relevant_chunk_id=relevant_chunk.chunk_id,
            strongest_neighbor_chunk_id=None,
            relevant_anchor_overlap=0,
            neighbor_anchor_overlap=0,
            explanation=(
                "The relevant evidence chunk was retrieved, "
                "so contextual anchors were sufficient for "
                "the evaluated retrieval window."
            ),
        )

    question_tokens = _context_tokens(
        question
    )

    best_relevant = None
    best_neighbor = None
    best_relevant_overlap = 0
    best_neighbor_overlap = 0

    for relevant_chunk in relevant_chunks:
        same_document = sorted(
            (
                chunk
                for chunk in chunks
                if chunk.document_id
                == relevant_chunk.document_id
            ),
            key=lambda chunk: (
                chunk.start_word,
                chunk.end_word,
            ),
        )

        try:
            index = same_document.index(
                relevant_chunk
            )
        except ValueError:
            continue

        neighbors = []

        if index > 0:
            neighbors.append(
                same_document[index - 1]
            )

        if index + 1 < len(same_document):
            neighbors.append(
                same_document[index + 1]
            )

        relevant_overlap = _anchor_overlap(
            question_tokens,
            relevant_chunk,
        )

        for neighbor in neighbors:
            neighbor_overlap = _anchor_overlap(
                question_tokens,
                neighbor,
            )

            if (
                best_neighbor is None
                or neighbor_overlap
                > best_neighbor_overlap
            ):
                best_relevant = relevant_chunk
                best_neighbor = neighbor
                best_relevant_overlap = (
                    relevant_overlap
                )
                best_neighbor_overlap = (
                    neighbor_overlap
                )

    context_loss = (
        ground_truth.complete_evidence_preserved
        and best_neighbor is not None
        and best_neighbor_overlap >= 2
        and best_neighbor_overlap
        > best_relevant_overlap
    )

    if context_loss:
        return ContextAutopsyResult(
            diagnosis=(
                ContextDiagnosisType.CHUNK_CONTEXT_LOSS
            ),
            relevant_chunk_id=(
                best_relevant.chunk_id
                if best_relevant
                else None
            ),
            strongest_neighbor_chunk_id=(
                best_neighbor.chunk_id
            ),
            relevant_anchor_overlap=(
                best_relevant_overlap
            ),
            neighbor_anchor_overlap=(
                best_neighbor_overlap
            ),
            explanation=(
                "The answer evidence is preserved, but an "
                "adjacent chunk contains stronger question "
                "anchors than the evidence chunk. The chunk "
                "boundary likely separated identifying "
                "context from the answer."
            ),
        )

    relevant_chunk = (
        best_relevant
        if best_relevant is not None
        else relevant_chunks[0]
    )

    return ContextAutopsyResult(
        diagnosis=ContextDiagnosisType.CONTEXT_INTACT,
        relevant_chunk_id=relevant_chunk.chunk_id,
        strongest_neighbor_chunk_id=(
            best_neighbor.chunk_id
            if best_neighbor is not None
            else None
        ),
        relevant_anchor_overlap=(
            best_relevant_overlap
        ),
        neighbor_anchor_overlap=(
            best_neighbor_overlap
        ),
        explanation=(
            "Adjacent chunks do not contain substantially "
            "stronger question anchors than the relevant "
            "evidence chunk."
        ),
    )
