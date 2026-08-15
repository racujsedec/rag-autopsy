import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from rag_autopsy.chunking import Chunk


def tokenize(text: str) -> list[str]:
    """Normalize text into lowercase word tokens."""
    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )


def evidence_coverage(
    chunk_text: str,
    evidence_text: str,
) -> float:
    """
    Measure how much of the expected evidence appears
    contiguously inside a chunk.

    Returns a value between 0.0 and 1.0.

    1.0 means the entire evidence passage is preserved
    inside one chunk.
    """

    chunk_tokens = tokenize(chunk_text)
    evidence_tokens = tokenize(evidence_text)

    if not evidence_tokens:
        return 0.0

    matcher = SequenceMatcher(
        None,
        evidence_tokens,
        chunk_tokens,
        autojunk=False,
    )

    longest_match = matcher.find_longest_match(
        0,
        len(evidence_tokens),
        0,
        len(chunk_tokens),
    )

    return longest_match.size / len(evidence_tokens)


@dataclass(frozen=True)
class GroundTruthResult:
    relevant_chunk_ids: list[str]
    supporting_chunk_ids: list[str]
    max_coverage: float
    complete_evidence_preserved: bool


def resolve_ground_truth(
    chunks: list[Chunk],
    evidence_text: str,
    support_threshold: float = 0.50,
) -> GroundTruthResult:
    """
    Resolve evidence text against chunks created by any
    chunking strategy.

    relevant_chunk_ids:
        Chunk(s) with the highest evidence coverage.

    supporting_chunk_ids:
        Chunks containing at least support_threshold of
        the expected evidence.

    complete_evidence_preserved:
        True when at least one chunk contains the entire
        evidence passage.
    """

    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError(
            "support_threshold must be between 0 and 1"
        )

    if not chunks:
        return GroundTruthResult(
            relevant_chunk_ids=[],
            supporting_chunk_ids=[],
            max_coverage=0.0,
            complete_evidence_preserved=False,
        )

    coverage_by_chunk = {
        chunk.chunk_id: evidence_coverage(
            chunk.text,
            evidence_text,
        )
        for chunk in chunks
    }

    max_coverage = max(
        coverage_by_chunk.values()
    )

    relevant_chunk_ids = [
        chunk_id
        for chunk_id, coverage
        in coverage_by_chunk.items()
        if abs(coverage - max_coverage) < 1e-9
        and coverage > 0
    ]

    supporting_chunk_ids = [
        chunk_id
        for chunk_id, coverage
        in coverage_by_chunk.items()
        if coverage >= support_threshold
    ]

    return GroundTruthResult(
        relevant_chunk_ids=relevant_chunk_ids,
        supporting_chunk_ids=supporting_chunk_ids,
        max_coverage=max_coverage,
        complete_evidence_preserved=(
            max_coverage >= 0.999
        ),
    )
