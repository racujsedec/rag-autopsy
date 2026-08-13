from rag_autopsy.retrieval import SearchResult


def recall_at_k(
    results: list[SearchResult],
    relevant_chunk_ids: list[str],
    k: int,
) -> float:
    """
    Return 1.0 if at least one relevant evidence chunk
    appears within the top-k results.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant_chunk_ids)

    for result in results[:k]:
        if result.chunk.chunk_id in relevant:
            return 1.0

    return 0.0


def reciprocal_rank(
    results: list[SearchResult],
    relevant_chunk_ids: list[str],
) -> float:
    """
    Return the reciprocal rank of the first relevant chunk.

    Example:
        relevant chunk at rank 1 -> 1.0
        relevant chunk at rank 2 -> 0.5
        relevant chunk at rank 3 -> 0.333...
        no relevant chunk        -> 0.0
    """

    relevant = set(relevant_chunk_ids)

    for rank, result in enumerate(results, start=1):
        if result.chunk.chunk_id in relevant:
            return 1.0 / rank

    return 0.0
