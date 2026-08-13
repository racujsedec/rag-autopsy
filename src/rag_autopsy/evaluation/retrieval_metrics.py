from rag_autopsy.retrieval import SearchResult


def recall_at_k(
    results: list[SearchResult],
    relevant_document: str,
    k: int,
) -> float:
    """
    Return 1.0 when a relevant document appears
    within the top-k results, otherwise 0.0.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_results = results[:k]

    for result in top_results:
        if result.chunk.document_id == relevant_document:
            return 1.0

    return 0.0
