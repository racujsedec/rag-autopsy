from collections import defaultdict

from rag_autopsy.chunking import Chunk
from rag_autopsy.retrieval.bm25 import (
    BM25Retriever,
    SearchResult,
)
from rag_autopsy.retrieval.semantic import SemanticRetriever


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]],
    rrf_k: int = 60,
) -> list[SearchResult]:
    """
    Combine multiple ranked result lists using
    Reciprocal Rank Fusion (RRF).
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0")

    fused_scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.chunk_id

            fused_scores[chunk_id] += (
                1.0 / (rrf_k + rank)
            )

            chunks[chunk_id] = result.chunk

    fused_results = [
        SearchResult(
            score=score,
            chunk=chunks[chunk_id],
        )
        for chunk_id, score in fused_scores.items()
    ]

    fused_results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return fused_results


class HybridRetriever:
    """
    Hybrid lexical + semantic retrieval using RRF.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        rrf_k: int = 60,
        semantic_model_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    ) -> None:
        self.chunks = chunks
        self.rrf_k = rrf_k

        self.bm25 = BM25Retriever(chunks)

        self.semantic = SemanticRetriever(
            chunks,
            model_name=semantic_model_name,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if not self.chunks:
            return []

        candidate_k = len(self.chunks)

        bm25_results = self.bm25.search(
            query,
            top_k=candidate_k,
        )

        semantic_results = self.semantic.search(
            query,
            top_k=candidate_k,
        )

        fused_results = reciprocal_rank_fusion(
            [
                bm25_results,
                semantic_results,
            ],
            rrf_k=self.rrf_k,
        )

        return fused_results[:top_k]
