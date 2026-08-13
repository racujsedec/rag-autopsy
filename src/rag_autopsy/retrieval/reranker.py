from sentence_transformers import CrossEncoder

from rag_autopsy.retrieval.bm25 import SearchResult


class RerankingRetriever:
    """
    Second-stage retriever that reranks candidates
    produced by another retrieval system.
    """

    def __init__(
        self,
        base_retriever,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        candidate_k: int = 6,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        self.model = CrossEncoder(model_name)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        candidate_count = max(
            self.candidate_k,
            top_k,
        )

        candidates = self.base_retriever.search(
            query=query,
            top_k=candidate_count,
        )

        if not candidates:
            return []

        pairs = [
            [query, result.chunk.text]
            for result in candidates
        ]

        reranker_scores = self.model.predict(
            pairs
        )

        reranked_results = [
            SearchResult(
                score=float(score),
                chunk=result.chunk,
            )
            for result, score in zip(
                candidates,
                reranker_scores,
            )
        ]

        reranked_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked_results[:top_k]
