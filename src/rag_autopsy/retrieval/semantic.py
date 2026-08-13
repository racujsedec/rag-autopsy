from sentence_transformers import SentenceTransformer, util

from rag_autopsy.chunking import Chunk
from rag_autopsy.retrieval.bm25 import SearchResult


class SemanticRetriever:
    """Embedding-based semantic retrieval baseline."""

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        texts = [chunk.text for chunk in chunks]

        self.corpus_embeddings = self.model.encode_document(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not self.chunks:
            return []

        query_embedding = self.model.encode_query(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = util.cos_sim(
            query_embedding,
            self.corpus_embeddings,
        )[0]

        number_of_results = min(
            top_k,
            len(self.chunks),
        )

        top_results = scores.topk(
            k=number_of_results,
        )

        results = []

        for score, index in zip(
            top_results.values,
            top_results.indices,
        ):
            results.append(
                SearchResult(
                    score=float(score),
                    chunk=self.chunks[int(index)],
                )
            )

        return results
