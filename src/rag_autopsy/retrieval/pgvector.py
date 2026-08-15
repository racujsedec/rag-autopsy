from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from rag_autopsy.chunking import Chunk
from rag_autopsy.retrieval.bm25 import SearchResult


class PgVectorRetriever:
    """PostgreSQL + pgvector semantic retriever."""

    def __init__(
        self,
        connection,
        model=None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.connection = connection

        register_vector(
            self.connection
        )

        self.model = (
            model
            if model is not None
            else SentenceTransformer(model_name)
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

        query_embedding = (
            self.model.encode_query(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    text,
                    start_word,
                    end_word,
                    1 - (embedding <=> %s) AS score
                FROM rag_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    top_k,
                ),
            )

            rows = cursor.fetchall()

        return [
            SearchResult(
                score=float(row[5]),
                chunk=Chunk(
                    chunk_id=row[0],
                    document_id=row[1],
                    text=row[2],
                    start_word=row[3],
                    end_word=row[4],
                ),
            )
            for row in rows
        ]
