from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from rag_autopsy.chunking import Chunk


class PgVectorIndexer:
    """Persist chunk embeddings in PostgreSQL with pgvector."""

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

    def upsert_chunks(
        self,
        chunks: list[Chunk],
    ) -> int:
        if not chunks:
            return 0

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = (
            self.model.encode_document(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )

        rows = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.text,
                chunk.start_word,
                chunk.end_word,
                embedding,
            )
            for chunk, embedding in zip(
                chunks,
                embeddings,
                strict=True,
            )
        ]

        with self.connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO rag_chunks (
                    chunk_id,
                    document_id,
                    text,
                    start_word,
                    end_word,
                    embedding
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    text = EXCLUDED.text,
                    start_word = EXCLUDED.start_word,
                    end_word = EXCLUDED.end_word,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )

        self.connection.commit()

        return len(rows)
