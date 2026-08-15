from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from rag_autopsy.chunking import Chunk


class PgVectorIndexer:
    """Persist canonical chunks and retrieval embeddings in pgvector."""

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
        retrieval_chunks: list[Chunk] | None = None,
    ) -> int:
        if not chunks:
            return 0

        if retrieval_chunks is None:
            retrieval_chunks = chunks

        canonical_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        retrieval_ids = [
            chunk.chunk_id
            for chunk in retrieval_chunks
        ]

        if (
            len(canonical_ids) != len(retrieval_ids)
            or set(canonical_ids) != set(retrieval_ids)
        ):
            raise ValueError(
                "retrieval chunk ids must match canonical chunk ids"
            )

        retrieval_by_id = {
            chunk.chunk_id: chunk
            for chunk in retrieval_chunks
        }

        ordered_retrieval_chunks = [
            retrieval_by_id[chunk.chunk_id]
            for chunk in chunks
        ]

        retrieval_texts = [
            chunk.text
            for chunk in ordered_retrieval_chunks
        ]

        embeddings = self.model.encode_document(
            retrieval_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        rows = [
            (
                canonical_chunk.chunk_id,
                canonical_chunk.document_id,
                canonical_chunk.text,
                retrieval_chunk.text,
                canonical_chunk.start_word,
                canonical_chunk.end_word,
                embedding,
            )
            for (
                canonical_chunk,
                retrieval_chunk,
                embedding,
            ) in zip(
                chunks,
                ordered_retrieval_chunks,
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
                    retrieval_text,
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
                    %s,
                    %s
                )
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    text = EXCLUDED.text,
                    retrieval_text = EXCLUDED.retrieval_text,
                    start_word = EXCLUDED.start_word,
                    end_word = EXCLUDED.end_word,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )

        self.connection.commit()

        return len(rows)
