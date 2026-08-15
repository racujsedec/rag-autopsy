import os
from pathlib import Path

import psycopg

from rag_autopsy.chunking import ParagraphChunker, PreviousChunkContextEnricher
from rag_autopsy.indexing import PgVectorIndexer


def load_chunks():
    chunker = ParagraphChunker(
        max_words=120
    )

    chunks = []

    for path in sorted(
        Path("data/raw").glob("*.txt")
    ):
        chunks.extend(
            chunker.chunk(
                document_id=path.stem,
                text=path.read_text(),
            )
        )

    return chunks


def main():
    database_url = os.getenv(
        "RAG_AUTOPSY_DATABASE_URL",
        "dbname=rag_autopsy",
    )

    chunks = load_chunks()

    retrieval_chunks = (
        PreviousChunkContextEnricher().enrich(
            chunks
        )
    )

    print(
        f"Loaded {len(chunks)} paragraph chunks."
    )

    with psycopg.connect(
        database_url
    ) as connection:
        indexer = PgVectorIndexer(
            connection=connection
        )

        count = indexer.upsert_chunks(
            chunks,
            retrieval_chunks=retrieval_chunks,
        )

    print(
        f"Upserted {count} chunks into pgvector."
    )


if __name__ == "__main__":
    main()
