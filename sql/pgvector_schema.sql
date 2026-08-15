CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    start_word INTEGER NOT NULL,
    end_word INTEGER NOT NULL,
    embedding vector(384) NOT NULL
);
