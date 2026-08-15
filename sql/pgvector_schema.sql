CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    text TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    start_word INTEGER NOT NULL,
    end_word INTEGER NOT NULL,
    embedding vector(384) NOT NULL
);

ALTER TABLE rag_chunks
ADD COLUMN IF NOT EXISTS retrieval_text TEXT;

UPDATE rag_chunks
SET retrieval_text = text
WHERE retrieval_text IS NULL;

ALTER TABLE rag_chunks
ALTER COLUMN retrieval_text SET NOT NULL;
