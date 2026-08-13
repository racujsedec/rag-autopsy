from dataclasses import dataclass
from rag_autopsy.config import ChunkingConfig

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    start_word: int
    end_word: int

class FixedSizeChunker:
    """Simple word-based chunker used as the Phase 1 baseline."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document_id: str, text: str) -> list[Chunk]:
        words = text.split()
        if not words:
            return []
        chunks: list[Chunk] = []
        step = self.config.chunk_size - self.config.overlap
        chunk_number = 0
        for start in range(0, len(words), step):
            end = min(start + self.config.chunk_size, len(words))
            chunk_words = words[start:end]
            if not chunk_words:
                break
            chunks.append(Chunk(
                chunk_id=f"{document_id}::chunk-{chunk_number:04d}",
                document_id=document_id,
                text=" ".join(chunk_words),
                start_word=start,
                end_word=end,
            ))
            chunk_number += 1
            if end == len(words):
                break
        return chunks
