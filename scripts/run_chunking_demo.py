from pathlib import Path

from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig


def main() -> None:
    raw_dir = Path("data/raw")
    chunker = FixedSizeChunker(ChunkingConfig(chunk_size=45, overlap=8))
    for path in sorted(raw_dir.glob("*.txt")):
        text = path.read_text()
        chunks = chunker.chunk(path.stem, text)
        print(f"\n{path.name}: {len(chunks)} chunks")
        for chunk in chunks:
            preview = chunk.text[:90].replace("\n", " ")
            print(f"  {chunk.chunk_id} [{chunk.start_word}:{chunk.end_word}] {preview}...")

if __name__ == "__main__":
    main()
