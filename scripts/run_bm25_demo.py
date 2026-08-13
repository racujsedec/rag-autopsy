from pathlib import Path

from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig
from rag_autopsy.retrieval.bm25 import BM25Retriever


def main() -> None:
    raw_dir = Path("data/raw")

    chunker = FixedSizeChunker(
        ChunkingConfig(
            chunk_size=45,
            overlap=8,
        )
    )

    all_chunks = []

    for path in sorted(raw_dir.glob("*.txt")):
        text = path.read_text()

        chunks = chunker.chunk(
            document_id=path.stem,
            text=text,
        )

        all_chunks.extend(chunks)

    retriever = BM25Retriever(all_chunks)

    query = "Why did Arcadia Components operating margin decline?"

    results = retriever.search(
        query=query,
        top_k=3,
    )

    print(f"\nQUERY: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(f"RESULT #{rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Document: {result.chunk.document_id}")
        print(f"Chunk: {result.chunk.chunk_id}")
        print(f"Text: {result.chunk.text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
