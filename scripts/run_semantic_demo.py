from pathlib import Path

from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig
from rag_autopsy.retrieval import SemanticRetriever


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
        chunks = chunker.chunk(
            document_id=path.stem,
            text=path.read_text(),
        )

        all_chunks.extend(chunks)

    retriever = SemanticRetriever(all_chunks)

    query = "What caused Arcadia's profitability to deteriorate?"

    results = retriever.search(
        query=query,
        top_k=3,
    )

    print("\nSEMANTIC SEARCH")
    print("=" * 80)
    print(f"Query: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(f"RESULT #{rank}")
        print(f"Similarity: {result.score:.4f}")
        print(f"Chunk: {result.chunk.chunk_id}")
        print(f"Text: {result.chunk.text}")
        print("-" * 80)


if __name__ == "__main__":
    main()
