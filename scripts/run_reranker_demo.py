from pathlib import Path

from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig
from rag_autopsy.retrieval import (
    HybridRetriever,
    RerankingRetriever,
)


def main():
    raw_dir = Path("data/raw")

    chunker = FixedSizeChunker(
        ChunkingConfig(
            chunk_size=45,
            overlap=8,
        )
    )

    chunks = []

    for path in sorted(raw_dir.glob("*.txt")):
        chunks.extend(
            chunker.chunk(
                document_id=path.stem,
                text=path.read_text(),
            )
        )

    hybrid = HybridRetriever(chunks)

    reranker = RerankingRetriever(
        base_retriever=hybrid,
        candidate_k=6,
    )

    query = (
        "How did engineers mitigate "
        "the data platform incident?"
    )

    print("\nHYBRID BEFORE RERANKING")
    print("=" * 80)

    hybrid_results = hybrid.search(
        query,
        top_k=3,
    )

    for rank, result in enumerate(
        hybrid_results,
        start=1,
    ):
        print(
            f"#{rank} "
            f"{result.chunk.chunk_id} "
            f"{result.score:.4f}"
        )

    print("\nAFTER CROSS-ENCODER RERANKING")
    print("=" * 80)

    reranked_results = reranker.search(
        query,
        top_k=3,
    )

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):
        print(
            f"#{rank} "
            f"{result.chunk.chunk_id} "
            f"{result.score:.4f}"
        )

        print(result.chunk.text)
        print("-" * 80)


if __name__ == "__main__":
    main()
