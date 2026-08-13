from pathlib import Path

from rag_autopsy.chunking import (
    FixedSizeChunker,
    ParagraphChunker,
)
from rag_autopsy.config import ChunkingConfig


def print_chunks(
    title: str,
    chunks,
) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for chunk in chunks:
        print(f"\n{chunk.chunk_id}")
        print(
            f"Words: "
            f"{chunk.start_word}:{chunk.end_word}"
        )
        print(chunk.text)


def main() -> None:
    path = Path(
        "data/raw/retail_ops.txt"
    )

    text = path.read_text()

    fixed_chunker = FixedSizeChunker(
        ChunkingConfig(
            chunk_size=45,
            overlap=8,
        )
    )

    paragraph_chunker = ParagraphChunker(
        max_words=120,
    )

    fixed_chunks = fixed_chunker.chunk(
        document_id="retail_ops",
        text=text,
    )

    paragraph_chunks = paragraph_chunker.chunk(
        document_id="retail_ops",
        text=text,
    )

    print_chunks(
        "FIXED-SIZE CHUNKING",
        fixed_chunks,
    )

    print_chunks(
        "PARAGRAPH-AWARE CHUNKING",
        paragraph_chunks,
    )


if __name__ == "__main__":
    main()
