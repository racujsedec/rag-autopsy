import re

from rag_autopsy.chunking.fixed import Chunk


class ParagraphChunker:
    """
    Split documents on paragraph boundaries instead of
    blindly cutting every N words.
    """

    def __init__(
        self,
        max_words: int = 120,
    ) -> None:
        if max_words <= 0:
            raise ValueError(
                "max_words must be greater than 0"
            )

        self.max_words = max_words

    def chunk(
        self,
        document_id: str,
        text: str,
    ) -> list[Chunk]:

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                text,
            )
            if paragraph.strip()
        ]

        chunks = []
        chunk_number = 0
        word_cursor = 0

        for paragraph in paragraphs:
            words = paragraph.split()

            # If one paragraph itself is too large,
            # split that paragraph into smaller pieces.
            for start in range(
                0,
                len(words),
                self.max_words,
            ):
                piece = words[
                    start:start + self.max_words
                ]

                if not piece:
                    continue

                chunk_text = " ".join(piece)

                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{document_id}::paragraph-"
                            f"{chunk_number:04d}"
                        ),
                        document_id=document_id,
                        text=chunk_text,
                        start_word=word_cursor + start,
                        end_word=(
                            word_cursor
                            + start
                            + len(piece)
                        ),
                    )
                )

                chunk_number += 1

            word_cursor += len(words)

        return chunks
