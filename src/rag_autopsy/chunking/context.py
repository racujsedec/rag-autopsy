from collections import defaultdict

from rag_autopsy.chunking.fixed import Chunk


class PreviousChunkContextEnricher:
    """
    Add the previous chunk's text as retrieval context
    while preserving the current chunk's identity and
    original word boundaries.

    Context never crosses document boundaries.
    """

    def enrich(
        self,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        chunks_by_document: dict[
            str,
            list[Chunk],
        ] = defaultdict(list)

        for chunk in chunks:
            chunks_by_document[
                chunk.document_id
            ].append(chunk)

        enriched_by_id: dict[str, Chunk] = {}

        for document_chunks in (
            chunks_by_document.values()
        ):
            ordered_chunks = sorted(
                document_chunks,
                key=lambda chunk: (
                    chunk.start_word,
                    chunk.end_word,
                ),
            )

            for index, chunk in enumerate(
                ordered_chunks
            ):
                if index == 0:
                    enriched_text = chunk.text
                else:
                    previous_chunk = (
                        ordered_chunks[index - 1]
                    )

                    enriched_text = (
                        previous_chunk.text
                        + "\n\n"
                        + chunk.text
                    )

                enriched_by_id[
                    chunk.chunk_id
                ] = Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=enriched_text,
                    start_word=chunk.start_word,
                    end_word=chunk.end_word,
                )

        return [
            enriched_by_id[chunk.chunk_id]
            for chunk in chunks
        ]
